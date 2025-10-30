from __future__ import annotations
import os
import json
import math
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import ee
import geemap
import geopandas as gpd
from pyproj import CRS

try:
    import rasterio  # NEW
    from rasterio.warp import transform_bounds  # NEW
except Exception as _e:
    rasterio = None  # NEW

# ===== NEW: 读取 TIF 辅助 =====
# 仅用于 tif 模式下构造 ROI
# --------------------------------
# 安装依赖：rasterio
try:
    import rasterio  # NEW
    from rasterio.warp import transform_bounds  # NEW
except Exception as _e:
    rasterio = None  # NEW


# ============================
# 0) 读取 JSON & 初始化 EE
# ============================
def read_service_account_json(json_path: str) -> Tuple[str, Optional[str]]:
    """
    从 Service Account JSON 文件解析 client_email 与 project_id。
    返回: (client_email, project_id or None)
    """
    p = Path(json_path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Service account JSON not found: {p}")

    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)

    client_email = data.get("client_email")
    project_id = data.get("project_id")

    if not client_email:
        raise ValueError(f"'client_email' not found in JSON: {p}")
    print(f"从 {p} 读取凭据: {client_email}, project_id={project_id}")
    return client_email, project_id


def ee_init_from_json(json_path: str) -> Tuple[str, str]:
    """用 JSON 初始化 Earth Engine，并做一次轻量连通性测试。"""
    client_email, project_id = read_service_account_json(json_path)
    creds = ee.ServiceAccountCredentials(client_email, str(Path(json_path).expanduser().resolve()))
    if project_id:
        ee.Initialize(creds, project=project_id)
    else:
        ee.Initialize(creds)

    _ = ee.Number(1).getInfo()  # ping 一下
    print(f"EE 初始化完成 | sa={client_email} | project={project_id or '<none>'}")
    return client_email, (project_id or "")


def infer_utm_epsg_from_lonlat(lon: float, lat: float) -> str:
    zone = int(math.floor((lon + 180) / 6) + 1)
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return f"EPSG:{epsg}"


# ============================
# 1) 完全对齐 ipynb 的 8-bit 可视化
# ============================
def build_and_visualize_s2_rgb8_exact(roi_geom,
                                      s2_start: str,
                                      s2_end: str,
                                      cloud_max: int,
                                      *,
                                      use_harmonized: bool = False,
                                      pct_bounds=(2, 98)) -> ee.Image:
    """
    与 ipynb 保持一致：
    - 集合：COPERNICUS/S2_SR（默认）
    - 去云：仅 SCL 的 3/8/9/10
    - 合成：median
    - 拉伸：在 ROI 上做 2–98% 百分位；用 ee.Image.visualize 生成 8-bit RGB
    返回：uint8（3-band）影像
    """
    s2_col = 'COPERNICUS/S2_SR' if not use_harmonized else 'COPERNICUS/S2_SR_HARMONIZED'

    def mask_scl(image):
        scl = image.select('SCL')
        cloud = scl.eq(3).Or(scl.eq(8)).Or(scl.eq(9)).Or(scl.eq(10))  # 3=shadow, 8/9/10=clouds
        # 只保留原始反射率波段 + SCL，用 SCL 掩云，并做 DN->Reflectance
        return (
            image.select(['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12','SCL'])
                 .updateMask(cloud.Not())
                 .divide(10000.0)
                 .copyProperties(image, ['system:time_start'])
        )

    img = (ee.ImageCollection(s2_col)
           .filterDate(s2_start, s2_end)
           .filterBounds(roi_geom)
           .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', cloud_max))
           .map(mask_scl)
           .median()
           .clip(roi_geom))

    rgb_float = img.select(['B4','B3','B2'])  # 可见光

    # 在原始 ROI 上统计百分位（与 ipynb 保持一致）
    p_lo, p_hi = pct_bounds
    stats = rgb_float.reduceRegion(
        reducer=ee.Reducer.percentile([p_lo, p_hi]),
        geometry=roi_geom,
        scale=20,
        bestEffort=True
    )

    vis_params = {
        'bands': ['B4','B3','B2'],
        'min': [
            ee.Number(stats.get(f'B4_p{p_lo}')),
            ee.Number(stats.get(f'B3_p{p_lo}')),
            ee.Number(stats.get(f'B2_p{p_lo}')),
        ],
        'max': [
            ee.Number(stats.get(f'B4_p{p_hi}')),
            ee.Number(stats.get(f'B3_p{p_hi}')),
            ee.Number(stats.get(f'B2_p{p_hi}')),
        ],
    }
    # 关键：服务器端 visualize → 直接 8-bit，避免本地量化差异导致“炸白”
    rgb8 = rgb_float.visualize(**vis_params)
    return rgb8  # uint8, 3-band


# ============================
# 2) DynamicWorld 选地块
# ============================
def select_polygons_in_roi(cadastre_asset_id: str,
                                      roi_geom,
                                      id_col: str):
    
    cadastre_fc = ee.FeatureCollection(cadastre_asset_id)
    cad_in_roi = cadastre_fc.filterBounds(roi_geom)
    id_list = cad_in_roi.aggregate_array(id_col)
    return cad_in_roi, id_list


# ===== NEW: ROI 构造器（bbox / tif 两种模式） =====
def _meters_per_degree(lat_deg: float) -> Tuple[float, float]:
    """估算该纬度处每度的米数 (lon_m_per_deg, lat_m_per_deg)。简化近似即可。"""
    from math import cos, pi
    lat_rad = lat_deg * pi / 180.0
    lon_m = 111320.0 * cos(lat_rad)
    lat_m = 110574.0
    return lon_m, lat_m


def _tif_bounds_wgs84_and_pixm(tif_path: str) -> Tuple[Tuple[float,float,float,float], float, Tuple[float,float]]:
    """
    读取 TIF 的外接矩形（转为 WGS84），估算像元米尺度。
    返回: (minx, miny, maxx, maxy) in WGS84, pixel_size_m (平均), (center_lon, center_lat)
    """
    if rasterio is None:
        raise RuntimeError("rasterio 未安装，无法在 tif 模式下构造 ROI。")
    if not os.path.exists(tif_path):
        raise FileNotFoundError(f"TIF 不存在: {tif_path}")

    with rasterio.open(tif_path) as ds:
        if ds.count < 1:
            raise ValueError(f"TIF 无有效波段: {tif_path}")
        b = ds.bounds
        src_crs = ds.crs
        if src_crs is None:
            raise ValueError(f"TIF 缺失 CRS: {tif_path}")
        # 转为 WGS84
        bounds_wgs84 = transform_bounds(src_crs, "EPSG:4326", b.left, b.bottom, b.right, b.top, densify_pts=21)

        # 估算像元米尺度
        # 若是投影坐标（米），直接取仿射的像元边长；若是经纬度，按纬度做度->米近似
        a = ds.transform.a
        e = ds.transform.e  # 通常为负，取绝对值
        pix_dx = abs(a)
        pix_dy = abs(e)
        center_lon = (bounds_wgs84[0] + bounds_wgs84[2]) / 2.0
        center_lat = (bounds_wgs84[1] + bounds_wgs84[3]) / 2.0

        if src_crs.is_projected:
            pixel_size_m = (pix_dx + pix_dy) / 2.0
        else:
            lon_m_per_deg, lat_m_per_deg = _meters_per_degree(center_lat)
            pixel_size_m = (pix_dx * lon_m_per_deg + pix_dy * lat_m_per_deg) / 2.0

    minx, miny, maxx, maxy = bounds_wgs84
    return (minx, miny, maxx, maxy), pixel_size_m, (center_lon, center_lat)


def make_rois(*,
              roi_mode: str,                # "bbox" | "tif"
              bbox: Optional[tuple] = None, # (minx, miny, maxx, maxy) in WGS84
              user_tif: Optional[str] = None,
              pad_m: int = 5000,
              shrink_margin_m: Optional[float] = None,
              shrink_margin_pixels: Optional[int] = 3) -> Tuple[ee.Geometry, ee.Geometry, Dict[str, Any]]:
    """
    返回: (roi, export_region, info)
    - bbox 模式：roi = Rectangle(bbox)；export_region = roi.buffer(+pad_m).bounds()
    - tif 模式：roi = Rectangle(tif_bounds_wgs84).buffer(-margin_m).bounds()；export_region = roi
    info: {mode, bbox_wgs84, margin_m, pixel_size_m, center}
    """
    if roi_mode not in ("bbox", "tif"):
        raise ValueError("roi_mode 仅支持 'bbox' 或 'tif'")

    info: Dict[str, Any] = {"mode": roi_mode}

    if roi_mode == "bbox":
        if not bbox or len(bbox) != 4:
            raise ValueError("bbox 模式需要提供 bbox=(minx, miny, maxx, maxy) (WGS84)")
        minx, miny, maxx, maxy = bbox
        info.update({"bbox_wgs84": bbox, "margin_m": pad_m})
        roi = ee.Geometry.Rectangle([minx, miny, maxx, maxy])
        export_region = roi.buffer(pad_m).bounds()
        return roi, export_region, info

    # tif 模式
    if not user_tif:
        raise ValueError("tif 模式需要提供 user_tif 路径")
    bbox_wgs84, pixel_size_m, center = _tif_bounds_wgs84_and_pixm(user_tif)
    minx, miny, maxx, maxy = bbox_wgs84
    roi_base = ee.Geometry.Rectangle([minx, miny, maxx, maxy])

    # 计算向内收缩的 margin（优先使用米；否则按像元数量折算）
    if shrink_margin_m is not None:
        margin_m = float(shrink_margin_m)
    else:
        px = shrink_margin_pixels if shrink_margin_pixels is not None else 0
        margin_m = max(0.0, px * float(pixel_size_m))

    # 兜底：避免收缩过大导致几何为空
    roi_shrunk = roi_base.buffer(-margin_m).bounds() if margin_m > 0 else roi_base
    # 注意：EE 的 buffer 单位为米，这里即使几何是 WGS84，依然用米做缓冲

    info.update({
        "bbox_wgs84": bbox_wgs84,
        "pixel_size_m": pixel_size_m,
        "margin_m": margin_m,
        "center": {"lon": center[0], "lat": center[1]}
    })
    roi = roi_shrunk
    export_region = roi  # tif 模式通常无需另外的导出范围
    return roi, export_region, info


# ============================
# 3) 顶层封装：完全按 ipynb 流程导出 8-bit
# ============================
def run_gee_fetch(
    *,
    cadastre_asset: str,
    id_col: str,
    bbox: tuple = None,               # (minx, miny, maxx, maxy) in WGS84  # CHANGED: 默认 None
    json_path: str,
    s2_start: str,
    s2_end: str,
    cloud_max: int = 40,
    dw_start: Optional[str] = None,
    dw_end: Optional[str] = None,
    farmland_th: float = 0.25,
    out_tif: str = "./gee_out/S2_RGB8.tif",
    out_ids: str = "./gee_out/id_list.txt",
    pad_m: int = 5000,
    crs_out: str = "auto",
    # ===== NEW: ROI 模式与 tif 参数 =====
    roi_mode: str = "bbox",                 # "bbox" | "tif"   # NEW
    user_tif: Optional[str] = None,         # NEW
    shrink_margin_m: Optional[float] = None,# NEW
    shrink_margin_pixels: Optional[int] = 3 # NEW
) -> Dict[str, Any]:
    """
    主流程（完全复刻 ipynb 的可视化导出链路）：
    - S2_SR + SCL 掩云 + median
    - 在原始 ROI 上统计 2–98% 并 visualize 成 8-bit
    - 下载到 out_tif（bbox 模式：region 用 expanded_roi；tif 模式跳过下载）
    - DW 按 4/7 估算农地占比，写出 id_list 和 filter_lot.gpkg
    """
    ee_init_from_json(json_path=json_path)

    # ===== CHANGED: 统一通过 ROI 构造器生成 roi / export_region =====
    roi, export_region, roi_info = make_rois(
        roi_mode=roi_mode,
        bbox=bbox,
        user_tif=user_tif,
        pad_m=pad_m,
        shrink_margin_m=shrink_margin_m,
        shrink_margin_pixels=shrink_margin_pixels
    )
    print(f"[ROI] mode={roi_info['mode']} | bbox(WGS84)={roi_info['bbox_wgs84']} | margin_m={roi_info['margin_m']}")

    # 输出坐标系（仅 bbox 模式用于下载 S2）
    if crs_out.lower() == "auto" and roi_mode == "bbox":  # CHANGED: 仅 bbox 需要导出 CRS
        minx, miny, maxx, maxy = roi_info["bbox_wgs84"]
        crs_out_val = infer_utm_epsg_from_lonlat((minx + maxx) / 2, (miny + maxy) / 2)
    elif roi_mode == "bbox":
        CRS.from_user_input(crs_out)
        crs_out_val = crs_out
    else:
        crs_out_val = None  # tif 模式不下载 S2，无需导出 CRS
    if roi_mode == "bbox":
        print(f"[CRS] 输出：{crs_out_val} | 外扩：{pad_m} m")

    # 选地块 & 写 id 列表（保持原逻辑）
    selected_polys, ids = select_polygons_in_roi(
        cadastre_asset_id=cadastre_asset, roi_geom=roi, id_col=id_col
    )
    id_list_py = ids.getInfo()
    print(f"[IDs] 命中 {len(id_list_py)} 个地块 | 前10：{id_list_py[:10]}")

    os.makedirs(os.path.dirname(out_ids), exist_ok=True)
    with open(out_ids, "w", encoding="utf-8") as f:
        f.write("\n".join(map(str, id_list_py)))
    print(f"[IDs] 已写入：{out_ids}")

    # ===== CHANGED: bbox 模式下保持下载 S2；tif 模式下跳过下载（用户自带影像）
    if roi_mode == "bbox":
        # ★ 完全对齐 ipynb：服务端 visualize → 8-bit
        rgb8_exact = build_and_visualize_s2_rgb8_exact(
            roi, s2_start, s2_end, cloud_max,
            use_harmonized=False, pct_bounds=(2, 98)
        ).clip(export_region)

        os.makedirs(os.path.dirname(out_tif), exist_ok=True)
        print(f"[DL] 下载影像（8-bit）：{out_tif}")

        geemap.download_ee_image(
            image=rgb8_exact,
            filename=out_tif,
            region=export_region,
            scale=10,
            crs=crs_out_val,
            overwrite=True   # visualize 已经是 uint8，这里不要再指定 dtype
        )
        print("Sentinel-2 RGB8 下载完成")
        tif_to_report = out_tif
        base_dir = os.path.dirname(out_tif)
    else:
        print("[DL] tif 模式：跳过 S2 下载，使用用户上传影像作为参考范围")
        tif_to_report = user_tif
        base_dir = os.path.dirname(user_tif) if user_tif else "."

    # 导出筛选地块 GPKG（与影像同目录或与 user_tif 同目录）
    gdf = geemap.ee_to_gdf(selected_polys)
    gpkg_path = os.path.join(base_dir, "filter_lot.gpkg")  # CHANGED: 目录取决于模式
    gdf.to_file(gpkg_path, driver="GPKG")
    print(f"已导出地籍：{gpkg_path}")

    return {
        "tif": tif_to_report,            # CHANGED: bbox=out_tif；tif=user_tif
        "ids": out_ids,
        "gpkg": gpkg_path,
        "id_list": id_list_py,
        "roi_info": roi_info            # NEW: 返回 ROI 信息便于调试/复现
    }
