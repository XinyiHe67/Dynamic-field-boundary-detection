# model/gee_pipeline.py
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


# ============================
# 3) 顶层封装：完全按 ipynb 流程导出 8-bit
# ============================
def run_gee_fetch_and_export(
    *,
    cadastre_asset: str,
    id_col: str,
    bbox: tuple,               # (minx, miny, maxx, maxy) in WGS84
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
    crs_out: str = "auto"
) -> Dict[str, Any]:
    """
    主流程（完全复刻 ipynb 的可视化导出链路）：
    - S2_SR + SCL 掩云 + median
    - 在原始 ROI 上统计 2–98% 并 visualize 成 8-bit
    - 下载到 out_tif（region 用 expanded_roi）
    - DW 按 4/7 估算农地占比，写出 id_list 和 filter_lot.gpkg
    """
    ee_init_from_json(json_path=json_path)
    minx, miny, maxx, maxy = bbox
    roi = ee.Geometry.Rectangle([minx, miny, maxx, maxy])
    expanded_roi = roi.buffer(pad_m).bounds()

    # 输出坐标系
    if crs_out.lower() == "auto":
        crs_out_val = infer_utm_epsg_from_lonlat((minx + maxx) / 2, (miny + maxy) / 2)
    else:
        CRS.from_user_input(crs_out)
        crs_out_val = crs_out
    print(f"[CRS] 输出：{crs_out_val} | 外扩：{pad_m} m")

    # 选地块 & 写 id 列表
    selected_polys, ids = select_polygons_in_roi(
        cadastre_asset_id=cadastre_asset, roi_geom=roi, id_col=id_col
    )
    id_list_py = ids.getInfo()
    print(f"[IDs] 命中 {len(id_list_py)} 个地块 | 前10：{id_list_py[:10]}")

    os.makedirs(os.path.dirname(out_ids), exist_ok=True)
    with open(out_ids, "w", encoding="utf-8") as f:
        f.write("\n".join(map(str, id_list_py)))
    print(f"[IDs] 已写入：{out_ids}")

    # ★ 完全对齐 ipynb：服务端 visualize → 8-bit
    rgb8_exact = build_and_visualize_s2_rgb8_exact(
        roi, s2_start, s2_end, cloud_max,
        use_harmonized=False, pct_bounds=(2, 98)
    ).clip(expanded_roi)

    os.makedirs(os.path.dirname(out_tif), exist_ok=True)
    print(f"[DL] 下载影像（8-bit）：{out_tif}")

    geemap.download_ee_image(
        image=rgb8_exact,
        filename=out_tif,
        region=expanded_roi,
        scale=10,
        crs=crs_out_val,
        overwrite=True   # visualize 已经是 uint8，这里不要再指定 dtype
    )
    print("Sentinel-2 RGB8 下载完成")

    # Export the screening plot GPKG
    gdf = geemap.ee_to_gdf(selected_polys)
    minx, miny, maxx, maxy = bbox
    cen_lon, cen_lat = (minx + maxx) / 2, (miny + maxy) / 2
    utm_crs = infer_utm_epsg_from_lonlat(cen_lon, cen_lat)
    gdf = gdf.to_crs(utm_crs)
    gpkg_path = os.path.join(os.path.dirname(out_tif), "filter_lot.gpkg")
    gdf.to_file(gpkg_path, driver="GPKG")
    print(f"The cadastral has been exported：{gpkg_path}")

    return {"tif": out_tif, "ids": out_ids, "gpkg": gpkg_path, "id_list": id_list_py}


