from __future__ import annotations
import os, re, glob
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio import features as rio_features
from rasterio.merge import merge as rio_merge
from rasterio.transform import Affine
from shapely.geometry import shape, Polygon, LineString
from shapely.ops import unary_union
from pyproj import CRS
import fiona
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely.geometry import box
import multiprocessing as mp
from tqdm import tqdm
from pathlib import Path
import warnings 

# 可选依赖：cv2
try:
    import cv2
except Exception:
    cv2 = None

# 允许 torch 反序列化某些类型（你自己的 .pt）
from torch.serialization import add_safe_globals
from rasterio.crs import CRS as RioCRS
from affine import Affine as _Affine
add_safe_globals([RioCRS, _Affine])

# --------------------------
# Helper functions
# --------------------------
# 顶部 helper 区域里：把单条 _ID_REGEX 与 extract_* 换成下面这段
_PATTERNS = [
    re.compile(r"patch_(\d+)_S", re.IGNORECASE),          # patch_1781869_S.pt
    re.compile(r"patch_(\d+)(?:\.|_|$)", re.IGNORECASE),  # patch_1781869.pt / patch_1781869_xxx.pt
    re.compile(r"poly_(\d+)", re.IGNORECASE),             # poly_1781869_*.pt
    re.compile(r"(\d{6,})"),                              # 兜底：连续6位以上数字
]

def extract_polygon_id_from_name(name: str) -> str:
    base = os.path.basename(name)
    for pat in _PATTERNS:
        m = pat.search(base)
        if m:
            return m.group(1)
    return "unknown"

def load_meta_from_tif(tif_path: str) -> Dict:
    with rasterio.open(tif_path) as src:
        return {"transform": src.transform, "crs": src.crs, "path": tif_path}

def mask_logits_to_binary(masks, thresh: float = 0.5) -> np.ndarray:
    if isinstance(masks, np.ndarray):
        arr = torch.from_numpy(masks)
    else:
        arr = masks

    if arr.ndim == 4 and arr.shape[1] == 1:
        arr = arr[:, 0]

    arr = arr.float()
    if (arr.min() < 0) or (arr.max() > 1.0):
        probs = torch.sigmoid(arr)
    else:
        probs = arr

    binm = (probs >= thresh).to(torch.uint8).cpu().numpy()
    return binm

def vectorize_binary_mask(bin_mask: np.ndarray, transform: Affine, crs) -> gpd.GeoDataFrame:
    geoms = []
    for geom, val in rio_features.shapes(bin_mask.astype(np.uint8), transform=transform):
        if val != 1:
            continue
        shp = shape(geom).buffer(0)
        if not shp.is_empty:
            geoms.append(shp)
    return gpd.GeoDataFrame(geometry=geoms, crs=crs) if geoms else gpd.GeoDataFrame(geometry=[], crs=crs)

def ensure_proj_meters(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        raise ValueError("GeoDataFrame has no CRS.")
    crs = CRS.from_user_input(gdf.crs)
    if crs.is_projected:
        return gdf
    c = gdf.unary_union.centroid
    lon, lat = float(c.x), float(c.y)
    zone = int((lon + 180) // 6) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return gdf.to_crs(epsg)

# --------------------------
# Load the data
# --------------------------
def load_cadastre_any(path_or_dir: str, layer: Optional[str] = None) -> gpd.GeoDataFrame:
    if os.path.isdir(path_or_dir):
        paths = sorted(glob.glob(os.path.join(path_or_dir, "*.gpkg")))
    else:
        if any(ch in path_or_dir for ch in ["*", "?", "["]):
            paths = sorted(glob.glob(path_or_dir))
        else:
            paths = [path_or_dir]

    if not paths:
        raise RuntimeError(f"No GPKG found from: {path_or_dir}")

    frames = []
    base_crs = None
    for p in paths:
        lyr = layer
        if lyr is None:
            layers = fiona.listlayers(p)
            if not layers:
                continue
            lyr = layers[0]
        gdf = gpd.read_file(p, layer=lyr)
        if gdf.empty:
            continue
        if base_crs is None:
            base_crs = gdf.crs
        elif str(gdf.crs) != str(base_crs):
            gdf = gdf.to_crs(base_crs)
        frames.append(gdf)

    if not frames:
        raise RuntimeError(f"No features read from: {paths}")

    cad = pd.concat(frames, ignore_index=True)
    cad = gpd.GeoDataFrame(cad, geometry="geometry", crs=base_crs)
    return cad

def load_buckets_from_pt_dir(pred_dir: str, img_root: Optional[str], mask_thresh: float) -> Dict[str, List[Tuple[np.ndarray, Affine, Dict]]]:
    files = [os.path.join(pred_dir, f) for f in os.listdir(pred_dir) if f.endswith(".pt")]
    files.sort()
    if not files:
        raise RuntimeError(f"No .pt files found in {pred_dir}")

    buckets: Dict[str, List[Tuple[np.ndarray, Affine, Dict]]] = {}
    for fp in files:
        obj = torch.load(fp, map_location="cpu")
        pred = obj["pred"] if isinstance(obj, dict) and "pred" in obj else obj
        masks = pred.get("masks", torch.empty(0))
        if isinstance(masks, torch.Tensor) and masks.numel() == 0:
            continue

        # polygon_id
        if isinstance(obj, dict) and ("polygon_id" in obj or ("meta" in obj and obj["meta"] and "polygon_id" in obj["meta"])):
            pid = str(obj.get("polygon_id", obj.get("meta", {}).get("polygon_id", "unknown")))
        else:
            pid = extract_polygon_id_from_name(fp)

        # meta
        if isinstance(obj, dict) and "meta" in obj and obj["meta"] and ("transform" in obj["meta"] and "crs" in obj["meta"]):
            meta = obj["meta"]
        else:
            if img_root is None:
                raise RuntimeError(f"{fp} lacks meta; set IMG_ROOT to recover .tif meta.")
            stem = os.path.splitext(os.path.basename(fp))[0]
            tif_guess = os.path.join(img_root, f"{stem}.tif")
            if not os.path.exists(tif_guess):
                tif_guess = os.path.join(img_root, f"{stem}.tiff")
                if not os.path.exists(tif_guess):
                    raise FileNotFoundError(f"Image for {fp} not found under {img_root} (tried .tif/.tiff)")
            meta = load_meta_from_tif(tif_guess)
            meta["polygon_id"] = pid

        binm = mask_logits_to_binary(masks, thresh=mask_thresh)
        if binm.size == 0:
            continue
        patch_mask = binm.max(axis=0).astype(np.uint8)
        buckets.setdefault(pid, []).append((patch_mask, meta["transform"], {"crs": meta["crs"], "path": meta.get("path", None)}))

    return buckets

def mosaic_binary_masks(masks_with_meta: List[Tuple[np.ndarray, Affine, Dict]]):
    if len(masks_with_meta) == 1:
        m, tr, md = masks_with_meta[0]
        return m, tr, CRS.from_user_input(md["crs"])
    srcs = []
    try:
        for arr, tr, md in masks_with_meta:
            H, W = arr.shape
            profile = {"driver":"GTiff","height":H,"width":W,"count":1,"dtype":"uint8","transform":tr,"crs":md["crs"]}
            mem = rasterio.io.MemoryFile(); ds = mem.open(**profile); ds.write(arr, 1)
            srcs.append((mem, ds))
        datasets = [ds for _, ds in srcs]
        merged, out_tr = rio_merge(datasets, method="max")
        out = merged[0].astype(np.uint8)
        out_crs = datasets[0].crs
        return out, out_tr, CRS.from_user_input(out_crs)
    finally:
        for mem, ds in srcs:
            ds.close(); mem.close()

# --------------------------
# 形态学拆分
# --------------------------
def split_touching_components(binary_mask: np.ndarray,
                              erode_px: int = 3,
                              min_area_px: int = 120,
                              connectivity: int = 4,
                              open_iter: int = 1):
    if cv2 is None:
        raise ImportError("cv2 not available. Install opencv-python.")
    m = (binary_mask.astype(np.uint8) > 0).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    if open_iter > 0:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=open_iter)
    m_erode = cv2.erode(m, kernel, iterations=erode_px) if erode_px > 0 else m
    conn = 4 if connectivity == 4 else 8
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m_erode, connectivity=conn)
    comps = []
    for cid in range(1, num):
        area = stats[cid, cv2.CC_STAT_AREA]
        if area < min_area_px:
            continue
        comp = (labels == cid).astype(np.uint8)
        if erode_px > 0:
            comp = cv2.dilate(comp, kernel, iterations=erode_px)
        comps.append(comp)
    return comps

# --------------------------
# 裁剪 & 导出
# --------------------------
def clip_pred_by_cadastre(pred_mask: np.ndarray,
                          transform: Affine,
                          crs: CRS,
                          cadastre_gdf: gpd.GeoDataFrame,
                          poly_id: str,
                          id_field: str,
                          *,
                          erode_px: int = 3,
                          min_area_px: int = 100,
                          connectivity: int = 4):
    cad = cadastre_gdf
    if cad.crs is None:
        raise ValueError("Cadastre GPKG has no CRS.")
    if str(cad.crs) != str(crs):
        cad = cad.to_crs(crs)

    row = cad.loc[cad[id_field].astype(str) == str(poly_id)]
    if row.empty:
        raise KeyError(f"poly_id={poly_id} not found in cadastre (field={id_field}).")
    target_poly: Polygon = row.geometry.values[0]

    components = split_touching_components(
        pred_mask, erode_px=erode_px, min_area_px=min_area_px, connectivity=connectivity
    )

    geoms = []
    for comp in components:
        for geom, val in rio_features.shapes(
            comp.astype(np.uint8),
            mask=comp.astype(bool),
            transform=transform
        ):
            if int(val) == 1:
                shp = shape(geom).buffer(0)
                if not shp.is_empty:
                    geoms.append(shp)

    if not geoms:
        return gpd.GeoDataFrame(geometry=[], crs=crs), target_poly

    pred_gdf = gpd.GeoDataFrame(geometry=geoms, crs=crs)
    target_gdf = gpd.GeoDataFrame(geometry=[target_poly], crs=crs)
    pred_clip = gpd.overlay(pred_gdf, target_gdf, how="intersection")
    return pred_clip, target_poly

def export_outputs_for_poly(out_dir: str,
                            poly_id: str,
                            pred_polys: gpd.GeoDataFrame,
                            target_poly: Polygon,
                            small_area_thresh_m2: float,
                            gpkg_path: Optional[str] = None):
    os.makedirs(out_dir, exist_ok=True)
    if pred_polys.crs is None:
        raise ValueError("pred_polys has no CRS.")
    pred_proj = ensure_proj_meters(pred_polys)
    pred_proj["area_m2"] = pred_proj.geometry.area
    smalls = pred_proj.loc[pred_proj["area_m2"] <= small_area_thresh_m2].copy().drop(columns=["area_m2"])
    smalls["poly_id"] = poly_id
    smalls = smalls.to_crs(pred_polys.crs)

    tgt_proj = ensure_proj_meters(gpd.GeoDataFrame(geometry=[target_poly], crs=pred_polys.crs))
    lines = [tgt_proj.geometry.values[0].boundary] + [g.boundary for g in pred_proj.geometry]
    lines_union = unary_union(lines)
    line_geoms = [lines_union] if isinstance(lines_union, LineString) else list(getattr(lines_union, "geoms", []))
    lines_gdf = gpd.GeoDataFrame({"poly_id": [poly_id]*len(line_geoms)}, geometry=line_geoms, crs=pred_proj.crs).to_crs(pred_polys.crs)

    if gpkg_path is None:
        gpkg_path = os.path.join(out_dir, f"poly_{poly_id}.gpkg")
    if os.path.exists(gpkg_path):
        os.remove(gpkg_path)
    lines_gdf.to_file(gpkg_path, layer="boundaries_all", driver="GPKG")
    smalls.to_file(gpkg_path, layer="small_fields", driver="GPKG")
    return {"gpkg": gpkg_path, "boundaries_layer": "boundaries_all", "small_fields_layer": "small_fields"}

# --------------------------
# 顶层管道
# --------------------------
def postprocess_from_pt_dir(pred_dir: str,
                            cadastre_gpkg_path: str,
                            out_dir: str,
                            id_field: str,
                            img_root: Optional[str],
                            mask_thresh: float,
                            small_area_thresh_m2: float) -> Dict[str, Dict[str, str]]:
    cad = load_cadastre_any(cadastre_gpkg_path, layer=None)
    buckets = load_buckets_from_pt_dir(pred_dir, img_root=img_root, mask_thresh=mask_thresh)

    results: Dict[str, Dict[str, str]] = {}
    for pid, items in buckets.items():
        merged_mask, merged_tr, merged_crs = mosaic_binary_masks(items)
        pred_polys, target_poly = clip_pred_by_cadastre(
            merged_mask, merged_tr, merged_crs, cad, poly_id=pid, id_field=id_field
        )
        if pred_polys.empty:
            print(f"[{pid}] empty after clipping.")
            continue
        info = export_outputs_for_poly(
            out_dir=out_dir, poly_id=pid, pred_polys=pred_polys,
            target_poly=target_poly, small_area_thresh_m2=small_area_thresh_m2
        )
        results[pid] = info
        print(f"[{pid}] -> {info['gpkg']}  ({info['boundaries_layer']}, {info['small_fields_layer']})")
            # === 并行拼接所有地块 GPKG 为一张 GeoTIFF ===
    try:
        print("\n>>> 并行拼接所有地块 GPKG 到单一 GeoTIFF ...")

        gpkg_list = [Path(v["gpkg"]) for v in results.values() if Path(v["gpkg"]).exists()]
        if not gpkg_list:
            print("⚠️ 未找到可拼接的 GPKG 文件，跳过全局合并。")
            return results

        # === Step 1: 获取所有边界的总范围和CRS ===
        sample_gdf = gpd.read_file(gpkg_list[0], layer="boundaries")
        crs = sample_gdf.crs
        bounds_list = [sample_gdf.total_bounds]

        # 为了避免单个文件 I/O 瓶颈，我们并行提取 bounds
        def get_bounds(path):
            try:
                gdf = gpd.read_file(path, layer="boundaries")
                return gdf.total_bounds
            except Exception:
                return None

        with mp.Pool(processes=os.cpu_count()) as pool:
            for b in tqdm(pool.imap(get_bounds, gpkg_list), total=len(gpkg_list)):
                if b is not None:
                    bounds_list.append(b)

        # 计算总体范围
        bounds_array = np.array(bounds_list)
        minx, miny = bounds_array[:, 0].min(), bounds_array[:, 1].min()
        maxx, maxy = bounds_array[:, 2].max(), bounds_array[:, 3].max()
        print(f"Total mosaic bounds: ({minx:.2f}, {miny:.2f}, {maxx:.2f}, {maxy:.2f})")

        # === Step 2: 设置分辨率与输出影像尺寸 ===
        res = 10  # 每像元 10 米，可按你的地块 CRS 调整
        width = int((maxx - minx) / res)
        height = int((maxy - miny) / res)
        transform = from_origin(minx, maxy, res, res)
        print(f"Output size: {width} x {height} px")

        # === Step 3: 并行 rasterize ===
        def rasterize_one(path):
            try:
                gdf = gpd.read_file(path, layer="boundaries")
                if gdf.empty:
                    return np.zeros((height, width), dtype=np.uint8)
                shapes = [(geom, 1) for geom in gdf.geometry if geom.is_valid]
                arr = rasterize(
                    shapes,
                    out_shape=(height, width),
                    transform=transform,
                    fill=0,
                    dtype="uint8"
                )
                return arr
            except Exception:
                return np.zeros((height, width), dtype=np.uint8)

        print(f"Using {os.cpu_count()} CPU cores for parallel rasterization...")

        with mp.Pool(processes=os.cpu_count()) as pool:
            partials = list(tqdm(pool.imap(rasterize_one, gpkg_list), total=len(gpkg_list)))

        # === Step 4: 合并所有块（取最大值）===
        print("Merging partial rasters...")
        mosaic = np.maximum.reduce(partials)

        # === Step 5: 写出 GeoTIFF ===
        merged_tif = Path(out_dir) / "merged_all_fields.tif"
        with rasterio.open(
            merged_tif, "w",
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype="uint8",
            crs=crs,
            transform=transform,
            compress="lzw",
            BIGTIFF="IF_SAFER"
        ) as dst:
            dst.write(mosaic, 1)

        print(f"✅ 全局拼接完成，输出文件：{merged_tif}")

    except Exception as e:
        print(f"⚠️ 拼接 TIF 失败：{e}")

    return results
