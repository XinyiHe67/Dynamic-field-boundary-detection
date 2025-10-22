# model/postprocessing.py
from __future__ import annotations
import os, re, glob, json, warnings
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import geopandas as gpd
import rasterio
from rasterio import features as rio_features
from rasterio.crs import CRS as RioCRS
from shapely.geometry import shape
from pyproj import CRS
from affine import Affine as _Affine
from shapely.validation import make_valid
from shapely.ops import unary_union

# 兼容 torch 安全反序列化（.pt 里常见 transform/crs）
from torch.serialization import add_safe_globals
add_safe_globals([RioCRS, _Affine])

# OpenCV（可选）用于轻微腐蚀
try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False
    warnings.warn("[WARN] OpenCV 未安装，无法形态学腐蚀；相邻实例更容易粘连。")

# ----------------------------
# 小工具函数
# ----------------------------
def _extract_polygon_id_from_name(pt_path: str) -> Optional[str]:
    """从文件名里猜 polygon_id，例如 pred_polygon_12345.pt -> 12345"""
    stem = os.path.splitext(os.path.basename(pt_path))[0]
    m = re.search(r'(\d{5,})', stem)  # 默认找 >=5 位数字
    return m.group(1) if m else None

def _load_meta_from_tif_guess(pred_path: str, img_root: str) -> Dict:
    """
    若预测缺少 meta，则在 IMG_ROOT 下用“同名 .tif/.tiff”回填 transform/crs。
    例如 pred_path = /runs/predictions/patch_123.pt -> 在 img_root 下寻找 patch_123.tif/.tiff
    """
    stem = os.path.splitext(os.path.basename(pred_path))[0]
    for ext in (".tif", ".tiff"):
        guess = os.path.join(img_root, f"{stem}{ext}")
        if os.path.exists(guess):
            with rasterio.open(guess) as src:
                return {"transform": src.transform, "crs": src.crs, "path": guess}
    raise FileNotFoundError(f"[Meta] 未在 {img_root} 下为 {pred_path} 找到同名 .tif/.tiff")

def _mask_logits_to_binary(masks, thresh: float = 0.5) -> np.ndarray:
    """支持 logits 或 [0,1] 概率；输出 [N,H,W] uint8(0/1)。"""
    arr = torch.from_numpy(masks) if isinstance(masks, np.ndarray) else masks
    if arr.ndim == 4 and arr.shape[1] == 1:
        arr = arr[:, 0]
    arr = arr.float()
    probs = torch.sigmoid(arr) if (arr.min() < 0 or arr.max() > 1.0) else arr
    return (probs >= thresh).to(torch.uint8).cpu().numpy()

def _erode_if_needed(binary: np.ndarray, iters: int = 1) -> np.ndarray:
    if not _HAS_CV2 or iters <= 0:
        return binary
    kernel = np.ones((3,3), np.uint8)
    return cv2.erode(binary, kernel, iterations=iters)

def _vectorize_single_mask(mask01: np.ndarray, transform, min_pixels=0,
                           simplify_ratio: float = 0.0, hw: Optional[Tuple[int,int]] = None):
    """把单个二值掩膜矢量化为 polygon 列表；像素面积过滤 + 顶点简化。"""
    H, W = mask01.shape
    polys = []
    for geom, val in rio_features.shapes(mask01.astype(np.uint8), transform=transform):
        if val != 1:
            continue
        poly = shape(geom)
        if min_pixels > 0 and poly.area < min_pixels:
            continue
        if simplify_ratio and simplify_ratio > 0:
            diag = (H**2 + W**2) ** 0.5
            poly = poly.simplify(diag * simplify_ratio, preserve_topology=True)
        if not poly.is_empty:
            polys.append(poly)
    return polys

def masks_non_overlapping(binm: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """实例间去重叠：高分优先；同分按索引靠前优先。"""
    N, H, W = binm.shape
    sc = np.nan_to_num(scores, nan=0.0).astype(np.float32)
    score_stack = sc[:, None, None] * binm.astype(np.float32)
    winner_idx = np.argmax(score_stack, axis=0)  # [H,W]
    winner_val = np.take_along_axis(score_stack, winner_idx[None, ...], axis=0)[0]
    out = np.zeros_like(binm, dtype=np.uint8)
    for i in range(N):
        out[i] = ((winner_idx == i) & (binm[i] == 1) & (winner_val > 0)).astype(np.uint8)
    return out

def _ensure_gpkg_path(base_dir: str, polygon_id: str) -> str:
    return os.path.join(base_dir, f"{polygon_id}.gpkg")

def _load_cadastre(source: str, layer: Optional[str], id_field: str) -> gpd.GeoDataFrame:
    """source 可为 .gpkg 或目录（批量 .gpkg）。"""
    if os.path.isdir(source):
        paths = sorted(glob.glob(os.path.join(source, "*.gpkg")))
        if not paths:
            raise FileNotFoundError(f"[Cadastre] 目录 {source} 下未找到 .gpkg")
        gdfs = []
        for p in paths:
            g = gpd.read_file(p) if layer is None else gpd.read_file(p, layer=layer)
            gdfs.append(g)
        cad = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs)
    else:
        cad = gpd.read_file(source) if layer is None else gpd.read_file(source, layer=layer)
    if id_field not in cad.columns:
        raise ValueError(f"[Cadastre] 缺少 ID 字段 {id_field}；可选列：{list(cad.columns)}")
    cad = cad[[id_field, cad.geometry.name]].dropna(subset=[cad.geometry.name]).reset_index(drop=True)
    cad["_id_str_"] = cad[id_field].astype(str)
    return cad.set_index("_id_str_", drop=False)

# ---- 关键：统一/还原 meta 的 crs/transform ----
def _normalize_meta(meta: dict):
    """
    把 meta 里的 crs/transform 统一为对象类型：
    - crs: 任意 pyproj 可识别输入（EPSG 字符串/WKT/对象）
    - transform: 支持 Affine 或 6/9 元 tuple
    """
    out_crs = None
    transform = None

    # CRS
    crs_in = meta.get("crs", None)
    if crs_in is not None:
        try:
            out_crs = CRS.from_user_input(crs_in)
        except Exception:
            out_crs = None

    # Transform
    tfm_in = meta.get("transform", None)
    if isinstance(tfm_in, _Affine):
        transform = tfm_in
    elif isinstance(tfm_in, (tuple, list)):
        try:
            transform = _Affine.from_gdal(*tfm_in) if len(tfm_in) == 6 else _Affine(*tfm_in)
        except Exception:
            transform = None

    return {"crs": out_crs, "transform": transform}

def _load_prediction_one(path: str, img_root: Optional[str]):
    """
    统一加载单个预测结果，支持 .pt 和 .npz：
      - .pt: torch.load，优先取 pred['meta']；不全则从 img_root 下同名 .tif 回填
      - .npz: 同名 .meta.json 中取 meta；不全同上回填
    返回: (pred, transform(Affine), out_crs(CRS), polygon_id(str|None))
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".pt":
        obj = torch.load(path, map_location="cpu", weights_only=False)
        pred = obj["pred"] if isinstance(obj, dict) and "pred" in obj else obj
        meta = pred.get("meta", None) if isinstance(pred, dict) else None
    else:  # .npz
        data = np.load(path)
        pred = {
            "boxes":  data.get("boxes"),
            "labels": data.get("labels"),
            "scores": data.get("scores"),
            "masks":  data.get("masks"),
            "image_id": int(data.get("image_id")) if data.get("image_id") is not None else -1,
        }
        meta_path = path.replace(".npz", ".meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"[Meta] 找不到 {os.path.basename(meta_path)}")
        with open(meta_path, "r", encoding="utf-8") as f:
            j = json.load(f)
        meta = j.get("meta", None)

    # 还原/规范化 meta
    transform = None
    out_crs = None
    if isinstance(meta, dict):
        nm = _normalize_meta(meta)
        transform = nm["transform"]
        out_crs = nm["crs"]

    # 缺少就从 img_root 回填
    if transform is None or out_crs is None:
        if img_root is None:
            raise RuntimeError(f"[Meta] {os.path.basename(path)} 缺少 transform/crs，且未提供 IMG_ROOT 回填 .tif")
        m2 = _load_meta_from_tif_guess(path, img_root)
        transform = m2["transform"]
        out_crs = CRS.from_user_input(m2["crs"]) if not isinstance(m2["crs"], CRS) else m2["crs"]

    # 取 polygon_id
    polygon_id = None
    if isinstance(pred, dict):
        polygon_id = pred.get("polygon_id") or ((pred.get("meta") or {}).get("polygon_id"))
    if polygon_id is None:
        polygon_id = _extract_polygon_id_from_name(path)
    polygon_id = str(polygon_id) if polygon_id is not None else None

    return pred, transform, out_crs, polygon_id


def _standardize_geoms_same_crs(gdf: gpd.GeoDataFrame, simplify_eps_m: float = 0.0) -> gpd.GeoDataFrame:
    """在当前 CRS 下做几何修复/轻简化；不改 CRS。单位=米（你当前 CRS 已经是米）。"""
    g = gdf.copy()
    g["geometry"] = g["geometry"].apply(lambda x: make_valid(x).buffer(0))
    if simplify_eps_m and simplify_eps_m > 0:
        g["geometry"] = g["geometry"].simplify(simplify_eps_m, preserve_topology=True)
    g = g[~g.geometry.is_empty & g.geometry.is_valid].reset_index(drop=True)
    return g

def _dedup_and_fuse_same_crs(
    sub_f: gpd.GeoDataFrame,
    iou_th: float = 0.90,           # IoU≥0.90 视为同一块
    overlap_th: float = 0.85,       # 交叠占小者≥0.85 也视为同一块
    prefer: str = "score",          # 先看 score，高者留；再看面积，大者留
    simplify_eps_m: float = 0.0,    # 几何轻简化（米）；S2 可先 0 或 2–3
    fuse_eps_m: float = 0.0         # 融合抹“台阶”（米）；需要平滑再开，例如 2–3
) -> gpd.GeoDataFrame:
    """在同一 polygon_id 子集内去重/（可选）融合。假设当前 CRS 单位为米。"""
    if sub_f.empty:
        return sub_f

    g = _standardize_geoms_same_crs(sub_f, simplify_eps_m)

    # 优先级：score 高 → 面积大
    scores = np.nan_to_num(g.get("score", pd.Series(index=g.index, data=0.0)).values, nan=0.0)
    areas  = g.geometry.area.values
    order  = np.lexsort((-areas, -scores))  # 先按score降序，再按面积降序

    sidx = g.sindex
    dropped = set()
    kept_rows = []

    for idx in order:
        if idx in dropped:
            continue
        a = g.geometry.iloc[idx]
        keep_idx = idx
        keep_geom = a

        # 候选（包围盒相交）
        cand = list(sidx.query(a))
        cand = [j for j in cand if j != idx and j not in dropped]

        for j in cand:
            b = g.geometry.iloc[j]
            if not keep_geom.intersects(b):
                continue
            inter = keep_geom.intersection(b)
            if inter.is_empty:
                continue
            uni = keep_geom.union(b)
            iou = inter.area / uni.area if uni.area > 0 else 0.0
            over = inter.area / min(keep_geom.area, b.area)

            if (iou >= iou_th) or (over >= overlap_th):
                # 选择保留者
                s_keep = scores[keep_idx]; s_j = scores[j]
                a_keep = keep_geom.area;   a_j = b.area

                # 规则：score 高者保留；同分面积大者保留
                take_j = (s_j > s_keep) or (np.isclose(s_j, s_keep) and a_j > a_keep)
                if take_j:
                    dropped.add(keep_idx)
                    keep_idx = j
                    keep_geom = b
                else:
                    dropped.add(j)
                    # 可选：做融合，抹掉像素台阶
                    if fuse_eps_m and fuse_eps_m > 0:
                        keep_geom = unary_union([keep_geom, b]).buffer(+fuse_eps_m).buffer(-fuse_eps_m)
                    # 否则就简单保留 keep_geom

        row = g.iloc[keep_idx].copy()
        row.geometry = keep_geom
        kept_rows.append(row)

    out = gpd.GeoDataFrame(kept_rows, crs=sub_f.crs).reset_index(drop=True)
    out = out[~out.geometry.is_empty & out.geometry.is_valid]

    # 再做一次“强裁剪”：保证最终互不重叠（winner-takes-all）
    if len(out) > 1:
        scores2 = np.nan_to_num(out.get("score", pd.Series(index=out.index, data=0.0)).values, nan=0.0)
        areas2  = out.geometry.area.values
        ord2    = np.lexsort((-areas2, -scores2))
        accepted = []
        rows2 = []
        for k in ord2:
            geom = out.geometry.iloc[k]
            for h in accepted:
                if geom.intersects(h):
                    geom = geom.difference(h)
                    if geom.is_empty:
                        geom = None
                        break
            if geom and not geom.is_empty:
                accepted.append(geom)
                r = out.iloc[k].copy()
                r.geometry = geom
                rows2.append(r)
        out = gpd.GeoDataFrame(rows2, crs=sub_f.crs).reset_index(drop=True)

    return out


# ----------------------------
# 核心处理：预测 → 多个 GPKG（每个 polygon_id 一个，含 fields/boundaries 两图层）
# ----------------------------
def postprocess_predictions_to_gpkg(
    pred_dir: str,
    cad_gdf: gpd.GeoDataFrame,
    out_dir: str,
    id_field: str,
    img_root: Optional[str],
    mask_thresh: float,
    erode_iter: int,
    min_pixels: int,
    simplify_ratio: float,
    score_thresh: float,
) -> bool:
    os.makedirs(out_dir, exist_ok=True)
    records_fields, records_bounds = [], []

    # 同时收集 .pt 和 .npz
    files = sorted([
        os.path.join(pred_dir, f) for f in os.listdir(pred_dir)
        if f.lower().endswith(".pt") or f.lower().endswith(".npz")
    ])
    if not files:
        raise RuntimeError(f"[Pred] {pred_dir} 下未找到 .pt/.npz 文件")

    out_crs = None              # 用第一个样本的 CRS
    cad_reproj_done = False     # cad_gdf 只重投影一次

    for pi, path in enumerate(files, 1):
        # 统一加载一个预测（含 meta 还原/回填）
        pred, transform, this_crs, polygon_id = _load_prediction_one(path, img_root)

        # 第一次拿到预测 CRS：把 cad_gdf 重投影到预测 CRS，避免 CRS 不一致导致相交为空
        if out_crs is None:
            out_crs = this_crs
            if cad_gdf.crs is not None and CRS.from_user_input(cad_gdf.crs) != CRS.from_user_input(out_crs):
                cad_gdf = cad_gdf.to_crs(out_crs)
                cad_reproj_done = True
            print(f"[INFO] 统一 cadastre CRS 到预测 CRS：{out_crs}（是否重投影: {cad_reproj_done}）")

        # 只打印前 3 个样本的调试信息
        if pi <= 3:
            print(f"[DBG] {os.path.basename(path)} | poly={polygon_id} | cad_crs={cad_gdf.crs} | pred_crs={out_crs} | tfm={type(transform)}")

        if polygon_id is None:
            print(f"[Warn] {os.path.basename(path)}: 无法确定 polygon_id，跳过")
            continue

        # 取 masks / scores
        masks = pred.get("masks", torch.empty(0))
        if (isinstance(masks, torch.Tensor) and masks.numel() == 0) or \
           (isinstance(masks, np.ndarray) and masks.size == 0):
            print(f"[Skip] {os.path.basename(path)}: 无 masks")
            continue

        scores = None
        if isinstance(pred, dict) and "scores" in pred:
            s = pred["scores"]
            scores = s.detach().cpu().numpy() if isinstance(s, torch.Tensor) else s
        N = (masks.shape[0] if isinstance(masks, (np.ndarray, torch.Tensor)) else 0)
        if scores is None or len(scores) != N:
            scores = np.full((N,), np.nan, dtype=float)

        # 找目标 cadastre 几何
        if polygon_id not in cad_gdf.index:
            print(f"[Warn] {os.path.basename(path)}: 在 cadastre 中找不到 ID={polygon_id}，跳过")
            continue
        target_geom = cad_gdf.loc[polygon_id, cad_gdf.geometry.name]

        # 掩膜 → 二值 → 去重叠
        binm = _mask_logits_to_binary(masks, thresh=mask_thresh)
        binm = masks_non_overlapping(binm, scores)

        # 逐实例矢量化 + 裁到目标多边形
        for i in range(binm.shape[0]):
            s_i = float(scores[i]) if not np.isnan(scores[i]) else 0.0
            if s_i < score_thresh:
                continue
            bi = binm[i].astype(np.uint8)
            if bi.sum() < min_pixels:
                continue
            bi = _erode_if_needed(bi, iters=erode_iter)
            polys = _vectorize_single_mask(
                bi, transform=transform,
                min_pixels=min_pixels,
                simplify_ratio=simplify_ratio,
                hw=binm.shape[-2:]
            )
            if not polys:
                continue

            for g_poly in polys:
                inter = g_poly.intersection(target_geom)
                if inter.is_empty:
                    continue
                geoms = [inter] if inter.geom_type == "Polygon" else list(inter.geoms)
                for g in geoms:
                    if g.is_empty:
                        continue
                    records_fields.append({
                        "polygon_id": polygon_id,
                        "instance": int(i),
                        "score": s_i,
                        "src_pt": os.path.basename(path),
                        "geometry": g
                    })
                    boundary = g.boundary
                    if boundary.is_empty:
                        continue
                    if boundary.geom_type == "MultiLineString":
                        for seg in boundary.geoms:
                            records_bounds.append({
                                "polygon_id": polygon_id,
                                "instance": int(i),
                                "score": s_i,
                                "src_pt": os.path.basename(path),
                                "geometry": seg
                            })
                    elif boundary.geom_type == "LineString":
                        records_bounds.append({
                            "polygon_id": polygon_id,
                            "instance": int(i),
                            "score": s_i,
                            "src_pt": os.path.basename(path),
                            "geometry": boundary
                        })

        print(f"[OK] {pi}/{len(files)} 处理完 {os.path.basename(path)} → {polygon_id}")

    if not records_fields:
        raise RuntimeError("[Result] 没有生成任何 field polygon。检查阈值/匹配。")

    gdf_fields = gpd.GeoDataFrame(records_fields, crs=out_crs)
    gdf_bounds = gpd.GeoDataFrame(records_bounds, crs=out_crs)

    # 按 polygon_id 分 gpkg 输出，方便逐地籍块查看；也可改成一个总 gpkg

    for pid, sub_f in gdf_fields.groupby("polygon_id"):
        out_path = _ensure_gpkg_path(out_dir, pid)

    # ★ 在同一 polygon_id 内做去重/（可选）融合 —— 全程使用当前 CRS（米）
        sub_f = _dedup_and_fuse_same_crs(
            sub_f,
            iou_th=0.90,
            overlap_th=0.85,
            prefer="score",
            simplify_eps_m=0.0,   # 先关；若边界有“台阶”，可设 2~3（米）
            fuse_eps_m=0.0        # 需要把重叠几何融合再开，例：2~3（米）
        )

        # 用去重后的面重建边界（避免旧的 boundary 与新面不一致）
        if len(sub_f):
            sub_b = gpd.GeoDataFrame({
                "polygon_id": sub_f["polygon_id"].values,
                "instance":   sub_f.get("instance", pd.Series(index=sub_f.index, data=-1)).values,
                "score":      sub_f.get("score", pd.Series(index=sub_f.index, data=np.nan)).values,
                "src_pt":     sub_f.get("src_pt", pd.Series(index=sub_f.index, data="")).values,
                "geometry":   sub_f.boundary
            }, crs=sub_f.crs)
        else:
            sub_b = gpd.GeoDataFrame(columns=["polygon_id","instance","score","src_pt","geometry"], crs=sub_f.crs)
    
        # 写出
        sub_f.to_file(out_path, driver="GPKG", layer="fields")
        if len(sub_b):
            sub_b.to_file(out_path, driver="GPKG", layer="boundaries")
    
        print(f"[Write] {pid}: {out_path}  (fields={len(sub_f)}, boundaries={len(sub_b)})")

    return True

# ----------------------------
# 合并多个 GPKG → 一个 merged.gpkg（fields_all + boundaries_all）
# ----------------------------
def _read_layer_safe(path, layer):
    try:
        return gpd.read_file(path, layer=layer)
    except Exception:
        return None

def _union_columns(gdfs):
    cols = set()
    for g in gdfs:
        if g is None or g.empty:
            continue
        cols.update([c for c in g.columns if c != g.geometry.name])
    cols = list(cols)
    out = []
    for g in gdfs:
        if g is None or g.empty:
            continue
        miss = [c for c in cols if c not in g.columns]
        for c in miss:
            g[c] = pd.NA
        out.append(g[[*cols, g.geometry.name]])
    return out, cols

def merge_gpkgs(out_dir, out_merged, target_crs=None):
    gpkg_files = sorted(glob.glob(os.path.join(out_dir, "*.gpkg")))
    if not gpkg_files:
        raise FileNotFoundError(f"No .gpkg in {out_dir}")

    fields_list, bounds_list = [], []
    crs_target = None

    for p in gpkg_files:
        gf = _read_layer_safe(p, "fields")
        gb = _read_layer_safe(p, "boundaries")
        if gf is not None and not gf.empty:
            gf = gf.copy(); gf["src_gpkg"] = os.path.basename(p); fields_list.append(gf)
        if gb is not None and not gb.empty:
            gb = gb.copy(); gb["src_gpkg"] = os.path.basename(p); bounds_list.append(gb)

    if not fields_list and not bounds_list:
        raise RuntimeError("No layers found to merge.")

    if target_crs is not None:
        crs_target = CRS.from_user_input(target_crs)
    else:
        for g in (fields_list + bounds_list):
            if g is not None and g.crs is not None:
                crs_target = CRS.from_user_input(g.crs); break
        if crs_target is None:
            crs_target = CRS.from_user_input("EPSG:4326")
            print("[WARN] All inputs missing CRS; default to EPSG:4326")

    def _to_target(g):
        if g is None or g.empty:
            return g
        if g.crs is None:
            return g.set_crs(crs_target, allow_override=True)
        if CRS.from_user_input(g.crs) != crs_target:
            return g.to_crs(crs_target)
        return g

    fields_list = [_to_target(g) for g in fields_list]
    bounds_list = [_to_target(g) for g in bounds_list]
    fields_list, _ = _union_columns(fields_list)
    bounds_list, _ = _union_columns(bounds_list)

    gdf_fields_all = gpd.GeoDataFrame(pd.concat(fields_list, ignore_index=True), crs=crs_target) if fields_list else None
    gdf_bounds_all = gpd.GeoDataFrame(pd.concat(bounds_list, ignore_index=True), crs=crs_target) if bounds_list else None

    if os.path.exists(out_merged):
        try: os.remove(out_merged)
        except Exception: pass

    if gdf_fields_all is not None and not gdf_fields_all.empty:
        gdf_fields_all.to_file(out_merged, driver="GPKG", layer="fields_all")
        print(f"[Write] fields_all: {len(gdf_fields_all)} features")
    if gdf_bounds_all is not None and not gdf_bounds_all.empty:
        gdf_bounds_all.to_file(out_merged, driver="GPKG", layer="boundaries_all")
        print(f"[Write] boundaries_all: {len(gdf_bounds_all)} features")

    print(f"[OK] merged → {out_merged} | CRS={crs_target}")
    return out_merged

# ----------------------------
# 一键入口（给 main.py 用）
# ----------------------------
def run_postprocessing(
    pred_dir: str,
    cad_source: str,
    layer_name: Optional[str],
    id_field: str,
    out_dir: str,
    img_root: Optional[str],
    mask_thresh: float = 0.7,
    erode_iter: int = 1,
    min_pixels: int = 20,
    simplify_ratio: float = 0.002,
    score_thresh: float = 0.65,
    target_crs: Optional[str] = None,
    merged_out_dir: Optional[str] = None,     # 新增：合并结果单独目录（如 final_result）
    merged_filename: str = "merged.gpkg",     # 新增：合并文件名
):
    # 1) 读 cadastre
    cad_gdf = _load_cadastre(cad_source, layer_name, id_field)
    print(f"[Cadastre] 加载 {len(cad_gdf)} 个 target polygon；CRS={cad_gdf.crs}")

    # 2) 逐图写出单个 GPKG（fields/boundaries 两图层）
    ok = postprocess_predictions_to_gpkg(
        pred_dir=pred_dir,
        cad_gdf=cad_gdf,
        out_dir=out_dir,
        id_field=id_field,
        img_root=img_root,
        mask_thresh=mask_thresh,
        erode_iter=erode_iter,
        min_pixels=min_pixels,
        simplify_ratio=simplify_ratio,
        score_thresh=score_thresh,
    )
    if not ok:
        raise RuntimeError("Postprocess failed to generate polygons.")

    # 3) 合并所有 GPKG 到一个 merged.gpkg（可放到单独目录）
    merged_base_dir = merged_out_dir or out_dir
    os.makedirs(merged_base_dir, exist_ok=True)
    merged_path = os.path.join(merged_base_dir, merged_filename)
    merged_path = merge_gpkgs(out_dir, merged_path, target_crs=target_crs)

    return {"out_dir": out_dir, "merged": merged_path}


