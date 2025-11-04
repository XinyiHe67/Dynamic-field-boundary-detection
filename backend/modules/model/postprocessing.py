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

# Make torch.load safe for common GIS types (transform/crs in .pt files)
from torch.serialization import add_safe_globals
add_safe_globals([RioCRS, _Affine])

# OpenCV (optional) for mild erosion
try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False
    warnings.warn("[WARN] OpenCV is not installed; morphological erosion will be skipped, "
                  "so neighboring instances are more likely to stick together.")

# ----------------------------
# Utility helpers
# ----------------------------
def _extract_polygon_id_from_name(pt_path: str) -> Optional[str]:
    """
    Try to infer a polygon_id from filename,
    e.g. 'pred_polygon_12345.pt' -> '12345' (any sequence of ≥5 digits).
    """
    stem = os.path.splitext(os.path.basename(pt_path))[0]
    m = re.search(r'(\d{5,})', stem)  
    return m.group(1) if m else None

def _load_meta_from_tif_guess(pred_path: str, img_root: str) -> Dict:
    """
    If prediction has no meta, try to recover transform/CRS from a same-name TIF/TIFF in IMG_ROOT.

    Example:
        pred_path = /runs/predictions/patch_123.pt
        will search for patch_123.tif / patch_123.tiff under img_root.
    """
    stem = os.path.splitext(os.path.basename(pred_path))[0]
    for ext in (".tif", ".tiff"):
        guess = os.path.join(img_root, f"{stem}{ext}")
        if os.path.exists(guess):
            with rasterio.open(guess) as src:
                return {"transform": src.transform, "crs": src.crs, "path": guess}
    raise FileNotFoundError(f"[Meta] Could not find a same-name .tif/.tiff for {pred_path} under {img_root}")

def _mask_logits_to_binary(masks, thresh: float = 0.5) -> np.ndarray:
    """
    Convert mask logits or probabilities to a binary mask.

    Supports:
        masks: [N, H, W] or [N, 1, H, W]
        Returns:
        np.ndarray of shape [N, H, W] with dtype uint8 (0/1).
    """
    arr = torch.from_numpy(masks) if isinstance(masks, np.ndarray) else masks
    if arr.ndim == 4 and arr.shape[1] == 1:
        arr = arr[:, 0]
    arr = arr.float()
    # Heuristic: treat as logits if outside [0,1]; otherwise assume probabilities
    probs = torch.sigmoid(arr) if (arr.min() < 0 or arr.max() > 1.0) else arr
    return (probs >= thresh).to(torch.uint8).cpu().numpy()

def _erode_if_needed(binary: np.ndarray, iters: int = 1) -> np.ndarray:
    if not _HAS_CV2 or iters <= 0:
        return binary
    kernel = np.ones((3,3), np.uint8)
    return cv2.erode(binary, kernel, iterations=iters)

def _vectorize_single_mask(mask01: np.ndarray, transform, min_pixels=0,
                           simplify_ratio: float = 0.0, hw: Optional[Tuple[int,int]] = None):
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
    """
    Vectorize a single binary mask (0/1) into polygons.

    Args:
        mask01: 2D binary mask [H, W]
        transform: Affine transform
        min_pixels: Area threshold in pixels; polygons smaller than this are dropped
        simplify_ratio: Simplify tolerance as a fraction of image diagonal
        hw: Optional (H, W), reserved for future use

    Returns:
        List of shapely Polygons.
    """
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
    """
    Load cadastre data.

    Args:
        source: .gpkg file path or directory containing multiple .gpkg files
        layer: optional layer name if reading from a multi-layer .gpkg
        id_field: column name of the polygon ID

    Returns:
        GeoDataFrame with index set to string version of ID.
    """
    if os.path.isdir(source):
        paths = sorted(glob.glob(os.path.join(source, "*.gpkg")))
        if not paths:
            raise FileNotFoundError(f"[Cadastre] No .gpkg files found in directory {source}")
        gdfs = []
        for p in paths:
            g = gpd.read_file(p) if layer is None else gpd.read_file(p, layer=layer)
            gdfs.append(g)
        cad = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs)
    else:
        cad = gpd.read_file(source) if layer is None else gpd.read_file(source, layer=layer)
    if id_field not in cad.columns:
        raise ValueError(f"[Cadastre] ID field '{id_field}' not found; available columns: {list(cad.columns)}")
    cad = cad[[id_field, cad.geometry.name]].dropna(subset=[cad.geometry.name]).reset_index(drop=True)
    cad["_id_str_"] = cad[id_field].astype(str)
    return cad.set_index("_id_str_", drop=False)

# ---- Normalize/restore meta CRS/transform ----
def _normalize_meta(meta: dict):
    """
    Normalize CRS/transform in meta, returning:
        {
            "crs": pyproj.CRS or None,
            "transform": Affine or None
        }
    meta["crs"] may be any pyproj-compatible input (EPSG string/WKT/object).
    meta["transform"] may be Affine or a 6/9-element tuple.
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
    Unified loader for a single prediction file (.pt or .npz).

    - .pt:
        torch.load, prefer pred['meta'] if available.
        If CRS/transform missing, fill from a same-name .tif in img_root.
    - .npz:
        read arrays, then load meta from same-stem .meta.json.
        If CRS/transform missing, fill from img_root as above.

    Returns:
        (pred_dict, transform(Affine), out_crs(CRS), polygon_id(str or None))
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
            raise FileNotFoundError(f"[Meta] Sidecar meta file not found: {os.path.basename(meta_path)}")
        with open(meta_path, "r", encoding="utf-8") as f:
            j = json.load(f)
        meta = j.get("meta", None)

    # Normalize CRS/transform from meta
    transform = None
    out_crs = None
    if isinstance(meta, dict):
        nm = _normalize_meta(meta)
        transform = nm["transform"]
        out_crs = nm["crs"]

    # Fill from img_root if missing
    if transform is None or out_crs is None:
        if img_root is None:
            raise RuntimeError(
                f"[Meta] {os.path.basename(path)} has no transform/CRS in meta, "
                f"and IMG_ROOT was not provided to recover from a TIF."
            )
        m2 = _load_meta_from_tif_guess(path, img_root)
        transform = m2["transform"]
        out_crs = CRS.from_user_input(m2["crs"]) if not isinstance(m2["crs"], CRS) else m2["crs"]

    # Determine polygon_id
    polygon_id = None
    if isinstance(pred, dict):
        polygon_id = pred.get("polygon_id") or ((pred.get("meta") or {}).get("polygon_id"))
    if polygon_id is None:
        polygon_id = _extract_polygon_id_from_name(path)
    polygon_id = str(polygon_id) if polygon_id is not None else None

    return pred, transform, out_crs, polygon_id


def _standardize_geoms_same_crs(gdf: gpd.GeoDataFrame, simplify_eps_m: float = 0.0) -> gpd.GeoDataFrame:
    """
    Fix and (optionally) lightly simplify geometries in-place, without changing CRS.

    Assumes current CRS units are meters.
    """
    g = gdf.copy()
    g["geometry"] = g["geometry"].apply(lambda x: make_valid(x).buffer(0))
    if simplify_eps_m and simplify_eps_m > 0:
        g["geometry"] = g["geometry"].simplify(simplify_eps_m, preserve_topology=True)
    g = g[~g.geometry.is_empty & g.geometry.is_valid].reset_index(drop=True)
    return g

def _dedup_and_fuse_same_crs(
    sub_f: gpd.GeoDataFrame,
    iou_th: float = 0.90,           # IoU >= 0.90 => considered the same piece
    overlap_th: float = 0.85,       # Intersection over smaller polygon >= 0.85 => same piece
    prefer: str = "score",          # Decide which to keep based on score, then area
    simplify_eps_m: float = 0.0,    # Optional geometry simplification (meters)
    fuse_eps_m: float = 0.0         # Optional smoothing buffer for merged polygons (meters)
) -> gpd.GeoDataFrame:
    """
    Deduplicate and optionally fuse polygons within the same polygon_id subset.

    Assumptions:
        - Current CRS units are meters.
        - 'score' column exists or will be treated as 0.0 if missing.
    """
    if sub_f.empty:
        return sub_f

    g = _standardize_geoms_same_crs(sub_f, simplify_eps_m)

    # Priority order: higher score, then larger area
    scores = np.nan_to_num(g.get("score", pd.Series(index=g.index, data=0.0)).values, nan=0.0)
    areas  = g.geometry.area.values
    # Sort by score desc, then area desc
    order  = np.lexsort((-areas, -scores))
    
    sidx = g.sindex
    dropped = set()
    kept_rows = []

    for idx in order:
        if idx in dropped:
            continue
        a = g.geometry.iloc[idx]
        keep_idx = idx
        keep_geom = a

        # Candidate overlaps based on bounding-box intersection
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
                # Decide which polygon to keep
                s_keep = scores[keep_idx]; s_j = scores[j]
                a_keep = keep_geom.area;   a_j = b.area

                # Keep polygon with higher score; if tied, keep larger area
                take_j = (s_j > s_keep) or (np.isclose(s_j, s_keep) and a_j > a_keep)
                if take_j:
                    dropped.add(keep_idx)
                    keep_idx = j
                    keep_geom = b
                else:
                    dropped.add(j)
                    # Optional: fuse geometries to smooth pixel staircases
                    if fuse_eps_m and fuse_eps_m > 0:
                        keep_geom = unary_union([keep_geom, b]).buffer(+fuse_eps_m).buffer(-fuse_eps_m)
                    # Otherwise, simply keep keep_geom as-is

        row = g.iloc[keep_idx].copy()
        row.geometry = keep_geom
        kept_rows.append(row)

    out = gpd.GeoDataFrame(kept_rows, crs=sub_f.crs).reset_index(drop=True)
    out = out[~out.geometry.is_empty & out.geometry.is_valid]

    # Final "hard" enforcement: ensure non-overlapping polygons (winner-takes-all)
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


# -----------------------------------------------------------------
# Core processing:
# predictions → many GPKGs
# (one per polygon_id, with `fields` & `boundaries` layers)
# -----------------------------------------------------------------
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

    # Collect both .pt and .npz prediction files
    files = sorted([
        os.path.join(pred_dir, f) for f in os.listdir(pred_dir)
        if f.lower().endswith(".pt") or f.lower().endswith(".npz")
    ])
    if not files:
        raise RuntimeError(f"[Pred] No .pt/.npz files found in {pred_dir}")

    out_crs = None              # Use CRS of the first prediction as target
    cad_reproj_done = False     # Reproject cad_gdf only once

    for pi, path in enumerate(files, 1):
        # Load one prediction (with meta normalization/backfill)
        pred, transform, this_crs, polygon_id = _load_prediction_one(path, img_root)

        # On the first sample, align cadastre CRS to prediction CRS
        if out_crs is None:
            out_crs = this_crs
            if cad_gdf.crs is not None and CRS.from_user_input(cad_gdf.crs) != CRS.from_user_input(out_crs):
                cad_gdf = cad_gdf.to_crs(out_crs)
                cad_reproj_done = True
            print(f"[INFO] Cadastre CRS unified to prediction CRS: {out_crs} "
                  f"(reprojected: {cad_reproj_done})")

        # Print debug info for first few samples
        if pi <= 3:
            print(f"[DBG] {os.path.basename(path)} | poly={polygon_id} | cad_crs={cad_gdf.crs} | pred_crs={out_crs} | tfm={type(transform)}")

        if polygon_id is None:
            print(f"[Warn] {os.path.basename(path)}: could not determine polygon_id, skipping.")
            continue

        # Extract masks and scores
        masks = pred.get("masks", torch.empty(0))
        if (isinstance(masks, torch.Tensor) and masks.numel() == 0) or \
           (isinstance(masks, np.ndarray) and masks.size == 0):
            print(f"[Skip] {os.path.basename(path)}: no masks found.")
            continue

        scores = None
        if isinstance(pred, dict) and "scores" in pred:
            s = pred["scores"]
            scores = s.detach().cpu().numpy() if isinstance(s, torch.Tensor) else s
        N = (masks.shape[0] if isinstance(masks, (np.ndarray, torch.Tensor)) else 0)
        if scores is None or len(scores) != N:
            scores = np.full((N,), np.nan, dtype=float)

        # Lookup target cadastre geometry for this polygon_id
        if polygon_id not in cad_gdf.index:
            print(f"[Warn] {os.path.basename(path)}: polygon ID={polygon_id} not found in cadastre, skipping.")
            continue
        target_geom = cad_gdf.loc[polygon_id, cad_gdf.geometry.name]

        # Masks → binary → de-overlap
        binm = _mask_logits_to_binary(masks, thresh=mask_thresh)
        binm = masks_non_overlapping(binm, scores)

        # Vectorize each instance and clip to target cadastre polygon
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

        print(f"[OK] {pi}/{len(files)} processed {os.path.basename(path)} → {polygon_id}")

    if not records_fields:
        raise RuntimeError("[Result] No field polygons were generated. Check thresholds/matching logic.")

    gdf_fields = gpd.GeoDataFrame(records_fields, crs=out_crs)
    gdf_bounds = gpd.GeoDataFrame(records_bounds, crs=out_crs)

    # Write one GPKG per polygon_id (with `fields` and `boundaries` layers).
    # This makes it easy to inspect per-parcel results; you can also merge into a single GPKG later.

    for pid, sub_f in gdf_fields.groupby("polygon_id"):
        out_path = _ensure_gpkg_path(out_dir, pid)

        # Deduplicate and (optionally) fuse within this polygon_id,
        # entirely in the current CRS (meters).
        sub_f = _dedup_and_fuse_same_crs(
            sub_f,
            iou_th=0.90,
            overlap_th=0.85,
            prefer="score",
            simplify_eps_m=0.0,   # set to 2–3 meters to lightly simplify if boundaries look stair-stepped
            fuse_eps_m=0.0        # set to 2–3 meters if you want merging to smooth overlaps
        )

        # Rebuild boundaries from the deduplicated polygons
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
    
        # Write to GPKG
        sub_f.to_file(out_path, driver="GPKG", layer="fields")
        if len(sub_b):
            sub_b.to_file(out_path, driver="GPKG", layer="boundaries")
    
        print(f"[Write] {pid}: {out_path}  (fields={len(sub_f)}, boundaries={len(sub_b)})")

    return True

# --------------------------------------------
# Merge multiple GPKGs → one merged.gpkg
# (fields_all + boundaries_all layers)
# --------------------------------------------
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

# ------------------------------------
# One-shot entrypoint for main.py
# ------------------------------------
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
    merged_out_dir: Optional[str] = None,     # optional: separate directory for merged result (e.g., final_result)
    merged_filename: str = "merged.gpkg",     # name of merged GPKG
):
    # 1) Load cadastre
    cad_gdf = _load_cadastre(cad_source, layer_name, id_field)
    print(f"[Cadastre] Loaded {len(cad_gdf)} target polygons; CRS={cad_gdf.crs}")

    # 2) Per-image → per-polygon GPKGs
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

    # 3) Merge all GPKGs into a single merged.gpkg (can be placed in a separate directory)
    merged_base_dir = merged_out_dir or out_dir
    os.makedirs(merged_base_dir, exist_ok=True)
    merged_path = os.path.join(merged_base_dir, merged_filename)
    merged_path = merge_gpkgs(out_dir, merged_path, target_crs=target_crs)

    return {"out_dir": out_dir, "merged": merged_path}


