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


# ===== TIF-related helper imports =====
# Only used to construct ROI in "tif" mode
# -------------------------------------------
# Dependency: rasterio
try:
    import rasterio 
    from rasterio.warp import transform_bounds
except Exception as _e:
    rasterio = None 


# ============================
# 0) Read JSON & initialize EE
# ============================
def read_service_account_json(json_path: str) -> Tuple[str, Optional[str]]:
    """
    Parse client_email and project_id from a Service Account JSON file.

    Returns:
        (client_email, project_id or None)
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
    print(f"Loaded service account from {p}: {client_email}, project_id={project_id}")
    return client_email, project_id


def ee_init_from_json(json_path: str) -> Tuple[str, str]:
    """
    Initialize Google Earth Engine using a service account JSON and perform
    a lightweight connectivity test.
    """
    client_email, project_id = read_service_account_json(json_path)
    creds = ee.ServiceAccountCredentials(client_email, str(Path(json_path).expanduser().resolve()))
    if project_id:
        ee.Initialize(creds, project=project_id)
    else:
        ee.Initialize(creds)
        
    # Simple ping to verify that EE client is working
    _ = ee.Number(1).getInfo() 
    print(f"EE initialized | sa={client_email} | project={project_id or '<none>'}")
    return client_email, (project_id or "")


def infer_utm_epsg_from_lonlat(lon: float, lat: float) -> str:
    zone = int(math.floor((lon + 180) / 6) + 1)
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return f"EPSG:{epsg}"


# ======================================================
# 1) S2 RGB 8-bit visualization (aligned with notebook)
# ======================================================
def build_and_visualize_s2_rgb8_exact(roi_geom,
                                      s2_start: str,
                                      s2_end: str,
                                      cloud_max: int,
                                      *,
                                      use_harmonized: bool = False,
                                      pct_bounds=(2, 98)) -> ee.Image:
    """
    Build a Sentinel-2 RGB image exactly as in the reference notebook:

    - Collection: COPERNICUS/S2_SR (or COPERNICUS/S2_SR_HARMONIZED)
    - Cloud masking: remove pixels with SCL in {3, 8, 9, 10}
      (3 = shadow; 8/9/10 = clouds and cloud shadows)
    - Compositing: median
    - Stretch: compute percentiles (e.g., 2–98%) over the ROI
    - Visualization: use ee.Image.visualize on the server side to get 8-bit RGB

    Returns:
        ee.Image: uint8, 3-band RGB image.
    """
    s2_col = 'COPERNICUS/S2_SR' if not use_harmonized else 'COPERNICUS/S2_SR_HARMONIZED'

    def mask_scl(image):
        scl = image.select('SCL')
        cloud = scl.eq(3).Or(scl.eq(8)).Or(scl.eq(9)).Or(scl.eq(10))  # 3=shadow, 8/9/10=clouds
        # Keep reflectance bands + SCL; use SCL to mask clouds; convert DN to reflectance
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

    # Visible RGB
    rgb_float = img.select(['B4','B3','B2'])

    # Compute percentiles over the original ROI (to match the notebook behavior)
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
    # Key point: server-side visualize -> direct 8-bit to avoid local quantization issues
    rgb8 = rgb_float.visualize(**vis_params)
    return rgb8  # uint8, 3-band


# ==================================
# 2) Select polygons within ROI
# ==================================
def select_polygons_in_roi(cadastre_asset_id: str,
                                      roi_geom,
                                      id_col: str):
    
    cadastre_fc = ee.FeatureCollection(cadastre_asset_id)
    cad_in_roi = cadastre_fc.filterBounds(roi_geom)
    id_list = cad_in_roi.aggregate_array(id_col)
    return cad_in_roi, id_list


# ===== ROI builder (two modes: "bbox" and "tif") =====
def _meters_per_degree(lat_deg: float) -> Tuple[float, float]:
    """
    Approximate conversion from degrees to meters at a given latitude.

    Returns:
        (meters_per_degree_lon, meters_per_degree_lat)
    """
    from math import cos, pi
    lat_rad = lat_deg * pi / 180.0
    lon_m = 111320.0 * cos(lat_rad)
    lat_m = 110574.0
    return lon_m, lat_m


def _tif_bounds_wgs84_and_pixm(tif_path: str) -> Tuple[Tuple[float,float,float,float], float, Tuple[float,float]]:
    """
    Read the bounding box of a TIF (converted to WGS84) and estimate pixel size in meters.

    Returns:
        (minx, miny, maxx, maxy) in WGS84,
        pixel_size_m (average),
        (center_lon, center_lat)
    """
    if rasterio is None:
        raise RuntimeError("rasterio is not installed; cannot construct ROI in 'tif' mode.")
    if not os.path.exists(tif_path):
        raise FileNotFoundError(f"TIF not found: {tif_path}")

    with rasterio.open(tif_path) as ds:
        if ds.count < 1:
            raise ValueError(f"TIF has no valid bands: {tif_path}")
        b = ds.bounds
        src_crs = ds.crs
        if src_crs is None:
            raise ValueError(f"TIF is missing CRS: {tif_path}")
        # Reproject bounds to WGS84
        bounds_wgs84 = transform_bounds(src_crs, "EPSG:4326", b.left, b.bottom, b.right, b.top, densify_pts=21)

        # Estimate pixel size in meters
        # If projected (meters), use affine pixel size directly;
        # if geographic (degrees), approximate using meters-per-degree at center latitude.
        a = ds.transform.a
        e = ds.transform.e  # usually negative; take abs
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
    Build ROI and export region.

    Returns:
        (roi, export_region, info)

    - "bbox" mode:
        roi = Rectangle(bbox)
        export_region = roi.buffer(+pad_m).bounds()
    - "tif" mode:
        roi = Rectangle(tif_bounds_wgs84).buffer(-margin_m).bounds()
        export_region = roi

    info contains at least:
        {
          "mode": roi_mode,
          "bbox_wgs84": (minx, miny, maxx, maxy),
          "margin_m": margin_m,
          "pixel_size_m": pixel_size_m,
          "center": {"lon": ..., "lat": ...}
        }
    """
    if roi_mode not in ("bbox", "tif"):
        raise ValueError("roi_mode must be either 'bbox' or 'tif'")

    info: Dict[str, Any] = {"mode": roi_mode}

    if roi_mode == "bbox":
        if not bbox or len(bbox) != 4:
            raise ValueError("In 'bbox' mode, bbox=(minx, miny, maxx, maxy) (WGS84) must be provided.")
        minx, miny, maxx, maxy = bbox
        info.update({"bbox_wgs84": bbox, "margin_m": pad_m})
        roi = ee.Geometry.Rectangle([minx, miny, maxx, maxy])
        export_region = roi.buffer(pad_m).bounds()
        return roi, export_region, info

    # "tif" mode
    if not user_tif:
        raise ValueError("In 'tif' mode, user_tif path must be provided.")
    bbox_wgs84, pixel_size_m, center = _tif_bounds_wgs84_and_pixm(user_tif)
    minx, miny, maxx, maxy = bbox_wgs84
    roi_base = ee.Geometry.Rectangle([minx, miny, maxx, maxy])

    # Compute shrink margin (prefer meters; otherwise use pixel count * pixel_size_m)
    if shrink_margin_m is not None:
        margin_m = float(shrink_margin_m)
    else:
        px = shrink_margin_pixels if shrink_margin_pixels is not None else 0
        margin_m = max(0.0, px * float(pixel_size_m))

    # Safety: avoid shrinking too much such that geometry becomes empty
    roi_shrunk = roi_base.buffer(-margin_m).bounds() if margin_m > 0 else roi_base
    # Note: EE buffer distance is always in meters, even when geometry is in WGS84.

    info.update({
        "bbox_wgs84": bbox_wgs84,
        "pixel_size_m": pixel_size_m,
        "margin_m": margin_m,
        "center": {"lon": center[0], "lat": center[1]}
    })
    roi = roi_shrunk
    export_region = roi  # In "tif" mode, using ROI as export region is usually enough
    return roi, export_region, info


# ==============================================================
# 3) High-level pipeline: export 8-bit RGB and filtered lots
# ==============================================================
def run_gee_fetch(
    *,
    cadastre_asset: str,
    id_col: str,
    bbox: tuple = None,               # (minx, miny, maxx, maxy) in WGS84 (default None)
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
    # ===== ROI mode and TIF-based parameters =====
    roi_mode: str = "bbox",                 # "bbox" | "tif"
    user_tif: Optional[str] = None,         
    shrink_margin_m: Optional[float] = None,
    shrink_margin_pixels: Optional[int] = 3 
) -> Dict[str, Any]:
    """
    Main pipeline (aligned with the notebook visualization/export chain):

    - S2_SR + SCL-based cloud masking + median composite
    - Compute 2–98% percentiles over the original ROI and visualize to 8-bit
    - Download visualized RGB8 to out_tif
      (in "bbox" mode: region = expanded ROI; in "tif" mode: download is skipped)
    - Select parcels in the ROI and export:
        * ID list to a text file
        * Filtered parcels to filter_lot.gpkg
    """
    ee_init_from_json(json_path=json_path)

    # ===== Unified ROI builder for both bbox and tif modes =====
    roi, export_region, roi_info = make_rois(
        roi_mode=roi_mode,
        bbox=bbox,
        user_tif=user_tif,
        pad_m=pad_m,
        shrink_margin_m=shrink_margin_m,
        shrink_margin_pixels=shrink_margin_pixels
    )
    print(f"[ROI] mode={roi_info['mode']} | bbox(WGS84)={roi_info['bbox_wgs84']} | margin_m={roi_info['margin_m']}")

    # Output CRS (required only when exporting S2 in bbox mode)
    if crs_out.lower() == "auto" and roi_mode == "bbox": 
        minx, miny, maxx, maxy = roi_info["bbox_wgs84"]
        crs_out_val = infer_utm_epsg_from_lonlat((minx + maxx) / 2, (miny + maxy) / 2)
    elif roi_mode == "bbox":
        CRS.from_user_input(crs_out)
        crs_out_val = crs_out
    else:
        crs_out_val = None  # "tif" mode: S2 download is skipped; no export CRS needed
    if roi_mode == "bbox":
        print(f"[CRS] Output CRS: {crs_out_val} | Buffer: {pad_m} m")

    # Select polygons and write ID list
    selected_polys, ids = select_polygons_in_roi(
        cadastre_asset_id=cadastre_asset, roi_geom=roi, id_col=id_col
    )
    id_list_py = ids.getInfo()
    print(f"[IDs] Found {len(id_list_py)} parcels | first 10: {id_list_py[:10]}")

    os.makedirs(os.path.dirname(out_ids), exist_ok=True)
    with open(out_ids, "w", encoding="utf-8") as f:
        f.write("\n".join(map(str, id_list_py)))
    print(f"[IDs] ID list written to: {out_ids}")

    # bbox mode: download S2; tif mode: skip download (user provides their own image)
    if roi_mode == "bbox":
        # Server-side visualize -> 8-bit, aligned with the notebook
        rgb8_exact = build_and_visualize_s2_rgb8_exact(
            roi, s2_start, s2_end, cloud_max,
            use_harmonized=False, pct_bounds=(2, 98)
        ).clip(export_region)

        os.makedirs(os.path.dirname(out_tif), exist_ok=True)
        print(f"[DL] Downloading 8-bit image to: {out_tif}")

        geemap.download_ee_image(
            image=rgb8_exact,
            filename=out_tif,
            region=export_region,
            scale=10,
            crs=crs_out_val,
            overwrite=True   # visualize already returns uint8; no need to specify dtype
        )
        print("Sentinel-2 RGB8 download completed.")
        tif_to_report = out_tif
        base_dir = os.path.dirname(out_tif)
    else:
        print("[DL] 'tif' mode: skipping S2 download, using user-provided image as reference.")
        tif_to_report = user_tif
        base_dir = os.path.dirname(user_tif) if user_tif else "."

    # Export filtered parcels GPKG (same directory as S2 TIF or user TIF)
    gdf = geemap.ee_to_gdf(selected_polys)
    gpkg_path = os.path.join(base_dir, "filter_lot.gpkg")
    gdf.to_file(gpkg_path, driver="GPKG")
    print(f"Cadastral layer exported to: {gpkg_path}")

    return {
        "tif": tif_to_report,            # bbox -> out_tif; tif -> user_tif
        "ids": out_ids,
        "gpkg": gpkg_path,
        "id_list": id_list_py,
        "roi_info": roi_info            # Returned for easier debugging/reproducibility
    }
