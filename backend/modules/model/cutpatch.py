import os
import math
import csv
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import rowcol
from rasterio import features
import geopandas as gpd
from shapely.geometry import box
from typing import Callable, Tuple, List, Optional

def prepare_data(tif_path: str, vector_path: str, output_dir: str):
    """
    加载影像和矢量文件，确保CRS一致，并创建输出目录。
    
    返回：
        - src: rasterio 数据源
        - gdf: CRS对齐后的 GeoDataFrame
        - transform: 仿射变换矩阵
        - profile: raster profile
        - shape: (高度, 宽度)
    """
    os.makedirs(output_dir, exist_ok=True)

    # 打开大图像
    src = rasterio.open(tif_path)
    transform = src.transform
    profile = src.profile
    shape = (src.height, src.width)
    crs = src.crs

    # 读取矢量图层
    gdf = gpd.read_file(vector_path)
    if gdf.crs is None:
        raise ValueError("No CRS found in vector file.")
    if gdf.crs != crs:
        gdf = gdf.to_crs(crs)

    return src, gdf, transform, profile, shape


def geom_bbox_px(geom, transform, shape, margin_px=0):
    r_height, r_width = shape
    minx, miny, maxx, maxy = geom.bounds
    r0, c0 = rowcol(transform, minx, maxy)
    r1, c1 = rowcol(transform, maxx, miny)
    ymin, ymax = min(r0, r1), max(r0, r1)
    xmin, xmax = min(c0, c1), max(c0, c1)

    xmin -= margin_px
    ymin -= margin_px
    xmax += margin_px
    ymax += margin_px

    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(r_width, xmax)
    ymax = min(r_height, ymax)

    w = max(0, xmax - xmin)
    h = max(0, ymax - ymin)
    return int(xmin), int(ymin), int(w), int(h)

def write_patch(src, window, out_path, profile_base, pad_to_full=False):
    win_w = int(window.width)
    win_h = int(window.height)
    data = src.read(window=window)

    if pad_to_full:
        pad_w, pad_h = int(window.width), int(window.height)
        if data.shape[1] != pad_h or data.shape[2] != pad_w:
            canvas = np.zeros((data.shape[0], pad_h, pad_w), dtype=data.dtype)
            canvas[:, :data.shape[1], :data.shape[2]] = data
            data = canvas
            win_w, win_h = pad_w, pad_h

    transform = rasterio.windows.transform(window, src.transform)
    profile = profile_base.copy()
    profile.update({
        "height": win_h,
        "width": win_w,
        "transform": transform
    })

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data)

def manifest() -> Tuple[str, int]:
    """
    主函数：不接受任何参数，内部自动加载路径、参数、数据，生成 patch 与 manifest.csv
    """
    # ====== 路径配置（可按需修改） ======
    BIG_TIF = "./modules/gee_out/S2_RGB8.tif"
    POLY_VEC = "./modules/gee_out/filter_lot.gpkg"
    OUT_DIR = "./modules/model/InferenceDataset"
    ID_FIELD = "OBJECTID"

    # ====== Patch 生成参数 ======
    PATCH_SMALL = 256
    PATCH_LARGE = 512
    TILE_OVERLAP = 256
    MARGIN_PX = 20
    PAD_TO_FULL = False

    # ====== 准备数据 ======
    # from patch_generator import prepare_data, geom_bbox_px, write_patch  # 或直接放当前脚本
    src, gdf, transform, profile, shape = prepare_data(BIG_TIF, POLY_VEC, OUT_DIR)
    r_height, r_width = shape

    def write_patch_adapter(window, out_path):
        write_patch(src, window, out_path, profile, pad_to_full=PAD_TO_FULL)

    def geom_bbox_px_adapter(geom):
        return geom_bbox_px(geom, transform, shape, margin_px=MARGIN_PX)

    os.makedirs(OUT_DIR, exist_ok=True)
    manifest_path = os.path.join(OUT_DIR, "manifest.csv")
    rows_written = 0

    if PATCH_LARGE <= TILE_OVERLAP:
        raise ValueError("tile_overlap 必须小于 patch_large")

    with open(manifest_path, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["patch_path", "poly_id", "x_off", "y_off", "width", "height", "patch_size", "tile_idx"])

        for idx, feat in gdf.iterrows():
            geom = feat.geometry
            poly_id = str(feat.get(ID_FIELD, f"poly_{idx}"))

            xmin, ymin, bw, bh = geom_bbox_px_adapter(geom)
            xmin, ymin, bw, bh = map(int, (xmin, ymin, bw, bh))
            if bw <= 0 or bh <= 0:
                continue

            max_dim = max(bw, bh)

            def _centered_window(psize: int):
                x_off = max(0, min(int(xmin + bw // 2 - psize // 2), r_width - psize))
                y_off = max(0, min(int(ymin + bh // 2 - psize // 2), r_height - psize))
                win_w = min(psize, r_width - x_off)
                win_h = min(psize, r_height - y_off)
                return x_off, y_off, win_w, win_h

            if max_dim <= PATCH_SMALL:
                psize = PATCH_SMALL
                x_off, y_off, win_w, win_h = _centered_window(psize)
                window = Window(x_off, y_off, win_w, win_h)
                out_name = os.path.join(OUT_DIR, f"patch_{poly_id}_S_{x_off}_{y_off}.tif")
                write_patch_adapter(window, out_name)
                writer.writerow([out_name, poly_id, x_off, y_off, win_w, win_h, psize, 0])
                rows_written += 1

            elif max_dim <= PATCH_LARGE:
                psize = PATCH_LARGE
                x_off, y_off, win_w, win_h = _centered_window(psize)
                window = Window(x_off, y_off, win_w, win_h)
                out_name = os.path.join(OUT_DIR, f"patch_{poly_id}_L_{x_off}_{y_off}.tif")
                write_patch_adapter(window, out_name)
                writer.writerow([out_name, poly_id, x_off, y_off, win_w, win_h, psize, 0])
                rows_written += 1

            else:
                psize = PATCH_LARGE
                stride = psize - TILE_OVERLAP
                tiles: List[Tuple[int, int, int, int]] = []

                for y_off in range(ymin, ymin + bh, stride):
                    for x_off in range(xmin, xmin + bw, stride):
                        x_off_clamp = max(0, min(int(x_off), r_width - psize))
                        y_off_clamp = max(0, min(int(y_off), r_height - psize))
                        win_w = min(psize, r_width - x_off_clamp)
                        win_h = min(psize, r_height - y_off_clamp)
                        tiles.append((x_off_clamp, y_off_clamp, win_w, win_h))

                seen = set()
                unique_tiles = []
                for t in tiles:
                    if t not in seen:
                        seen.add(t)
                        unique_tiles.append(t)

                for ti, (x_off, y_off, win_w, win_h) in enumerate(unique_tiles):
                    window = Window(x_off, y_off, win_w, win_h)
                    out_name = os.path.join(OUT_DIR, f"patch_{poly_id}_T{ti:02d}_{x_off}_{y_off}.tif")
                    write_patch_adapter(window, out_name)
                    writer.writerow([out_name, poly_id, x_off, y_off, win_w, win_h, psize, ti])
                    rows_written += 1

    try:
        src.close()
    except Exception:
        pass

    print(f"Complete. Output: {OUT_DIR}\n Manifest: {manifest_path}")
    return manifest_path, rows_written

