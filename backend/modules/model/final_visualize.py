# 放到 model/postprocessing.py 末尾（或新建 model/visualize.py）
import matplotlib.pyplot as plt
import geopandas as gpd
import rasterio
from rasterio.plot import show as rio_show
from pyproj import CRS

def visualize_overlay_window(
    geotiff_path: str,
    gpkg_path: str,
    layer_candidates_fields=("fields_all", "fields"),
    layer_candidates_bounds=("boundaries_all", "boundaries"),
    alpha_image: float = 1.0,
    edgecolor_fields: str = "lime",
    edgecolor_bounds: str = "red",
    linewidth: float = 0.6,
    figsize=(10, 10),
):
    """
    直接弹窗预览：底图为 GeoTIFF，叠加 GPKG 面/线，不保存PNG。
    - geotiff_path: 例如 'gee_out/S2_RGB8.tif'
    - gpkg_path:    例如 'final_result/merged.gpkg' 或单个 id 的 .gpkg
    """
    # 1) 打开影像
    with rasterio.open(geotiff_path) as src:
        rcrs = src.crs
        fig, ax = plt.subplots(figsize=figsize)
        rio_show(src, ax=ax, alpha=alpha_image)  # 自动按 transform 显示

    # 2) 读取并叠加图层（先尝试 *_all，再尝试单图层名）
    def _read_first_ok(path, names):
        for nm in names:
            try:
                return gpd.read_file(path, layer=nm)
            except Exception:
                continue
        return None

    gdf_fields = _read_first_ok(gpkg_path, layer_candidates_fields)
    gdf_bounds = _read_first_ok(gpkg_path, layer_candidates_bounds)

    def _to_raster_crs(gdf):
        if gdf is None or gdf.empty:
            return gdf
        if gdf.crs is None:
            # 如果缺CRS，直接设为影像CRS以保证能画（保守做法）
            return gdf.set_crs(rcrs, allow_override=True)
        if CRS.from_user_input(gdf.crs) != CRS.from_user_input(rcrs):
            return gdf.to_crs(rcrs)
        return gdf

    gdf_fields = _to_raster_crs(gdf_fields)
    gdf_bounds = _to_raster_crs(gdf_bounds)

    # 3) 绘制矢量（只画边界，不填充）
    if gdf_fields is not None and not gdf_fields.empty:
        gdf_fields.plot(ax=ax, facecolor="none", edgecolor=edgecolor_fields, linewidth=linewidth)
    if gdf_bounds is not None and not gdf_bounds.empty:
        gdf_bounds.plot(ax=ax, color=edgecolor_bounds, linewidth=linewidth)

    ax.set_title("Overlay preview (TIFF + GPKG)")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()  # ← 不保存，直接弹窗

