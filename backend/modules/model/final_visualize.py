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
    Quick visualization utility — display a GeoTIFF base image overlaid with
    vector data from a GPKG file (no PNG saving).
    """
    # 1) Open GeoTIFF and display as base map
    with rasterio.open(geotiff_path) as src:
        rcrs = src.crs
        fig, ax = plt.subplots(figsize=figsize)
        rio_show(src, ax=ax, alpha=alpha_image)  # Automatically uses the raster transform for correct georeferencing

    # 2) Read and overlay vector layers (try *_all first, fallback to single layer name)
    def _read_first_ok(path, names):
        for nm in names:
            try:
                return gpd.read_file(path, layer=nm)
            except Exception:
                continue
        return None

    gdf_fields = _read_first_ok(gpkg_path, layer_candidates_fields)
    gdf_bounds = _read_first_ok(gpkg_path, layer_candidates_bounds)

    # Reproject vector layers to match raster CRS if needed
    def _to_raster_crs(gdf):
        if gdf is None or gdf.empty:
            return gdf
        if gdf.crs is None:
            # If CRS is missing, assume the same as the raster (safe fallback)
            return gdf.set_crs(rcrs, allow_override=True)
        if CRS.from_user_input(gdf.crs) != CRS.from_user_input(rcrs):
            return gdf.to_crs(rcrs)
        return gdf

    gdf_fields = _to_raster_crs(gdf_fields)
    gdf_bounds = _to_raster_crs(gdf_bounds)

    # 3) Plot vector overlays (outline only, no fill)
    if gdf_fields is not None and not gdf_fields.empty:
        gdf_fields.plot(ax=ax, facecolor="none", edgecolor=edgecolor_fields, linewidth=linewidth)
    if gdf_bounds is not None and not gdf_bounds.empty:
        gdf_bounds.plot(ax=ax, color=edgecolor_bounds, linewidth=linewidth)

    ax.set_title("Overlay preview (TIFF + GPKG)")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.show() 

