import os, re
from pathlib import Path
import geopandas as gpd
import pandas as pd

try:
    import pyogrio
    HAS_PYOGRIO = True
except Exception:
    HAS_PYOGRIO = False


# ===================== Configuration =====================
PATCH_DIR = Path("./modules/model/InferenceDataset")                                      # Directory containing patch files
IN_GPKG = Path("./modules/gee_out/filter_lot.gpkg")                                       # Input GPKG file
LAYER_NAME = "filter_lot"                                                                 # GPKG layer name
OUT_DIR = Path(__file__).resolve().parent.parent / "./model/Preprocess_Target_polygon"    # Output directory
ID_COL = "OBJECTID"                                                                       # Column name representing target ID in GPKG
# ====================================================

def guess_id_col(cols, user_id=None):
    if user_id and user_id in cols: return user_id
    for c in ["objectID","OBJECTID","ObjectID","object_id","id","ID"]:
        if c in cols: return c
    raise ValueError(f"ID column not found. Available columns: {list(cols)}")

def extract_id_from_name(name: str):
    base = os.path.basename(name)
    m = re.match(r"patch_([^_]+)_", base)
    return m.group(1) if m else None

def collect_ids_from_dir(patch_dir: Path):
    exts = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
    ids = []
    for p in patch_dir.iterdir():
        if p.is_file() and p.suffix.lower() in exts and "patch_" in p.name:
            oid = extract_id_from_name(p.name)
            if oid is not None:
                ids.append(str(oid))
    ids = sorted(set(ids))
    if not ids:
        raise RuntimeError("No object IDs were parsed from the directory. "
                           "Please check that filenames look like patch_<ID>_*.tif")
    print(f"[INFO] Collected {len(ids)} unique object IDs: {ids[:8]}{' ...' if len(ids)>8 else ''}")
    return ids

def sanitize_filename(s: str) -> str:
    return re.sub(r"[^0-9A-Za-z_\-]+", "_", str(s).strip()) or "unknown"

def fix_geometry_if_needed(gdf):
    if not gdf.geometry.is_valid.all():
        gdf = gdf.copy()
        gdf["geometry"] = gdf.geometry.buffer(0)
    return gdf

def export_split_by_id():
    ids = collect_ids_from_dir(PATCH_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if HAS_PYOGRIO:
        head = pyogrio.read_dataframe(IN_GPKG, layer=LAYER_NAME, max_features=5)
        id_col = guess_id_col(head.columns, ID_COL)
        print(f"[INFO] Using pyogrio for streaming read, ID column = {id_col}")

        for oid in ids:
            # Escape single quotes to avoid SQL injection issues
            oid_safe = oid.replace("'", "''")  
            where = f"{id_col} = '{oid_safe}'"
            sub = pyogrio.read_dataframe(IN_GPKG, layer=LAYER_NAME, where=where)
            if sub.empty:
                print(f"[WARN] ID {oid} not found, skipped.")
                continue
            sub = fix_geometry_if_needed(sub)
            out_path = OUT_DIR / f"{sanitize_filename(oid)}.gpkg"
            pyogrio.write_dataframe(sub, out_path, layer="polygon", driver="GPKG")
            print(f"[OK] Wrote {out_path} (features={len(sub)})")
    else:
        print("[INFO] pyogrio not installed, using geopandas full read.")
        gdf = gpd.read_file(IN_GPKG, layer=LAYER_NAME)
        id_col = guess_id_col(gdf.columns, ID_COL)
        gdf[id_col] = gdf[id_col].astype(str)
        subset = gdf[gdf[id_col].isin(set(ids))].copy()
        if subset.empty:
            raise RuntimeError("Filtered result is empty, no matching IDs found.")
        subset = fix_geometry_if_needed(subset)

        for oid, sub in subset.groupby(id_col):
            out_path = OUT_DIR / f"{sanitize_filename(oid)}.gpkg"
            sub.to_file(out_path, layer="polygon", driver="GPKG")
            print(f"[OK] Wrote {out_path} (features={len(sub)})")

    print("[DONE] All exports completed. ")


if __name__ == "__main__":
    export_split_by_id()
