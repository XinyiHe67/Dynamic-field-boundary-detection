import os, re
from pathlib import Path
import geopandas as gpd
import pandas as pd

try:
    import pyogrio
    HAS_PYOGRIO = True
except Exception:
    HAS_PYOGRIO = False


# ===================== 配置部分 =====================
PATCH_DIR = Path("./modules/model/InferenceDataset")          # 你的 patch 文件夹路径
IN_GPKG = Path("./modules/gee_out/filter_lot.gpkg")
LAYER_NAME = "filter_lot"                         # GPKG 图层名
OUT_DIR = Path(__file__).resolve().parent.parent / "./model/Preprocess_Target_polygon"    # 输出文件夹
ID_COL = "OBJECTID"                                     # GPKG 里代表目标ID的列名
# ====================================================

def guess_id_col(cols, user_id=None):
    if user_id and user_id in cols: return user_id
    for c in ["objectID","OBJECTID","ObjectID","object_id","id","ID"]:
        if c in cols: return c
    raise ValueError(f"未找到 objectID 列，可选列有：{list(cols)}")

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
        raise RuntimeError("未在目录中解析到任何 objectID，请检查文件命名是否类似 patch_<ID>_*.tif")
    print(f"[INFO] 收集到 {len(ids)} 个唯一 objectID：{ids[:8]}{' ...' if len(ids)>8 else ''}")
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
        print(f"[INFO] 使用 pyogrio 流式读取，ID列={id_col}")

        for oid in ids:
            oid_safe = oid.replace("'", "''")  # 先替换单引号，防止 SQL 注入
            where = f"{id_col} = '{oid_safe}'"
            sub = pyogrio.read_dataframe(IN_GPKG, layer=LAYER_NAME, where=where)
            if sub.empty:
                print(f"[WARN] ID {oid} 未找到，跳过。")
                continue
            sub = fix_geometry_if_needed(sub)
            out_path = OUT_DIR / f"{sanitize_filename(oid)}.gpkg"
            pyogrio.write_dataframe(sub, out_path, layer="polygon", driver="GPKG")
            print(f"[OK] 写出 {out_path} (要素数={len(sub)})")
    else:
        print("[INFO] 未安装 pyogrio，使用 geopandas 一次性读取。")
        gdf = gpd.read_file(IN_GPKG, layer=LAYER_NAME)
        id_col = guess_id_col(gdf.columns, ID_COL)
        gdf[id_col] = gdf[id_col].astype(str)
        subset = gdf[gdf[id_col].isin(set(ids))].copy()
        if subset.empty:
            raise RuntimeError("筛选结果为空，未找到匹配的ID。")
        subset = fix_geometry_if_needed(subset)

        for oid, sub in subset.groupby(id_col):
            out_path = OUT_DIR / f"{sanitize_filename(oid)}.gpkg"
            sub.to_file(out_path, layer="polygon", driver="GPKG")
            print(f"[OK] 写出 {out_path} (要素数={len(sub)})")

    print("[DONE] 全部完成 ")


if __name__ == "__main__":
    export_split_by_id()
