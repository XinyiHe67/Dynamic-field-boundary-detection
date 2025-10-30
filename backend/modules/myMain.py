import argparse
import os
from glob import glob
from pathlib import Path
import datetime as _dt
import sys
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "model"))
from zoneinfo import ZoneInfo
import pandas as pd
import geopandas as gpd
import shutil

from model.dataset import DataConfig, build_loaders
from model import train as train_mod
from model import infer as infer_mod
from model import visualize as viz_mod  # 可视化模块
from model import evaluation as eval_mod
from model import postprocessing as post_mod
# bbox 模式：走 gee_pipeline.py
from model.gee_pipeline import run_gee_fetch_and_export as run_gee_bbox
from model.gee_pipeline import read_service_account_json  # 统一用这边的读 SA

# tif 模式：走 image_input.py
from model.image_input import run_gee_fetch as run_gee_tif
from model import cutpatch
from model import exportTargetPolygon
from model.data_loader_auto import Infer_DataConfig
from model.data_loader_auto import build_test_loaders
from model.final_visualize import visualize_overlay_window

import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show
import base64
from io import BytesIO


def parse_args():
    p = argparse.ArgumentParser()

    # mode 增加 preprocess
    p.add_argument("--mode", choices=["all", "train", "infer", 
        "visualize", "eval", "post", "gee", "auto", "preprocess",
        "auto_data"], 
        default="auto")
    

    # ===== GEE 专用参数 =====
    p.add_argument("--gee_key_path",
    default="./modules/model/cs88-468908-d6ce3af55bb8.json")
    p.add_argument("--gee_service_account", default="")
    p.add_argument("--gee_project_id", default="")
    p.add_argument("--gee_cad_asset",
    default="projects/cs88-468908/assets/polygon")
    p.add_argument("--gee_id_col", default="OBJECTID")
    p.add_argument("--gee_bbox", default = None)
    p.add_argument("--gee_s2_start", default=None)
    p.add_argument("--gee_s2_end", default=None)
    p.add_argument("--gee_cloud_max", type=int, default=40)
    p.add_argument("--gee_farmland_th", type=float, default=0.25)
    p.add_argument("--gee_out_tif", default="./modules/gee_out/S2_RGB8.tif")
    p.add_argument("--gee_out_ids", default="./modules/gee_out/id_list.txt")
    p.add_argument("--gee_pad_m", type=int, default=5000,
                   help="下载区域外扩距离（米）")
    p.add_argument("--gee_crs_out", default="auto",
                   help="'auto' 或 'EPSG:xxxx'")
    p.add_argument("--roi_mode", choices=["bbox", "tif"], default="tif",
               help="bbox=坐标模式(gee_pipeline.py)，tif=图片模式(image_input.py)")
    p.add_argument("--user_tif", default="./modules/gee_out/S2_RGB8.tif",
               help="tif 模式下使用的 GeoTIFF 路径（作为 ROI，不会下载 S2）")
    
    # ===== Patch 生成（cutpatch.py）=====
    p.add_argument("--patch_tif", default="./modules/gee_out/S2_RGB8.tif")
    p.add_argument("--patch_vector", default="./modules/gee_out/filter_lot.gpkg")
    p.add_argument("--patch_out_dir", default="./modules/model/InferenceDataset")
    p.add_argument("--patch_id_field", default="OBJECTID")
    p.add_argument("--patch_small", type=int, default=256)
    p.add_argument("--patch_large", type=int, default=512)
    p.add_argument("--patch_overlap", type=int, default=256)
    p.add_argument("--patch_margin", type=int, default=20)
    p.add_argument("--patch_pad_full", action="store_true")

    # ===== 按ID导出（exportTargetPolygon.py）=====
    p.add_argument("--exp_in_gpkg", default="./modules/gee_out/filter_lot.gpkg")
    p.add_argument("--exp_layer_name", default="filter_lot")
    p.add_argument("--exp_out_dir", default="./modules/model/Preprocess_Target_polygon")
    p.add_argument("--exp_id_col", default="OBJECTID")

    # dataset
    p.add_argument("--img_dir", default="./modules/model/Dataset")
    p.add_argument("--ann_dir", default="./modules/model/Dataset/withID")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--split_file", default="data/split_info.json")
    p.add_argument("--split_method", default="random", choices=["random", "pattern"])
    p.add_argument("--train_ratio", type=float, default=0.6)
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--test_ratio", type=float, default=0.2)
    p.add_argument("--random_seed", type=int, default=42)

    # inference_dataset_only
    p.add_argument("--inference_img_dir", default="./modules/model/InferenceDataset")

    # train
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--accumulate_grad_batches", type=int, default=2)
    p.add_argument("--warmup_epochs", type=int, default=1)
    p.add_argument("--mask_loss_weight", type=float, default=2.0)
    p.add_argument("--box_loss_weight", type=float, default=1.0)
    p.add_argument("--use_dice_loss", action="store_true", default=True)
    p.add_argument("--use_iou_loss", action="store_true", default=True)
    p.add_argument("--ckpt_dir", default="./modules/checkpoints")

    # infer
    p.add_argument("--ckpt", default="")              
    p.add_argument("--out_dir", default="./modules/runs/predictions")
    p.add_argument("--out_format", default="pt", choices=["pt", "npz"])

    # visualize
    p.add_argument("--viz_out_dir", default="./modules/runs/visualizations")
    p.add_argument("--viz_mode", default="side_by_side", choices=["side_by_side", "boundary"])
    p.add_argument("--viz_limit", type=int, default=0)  # 0 = no restriction
    p.add_argument("--score_thresh", type=float, default=0.5)

    # evaluation
    p.add_argument("--eval_csv", default="./modules/runs/eval_results.csv")
    p.add_argument("--eval_score_thresh", type=float, default=0.5)
    p.add_argument("--eval_iou_thr", type=float, default=0.5)
    p.add_argument("--eval_boundary_tol", type=int, default=3)

    # --- Postprocessing params ---
    p.add_argument("--pred_dir", default="./modules/runs/predictions")
    p.add_argument("--cad_source", default="./modules/model/Target_polygon")      
    p.add_argument("--layer_name", default=None)                  
    p.add_argument("--cad_id_field", default="OBJECTID")
    p.add_argument("--post_out_dir", default="./modules/post_outputs")
    p.add_argument("--img_root", default=None)                     
    p.add_argument("--mask_thresh", type=float, default=0.7)
    p.add_argument("--erode_iter", type=int, default=1)
    p.add_argument("--min_pixels", type=int, default=20)
    p.add_argument("--simplify_ratio", type=float, default=0.002)
    p.add_argument("--post_score_thresh", type=float, default=0.65)
    p.add_argument("--target_crs", default=None)                    # e.g. "EPSG:32755"
    p.add_argument("--final_result_dir", default="./modules/final_result")
    
    return p.parse_args()

def _auto_find_ckpt(ckpt_dir: str) -> str:
    """优先匹配常见命名；否则取目录下最新的 .pth。"""
    ckpt_dir_p = Path(ckpt_dir)
    for name in ("maskrcnn_best.pth", "best.pth", "last.pth"):
        p = ckpt_dir_p / name
        if p.exists():
            return str(p)
    cands = sorted(ckpt_dir_p.glob("*.pth"), key=lambda x: x.stat().st_mtime, reverse=True)
    return str(cands[0]) if cands else ""

def visualize_overlay_window_png(
    geotiff_path, gpkg_path, layer_candidates_bounds=("boundaries_all", "boundaries")
):
    """绘制 GeoTIFF + GPKG 边界叠加图，直接保存 PNG 到 final_result/preview.png"""
    try:
        out_png = "./modules/final_result/preview.png"
        os.makedirs(os.path.dirname(out_png), exist_ok=True)

        # === 底图 ===
        with rasterio.open(geotiff_path) as src:
            fig, ax = plt.subplots(figsize=(10, 10))
            show(src, ax=ax)
            r_crs = src.crs

        # === 叠加矢量 ===
        plotted = False
        for layer in layer_candidates_bounds:
            try:
                gdf = gpd.read_file(gpkg_path, layer=layer)
                if gdf is None or len(gdf) == 0:
                    print(f"[WARN] layer '{layer}' empty, try next")
                    continue

                # CRS 对齐
                if gdf.crs is not None and r_crs is not None and gdf.crs != r_crs:
                    gdf = gdf.to_crs(r_crs)

                # 按几何类型选择绘制对象
                geom_types = set(gdf.geom_type.unique())
                if any("Polygon" in t for t in geom_types):
                    to_plot = gdf.boundary          # 面→画边界线
                elif any("LineString" in t for t in geom_types):
                    to_plot = gdf                    # 线→直接画线
                else:
                    to_plot = gdf.geometry.boundary  # 兜底

                # 画线：提高 zorder 和 linewidth 以便压在底图上清晰可见
                to_plot.plot(ax=ax, linewidth=1.2, color="red", zorder=10)
                print(f"[OK] plotted layer '{layer}' | types={geom_types} | crs={gdf.crs}")
                plotted = True
                break
            except Exception as e:
                print(f"[WARN] failed layer '{layer}': {e}")
                continue

        if not plotted:
            print("[ERR] no layer plotted; check layer names or geometry types")

        plt.axis('off')
        plt.tight_layout()
        plt.savefig(out_png, format='png', dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"[OK] Visualization saved to {out_png}")
        return out_png

    except Exception as e:
        print(f"[ERR] Visualization failed: {e}")
        return None

def main():
    args = parse_args()
    # ===== GEE Fetch =====
    if args.mode in ("auto", "gee"):
        json_sa, json_project = read_service_account_json(args.gee_key_path)
        sa_eff = args.gee_service_account or json_sa
        project_eff = args.gee_project_id or (json_project or "")
        if args.roi_mode == "bbox":
            # 基本校验
            if not args.gee_bbox:
                args.gee_bbox = input("Please input your coordinates range (minx,miny,maxx,maxy): ").strip()

            # 仅在 bbox 模式下解析 bbox
            def _parse_bbox(s: str):
                try:
                    vals = [float(x.strip()) for x in s.split(",")]
                    if len(vals) != 4:
                        raise ValueError
                    minx, miny, maxx, maxy = vals
                    if not (minx < maxx and miny < maxy):
                        raise ValueError
                    return (minx, miny, maxx, maxy)
                except Exception:
                    raise SystemExit("bbox 模式需要 --gee_bbox（minx,miny,maxx,maxy，WGS84），示例：146.15,-34.25,146.55,-33.95")

        def _ask_date(prompt: str) -> str:
            while True:
                s = input(prompt).strip()
                try:
                    _dt.date.fromisoformat(s)   # 校验 YYYY-MM-DD
                    return s
                except ValueError:
                    print("Invalid date. Use YYYY-MM-DD, e.g. 2024-02-01")

        # 处理时间窗（沿用你现有的参数名；若已有就删掉以下两行）
        s2_start = args.gee_s2_start or _ask_date("Please input S2 start date (YYYY-MM-DD): ")
        s2_end   = args.gee_s2_end   or _ask_date("Please input S2 end date   (YYYY-MM-DD): ")   
        # 顺序校验：若 start > end，则交换
        if _dt.date.fromisoformat(s2_start) > _dt.date.fromisoformat(s2_end):
            print("Start date is later than end date, swapping them.")
            s2_start, s2_end = s2_end, s2_start
        # ========= 分支：bbox 模式 =========
        if args.roi_mode == "bbox":
            if not args.gee_bbox:
                raise SystemExit("bbox 模式必须提供 --gee_bbox，例如 146.15,-34.25,146.55,-33.95")
            bbox = _parse_bbox(args.gee_bbox)

            results = run_gee_bbox(
                cadastre_asset=args.gee_cad_asset,   # 你的参数名可能是 gee_cad_asset / gee_cadastre_asset
                id_col=args.gee_id_col,            # 你的参数名可能是 gee_id_field / gee_id_col
                bbox=bbox,
                json_path=args.gee_key_path,
                s2_start=s2_start,
                s2_end=s2_end,
                cloud_max=args.gee_cloud_max,
                farmland_th=args.gee_farmland_th,
                out_tif=args.gee_out_tif,
                out_ids=args.gee_out_ids,
                pad_m=args.gee_pad_m,
                crs_out=args.gee_crs_out
            )
            print("[GEE] bbox 模式完成：")

        # ========= 分支：tif 模式 =========
        elif args.roi_mode == "tif":
            if not args.user_tif:
                raise SystemExit("tif 模式必须提供 --user_tif，指向你的 GeoTIFF")

            # 1) 调 image_input.py 的 tif 模式（注意：不要传 out_dir_for_tif_mode）
            results = run_gee_tif(
                roi_mode="tif",
                cadastre_asset=args.gee_cad_asset,
                id_col=args.gee_id_col,          # ← 用你在 argparse 里定义的 gee_id_col
                json_path=args.gee_key_path,
                s2_start=s2_start,               # 仅用于 DynamicWorld 时间窗
                s2_end=s2_end,
                cloud_max=args.gee_cloud_max,
                farmland_th=args.gee_farmland_th,
                out_ids=args.gee_out_ids,        # 这个参数 image_input 可能忽略；用 results['ids'] 兜底
                user_tif=args.user_tif
            )

            # 2) 直接使用 image_input.py 原地输出的文件路径
            tif_dir = os.path.dirname(args.user_tif)
            gpkg_path = results.get("gpkg") or os.path.join(tif_dir, "filter_lot.gpkg")
            ids_path  = results.get("ids")  or os.path.join(tif_dir, "id_list.txt")
            print("[GEE] tif 模式完成")


    # ===== 预处理（切 patch + 导出 target polygons）=====
    if args.mode in ("auto","preprocess"):
        # --- 把 CLI 参数写回 cutpatch.py 的模块变量 ---
        if hasattr(cutpatch, "BIG_TIF"):      cutpatch.BIG_TIF  = args.patch_tif
        if hasattr(cutpatch, "POLY_VEC"):     cutpatch.POLY_VEC = args.patch_vector
        if hasattr(cutpatch, "OUT_DIR"):      cutpatch.OUT_DIR  = args.patch_out_dir
        if hasattr(cutpatch, "ID_FIELD"):     cutpatch.ID_FIELD = args.patch_id_field
        if hasattr(cutpatch, "PATCH_SMALL"):  cutpatch.PATCH_SMALL  = args.patch_small
        if hasattr(cutpatch, "PATCH_LARGE"):  cutpatch.PATCH_LARGE  = args.patch_large
        if hasattr(cutpatch, "TILE_OVERLAP"): cutpatch.TILE_OVERLAP = args.patch_overlap
        if hasattr(cutpatch, "MARGIN_PX"):    cutpatch.MARGIN_PX    = args.patch_margin
        if hasattr(cutpatch, "PAD_TO_FULL"):  cutpatch.PAD_TO_FULL  = args.patch_pad_full

        print("\n==== [Preprocess 1/2] 切 Patch ====")
        manifest_path, rows = cutpatch.manifest()   # 直接用你现成的入口函数
        print(f">> manifest: {manifest_path} | rows={rows}")

        # --- 把 CLI 参数写回 exportTargetPolygon.py 的模块变量 ---
        if hasattr(exportTargetPolygon, "IN_GPKG"):     exportTargetPolygon.IN_GPKG    = Path(args.exp_in_gpkg)
        if hasattr(exportTargetPolygon, "LAYER_NAME"):  exportTargetPolygon.LAYER_NAME = args.exp_layer_name
        if hasattr(exportTargetPolygon, "OUT_DIR"):     exportTargetPolygon.OUT_DIR    = Path(args.exp_out_dir)
        if hasattr(exportTargetPolygon, "ID_COL"):      exportTargetPolygon.ID_COL     = args.exp_id_col
        if hasattr(exportTargetPolygon, "PATCH_DIR"):   exportTargetPolygon.PATCH_DIR  = Path(args.patch_out_dir)

        print("\n==== [Preprocess 2/2] 按 OBJECTID 导出多边形 ====")
        exportTargetPolygon.export_split_by_id()
        print(">> 预处理完成（Patch + 导出多边形）")

    # ------- Data loaders -------
    if args.mode == "all":
        dcfg = DataConfig(
            img_dir=args.img_dir,
            ann_dir=args.ann_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            split_file=args.split_file,
            split_method=args.split_method,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            random_seed=args.random_seed,
        )
        train_loader, val_loader, test_loader = build_loaders(dcfg)

        print(">> Using img_dir =", dcfg.img_dir)
        print(">> Using ann_dir =", dcfg.ann_dir)
        print(">> Using ckpt_dir =", args.ckpt_dir)

    elif args.mode in ("auto", "auto_data"):
        new_dcfg = Infer_DataConfig(
        img_dir=args.inference_img_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        )
        new_test_loader = build_test_loaders(new_dcfg)
        print(">> Using inference_img_dir =", new_dcfg.img_dir)
        print(">> Using ckpt_dir =", args.ckpt_dir)

    # 如果用户显式给了 ckpt，则 all/train 也跳过训练
    has_user_ckpt = bool(args.ckpt)

    best_ckpt_path = ""   # 供 infer/vis 使用
    latest_ckpt_path = ""

    # ===== [1/5] Training =====
    if args.mode in ("all", "train") and not has_user_ckpt:
        print("\n==== [1/3] Training ====")
        best_ckpt_path, latest_ckpt_path = train_mod.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            accumulate_grad_batches=args.accumulate_grad_batches,
            warmup_epochs=args.warmup_epochs,
            mask_loss_weight=args.mask_loss_weight,
            box_loss_weight=args.box_loss_weight,
            use_dice_loss=args.use_dice_loss,
            use_iou_loss=args.use_iou_loss,
            ckpt_dir=args.ckpt_dir,  # 传给训练用于落盘
        )
        print(f">> Train done. best_ckpt={best_ckpt_path} | latest_ckpt={latest_ckpt_path}")
    elif args.mode in ("all", "train") and has_user_ckpt:
        print(">> 检测到 --ckpt 参数，跳过训练，直接使用提供的权重进行后续步骤。")

    # ===== 选择后续步骤使用的 ckpt =====
    ckpt_for_next = args.ckpt or best_ckpt_path or latest_ckpt_path or _auto_find_ckpt(args.ckpt_dir)
    print(">> Selected ckpt =", ckpt_for_next if ckpt_for_next else "<None>")

    # 对于纯 infer / visualize，没有 ckpt 需要报错
    if args.mode in ("infer", "visualize") and not ckpt_for_next:
        raise FileNotFoundError("请通过 --ckpt 指定权重文件，或先运行 --mode all / train 产生权重。")
    # 对于 all 模式，如果训练没产出、也没找到现成 ckpt，一样需要提示
    if args.mode == "all" and not ckpt_for_next:
        raise FileNotFoundError("未找到可用的 ckpt 路径。请检查训练阶段是否成功保存权重，或通过 --ckpt 指定。")

    # ===== [2/5] Inference =====
    if args.mode in ("all", "infer"):
        print("\n==== [2/3] Inference & save predictions ====")
        info = infer_mod.predict_dataset(
            test_loader=test_loader,
            ckpt_path=ckpt_for_next,
            out_dir=args.out_dir,
            out_format=args.out_format,
            use_pretrained=False,
        )
        print(f">> Inference 完成：共保存 {info['num_items']} 个样本的预测到 {args.out_dir}")
        print(f">> 示例文件：{info['sample_paths'][:3]}")

    elif args.mode in ("auto", "auto_data"):
        print("\n==== Inference & save predictions ====")
        info = infer_mod.predict_dataset(
            test_loader=new_test_loader,
            ckpt_path=ckpt_for_next,
            out_dir=args.out_dir,
            out_format=args.out_format,
            use_pretrained=False,
        )
        print(f">> Inference 完成：共保存 {info['num_items']} 个样本的预测到 {args.out_dir}")
        print(f">> 示例文件：{info['sample_paths'][:3]}")

    # ===== [3/5] Visualization =====
    if args.mode in ("visualize"):
        print("\n==== [3/3] Visualization PNGs ====")

         # 1) 叠彩色掩膜
        Path(args.viz_out_dir).mkdir(parents=True, exist_ok=True)
        vis_info1 = viz_mod.predict_and_save_pngs(
            test_loader=test_loader,
            ckpt_path=ckpt_for_next,
            out_dir=args.viz_out_dir,
            limit=args.viz_limit,
            score_thresh=args.score_thresh,
            mode="side_by_side",   
        )
        print(f">> Side-by-side 保存 {vis_info1['num_images']} 张到 {args.viz_out_dir}")

        # 2) 边界图（单独目录）
        bdir = str(Path(args.viz_out_dir).parent / (Path(args.viz_out_dir).name + "_boundary"))
        Path(bdir).mkdir(parents=True, exist_ok=True)
        vis_info2 = viz_mod.predict_and_save_pngs(
            test_loader=test_loader,
            ckpt_path=ckpt_for_next,
            out_dir=bdir,
            limit=args.viz_limit,
            score_thresh=args.score_thresh,
            mode="boundary",
        )
        print(f">> Boundary 保存 {vis_info2['num_images']} 张到 {bdir}")

    # ===== [4/5] Evaluation =====
    if args.mode in ("all", "eval"):
        # 选用与推理/可视化同一份 ckpt
        if not ckpt_for_next:
            raise FileNotFoundError("评估阶段未找到可用的 ckpt，请通过 --ckpt 指定或先训练。")

        print("\n==== [4/4] Evaluation on test set ====")
        df, summary = eval_mod.evaluate_from_checkpoint(
            test_loader=test_loader,
            ckpt_path=ckpt_for_next,
            score_thresh=args.eval_score_thresh,
            iou_match_thr=args.eval_iou_thr,
            boundary_tol=args.eval_boundary_tol,
            save_csv=args.eval_csv,
        )
        # 打印一个简要汇总
        keys = ["Precision", "Recall", "F1_score", "IoU", "Dice"]
        msg = " | ".join(f"{k}: {summary.get(k, None):.4f}" for k in keys if summary.get(k, None) is not None)
        print(f">> Eval summary: {msg}")
        print(f">> Per-image csv: {args.eval_csv}")

    # ===== [5/5] Post-processing =====
    # 在 main() 里，准备调用后处理/导出前
    if args.mode == "all":
        cad_source = "./modules/model/Target_polygon"              # 训练完整流程用这个
    elif args.mode in ("auto", "auto_data", "post"):
        cad_source = "./modules/model/Preprocess_Target_polygon"   # 自动流程用这个
    print("\n==== [Postprocess] Vectorize predictions & merge ====")
    ts = _dt.datetime.now(ZoneInfo("Australia/Sydney")).strftime("%Y%m%d_%H%M%S")
    info = post_mod.run_postprocessing(
        pred_dir=args.pred_dir,
        cad_source=cad_source,
        layer_name=args.layer_name,
        id_field=args.cad_id_field,
        out_dir=args.post_out_dir,
        img_root=args.img_root,
        mask_thresh=args.mask_thresh,
        erode_iter=args.erode_iter,
        min_pixels=args.min_pixels,
        simplify_ratio=args.simplify_ratio,
        score_thresh=args.post_score_thresh,
        target_crs=args.target_crs,
        merged_out_dir=args.final_result_dir,    # ← 合并结果单独目录（比如 final_result）
        merged_filename=f"merged_{ts}.gpkg",
    )
    print("Merged at:", info["merged"])

 # ====== 全流程结束做一个预览 ======
  # 举例：根据模式选不同的 GPKG 路径
    if args.mode in ("auto", "post"):
        gpkg_for_view = info["merged"]
        tif_for_view  = args.gee_out_tif     # 你的底图tif
        
        print("\n==== [Visualization] Generating preview.png ====")
        preview_png = visualize_overlay_window_png(
            geotiff_path=tif_for_view,
            gpkg_path=gpkg_for_view,
            layer_candidates_bounds=("boundaries_all", "boundaries"),
        )
        print(f"[OK] Preview image saved at: {preview_png}")
        # === 输入：刚合并好的 GPKG 路径 ===
        Merged_gpkg = info["merged"]
        out_xlsx = f"./modules/final_result/field_summary_{ts}.xlsx"

        # === 读取 fields 图层 ===
        gdf = gpd.read_file(Merged_gpkg, layer="fields_all")

        # 确保 CRS 是投影坐标系（单位是米）
        if not gdf.crs or not gdf.crs.is_projected:
            raise ValueError(f"当前 CRS={gdf.crs} 不是投影坐标系，请先转换为米制（如 EPSG:32755）")

        # === 计算每个 polygon_id 的字段统计 ===
        stats = (
            gdf.groupby("polygon_id")
            .agg(
                field_count=("geometry", "count"),
                total_area=("geometry", lambda x: sum(g.area for g in x if g is not None))
            )
            .reset_index()
        )

        # === 保存为 Excel ===
        os.makedirs(os.path.dirname(out_xlsx), exist_ok=True)
        stats.to_excel(out_xlsx, index=False)

        # ===== [Clean up temporary folders] =====
        to_delete = [
            "./modules/gee_out",
            "./modules/post_outputs",
            "./modules/runs/predictions",
            "./modules/model/InferenceDataset",
            "./modules/model/Preprocess_Target_polygon",
            "./model/InferenceDataset"
        ]

        print("\n==== [Clean-up] Removing temporary folders ====")
        for path in to_delete:
            if os.path.exists(path):
                try:
                    shutil.rmtree(path)
                    print(f"[OK] Deleted: {path}")
                except Exception as e:
                    print(f"[ERR] Failed to delete {path}: {e}")
            else:
                print(f"[SKIP] Not found: {path}")
        print("==== [Clean-up Completed] ====\n")

        return {
            "merged_gpkg": os.path.basename(info["merged"]),
            "preview_png": "preview.png",
            "field_summary_xlsx": os.path.basename(out_xlsx)
        }

if __name__ == "__main__":
    main()



