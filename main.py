# model/main.py
import argparse
import os
from glob import glob
from pathlib import Path

from model.dataset import DataConfig, build_loaders
from model import train as train_mod
from model import infer as infer_mod
from model import visualize as viz_mod  # 可视化模块
from model import evaluation as eval_mod

def parse_args():
    p = argparse.ArgumentParser()

    # choose your mode
    p.add_argument("--mode", choices=["all", "train", "infer", "visualize", "eval"], default="all")

    # dataset
    p.add_argument("--img_dir", default="./model/Dataset")
    p.add_argument("--ann_dir", default="./model/Dataset/withID")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--split_file", default="data/split_info.json")
    p.add_argument("--split_method", default="random", choices=["random", "pattern"])
    p.add_argument("--train_ratio", type=float, default=0.6)
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--test_ratio", type=float, default=0.2)
    p.add_argument("--random_seed", type=int, default=42)

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
    p.add_argument("--ckpt_dir", default="./checkpoints")

    # infer
    p.add_argument("--ckpt", default="")              # all/visualize/infer 都可显式指定
    p.add_argument("--out_dir", default="./runs/predictions")
    p.add_argument("--out_format", default="pt", choices=["pt", "npz"])

    # visualize
    p.add_argument("--viz_out_dir", default="./runs/visualizations")
    p.add_argument("--viz_mode", default="side_by_side", choices=["side_by_side", "boundary"])
    p.add_argument("--viz_limit", type=int, default=0)  # 0=不限制
    p.add_argument("--score_thresh", type=float, default=0.5)

    # evaluation
    p.add_argument("--eval_csv", default="./runs/eval_results.csv")
    p.add_argument("--eval_score_thresh", type=float, default=0.5)
    p.add_argument("--eval_iou_thr", type=float, default=0.5)
    p.add_argument("--eval_boundary_tol", type=int, default=3)

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

def main():
    args = parse_args()

    # ------- Data loaders -------
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

    # 如果用户显式给了 ckpt，则 all/train 也跳过训练
    has_user_ckpt = bool(args.ckpt)

    best_ckpt_path = ""   # 供 infer/vis 使用
    latest_ckpt_path = ""

    # ===== [1/3] Training =====
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

    # ===== [2/3] Inference =====
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

    # ===== [3/3] Visualization =====
    if args.mode in ("all", "visualize"):
        print("\n==== [3/3] Visualization PNGs ====")

         # 1) 叠彩色掩膜
        Path(args.viz_out_dir).mkdir(parents=True, exist_ok=True)
        vis_info1 = viz_mod.predict_and_save_pngs(
            test_loader=test_loader,
            ckpt_path=ckpt_for_next,
            out_dir=args.viz_out_dir,
            limit=args.viz_limit,
            score_thresh=args.score_thresh,
            mode="side_by_side",   # ✅ 用下划线，不要写 "side by side"
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

    # ===== [4/4] Evaluation =====
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


if __name__ == "__main__":
    main()



