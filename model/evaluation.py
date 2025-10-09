# model/evaluation.py
import os
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import torch
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score


# =============== 单图评估（像素级） ===============
def evaluate_single_image(
    gt_masks,                      # (M,H,W) torch/numpy
    pred_masks,                    # (N,H,W) torch/numpy
) -> Dict[str, Any]:
    """
    把多实例掩膜合成前景二值图，计算 Precision / Recall / F1 / IoU / Dice。
    """
    if isinstance(gt_masks, torch.Tensor):
        gt_masks = gt_masks.detach().cpu().numpy()
    if isinstance(pred_masks, torch.Tensor):
        pred_masks = pred_masks.detach().cpu().numpy()

    # 合并实例为语义前景
    gt   = (gt_masks.sum(0)  > 0).astype(np.uint8) if gt_masks.size  else 0
    pred = (pred_masks.sum(0) > 0).astype(np.uint8) if pred_masks.size else 0

    # 兼容无 GT 情况
    if isinstance(gt, int):
        if isinstance(pred, int):
            return {"Precision":0.0,"Recall":0.0,"F1_score":0.0,"IoU":0.0,"Dice":0.0}
        gt = np.zeros_like(pred, dtype=np.uint8)

    inter = np.logical_and(gt, pred).sum()
    union = np.logical_or(gt, pred).sum()
    iou   = inter / union if union > 0 else 0.0
    dice  = (2.0 * inter) / (gt.sum() + pred.sum() + 1e-6)

    y_true = gt.ravel()
    y_pred = pred.ravel()
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)

    return {
        "Precision":  float(prec),
        "Recall":     float(rec),
        "F1_score":   float(f1),
        "IoU":        float(iou),
        "Dice":       float(dice),
    }


# =============== 整个 DataLoader 评估（在线推断） ===============
def evaluate_dataset(
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    score_thresh: float = 0.5,
    iou_match_thr: float = 0.5,    # 预留参数，占位不影响现有逻辑
    boundary_tol: int = 3,         # 预留参数，占位不影响现有逻辑
    save_csv: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    遍历 data_loader：前向 -> 取预测 masks（按 score_thresh 过滤）-> 与 GT 计算指标。
    """
    model.eval()
    results: List[Dict[str, Any]] = []

    with torch.no_grad():
        for bidx, (images, targets) in enumerate(data_loader):
            images = [im.to(device) for im in images]
            outputs = model(images)

            for i, (pred, tgt) in enumerate(zip(outputs, targets)):
                # 预测 masks
                masks = pred.get("masks", torch.empty(0))
                if isinstance(masks, torch.Tensor):
                    if masks.ndim == 4 and masks.shape[1] == 1:
                        masks = masks.squeeze(1)  # (N,1,H,W)->(N,H,W)
                else:
                    masks = torch.as_tensor(masks)

                scores = pred.get("scores", torch.empty(0))
                if not isinstance(scores, torch.Tensor):
                    scores = torch.as_tensor(scores)

                if scores.numel() > 0:
                    keep = scores.detach().cpu().numpy() >= score_thresh
                    pred_masks = masks[keep] if keep.any() else torch.empty((0, *masks.shape[-2:]))
                else:
                    pred_masks = masks if masks.numel() > 0 else torch.empty((0, *images[0].shape[-2:]))

                # GT masks
                gt_masks = tgt["masks"]
                if gt_masks.ndim == 4 and gt_masks.shape[1] == 1:
                    gt_masks = gt_masks.squeeze(1)
                gt_masks = gt_masks.detach().cpu()

                rep = evaluate_single_image(gt_masks, pred_masks)

                # 附带 image_id/path 便于回查
                img_id = tgt.get("image_id", bidx * len(images) + i)
                if torch.is_tensor(img_id):
                    try:
                        img_id = int(img_id.item())
                    except Exception:
                        img_id = img_id.tolist()
                rep["image_id"] = img_id

                meta = tgt.get("meta", {})
                if isinstance(meta, dict) and "path" in meta:
                    rep["path"] = meta["path"]

                results.append(rep)

    df = pd.DataFrame(results)
    summary = df.mean(numeric_only=True).to_dict()

    if save_csv:
        os.makedirs(os.path.dirname(save_csv) or ".", exist_ok=True)
        df.to_csv(save_csv, index=False)
        print(f">> Saved results CSV: {save_csv}")

    return df, summary


# =============== 便捷入口：从 ckpt 加载后评估（与你 main.py 的调用一致） ===============
def evaluate_from_checkpoint(
    test_loader,
    ckpt_path: str,
    score_thresh: float = 0.5,
    iou_match_thr: float = 0.5,
    boundary_tol: int = 3,
    save_csv: Optional[str] = None,
):
    """
    加载权重 -> 构建模型 -> 在 test_loader 上前向并评估 -> 返回 (df, summary)。
    完全匹配 main.py 调用签名。
    """
    from model.engine import pick_device_and_amp
    from model.maskrcnn import build_model_with_custom_loss

    device, _ = pick_device_and_amp()

    # 推断类别数（背景 + N 类）
    base = test_loader.dataset
    while hasattr(base, "dataset"):  # 兼容 Subset
        base = base.dataset
    # 如果你的数据集类名保存在 dataset.classes，中间件里已经这么做了
    n_classes = 1 + (len(getattr(base, "classes", ["field"])) or 1)

    model = build_model_with_custom_loss(num_classes=n_classes, use_pretrained=False)
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=False)
    model.to(device).eval()

    return evaluate_dataset(
        model=model,
        data_loader=test_loader,
        device=device,
        score_thresh=score_thresh,
        iou_match_thr=iou_match_thr,
        boundary_tol=boundary_tol,
        save_csv=save_csv,
    )
