import os
from typing import Dict, Any, Optional
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 可选依赖：OpenCV 用于边界渲染（没有也能跑）
try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

# === 工具 ===
def tensor_to_uint8_img(img_tensor):
    """(C,H,W) tensor -> (H,W,3) uint8 RGB"""
    if isinstance(img_tensor, torch.Tensor):
        img = img_tensor.detach().cpu().float()
        if img.ndim == 3 and img.shape[0] in (1,3,4):
            img = img[:3, ...].permute(1,2,0)
        img = img.numpy()
    else:
        img = img_tensor
    if img.max() <= 1.0:
        img = img * 255.0
    return np.clip(img, 0, 255).astype(np.uint8)

def _stem_from_target(tgt: Dict[str, Any], fallback: str) -> str:
    """根据目标里的 meta.path 或 image_id 生成文件名 stem"""
    meta = tgt.get("meta", {})
    p = meta.get("path", None)
    if p:
        base = os.path.basename(p)
        return os.path.splitext(base)[0]
    if "image_id" in tgt and torch.is_tensor(tgt["image_id"]):
        return f"img_{int(tgt['image_id'].item())}"
    return fallback

def _resolve_class_names(loader) -> Dict[int, str]:
    base = loader.dataset
    while hasattr(base, "dataset"):  # 兼容 Subset
        base = base.dataset
    if hasattr(base, "classes"):
        names = list(getattr(base, "classes"))
        return {0: "__background__", **{i+1: n for i, n in enumerate(names)}}
    return {0: "__background__", 1: "field"}

# === 你的两种可视化写法：叠加半透明 mask / 只画边界 ===
def render_side_by_side(img_tensor, pred, class_names: dict,
                        score_thresh: float = 0.5, title: str = "Prediction",
                        save_path: Optional[str] = None, show: bool = False):
    """左原图、右预测（半透明 mask + bbox + label）——和你 notebook 里的效果一致"""
    img = img_tensor.detach().cpu().float()
    if img.max() > 1.5:
        img = img / 255.0
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()

    boxes  = pred.get("boxes",  torch.empty(0)).detach().cpu().numpy()
    labels = pred.get("labels", torch.empty(0, dtype=torch.long)).detach().cpu().numpy()
    scores = pred.get("scores", torch.empty(0)).detach().cpu().numpy()
    masks  = pred.get("masks",  None)
    if masks is not None:
        masks = masks.detach().cpu().numpy()

    fig, axs = plt.subplots(1, 2, figsize=(14, 7))
    axs[0].imshow(img); axs[0].set_title("Input image"); axs[0].axis("off")
    axs[1].imshow(img); axs[1].set_title(title);          axs[1].axis("off")

    for i in range(len(boxes)):
        if scores[i] < score_thresh: 
            continue
        color = (np.random.rand(), np.random.rand(), np.random.rand())

        if masks is not None and i < masks.shape[0]:
            mask_bin = (masks[i, 0] > 0.5).astype(np.uint8)
            overlay = np.zeros((mask_bin.shape[0], mask_bin.shape[1], 4), dtype=np.float32)
            overlay[..., :3] = color
            overlay[..., 3] = 0.45 * mask_bin
            axs[1].imshow(overlay, interpolation="none")

        xmin, ymin, xmax, ymax = boxes[i]
        rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                                 linewidth=1.5, edgecolor=color, facecolor='none')
        axs[1].add_patch(rect)

        cls_id = int(labels[i])
        cls_name = class_names.get(cls_id, str(cls_id))
        axs[1].text(xmin, max(ymin-2, 0), f"{cls_name} {scores[i]:.2f}",
                    fontsize=11, color="white",
                    bbox=dict(facecolor=color, alpha=0.6, edgecolor='none', pad=2))

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    plt.close(fig)

def draw_boundaries_on_image(img_rgb, masks, scores,
                             score_thresh=0.5, prob_thresh=0.5,
                             color=(255,0,0), thickness=2):
    """只画边界的版本（用 OpenCV 轮廓）"""
    vis_rgb = img_rgb.copy()
    if isinstance(masks, torch.Tensor):
        masks = masks.detach().cpu().float().numpy()
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().float().numpy()
    if masks.ndim == 4:
        masks = np.squeeze(masks, 1)

    keep = scores >= score_thresh
    masks = masks[keep]
    if masks.size == 0 or not _HAS_CV2:
        return vis_rgb

    vis_bgr = vis_rgb[..., ::-1].copy()
    bgr = (color[2], color[1], color[0])

    for m in masks:
        if m.max() <= 1.0:
            m = (m >= prob_thresh).astype(np.uint8)
        else:
            m = (m > 0).astype(np.uint8)

        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            continue
        cv2.drawContours(vis_bgr, contours, contourIdx=-1, color=bgr, thickness=thickness)

    return vis_bgr[..., ::-1]

def save_boundaries_png(img_tensor, pred, save_path: str,
                        score_thresh=0.5, prob_thresh=0.5, color=(255,0,0), thickness=2):
    """把“边界叠加图”直接存 PNG"""
    rgb = tensor_to_uint8_img(img_tensor)
    masks = pred.get("masks", torch.empty(0))
    scores = pred.get("scores", torch.empty(0))
    vis = draw_boundaries_on_image(rgb, masks, scores, score_thresh, prob_thresh, color, thickness)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.figure(figsize=(7,7)); plt.imshow(vis); plt.axis("off")
    plt.savefig(save_path, bbox_inches="tight", dpi=150); plt.close()

# === 批量可视化入口：加载模型->跑 loader->保存图片 ===
def predict_and_save_pngs(test_loader,
                          ckpt_path: str,
                          out_dir: str,
                          limit: int = 0,
                          score_thresh: float = 0.5,
                          mode: str = "side_by_side"):
    """
    mode: "side_by_side" 用 render_side_by_side；"boundary" 用边界图
    """
    from model.engine import pick_device_and_amp
    from model.maskrcnn import build_model_with_custom_loss

    device, _ = pick_device_and_amp()
    class_names = _resolve_class_names(test_loader)
    num_classes = len(class_names)

    model = build_model_with_custom_loss(num_classes=num_classes, use_pretrained=False)
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=False)
    model.to(device).eval()

    os.makedirs(out_dir, exist_ok=True)
    saved = 0
    with torch.no_grad():
        for images, targets in test_loader:
            outputs = model([im.to(device) for im in images])
            for img_t, pred, tgt in zip(images, outputs, targets):
                stem = _stem_from_target(tgt, fallback=f"{saved:06d}")
                if mode == "boundary":
                    png = os.path.join(out_dir, f"{stem}_boundary.png")
                    save_boundaries_png(img_t, pred, png, score_thresh=score_thresh)
                else:
                    png = os.path.join(out_dir, f"{stem}_viz.png")
                    render_side_by_side(img_t, pred, class_names, score_thresh=score_thresh,
                                        title="Prediction", save_path=png, show=False)
                saved += 1
                if limit and saved >= limit:
                    return {"num_images": saved, "out_dir": out_dir}
    return {"num_images": saved, "out_dir": out_dir}
