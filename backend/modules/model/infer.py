import os
import torch
import numpy as np
from typing import Dict, Any, List
from model.engine import pick_device_and_amp
from model.maskrcnn import build_model_with_custom_loss

def _resolve_class_names(test_loader) -> Dict[int, str]:
    base = test_loader.dataset
    while hasattr(base, "dataset"):   # 兼容 Subset
        base = base.dataset
    if hasattr(base, "classes"):
        names = list(getattr(base, "classes"))
        return {0: "__background__", **{i+1: n for i, n in enumerate(names)}}
    return {0: "__background__", 1: "field"}

def _load_ckpt(model, ckpt_path: str):
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=False)

def _stem_from_target(tgt: Dict[str, Any], fallback: str) -> str:
    # 尽力从 meta.path 拿文件名；没有就用 image_id
    meta = tgt.get("meta", {})
    p = meta.get("path", None)
    if p:
        base = os.path.basename(p)
        return os.path.splitext(base)[0]
    if "image_id" in tgt and torch.is_tensor(tgt["image_id"]):
        return f"img_{int(tgt['image_id'].item())}"
    return fallback

def _to_cpu_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def _save_one(out_dir: str, stem: str, pred: Dict[str, Any],
              target: Dict[str, Any], class_names: Dict[int, str],
              out_format: str = "pt") -> str:
    os.makedirs(out_dir, exist_ok=True)
    item = {
        "boxes": pred["boxes"].detach().cpu(),
        "labels": pred["labels"].detach().cpu(),
        "scores": pred.get("scores", torch.empty(0)).detach().cpu(),
        "masks": pred.get("masks", torch.empty(0)).detach().cpu(),  # (N,1,H,W)
        "class_names": class_names,
        "image_id": int(target.get("image_id", torch.tensor([-1])).item()),
        "meta": target.get("meta", {}),
    }
    if out_format == "pt":
        path = os.path.join(out_dir, f"{stem}.pt")
        torch.save(item, path)
    else:  # npz
        np_item = {k: (_to_cpu_np(v) if isinstance(v, (torch.Tensor, np.ndarray)) else v)
                   for k, v in item.items() if k not in ("class_names", "meta")}
        path = os.path.join(out_dir, f"{stem}.npz")
        np.savez_compressed(path, **np_item)
        # 同名保存 meta 与 class_names
        with open(os.path.join(out_dir, f"{stem}.meta.json"), "w", encoding="utf-8") as f:
            import json
            json.dump({"class_names": class_names, "meta": item["meta"]}, f, ensure_ascii=False, indent=2)
    return path

def predict_dataset(test_loader,
                    ckpt_path: str,
                    out_dir: str = "./runs/predictions",
                    out_format: str = "pt",
                    use_pretrained: bool = False) -> Dict[str, Any]:
    """
    对整个 test_loader 进行推理，并把**每张图**的预测保存到磁盘。
    返回 {'num_items': N, 'sample_paths': [...], 'class_names': {...}}
    """
    device, _ = pick_device_and_amp()
    class_names = _resolve_class_names(test_loader)
    num_classes = len(class_names)

    model = build_model_with_custom_loss(num_classes=num_classes,
                                         use_pretrained=use_pretrained)
    _load_ckpt(model, ckpt_path)
    model.to(device).eval()

    saved_paths: List[str] = []
    with torch.no_grad():
        for images, targets in test_loader:
            imgs = [im.to(device) for im in images]
            outputs = model(imgs)  # list[dict]
            for i, (pred, tgt) in enumerate(zip(outputs, targets)):
                stem = _stem_from_target(tgt, fallback=f"batch_{len(saved_paths)}")
                path = _save_one(out_dir, stem, pred, tgt, class_names, out_format)
                saved_paths.append(path)

    return {"num_items": len(saved_paths),
            "sample_paths": saved_paths[:5],
            "class_names": class_names}
