from __future__ import annotations
import os, json, random
import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
import rasterio
from PIL import Image, ImageDraw
from .rs_subfield_dataset import RSSubfieldDataset 
from dataclasses import dataclass
from torch.utils.data import DataLoader

def _extract_polygon_id_from_patchname(name: str) -> Optional[str]:

    base = os.path.basename(name)
    base = os.path.splitext(base)[0]

    # 1. 优先匹配 patch_<数字>_S 这种格式
    m = re.search(r"patch_(\d+)_S", base, re.IGNORECASE)
    if m:
        return m.group(1)

    # 2. 否则匹配 patch_<数字> 这种格式
    m = re.search(r"patch_(\d+)", base, re.IGNORECASE)
    if m:
        return m.group(1)

    # 3. 都没有匹配上就返回 None
    return None

def _extract_polygon_id(image_basename: str, id_regex: Optional[str] = None) -> Optional[str]:

    # 自定义正则
    if id_regex:
        m = re.search(id_regex, image_basename)
        if m:
            return m.group(1) if m.groups() else m.group(0)

    return _extract_polygon_id_from_patchname(image_basename)


class GeoPatchTestDataset(Dataset):
    """
    测试专用 Dataset：仅读取 GeoTIFF (.tif)，不依赖标注文件。
    - 输出 (image_tensor, target)，其中 target 仅包含 meta 字段（兼容 predict_dataset）
    """
    def __init__(self, img_dir, samples, bands=(1, 2, 3)):
        self.img_dir = img_dir
        self.samples = samples  # List[str]：仅包含图像文件名
        self.bands = bands

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_name = self.samples[idx]
        ip = os.path.join(self.img_dir, img_name)

        with rasterio.open(ip) as src:
            bands = tuple(b for b in self.bands if 1 <= b <= src.count)
            arr = src.read(bands).astype(np.float32)
            if arr.max() > 1.5:
                arr = arr / 255.0
            H, W = src.height, src.width
            crs = src.crs
            transform = src.transform

        image_tensor = torch.from_numpy(arr)
        image_basename = os.path.basename(ip)
        polygon_id = _extract_polygon_id(image_basename, None)

        target = {
            "meta": {
                "crs": crs,
                "transform": transform,
                "height": H,
                "width": W,
                "path": ip,
                "polygon_id": polygon_id,
            }
        }

        return image_tensor, target


def inference_collate_fn(batch):
    images, targets = zip(*batch)  # 拆分为两个 list
    # images = torch.stack(images, dim=0)  # 拼成 batch tensor
    return images, list(targets)  # 保留 target 为 list of dict


def create_inference_datasets(img_dir: str, **dataset_kwargs):
    """
    创建仅包含测试数据的 Dataset。
    - 自动扫描 img_dir 下所有 .tif/.tiff 文件。
    - 返回 GeoPatchTestDataset。
    """
    tif_files = sorted([
        f for f in os.listdir(img_dir)
        if f.lower().endswith((".tif", ".tiff"))
    ])
    if not tif_files:
        raise RuntimeError(f"No .tif files found in {img_dir}")

    test_dataset = GeoPatchTestDataset(img_dir=img_dir, samples=tif_files, **dataset_kwargs)
    return test_dataset

@dataclass
class Infer_DataConfig:
    img_dir: str = "./model/Dataset"
    batch_size: int = 4
    num_workers: int = 2


def build_test_loaders(cfg):
    test_ds = create_inference_datasets(
        img_dir=cfg.img_dir,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=inference_collate_fn,  # 若推理不需要 batch 拼接，可改为 None
    )

    return test_loader

