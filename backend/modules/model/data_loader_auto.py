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

    # 1. Prefer matching the pattern patch_<number>_S
    m = re.search(r"patch_(\d+)_S", base, re.IGNORECASE)
    if m:
        return m.group(1)

    # 2. Otherwise match the pattern patch_<number>
    m = re.search(r"patch_(\d+)", base, re.IGNORECASE)
    if m:
        return m.group(1)

    # 3. If neither pattern matches, return None
    return None

def _extract_polygon_id(image_basename: str, id_regex: Optional[str] = None) -> Optional[str]:

    # Custom regex
    if id_regex:
        m = re.search(id_regex, image_basename)
        if m:
            return m.group(1) if m.groups() else m.group(0)

    return _extract_polygon_id_from_patchname(image_basename)


class GeoPatchTestDataset(Dataset):
    """
    Test-only Dataset: reads GeoTIFF (.tif) files without relying on annotation files.
    - Outputs (image_tensor, target), where target only contains a meta field
      (compatible with predict_dataset).
    """
    def __init__(self, img_dir, samples, bands=(1, 2, 3)):
        self.img_dir = img_dir
        self.samples = samples  
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
    # Split into two lists: images and targets
    images, targets = zip(*batch)  
    # images = torch.stack(images, dim=0)  # If needed, stack into a batch tensor
    return images, list(targets)  # Keep targets as a list of dicts


def create_inference_datasets(img_dir: str, **dataset_kwargs):
    """
    Create a Dataset containing only test data.
    - Automatically scans all .tif/.tiff files under img_dir.
    - Returns a GeoPatchTestDataset.
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
        collate_fn=inference_collate_fn,  
    )

    return test_loader

