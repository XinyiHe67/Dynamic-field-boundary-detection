from __future__ import annotations
import os, json, random
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
import rasterio
from PIL import Image, ImageDraw
from .rs_subfield_dataset import RSSubfieldDataset 
from dataclasses import dataclass
from torch.utils.data import DataLoader

# Helper functions
def _polygon_to_mask(height: int, width: int, polygon_xy: List[Tuple[float, float]]) -> np.ndarray:
    """Convert a single polygon (list of (x,y)) raster into a binary mask (H, W) with a range of {0,1}."""
    m = Image.new(mode="1", size=(width, height), color=0)
    ImageDraw.Draw(m).polygon(polygon_xy, outline=1, fill=1)
    return np.array(m, dtype=np.uint8)


def _bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
    """Calculate the (xmin, ymin, xmax, ymax) border from a binary mask; Return None if empty."""
    ys, xs = np.where(mask > 0)
    if ys.size == 0 or xs.size == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def _load_xanylabeling_json(json_path: str) -> Dict[str, Any]:
    """
    Read the X-AnyLabeling/Labelme style JSON:
    {
      "imagePath": "...",
      "imageHeight": H,
      "imageWidth": W,
      "shapes": [
        {"label": "...", "shape_type": "polygon", "points": [[x,y], ...]},
        ...
      ]
    }
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "shapes" not in data and "annotations" in data:
        data["shapes"] = data["annotations"]
    return data

class GeoPatchDataset(Dataset):
    """
    读取 GeoTIFF (.tif) + X-anylabeling (.json)
    - 用 rasterio 读像素（默认取 1,2,3 波段），归一化到 0~1，得到 (C,H,W) float32
    - target 结构与 Mask R-CNN 兼容，并额外返回 target["meta"] = {crs, transform, height, width, path}
    """
    def __init__(self, img_dir, ann_dir, samples=None,  # ← 改：samples 可为 None
                 bands=(1,2,3), use_single_class=True, ignore_empty=False):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.bands = bands
        self.use_single_class = use_single_class
        self.ignore_empty = ignore_empty

        if samples is None:
            # 自动配对同名 .tif/.tiff 与 .json
            tifs = [f for f in os.listdir(img_dir) if f.lower().endswith((".tif", ".tiff"))]
            base2img = {os.path.splitext(f)[0]: f for f in tifs}
            jsons = [f for f in os.listdir(ann_dir) if f.lower().endswith(".json")]
            pairs = []
            for j in jsons:
                b = os.path.splitext(j)[0]
                if b in base2img:
                    pairs.append((base2img[b], j))
            if not pairs:
                raise RuntimeError("No (tif,json) pairs found in img_dir/ann_dir.")
            self.samples = sorted(pairs)
        else:
            self.samples = samples

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx: int):
        img_name, json_name = self.samples[idx]
        ip = os.path.join(self.img_dir, img_name)
        jp = os.path.join(self.ann_dir, json_name)

        # --- 用 rasterio 读取像素 + 地理元数据 ---
        with rasterio.open(ip) as src:
            bands = tuple(b for b in self.bands if 1 <= b <= src.count)
            arr = src.read(bands).astype(np.float32)    # (C,H,W)
            if arr.max() > 1.5: arr = arr / 255.0       # 简单归一化
            H, W = src.height, src.width
            meta = {
                "crs": src.crs,
                "transform": src.transform,
                "height": H,
                "width": W,
                "path": ip,
            }
        image_tensor = torch.from_numpy(arr)            # (C,H,W), float32

        # --- 读 JSON 多边形 -> 栅格 ---
        data = _load_xanylabeling_json(jp)
        shapes = data.get("shapes", [])
        polygons, labels = [], []
        for shp in shapes:
            if shp.get("shape_type", "polygon") != "polygon": continue
            pts = shp.get("points", [])
            if len(pts) < 3: continue
            clipped = []
            for x, y in pts:
                x = float(min(max(x, 0.0), W - 1.0))
                y = float(min(max(y, 0.0), H - 1.0))
                clipped.append((x, y))
            polygons.append(clipped)
            labels.append(1 if self.use_single_class else 1)

        # --- 组装 target ---
        if len(polygons) == 0:
            if self.ignore_empty:
                return self.__getitem__((idx + 1) % len(self))
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "masks": torch.zeros((0, H, W), dtype=torch.uint8),
                "image_id": torch.tensor([idx], dtype=torch.int64),
                "area": torch.zeros((0,), dtype=torch.float32),
                "iscrowd": torch.zeros((0,), dtype=torch.int64),
                "meta": meta,
            }
            return image_tensor, target

        masks, boxes = [], []
        for poly in polygons:
            m = _polygon_to_mask(H, W, poly)
            if m.sum() == 0: continue
            bbox = _bbox_from_mask(m)
            if bbox is None: continue
            masks.append(m); boxes.append(bbox)
        if len(masks) == 0:
            return self.__getitem__((idx + 1) % len(self))

        masks_np = np.stack(masks, axis=0).astype(np.uint8)         # (N,H,W)
        boxes_np = np.asarray(boxes, dtype=np.float32)              # (N,4)
        labels_np = np.asarray(labels[:len(masks)], dtype=np.int64)
        areas = masks_np.reshape(masks_np.shape[0], -1).sum(axis=1).astype(np.float32)
        iscrowd = np.zeros((masks_np.shape[0],), dtype=np.int64)

        target = {
            "boxes": torch.as_tensor(boxes_np, dtype=torch.float32),
            "labels": torch.as_tensor(labels_np, dtype=torch.int64),
            "masks": torch.as_tensor(masks_np, dtype=torch.uint8),
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": torch.as_tensor(areas, dtype=torch.float32),
            "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64),
            "meta": meta,                                        # ✅ 关键信息
        }
        return image_tensor, target
    def get_sample_info(self, idx: int):
        """
        兼容 DatasetSplitter.save_split_info 的查询接口。
        返回至少包含 'image_name'（你的保存函数会用到）。
        """
        img_name, json_name = self.samples[idx]
        return {
            "image_name": img_name,
            "ann_name": json_name,
            "image_path": os.path.join(self.img_dir, img_name),
            "ann_path": os.path.join(self.ann_dir, json_name),
        }

    """
    Adaptation
      - Image: GeoTIFF (RGB, 8-bit), size mixed 256/512
      - Annotation: JSON of the same name (X-any labeling output, polygon)
    Table of Contents example:
      root/
        images/
          patch_0001.tif
          patch_0002.tif
        ann/
          patch_0001.json
          patch_0002.json
    """


def detection_collate_fn(batch):
    images, targets = list(zip(*batch))
    return list(images), list(targets)


class DatasetSplitter:
    """Dataset splitter, supporting multiple segmentation strategies"""

    @staticmethod
    def random_split(
        dataset: RSSubfieldDataset,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        test_ratio: float = 0.2,
        random_seed: Optional[int] = 42
    ) -> Tuple[Subset, Subset, Subset]:
        """
        Randomly split the dataset
        Args:
          dataset: The dataset to be split
          train_ratio: Training set ratio
          val_ratio: Validation set ratio
          test_ratio: Test set ratio
          random_seed: Random seed

        Returns:
            (train_dataset, val_dataset, test_dataset)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "The sum of the proportions must equal 1"

        total_size = len(dataset)
        indices = list(range(total_size))

        if random_seed is not None:
            random.seed(random_seed)
        random.shuffle(indices)

        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)

        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]

        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        test_dataset = Subset(dataset, test_indices)

        return train_dataset, val_dataset, test_dataset

    @staticmethod
    def split_by_file_pattern(
        dataset: RSSubfieldDataset,
        train_pattern: str = "train",
        val_pattern: str = "val",
        test_pattern: str = "test"
    ) -> Tuple[Subset, Subset, Subset]:
        """
        Split the dataset according to the file name pattern
        Args:
          dataset: The dataset to be split
          train_pattern: The pattern contained in the file name of the training set
          val_pattern: The pattern contained in the file name of the validation set
          test_pattern: The pattern contained in the test set file name
          
        Returns:
            (train_dataset, val_dataset, test_dataset)
        """
        train_indices, val_indices, test_indices = [], [], []

        for idx in range(len(dataset)):
            sample_info = dataset.get_sample_info(idx)
            filename = sample_info["image_name"].lower()

            if train_pattern.lower() in filename:
                train_indices.append(idx)
            elif val_pattern.lower() in filename:
                val_indices.append(idx)
            elif test_pattern.lower() in filename:
                test_indices.append(idx)
            else:
                # By default, it is assigned to the training set
                train_indices.append(idx)

        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        test_dataset = Subset(dataset, test_indices)

        return train_dataset, val_dataset, test_dataset

    @staticmethod
    def save_split_info(
        train_dataset: Subset,
        val_dataset: Subset,
        test_dataset: Subset,
        save_path: str
    ):
        """
        Save the dataset segmentation information to a file
        Args:
          train_dataset: Training set
          val_dataset: Validation set
          test_dataset: Test set
          save_path: Save the path
        """
        split_info = {
            "train_indices": train_dataset.indices,
            "val_indices": val_dataset.indices,
            "test_indices": test_dataset.indices,
            "train_size": len(train_dataset),
            "val_size": len(val_dataset),
            "test_size": len(test_dataset)
        }

        # Obtain the file name information of the sample
        base_dataset = train_dataset.dataset
        for split_name, indices in [("train", train_dataset.indices),
                                  ("val", val_dataset.indices),
                                  ("test", test_dataset.indices)]:
            split_info[f"{split_name}_files"] = [
                base_dataset.get_sample_info(idx)["image_name"] for idx in indices
            ]

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(split_info, f, indent=2, ensure_ascii=False)

        print(f"The dataset segmentation information has been saved to: {save_path}")
        print(f"Train set: {len(train_dataset)} numbers of samples")
        print(f"Validation set: {len(val_dataset)} numbers of samples")
        print(f"Test set: {len(test_dataset)} numbers of samples")

    @staticmethod
    def load_split_from_file(
        dataset: RSSubfieldDataset,
        split_file: str
    ) -> Tuple[Subset, Subset, Subset]:
        """
       Load the predefined dataset split from the file
       Args:
         dataset: Complete dataset
         split_file: Split information file path

       Returns:
            (train_dataset, val_dataset, test_dataset)
        """
        with open(split_file, 'r', encoding='utf-8') as f:
            split_info = json.load(f)

        train_dataset = Subset(dataset, split_info["train_indices"])
        val_dataset = Subset(dataset, split_info["val_indices"])
        test_dataset = Subset(dataset, split_info["test_indices"])

        return train_dataset, val_dataset, test_dataset
    
def create_split_datasets(
    img_dir: str,
    ann_dir: str,
    split_method: str = "random",
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    random_seed: Optional[int] = 42,
    save_split_info: Optional[str] = None,
    **dataset_kwargs
) -> Tuple[Subset, Subset, Subset]:
    """
    Convenient function: Create the segmented dataset
    Args:
      img_dir: Image directory
      ann_dir: Label the directory
      split_method: Splitting method ("random" or "pattern")
      train_ratio: Training set ratio
      val_ratio: Validation set ratio
      test_ratio: Test set ratio
      random_seed: Random seed
      save_split_info: The file path where the split information is saved
      **dataset_kwargs: Other parameters of RSSubfieldDataset

    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    # Create the full dataset
    full_dataset = GeoPatchDataset(img_dir, ann_dir, **dataset_kwargs)

    # Split the dataset
    splitter = DatasetSplitter()

    if split_method == "random":
        train_ds, val_ds, test_ds = splitter.random_split(
            full_dataset, train_ratio, val_ratio, test_ratio, random_seed
        )
    elif split_method == "pattern":
        train_ds, val_ds, test_ds = splitter.split_by_file_pattern(full_dataset)
    else:
        raise ValueError(f"Unsupported segmentation methods: {split_method}")

    # Keep the information of splitting
    if save_split_info:
        splitter.save_split_info(train_ds, val_ds, test_ds, save_split_info)

    return train_ds, val_ds, test_ds

@dataclass
class DataConfig:
    img_dir: str = "./model/Dataset"
    ann_dir: str = "./model/Dataset/withID"
    batch_size: int = 4
    num_workers: int = 2
    split_file: str = "data/split_info.json"
    split_method: str = "random"  # "random" | "pattern"
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    random_seed: int = 42

def build_loaders(cfg: DataConfig):
    # 直接调用你已有的 create_split_datasets（最小改动）
    train_ds, val_ds, test_ds = create_split_datasets(
        img_dir=cfg.img_dir,
        ann_dir=cfg.ann_dir,
        split_method=cfg.split_method,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        test_ratio=cfg.test_ratio,
        random_seed=cfg.random_seed,
        save_split_info=cfg.split_file,
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, collate_fn=detection_collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, collate_fn=detection_collate_fn
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, collate_fn=detection_collate_fn
    )
    return train_loader, val_loader, test_loader

