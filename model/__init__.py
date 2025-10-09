# 先把各子模块都 import 进来（保持你原有的）
from . import dataset, engine, maskrcnn, train, infer, visualize, evaluation

# 若希望直接从 model 顶层拿到函数，也做一层 re-export（可选但方便）
from .evaluation import (
    evaluate_single_image,
    evaluate_dataset,
    evaluate_from_checkpoint,
)

__all__ = [
    # 模块
    "dataset",
    "engine",
    "maskrcnn",
    "train",
    "infer",
    "visualize",
    "evaluation",
    # 直接导出的评估函数（可选）
    "evaluate_single_image",
    "evaluate_dataset",
    "evaluate_from_checkpoint",
]

