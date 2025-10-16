# model/__init__.py

from . import dataset, engine, maskrcnn, train, infer
from . import visualize, evaluation, gee_pipeline, cutpatch
from . import exportTargetPolygon, data_loader_auto
from . import postprocessing  # ← 新增：整个模块可被 import

# 评估函数对外开放
from .evaluation import (
    evaluate_single_image,
    evaluate_dataset,
    evaluate_from_checkpoint,
)

# GEE 管道对外开放
from .gee_pipeline import (
    read_service_account_json,
    run_gee_fetch_and_export,
)

# 后处理对外 API（可选：直接把常用函数 re-export，方便 main.py 直接调用）
from .postprocessing import (
    run_postprocessing,
    merge_gpkgs,
    postprocess_predictions_to_gpkg,
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
    "gee_pipeline",
    "cutpatch",
    "exportTargetPolygon",
    "data_loader_auto",      # ← 去掉了原来的前导空格
    "postprocessing",        # ← 新增：模块也作为公共导出

    # 评估函数
    "evaluate_single_image",
    "evaluate_dataset",
    "evaluate_from_checkpoint",

    # GEE API
    "read_service_account_json",
    "run_gee_fetch_and_export",

    # 后处理 API（给 main.py 直接用）
    "run_postprocessing",
    "merge_gpkgs",
    "postprocess_predictions_to_gpkg",
]


