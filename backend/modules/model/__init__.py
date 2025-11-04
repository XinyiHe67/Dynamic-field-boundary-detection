# model/__init__.py

from . import dataset, engine, maskrcnn, train, infer
from . import visualize, evaluation, gee_pipeline, cutpatch
from . import exportTargetPolygon, data_loader_auto
from . import postprocessing  

# Evaluation functions exposed externally
from .evaluation import (
    evaluate_single_image,
    evaluate_dataset,
    evaluate_from_checkpoint,
)

# GEE pipeline exposed externally
from .gee_pipeline import (
    read_service_account_json,
    run_gee_fetch_and_export,
)

# Postprocessing public API
from .postprocessing import (
    run_postprocessing,
    merge_gpkgs,
    postprocess_predictions_to_gpkg,
)

__all__ = [
    # Modules
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
    "data_loader_auto",      
    "postprocessing",       

    # Evaluation functions
    "evaluate_single_image",
    "evaluate_dataset",
    "evaluate_from_checkpoint",

    # GEE API
    "read_service_account_json",
    "run_gee_fetch_and_export",

    # Postprocessing API
    "run_postprocessing",
    "merge_gpkgs",
    "postprocess_predictions_to_gpkg",
]


