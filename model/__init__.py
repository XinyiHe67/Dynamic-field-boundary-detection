# 建议把 __all__ 修改为
__all__ = [
    "dataset",     # 提供 DataConfig / build_loaders / create_split_datasets / GeoPatchDataset
    "engine",      # pick_device_and_amp / train_one_epoch / validate_one_epoch / fit_maskrcnn
    "maskrcnn",    # build_model_with_custom_loss
    "train",       # train(...)
    "infer",       # predict_dataset(...)
    "visualize",   # predict_and_save_pngs / render_side_by_side ...
]

