# model/train.py
from model.engine import fit_maskrcnn  # 你的训练主循环

def train(train_loader, val_loader,
          epochs=20, lr=1e-4, weight_decay=1e-4,
          accumulate_grad_batches=2, warmup_epochs=1,
          mask_loss_weight=2.0, box_loss_weight=1.0,
          use_dice_loss=True, use_iou_loss=True,
          ckpt_dir="./checkpoints"):   # ← 新增参数，给 main.py 用
    """
    封装一层，便于 main.py 调用。兼容 fit_maskrcnn 是否支持 ckpt_dir。
    """
    kwargs = dict(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        accumulate_grad_batches=accumulate_grad_batches,
        warmup_epochs=warmup_epochs,
        mask_loss_weight=mask_loss_weight,
        box_loss_weight=box_loss_weight,
        use_dice_loss=use_dice_loss,
        use_iou_loss=use_iou_loss,
    )

    try:
        # 如果 engine.fit_maskrcnn 支持 ckpt_dir，就传进去
        best_ckpt, latest_ckpt = fit_maskrcnn(ckpt_dir=ckpt_dir, **kwargs)
    except TypeError:
        # 如果不支持，就不传（向后兼容）
        best_ckpt, latest_ckpt = fit_maskrcnn(**kwargs)

    return best_ckpt, latest_ckpt
