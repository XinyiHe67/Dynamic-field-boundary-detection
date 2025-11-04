# model/train.py
from model.engine import fit_maskrcnn

def train(train_loader, val_loader,
          epochs=20, lr=1e-4, weight_decay=1e-4,
          accumulate_grad_batches=2, warmup_epochs=1,
          mask_loss_weight=2.0, box_loss_weight=1.0,
          use_dice_loss=True, use_iou_loss=True,
          ckpt_dir="./checkpoints"):  
    """
    Wrapper function for easy calling from main.py.
    Compatible with whether fit_maskrcnn supports ckpt_dir.
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
        # If engine.fit_maskrcnn supports ckpt_dir, pass it in
        best_ckpt, latest_ckpt = fit_maskrcnn(ckpt_dir=ckpt_dir, **kwargs)
    except TypeError:
        # If not supported, skip it (for backward compatibility)
        best_ckpt, latest_ckpt = fit_maskrcnn(**kwargs)

    return best_ckpt, latest_ckpt
