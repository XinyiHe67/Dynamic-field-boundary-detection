# model/train.py
from model.engine import fit_maskrcnn  # 你已有的训练主循环
# 如果你的 fit_maskrcnn 里会自己构建模型和 device，就不用在这里再建
# 若你想改为在这里建模，也可引入：from model.maskrcnn import build_model_with_custom_loss

def train(train_loader, val_loader,
          epochs=20, lr=1e-4, weight_decay=1e-4,
          accumulate_grad_batches=2, warmup_epochs=1,
          # 下面这几个跟你 engine.fit_maskrcnn 的签名保持一致
          mask_loss_weight=2.0, box_loss_weight=1.0,
          use_dice_loss=True, use_iou_loss=True):
    """
    封装一层，便于 main.py 调用
    """
    best_ckpt, latest_ckpt = fit_maskrcnn(
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
    return best_ckpt, latest_ckpt
