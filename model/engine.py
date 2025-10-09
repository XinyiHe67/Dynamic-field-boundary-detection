import time, math, os
import torch, torch.nn as nn
from torch.amp import autocast, GradScaler
from .maskrcnn import build_model_with_custom_loss


def pick_device_and_amp():
    if torch.cuda.is_available():
        return torch.device("cuda"), True
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps"), False
    else:
        return torch.device("cpu"), False

device, use_amp = pick_device_and_amp()
print(">> device:", device, "| use_amp:", use_amp)
if device.type == "cpu":
    torch.set_num_threads(4)

# ============================================
# 4. Training Functions (same as before but with custom model)
# ============================================

def optimize_batch_data(images, targets, device):
    images = [img.to(device, non_blocking=True) for img in images]
    new_targets = []
    for t in targets:
        new_t = {}
        for k, v in t.items():
            if isinstance(v, torch.Tensor):
                new_t[k] = v.to(device, non_blocking=True)
            else:
                new_t[k] = v  # Keep non-tensor values as-is
        new_targets.append(new_t)
    
    return images, new_targets


def train_one_epoch(model, loader, optimizer, device, scaler=None, max_norm=0.0,
                   debug_mode=False, accumulate_grad_batches=1):
    model.train()
    total_loss = 0.0
    n_batches = 0
    optimizer.zero_grad(set_to_none=True)
    
    for batch_idx, (images, targets) in enumerate(loader):
        images, targets = optimize_batch_data(images, targets, device)
        
        if debug_mode and batch_idx == 0:
            print(f">> Batch 0: {len(images)} images")
            for i, img in enumerate(images[:2]):
                print(f"   Image {i}: {img.shape}, dtype={img.dtype}, "
                      f"range=[{img.min():.3f}, {img.max():.3f}]")
            for i, tgt in enumerate(targets[:2]):
                print(f"   Target {i}: boxes={tgt['boxes'].shape}, "
                      f"labels={tgt['labels'].shape}, masks={tgt['masks'].shape}")
        
        if scaler is not None and device.type == "cuda":
            with autocast(device_type="cuda", dtype=torch.float16):
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values()) / accumulate_grad_batches
            scaler.scale(loss).backward()
        else:
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values()) / accumulate_grad_batches
            loss.backward()
        
        if (batch_idx + 1) % accumulate_grad_batches == 0:
            if scaler is not None and device.type == "cuda":
                if max_norm > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                if max_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        
        total_loss += float(loss.item()) * accumulate_grad_batches
        n_batches += 1
        
        if debug_mode and batch_idx % max(1, len(loader)//10) == 0:
            print(f"   Batch {batch_idx}/{len(loader)}: loss={loss.item():.4f}")
    
    return total_loss / max(1, n_batches)


@torch.no_grad()
def validate_one_epoch(model, loader, device, debug_mode=False):
    model.train()  # Keep in train mode to get losses
    sums = {
        "loss": 0.0,
        "loss_classifier": 0.0,
        "loss_box_reg": 0.0,
        "loss_mask": 0.0,
        "loss_objectness": 0.0,
        "loss_rpn_box_reg": 0.0,
    }
    n_batches = 0
    
    for batch_idx, (images, targets) in enumerate(loader):
        images, targets = optimize_batch_data(images, targets, device)
        loss_dict = model(images, targets)
        total = sum(loss_dict.values())
        
        sums["loss"] += float(total.item())
        for k in loss_dict:
            if k in sums:
                sums[k] += float(loss_dict[k].item())
        n_batches += 1
        
        if debug_mode and batch_idx % max(1, len(loader)//5) == 0:
            print(f"   Val Batch {batch_idx}/{len(loader)}: loss={total.item():.4f}")
    
    for k in sums:
        sums[k] /= max(1, n_batches)
    return sums


def quick_test_model(model, sample_loader, device, max_batches=2):
    print(">> Running quick model test...")
    model.train()
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(sample_loader):
            if batch_idx >= max_batches:
                break
            print(f"   Testing batch {batch_idx+1}/{max_batches}")
            images, targets = optimize_batch_data(images, targets, device)
            try:
                start_time = time.time()
                loss_dict = model(images, targets)
                forward_time = time.time() - start_time
                total_loss = sum(loss_dict.values())
                
                print(f"   Batch {batch_idx+1} OK: loss={total_loss:.4f}, "
                      f"time={forward_time:.2f}s")
                if device.type == "cuda":
                    memory_used = torch.cuda.memory_allocated() / 1024**3
                    print(f"   GPU Memory: {memory_used:.2f}GB")
            except Exception as e:
                print(f"   Batch {batch_idx+1} FAILED: {str(e)}")
                return False
    print(">> Model test completed successfully!")
    return True


def fit_maskrcnn(
    train_loader,
    val_loader,
    epochs: int = 5,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    max_grad_norm: float = 0.0,
    ckpt_dir: str = "./checkpoints",
    debug_mode: bool = True,
    accumulate_grad_batches: int = 2,
    warmup_epochs: int = 1,
    # New parameters for custom loss
    mask_loss_weight: float = 2.0,
    box_loss_weight: float = 1.0,
    use_dice_loss: bool = True,
    use_iou_loss: bool = True,
):
    """
    Train Mask R-CNN with custom loss functions
    
    Args:
        mask_loss_weight: Increase this (e.g., 2.0-3.0) if segmentation quality is most important
        box_loss_weight: Weight for bounding box regression
        use_dice_loss: Use Dice+BCE for masks (recommended for segmentation tasks)
        use_iou_loss: Use IoU-based loss for boxes (recommended)
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # Build model with custom losses
    model = build_model_with_custom_loss(
        num_classes=2,  # Background + 1 object class
        use_pretrained=True,
        mask_loss_weight=mask_loss_weight,
        box_loss_weight=box_loss_weight,
        use_dice_loss=use_dice_loss,
        use_iou_loss=use_iou_loss
    ).to(device)
    
    if not quick_test_model(model, train_loader, device):
        print("Model test failed! Please check your data.")
        return None, None
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        eps=1e-8,
        betas=(0.9, 0.999)
    )
    
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            import math
            progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
            return 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler(device="cuda", enabled=use_amp)
    
    best_val = float("inf")
    best_path = os.path.join(ckpt_dir, "maskrcnn_best.pth")
    latest_path = os.path.join(ckpt_dir, "maskrcnn_latest.pth")
    
    print(f">> Starting training: {epochs} epochs, lr={lr}, device={device}")
    print(f">> Batch accumulation: {accumulate_grad_batches}, warmup: {warmup_epochs} epochs")
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        print(f"\n[Epoch {epoch:03d}/{epochs}] Starting...")
        
        train_start = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, scaler, max_grad_norm,
            debug_mode=(debug_mode and epoch <= 2),
            accumulate_grad_batches=accumulate_grad_batches
        )
        train_time = time.time() - train_start
        
        val_start = time.time()
        val_metrics = validate_one_epoch(model, val_loader, device, 
                                         debug_mode=(debug_mode and epoch <= 2))
        val_time = time.time() - val_start
        val_loss = val_metrics["loss"]
        
        scheduler.step()
        
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "val_loss": val_loss,
            "config": {
                "lr": lr,
                "weight_decay": weight_decay,
                "accumulate_grad_batches": accumulate_grad_batches,
                "mask_loss_weight": mask_loss_weight,
                "box_loss_weight": box_loss_weight,
                "use_dice_loss": use_dice_loss,
                "use_iou_loss": use_iou_loss,
            }
        }
        torch.save(checkpoint, latest_path)
        
        if val_loss < best_val:
            best_val = val_loss
            torch.save(checkpoint, best_path)
            flag = " *** NEW BEST ***"
        else:
            flag = ""
        
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"[Epoch {epoch:03d}] "
              f"train_loss={train_loss:.4f} ({train_time:.1f}s) | "
              f"val_loss={val_loss:.4f} ({val_time:.1f}s) "
              f"[cls:{val_metrics['loss_classifier']:.3f} "
              f"box:{val_metrics['loss_box_reg']:.3f} "
              f"mask:{val_metrics['loss_mask']:.3f}] "
              f"| lr={current_lr:.2e} | total={epoch_time:.1f}s{flag}")
        
        if device.type == "cuda":
            torch.cuda.empty_cache()
    
    print(f"\n>> Training completed! Best val_loss: {best_val:.4f}")
    return best_path, latest_path

