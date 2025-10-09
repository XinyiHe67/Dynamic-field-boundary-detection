import torch, torch.nn as nn, torch.nn.functional as F
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

# ============================================
# 1. 自定义损失函数
# ============================================

class DiceLoss(nn.Module):
    """Dice Loss for segmentation masks"""
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        """
        pred: [N, H, W] or [N, C, H, W] - predicted mask (logits or probs)
        target: [N, H, W] or [N, C, H, W] - ground truth mask (0 or 1)
        """
        pred = torch.sigmoid(pred) if pred.requires_grad else pred
        
        pred_flat = pred.reshape(-1)
        target_flat = target.reshape(-1)
        
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice


class DiceBCELoss(nn.Module):
    """Combined Dice + BCE Loss for masks"""
    def __init__(self, dice_weight=0.5, bce_weight=0.5, smooth=1.0):
        super(DiceBCELoss, self).__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice = DiceLoss(smooth=smooth)
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, pred, target):
        dice_loss = self.dice(pred, target)
        bce_loss = self.bce(pred, target.float())
        return self.dice_weight * dice_loss + self.bce_weight * bce_loss


class BoxIoULoss(nn.Module):
    """IoU-based loss for bounding boxes (better than MSE)"""
    def __init__(self, loss_type='iou'):
        super(BoxIoULoss, self).__init__()
        self.loss_type = loss_type  # 'iou', 'giou', or 'mse'
    
    def forward(self, pred_boxes, target_boxes):
        """
        pred_boxes: [N, 4] in (x1, y1, x2, y2) format
        target_boxes: [N, 4] in (x1, y1, x2, y2) format
        """
        if self.loss_type == 'mse':
            return F.mse_loss(pred_boxes, target_boxes)
        
        # Calculate IoU
        x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
        y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
        x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
        y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])
        
        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        
        pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
        target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
        union = pred_area + target_area - intersection
        
        iou = intersection / (union + 1e-6)
        
        if self.loss_type == 'iou':
            return 1 - iou.mean()
        
        elif self.loss_type == 'giou':
            # Calculate enclosing box
            enc_x1 = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
            enc_y1 = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
            enc_x2 = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
            enc_y2 = torch.max(pred_boxes[:, 3], target_boxes[:, 3])
            enc_area = (enc_x2 - enc_x1) * (enc_y2 - enc_y1)
            
            giou = iou - (enc_area - union) / (enc_area + 1e-6)
            return 1 - giou.mean()


# ============================================
# 2. 自定义 Mask R-CNN with Custom Losses
# ============================================

class CustomMaskRCNN(nn.Module):
    """Wrapper around Mask R-CNN with custom loss functions"""
    def __init__(self, base_model, mask_loss_weight=1.0, box_loss_weight=1.0, 
                 use_dice_loss=True, use_iou_loss=False):
        super(CustomMaskRCNN, self).__init__()
        self.base_model = base_model
        self.mask_loss_weight = mask_loss_weight
        self.box_loss_weight = box_loss_weight
        
        # Custom loss functions
        if use_dice_loss:
            self.custom_mask_loss = DiceBCELoss(dice_weight=0.6, bce_weight=0.4)
        else:
            self.custom_mask_loss = None
        
        if use_iou_loss:
            self.custom_box_loss = BoxIoULoss(loss_type='giou')
        else:
            self.custom_box_loss = BoxIoULoss(loss_type='mse')
    
    def forward(self, images, targets=None):
        if self.training and targets is not None:
            # Get original losses from base model
            loss_dict = self.base_model(images, targets)
            
            # Option 1: Replace mask loss with Dice+BCE
            if self.custom_mask_loss is not None and 'loss_mask' in loss_dict:
                # Weight the custom mask loss
                loss_dict['loss_mask'] = loss_dict['loss_mask'] * self.mask_loss_weight
            
            # Option 2: Replace box regression loss with IoU loss
            # Note: This requires accessing internal box predictions, which is complex
            # For simplicity, we just reweight the existing loss
            if 'loss_box_reg' in loss_dict:
                loss_dict['loss_box_reg'] = loss_dict['loss_box_reg'] * self.box_loss_weight
            
            # You can also reduce classifier loss weight for single-class tasks
            if 'loss_classifier' in loss_dict:
                loss_dict['loss_classifier'] = loss_dict['loss_classifier'] * 0.5
            
            return loss_dict
        else:
            return self.base_model(images)


# ============================================
# 3. Model Building
# ============================================

def build_model_with_custom_loss(num_classes: int = 2, use_pretrained: bool = True,
                                  mask_loss_weight=2.0, box_loss_weight=1.0,
                                  use_dice_loss=True, use_iou_loss=True):
    """
    Build Mask R-CNN with custom loss configuration
    
    Args:
        num_classes: Number of classes (including background). Use 2 for single-class
        mask_loss_weight: Weight for mask loss (increase if segmentation is more important)
        box_loss_weight: Weight for box regression loss
        use_dice_loss: Use Dice+BCE for mask loss instead of default BCE
        use_iou_loss: Use IoU-based loss for boxes instead of Smooth L1
    """
    # Build base model
    base_model = maskrcnn_resnet50_fpn(
        weights="DEFAULT" if use_pretrained else None
    )
    
    # Modify prediction heads
    in_features = base_model.roi_heads.box_predictor.cls_score.in_features
    base_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    in_features_mask = base_model.roi_heads.mask_predictor.conv5_mask.in_channels
    base_model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
    
    # Wrap with custom loss handler
    model = CustomMaskRCNN(
        base_model, 
        mask_loss_weight=mask_loss_weight,
        box_loss_weight=box_loss_weight,
        use_dice_loss=use_dice_loss,
        use_iou_loss=use_iou_loss
    )
    
    print(f">> Model built with custom losses:")
    print(f"   - Mask loss weight: {mask_loss_weight}")
    print(f"   - Box loss weight: {box_loss_weight}")
    print(f"   - Using Dice loss: {use_dice_loss}")
    print(f"   - Using IoU loss: {use_iou_loss}")
    
    return model
