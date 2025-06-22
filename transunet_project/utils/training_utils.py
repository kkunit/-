import torch
import torch.nn as nn
import torch.optim as optim

# --- Loss Functions ---
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # inputs: (B, C, H, W), logits (before sigmoid/softmax)
        # targets: (B, C, H, W), ground truth {0,1}
        # For binary segmentation (C=1), apply sigmoid to inputs
        if inputs.shape[1] == 1: # Binary case
            inputs = torch.sigmoid(inputs)
        else: # Multi-class case, apply softmax
            inputs = torch.softmax(inputs, dim=1)
            # If targets are (B, H, W) of class indices, convert to one-hot
            if targets.ndim == inputs.ndim - 1:
                targets = torch.nn.functional.one_hot(targets, num_classes=inputs.shape[1]).permute(0, 3, 1, 2).float()

        # Flatten inputs and targets
        inputs = inputs.contiguous().view(inputs.size(0), -1)
        targets = targets.contiguous().view(targets.size(0), -1)

        intersection = (inputs * targets).sum(dim=1)
        dice_coeff = (2. * intersection + self.smooth) / (inputs.sum(dim=1) + targets.sum(dim=1) + self.smooth)

        return 1 - dice_coeff.mean() # Mean loss over batch

class JaccardLoss(nn.Module): # Also known as IoULoss
    def __init__(self, smooth=1e-6):
        super(JaccardLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # inputs: (B, C, H, W), logits
        # targets: (B, C, H, W), ground truth {0,1}
        if inputs.shape[1] == 1: # Binary
            inputs = torch.sigmoid(inputs)
        else: # Multi-class
            inputs = torch.softmax(inputs, dim=1)
            if targets.ndim == inputs.ndim - 1:
                targets = torch.nn.functional.one_hot(targets, num_classes=inputs.shape[1]).permute(0, 3, 1, 2).float()

        inputs = inputs.contiguous().view(inputs.size(0), -1)
        targets = targets.contiguous().view(targets.size(0), -1)

        intersection = (inputs * targets).sum(dim=1)
        total = inputs.sum(dim=1) + targets.sum(dim=1)
        union = total - intersection

        iou = (intersection + self.smooth) / (union + self.smooth)
        return 1 - iou.mean()

class DiceBCELoss(nn.Module):
    def __init__(self, smooth_dice=1e-6, smooth_bce=1e-7, weight_dice=0.5, weight_bce=0.5):
        super(DiceBCELoss, self).__init__()
        self.dice_loss = DiceLoss(smooth=smooth_dice)
        # BCEWithLogitsLoss is numerically more stable than Sigmoid + BCE
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.weight_dice = weight_dice
        self.weight_bce = weight_bce

    def forward(self, inputs, targets):
        # inputs: (B, C, H, W), logits
        # targets: (B, C, H, W), ground truth {0,1}
        # This loss is typically used for binary segmentation (C=1)
        if inputs.shape[1] != 1:
            # For multi-class, one might use CrossEntropy + Dice.
            # This implementation is primarily for binary.
            # Consider raising an error or adapting for multi-class if needed.
            # For multi-class, BCE is not appropriate. Use CrossEntropyLoss.
            raise NotImplementedError("DiceBCELoss as implemented is for binary segmentation (C=1).")

        dice_l = self.dice_loss(inputs, targets)
        # BCEWithLogitsLoss expects raw logits for inputs
        bce_l = self.bce_loss(inputs, targets)

        return self.weight_dice * dice_l + self.weight_bce * bce_l

# --- Optimizer Helper ---
def get_optimizer(model, optimizer_name='Adam', lr=1e-3, weight_decay=0, momentum=0.9):
    if optimizer_name.lower() == 'adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'adamw':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Optimizer {optimizer_name} not supported.")

# --- Learning Rate Scheduler Helper ---
def get_lr_scheduler(optimizer, scheduler_name='ReduceLROnPlateau',
                     patience=10, factor=0.1, min_lr=1e-6, # For ReduceLROnPlateau
                     step_size=30, gamma=0.1, # For StepLR
                     T_max=100, eta_min=0 # For CosineAnnealingLR
                    ):
    if scheduler_name.lower() == 'reducelronplateau':
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience, min_lr=min_lr, verbose=True)
    elif scheduler_name.lower() == 'steplr':
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma, verbose=True)
    elif scheduler_name.lower() == 'cosineannealinglr':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min, verbose=True)
    elif scheduler_name.lower() == 'none' or scheduler_name is None:
        return None
    else:
        raise ValueError(f"Scheduler {scheduler_name} not supported.")

# --- Training Loop (Simplified Example) ---
# A more complete training loop would be in main_train.py or a dedicated trainer class.
# This is just a helper function for one epoch.
def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch_num=0, total_epochs=0):
    model.train()
    running_loss = 0.0
    num_samples = 0

    for i, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        num_samples += inputs.size(0)

        if i % (len(dataloader) // 10 + 1) == 0 : # Print progress roughly 10 times per epoch
             print(f"Epoch [{epoch_num}/{total_epochs}] Batch [{i}/{len(dataloader)}] Loss: {loss.item():.4f}")

    epoch_loss = running_loss / num_samples
    return epoch_loss

def evaluate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    num_samples = 0

    # Additional metrics (e.g., Dice score) can be calculated here if needed for validation
    # For now, just validation loss

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            num_samples += inputs.size(0)

    epoch_loss = running_loss / num_samples
    return epoch_loss


if __name__ == '__main__':
    # --- Test Loss Functions ---
    print("Testing Loss Functions...")
    # Binary case
    dummy_logits_binary = torch.randn(4, 1, 32, 32) # B, C=1, H, W
    dummy_targets_binary = torch.randint(0, 2, (4, 1, 32, 32)).float()

    dice = DiceLoss()
    loss_d = dice(dummy_logits_binary, dummy_targets_binary)
    print(f"Dice Loss (binary): {loss_d.item()}")

    jaccard = JaccardLoss()
    loss_j = jaccard(dummy_logits_binary, dummy_targets_binary)
    print(f"Jaccard Loss (binary): {loss_j.item()}")

    dice_bce = DiceBCELoss()
    loss_db = dice_bce(dummy_logits_binary, dummy_targets_binary)
    print(f"DiceBCELoss (binary): {loss_db.item()}")

    # Multi-class case (for Dice/Jaccard)
    num_classes = 3
    dummy_logits_multi = torch.randn(4, num_classes, 32, 32) # B, C=3, H, W
    dummy_targets_multi_idx = torch.randint(0, num_classes, (4, 32, 32)).long() # B, H, W (class indices)
    # dummy_targets_multi_onehot = torch.nn.functional.one_hot(dummy_targets_multi_idx, num_classes).permute(0,3,1,2).float()


    loss_d_multi = dice(dummy_logits_multi, dummy_targets_multi_idx)
    print(f"Dice Loss (multi-class, target indices): {loss_d_multi.item()}")

    loss_j_multi = jaccard(dummy_logits_multi, dummy_targets_multi_idx)
    print(f"Jaccard Loss (multi-class, target indices): {loss_j_multi.item()}")

    # Test CrossEntropy + Dice (a common multi-class combo)
    ce_loss = nn.CrossEntropyLoss()
    loss_ce_multi = ce_loss(dummy_logits_multi, dummy_targets_multi_idx)
    print(f"CrossEntropy Loss (multi-class): {loss_ce_multi.item()}")

    combined_loss_multi = 0.5 * loss_ce_multi + 0.5 * loss_d_multi
    print(f"Combined CE + Dice (multi-class): {combined_loss_multi.item()}")

    # --- Test Optimizer and Scheduler ---
    print("\nTesting Optimizer and Scheduler...")
    dummy_model = nn.Linear(10, 1)
    adam_opt = get_optimizer(dummy_model, 'Adam', lr=1e-3)
    print(f"Adam Optimizer: {adam_opt}")

    sgd_opt = get_optimizer(dummy_model, 'SGD', lr=1e-2, momentum=0.99)
    print(f"SGD Optimizer: {sgd_opt}")

    plateau_scheduler = get_lr_scheduler(adam_opt, 'ReduceLROnPlateau')
    print(f"ReduceLROnPlateau Scheduler: {plateau_scheduler}")
    # To step ReduceLROnPlateau: scheduler.step(validation_loss)

    step_scheduler = get_lr_scheduler(sgd_opt, 'StepLR', step_size=5)
    print(f"StepLR Scheduler: {step_scheduler}")
    # To step StepLR: scheduler.step()

    print("\nBasic training_utils.py implemented.")
