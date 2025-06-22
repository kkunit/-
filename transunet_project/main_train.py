import argparse
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import random
import logging
import json
from datetime import datetime

# Project imports
from models.transunet import TransUNet, VIT_CONFIG # Assuming VIT_CONFIG provides defaults
from utils.data_utils import get_augmented_dataset # SegmentationDataset is used within get_augmented_dataset
from utils.training_utils import get_optimizer, get_lr_scheduler, DiceLoss, JaccardLoss, DiceBCELoss
from utils.eval_utils import dice_coefficient, jaccard_index # For validation metrics
from utils.viz_utils import plot_training_curves

# Setup basic logging
# Configure root logger once
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Get logger for this module


def parse_args():
    parser = argparse.ArgumentParser(description="TransUNet Training Script")

    # Dataset paths
    parser.add_argument('--image_dir', type=str, required=True, help="Directory for training images")
    parser.add_argument('--mask_dir', type=str, required=True, help="Directory for training masks")
    parser.add_argument('--val_image_dir', type=str, default=None, help="Directory for validation images (optional)")
    parser.add_argument('--val_mask_dir', type=str, default=None, help="Directory for validation masks (optional)")

    # Model parameters
    parser.add_argument('--cnn_backbone', type=str, default='resnet50', choices=['resnet18', 'resnet50'], help="CNN backbone type")
    parser.add_argument('--vit_config_name', type=str, default='ViT-B_16', help="ViT configuration name")

    # Transfer Learning / Fine-tuning
    parser.add_argument('--cnn_weights_path', type=str, default=None, help="Path to pretrained CNN backbone weights")
    parser.add_argument('--cnn_freeze_stages', type=int, default=-1, help="Freeze CNN stages up to this number (-1 unfreezes all)")
    parser.add_argument('--vit_weights_path', type=str, default=None, help="Path to pretrained ViT weights")
    parser.add_argument('--vit_freeze_patch_embed', action='store_true', help="Freeze ViT patch embedding")
    parser.add_argument('--vit_freeze_pos_embed', action='store_true', help="Freeze ViT positional embedding")
    parser.add_argument('--vit_freeze_blocks_upto', type=int, default=-1, help="Freeze ViT blocks up to this index (-1 unfreezes all)")

    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size for training")
    parser.add_argument('--lr', type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument('--optimizer_name', type=str, default='Adam', choices=['Adam', 'AdamW', 'SGD'], help="Optimizer type") # Renamed from --optimizer
    parser.add_argument('--weight_decay', type=float, default=1e-5, help="Weight decay for optimizer")
    parser.add_argument('--lr_scheduler_name', type=str, default='ReduceLROnPlateau', choices=['ReduceLROnPlateau', 'StepLR', 'CosineAnnealingLR', 'None'], help="Learning rate scheduler")
    # Params for ReduceLROnPlateau
    parser.add_argument('--lr_scheduler_patience', type=int, default=10, help="Patience for ReduceLROnPlateau")
    parser.add_argument('--lr_scheduler_factor', type=float, default=0.1, help="Factor for ReduceLROnPlateau")
    # Params for StepLR
    parser.add_argument('--lr_scheduler_step_size', type=int, default=30, help="Step size for StepLR")
    parser.add_argument('--lr_scheduler_gamma', type=float, default=0.1, help="Gamma for StepLR")
    # Params for CosineAnnealingLR
    parser.add_argument('--lr_scheduler_t_max', type=int, default=0, help="T_max for CosineAnnealingLR (if 0, uses epochs)")


    parser.add_argument('--loss_type', type=str, default='DiceBCE', choices=['Dice', 'Jaccard', 'DiceBCE', 'CrossEntropy'], help="Loss function type")

    # Dataset and Preprocessing
    parser.add_argument('--img_size_h', type=int, default=256, help="Image height for resizing")
    parser.add_argument('--img_size_w', type=int, default=256, help="Image width for resizing")
    parser.add_argument('--num_classes', type=int, default=1, help="Number of segmentation classes (1 for binary)")
    parser.add_argument('--mask_target_type', type=str, default='binary_float', choices=['binary_float', 'multiclass_long'], help="Target type for masks")
    parser.add_argument('--no_augmentation', action='store_true', help="Disable data augmentation for training set")

    # Output and Environment
    parser.add_argument('--output_dir', type=str, default='runs/transunet_experiment', help="Base directory to save models, logs, and plots. A timestamped subfolder will be created.")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="Device (cuda or cpu)")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument('--num_workers', type=int, default=2, help="Number of workers for DataLoader")

    return parser.parse_args()

def setup_environment(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == 'cuda' and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False # False for reproducibility

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # args.output_dir is the base. Create a timestamped subdirectory within it.
    args.output_dir_final = os.path.join(args.output_dir, timestamp)
    os.makedirs(args.output_dir_final, exist_ok=True)

    # Setup file logger
    file_handler = logging.FileHandler(os.path.join(args.output_dir_final, 'training.log'))
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(module)s:%(lineno)d] - %(message)s')
    file_handler.setFormatter(formatter)

    # Add handler to the root logger to capture logs from all modules
    root_logger = logging.getLogger()
    # Avoid adding multiple file handlers if this function is called multiple times (e.g. in a notebook)
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == file_handler.baseFilename for h in root_logger.handlers):
        root_logger.addHandler(file_handler)

    logger.info(f"Script arguments: {json.dumps(vars(args), indent=2)}")
    with open(os.path.join(args.output_dir_final, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    return torch.device(args.device)


def main():
    args = parse_args()
    device = setup_environment(args)

    logger.info("Starting TransUNet Training Process")
    logger.info(f"Using device: {device}")
    logger.info(f"Output will be saved to: {args.output_dir_final}")

    # 1. Data Loading
    logger.info("Loading datasets...")
    img_size_tuple = (args.img_size_h, args.img_size_w)

    train_dataset = get_augmented_dataset(
        image_dir=args.image_dir, mask_dir=args.mask_dir,
        img_size=img_size_tuple, augment=not args.no_augmentation,
        mask_type=args.mask_target_type
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=(device.type == 'cuda'))
    logger.info(f"Training dataset loaded: {len(train_dataset)} samples.")

    val_loader = None
    if args.val_image_dir and args.val_mask_dir:
        val_dataset = get_augmented_dataset(
            image_dir=args.val_image_dir, mask_dir=args.val_mask_dir,
            img_size=img_size_tuple, augment=False,
            mask_type=args.mask_target_type
        )
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=(device.type == 'cuda'))
        logger.info(f"Validation dataset loaded: {len(val_dataset)} samples.")
    else:
        logger.info("No validation dataset provided. Model saving will be based on last epoch or periodic.")

    # 2. Model Initialization
    logger.info("Initializing model...")
    current_vit_config = VIT_CONFIG
    if args.vit_config_name != 'ViT-B_16': # Default name in VIT_CONFIG
        # This is a placeholder for logic to select different ViT configurations if VIT_CONFIG were a dict of configs
        logger.warning(f"ViT config '{args.vit_config_name}' requested. TransUNet currently uses its internal default VIT_CONFIG. Modify models/transunet.py or pass a config dict directly if other ViT sizes are needed.")

    model = TransUNet(
        img_size=img_size_tuple, num_classes=args.num_classes,
        cnn_backbone_type=args.cnn_backbone, cnn_in_channels=3,
        cnn_pretrained_weights_path=args.cnn_weights_path, cnn_freeze_stages=args.cnn_freeze_stages,
        vit_patch_size=1, vit_config=current_vit_config,
        vit_pretrained_weights_path=args.vit_weights_path,
        vit_freeze_patch_embed=args.vit_freeze_patch_embed, vit_freeze_pos_embed=args.vit_freeze_pos_embed,
        vit_freeze_blocks_upto=args.vit_freeze_blocks_upto
    ).to(device)
    logger.info(f"Model: TransUNet with {args.cnn_backbone} backbone and ViT ({args.vit_config_name} like).")
    num_params_total = sum(p.numel() for p in model.parameters())
    num_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {num_params_total / 1e6:.2f} M")
    logger.info(f"Trainable parameters: {num_params_trainable / 1e6:.2f} M")

    # 3. Loss Function and Optimizer
    logger.info(f"Setting up loss function: {args.loss_type}")
    if args.loss_type == 'Dice': criterion = DiceLoss()
    elif args.loss_type == 'Jaccard': criterion = JaccardLoss()
    elif args.loss_type == 'DiceBCE': criterion = DiceBCELoss()
    elif args.loss_type == 'CrossEntropy':
        if args.mask_target_type != 'multiclass_long' and args.num_classes > 1:
            logger.error("CrossEntropyLoss for num_classes > 1 requires 'multiclass_long' mask_target_type.")
            raise ValueError("Invalid mask_target_type for CrossEntropyLoss with multi-class.")
        criterion = torch.nn.CrossEntropyLoss()
    else: raise ValueError(f"Unsupported loss type: {args.loss_type}")

    optimizer = get_optimizer(model, args.optimizer_name, lr=args.lr, weight_decay=args.weight_decay)

    scheduler_params = {}
    if args.lr_scheduler_name == 'ReduceLROnPlateau':
        scheduler_params = {'patience': args.lr_scheduler_patience, 'factor': args.lr_scheduler_factor, 'min_lr': 1e-7}
    elif args.lr_scheduler_name == 'StepLR':
        scheduler_params = {'step_size': args.lr_scheduler_step_size, 'gamma': args.lr_scheduler_gamma}
    elif args.lr_scheduler_name == 'CosineAnnealingLR':
        scheduler_params = {'T_max': args.lr_scheduler_t_max if args.lr_scheduler_t_max > 0 else args.epochs, 'eta_min': 1e-7}

    scheduler = get_lr_scheduler(optimizer, args.lr_scheduler_name, **scheduler_params)
    logger.info(f"Optimizer: {args.optimizer_name}, LR Scheduler: {args.lr_scheduler_name}")

    # 4. Training Loop
    logger.info("Starting training loop...")
    best_val_metric = -np.inf
    history = {'train_loss': [], 'val_loss': [], 'val_dice': [], 'val_iou': []}

    for epoch in range(args.epochs):
        model.train()
        running_train_loss = 0.0
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * inputs.size(0)
            if (i + 1) % (max(1, len(train_loader) // 10)) == 0: # Log approx 10 times
                logger.info(f"Epoch [{epoch+1}/{args.epochs}] Batch [{i+1}/{len(train_loader)}] Train Loss: {loss.item():.4f}")

        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        history['train_loss'].append(epoch_train_loss)
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch [{epoch+1}/{args.epochs}] Avg Train Loss: {epoch_train_loss:.4f} LR: {current_lr:.2e}")

        current_epoch_val_metric_for_saving = -1.0
        if val_loader:
            model.eval()
            running_val_loss = 0.0
            all_val_preds, all_val_targets = [], []
            with torch.no_grad():
                for inputs_val, targets_val in val_loader:
                    inputs_val, targets_val = inputs_val.to(device, non_blocking=True), targets_val.to(device, non_blocking=True)
                    outputs_val = model(inputs_val)
                    val_loss_item = criterion(outputs_val, targets_val)
                    running_val_loss += val_loss_item.item() * inputs_val.size(0)
                    all_val_preds.append(outputs_val.cpu())
                    all_val_targets.append(targets_val.cpu())

            epoch_val_loss = running_val_loss / len(val_loader.dataset)
            history['val_loss'].append(epoch_val_loss)

            val_preds_tensor = torch.cat(all_val_preds)
            val_targets_tensor = torch.cat(all_val_targets)

            epoch_val_dice = dice_coefficient(val_preds_tensor, val_targets_tensor, threshold=0.5)
            epoch_val_iou = jaccard_index(val_preds_tensor, val_targets_tensor, threshold=0.5)
            history['val_dice'].append(epoch_val_dice)
            history['val_iou'].append(epoch_val_iou)
            logger.info(f"Epoch [{epoch+1}/{args.epochs}] Val Loss: {epoch_val_loss:.4f}, Val Dice: {epoch_val_dice:.4f}, Val IoU: {epoch_val_iou:.4f}")
            current_epoch_val_metric_for_saving = epoch_val_dice # Use Dice for best model saving

            if args.lr_scheduler_name == 'ReduceLROnPlateau': scheduler.step(epoch_val_loss) # common to use val_loss
            elif args.lr_scheduler_name != 'None': scheduler.step()

            if current_epoch_val_metric_for_saving > best_val_metric:
                best_val_metric = current_epoch_val_metric_for_saving
                torch.save(model.state_dict(), os.path.join(args.output_dir_final, 'best_model.pth'))
                logger.info(f"Saved new best model (Val Dice: {best_val_metric:.4f}) at epoch {epoch+1}")
        else: # No validation loader
            if args.lr_scheduler_name != 'None' and args.lr_scheduler_name != 'ReduceLROnPlateau': scheduler.step()

        # Save model periodically (e.g. every 10 epochs or last epoch) if not relying on best_val_metric or if no val_loader
        if not val_loader and (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(args.output_dir_final, f'model_epoch_{epoch+1}.pth'))
            logger.info(f"Saved model checkpoint at epoch {epoch+1}")

    # 5. Post-Training
    logger.info("Training finished.")
    final_model_path = os.path.join(args.output_dir_final, 'final_model_epoch_last.pth')
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Final model (last epoch) saved to {final_model_path}")

    plot_metrics = ['loss']
    if val_loader and history['val_dice']: plot_metrics.extend(['dice', 'iou'])

    valid_history_for_plot = {k: v for k, v in history.items() if v}
    actual_metrics_to_plot = [m for m in plot_metrics if f'train_{m}' in valid_history_for_plot or f'val_{m}' in valid_history_for_plot]

    if actual_metrics_to_plot:
        fig = plot_training_curves(valid_history_for_plot, metrics=actual_metrics_to_plot)
        fig_path = os.path.join(args.output_dir_final, 'training_curves.png')
        try:
            fig.savefig(fig_path)
            logger.info(f"Training curves saved to {fig_path}")
        except Exception as e:
            logger.error(f"Could not save training curves: {e}")
        finally:
            import matplotlib.pyplot as plt
            plt.close(fig)
    else:
        logger.info("No data to plot for training curves (history might be empty or keys missing).")

if __name__ == '__main__':
    main()
