import argparse
import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import logging
import json
from datetime import datetime
import matplotlib.pyplot as plt # For closing figures

# Project imports
from models.transunet import TransUNet, VIT_CONFIG
from utils.data_utils import get_augmented_dataset
from utils.eval_utils import dice_coefficient, jaccard_index, calculate_classification_metrics
# calculate_classification_metrics also gives data for CM, ROC, PR
from utils.viz_utils import display_segmentation_results, plot_training_curves, plot_roc_curve, plot_pr_curve, plot_confusion_matrix

# Setup basic logging
if not logging.getLogger().hasHandlers(): # Ensure root logger is configured only once
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="TransUNet Testing Script")

    # Dataset paths
    parser.add_argument('--image_dir', type=str, required=True, help="Directory for test images")
    parser.add_argument('--mask_dir', type=str, required=True, help="Directory for test masks (ground truth)")

    # Model path and configuration (must match the trained model)
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained .pth model file")
    parser.add_argument('--cnn_backbone', type=str, default='resnet50', choices=['resnet18', 'resnet50'], help="CNN backbone type of the loaded model")
    parser.add_argument('--vit_config_name', type=str, default='ViT-B_16', help="ViT configuration name of the loaded model")
    parser.add_argument('--img_size_h', type=int, default=256, help="Image height used during training")
    parser.add_argument('--img_size_w', type=int, default=256, help="Image width used during training")
    parser.add_argument('--num_classes', type=int, default=1, help="Number of segmentation classes (1 for binary)")

    # Evaluation parameters
    parser.add_argument('--metrics_threshold', type=float, default=0.5, help="Threshold for converting probabilities to binary predictions for metrics")
    parser.add_argument('--mask_target_type', type=str, default='binary_float', choices=['binary_float', 'multiclass_long'], help="Target type for masks, should match training")
    parser.add_argument('--save_samples_count', type=int, default=5, help="Number of sample segmentation results to save as images. 0 to disable.")


    # Output and Environment
    parser.add_argument('--output_dir', type=str, default='test_results/transunet_test', help="Base directory to save results, metrics, and plots")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="Device (cuda or cpu)")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size for inference (can often be larger than training)")
    parser.add_argument('--num_workers', type=int, default=2, help="Number of workers for DataLoader")

    return parser.parse_args()

def setup_test_environment(args):
    device = torch.device(args.device)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    args.output_dir_final = os.path.join(args.output_dir, timestamp)
    os.makedirs(args.output_dir_final, exist_ok=True)

    # Setup file logger
    file_handler = logging.FileHandler(os.path.join(args.output_dir_final, 'testing.log'))
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(module)s:%(lineno)d] - %(message)s')
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == file_handler.baseFilename for h in root_logger.handlers):
        root_logger.addHandler(file_handler)

    logger.info(f"Script arguments: {json.dumps(vars(args), indent=2)}")
    with open(os.path.join(args.output_dir_final, 'args_test.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    return device

def main():
    args = parse_args()
    device = setup_test_environment(args)

    logger.info("Starting TransUNet Testing Process")
    logger.info(f"Using device: {device}")
    logger.info(f"Output will be saved to: {args.output_dir_final}")

    # 1. Data Loading
    logger.info("Loading test dataset...")
    img_size_tuple = (args.img_size_h, args.img_size_w)

    test_dataset = get_augmented_dataset(
        image_dir=args.image_dir, mask_dir=args.mask_dir,
        img_size=img_size_tuple, augment=False, # No augmentation for testing
        mask_type=args.mask_target_type
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=(device.type == 'cuda'))
    logger.info(f"Test dataset loaded: {len(test_dataset)} samples.")

    # 2. Model Loading
    logger.info(f"Loading model architecture and weights from: {args.model_path}")
    # Ensure VIT_CONFIG is appropriate or allow passing a custom one if needed
    current_vit_config = VIT_CONFIG
    if args.vit_config_name != 'ViT-B_16': # Default name in VIT_CONFIG
         logger.warning(f"ViT config '{args.vit_config_name}' specified. Ensure this matches the trained model's ViT config if it wasn't the default.")

    model = TransUNet(
        img_size=img_size_tuple, num_classes=args.num_classes,
        cnn_backbone_type=args.cnn_backbone, vit_config=current_vit_config
        # Other TransUNet params like decoder_channels use defaults, assuming they match trained model.
        # For robustness, model config should ideally be saved with weights during training.
    ).to(device)

    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        logger.info("Model weights loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading model weights: {e}. Ensure model architecture params match the checkpoint.")
        return

    model.eval()

    # 3. Inference Loop and Storing Results
    logger.info("Running inference on the test set...")
    all_preds_cpu = []
    all_targets_cpu = []
    saved_samples_count = 0

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(device, non_blocking=True)
            # Targets are kept on CPU for batch-wise saving, then moved to device for full eval if needed by a metric

            outputs = model(inputs)

            all_preds_cpu.append(outputs.cpu())
            all_targets_cpu.append(targets.cpu()) # targets were already on CPU from dataloader

            if args.save_samples_count > 0 and saved_samples_count < args.save_samples_count:
                num_to_save_this_batch = min(inputs.size(0), args.save_samples_count - saved_samples_count)
                if num_to_save_this_batch > 0:
                    try:
                        fig = display_segmentation_results(
                            images=inputs.cpu()[:num_to_save_this_batch],
                            true_masks=targets.cpu()[:num_to_save_this_batch], # Assuming targets are (B,1,H,W) or (B,H,W)
                            pred_masks=outputs.cpu()[:num_to_save_this_batch], # Pass logits
                            num_samples=num_to_save_this_batch,
                            threshold=args.metrics_threshold,
                            title_suffix=f"Batch{i}"
                        )
                        sample_fig_path = os.path.join(args.output_dir_final, f'sample_results_batch{i}_imgs.png')
                        fig.savefig(sample_fig_path)
                        plt.close(fig)
                        logger.info(f"Saved sample segmentation plot to {sample_fig_path}")
                        saved_samples_count += num_to_save_this_batch
                    except Exception as e:
                        logger.error(f"Error saving sample segmentation plot: {e}")


    if not all_preds_cpu:
        logger.warning("No predictions were made. Check test dataset or model.")
        return

    preds_tensor_all = torch.cat(all_preds_cpu)
    targets_tensor_all = torch.cat(all_targets_cpu)
    logger.info(f"Inference complete. Total samples processed: {len(preds_tensor_all)}")

    # 4. Metrics Calculation
    logger.info("Calculating evaluation metrics...")
    metrics_results = {}

    # Segmentation-specific metrics
    metrics_results['dice_coefficient'] = dice_coefficient(preds_tensor_all, targets_tensor_all, threshold=args.metrics_threshold)
    metrics_results['jaccard_index'] = jaccard_index(preds_tensor_all, targets_tensor_all, threshold=args.metrics_threshold)
    logger.info(f"Dice Coefficient: {metrics_results['dice_coefficient']:.4f}")
    logger.info(f"Jaccard Index (IoU): {metrics_results['jaccard_index']:.4f}")

    # Detailed classification-style metrics (pixel-wise)
    # Determine 'average' mode for multiclass based on num_classes
    avg_mode = 'binary' if args.num_classes == 1 else 'macro'
    detailed_metrics = calculate_classification_metrics(
        preds_tensor_all, targets_tensor_all,
        threshold=args.metrics_threshold,
        average=avg_mode,
        num_classes_for_roc_auc=args.num_classes if args.num_classes > 1 else None
    )
    metrics_results.update(detailed_metrics) # Add all detailed metrics

    for k, v in detailed_metrics.items():
        if k not in ['roc_curve', 'pr_curve', 'confusion_matrix', 'classification_report']:
            logger.info(f"  {k}: {v if isinstance(v, (int, float, str)) else 'Complex data (see JSON/plots)'}")
        elif isinstance(v, dict) and k == 'classification_report': # Pretty print classification report dict
            logger.info(f"  Classification Report:")
            for label, report_item in v.items():
                if isinstance(report_item, dict): # Class-specific report
                    logger.info(f"    Class '{label}': P: {report_item['precision']:.3f}, R: {report_item['recall']:.3f}, F1: {report_item['f1-score']:.3f}, Support: {report_item['support']}")
                else: # Overall averages like accuracy, macro avg, weighted avg
                    logger.info(f"    {label}: {report_item:.3f}")


    # 5. Results Reporting and Saving
    # Save numerical metrics to JSON
    metrics_file_path = os.path.join(args.output_dir_final, 'test_metrics.json')
    # Convert numpy arrays/tensors in metrics_results to lists for JSON serialization
    serializable_metrics = {}
    for k, v in metrics_results.items():
        if isinstance(v, np.ndarray):
            serializable_metrics[k] = v.tolist()
        elif isinstance(v, torch.Tensor):
            serializable_metrics[k] = v.tolist()
        elif isinstance(v, dict) and k not in ['roc_curve', 'pr_curve']: # Already handled for cl_report
             serializable_metrics[k] = v # classification_report is already dict
        elif k in ['roc_curve', 'pr_curve'] and isinstance(v, dict): # ROC/PR curves are dicts of lists
            serializable_metrics[k] = v
        elif isinstance(v, (int, float, str)):
             serializable_metrics[k] = v
        # else: logger.warning(f"Metric '{k}' type {type(v)} not directly serializable to JSON, skipping.")


    with open(metrics_file_path, 'w') as f:
        json.dump(serializable_metrics, f, indent=4)
    logger.info(f"Numerical metrics saved to {metrics_file_path}")

    # Plot and save figures
    if 'confusion_matrix' in metrics_results and isinstance(metrics_results['confusion_matrix'], list): # It was tolist()
        cm_data = np.array(metrics_results['confusion_matrix'])
        class_names_cm = [f'Class {i}' for i in range(args.num_classes)] if args.num_classes > 1 else ['Background', 'Foreground']
        if cm_data.size > 0 : # Ensure CM is not empty
            fig_cm = plot_confusion_matrix(cm_data, class_names_cm)
            cm_path = os.path.join(args.output_dir_final, 'confusion_matrix.png')
            fig_cm.savefig(cm_path); plt.close(fig_cm)
            logger.info(f"Confusion matrix plot saved to {cm_path}")

    if 'roc_curve' in metrics_results and isinstance(metrics_results['roc_curve'], dict) and \
       'fpr' in metrics_results['roc_curve'] and 'tpr' in metrics_results['roc_curve']:
        roc_auc_score = metrics_results.get('roc_auc', "N/A")
        if isinstance(roc_auc_score, float): roc_auc_score = f"{roc_auc_score:.3f}"

        fig_roc = plot_roc_curve(metrics_results['roc_curve']['fpr'], metrics_results['roc_curve']['tpr'], roc_auc_score)
        roc_path = os.path.join(args.output_dir_final, 'roc_curve.png')
        fig_roc.savefig(roc_path); plt.close(fig_roc)
        logger.info(f"ROC curve plot saved to {roc_path}")

    if 'pr_curve' in metrics_results and isinstance(metrics_results['pr_curve'], dict) and \
       'precision' in metrics_results['pr_curve'] and 'recall' in metrics_results['pr_curve']:
        fig_pr = plot_pr_curve(metrics_results['pr_curve']['recall'], metrics_results['pr_curve']['precision'])
        pr_path = os.path.join(args.output_dir_final, 'pr_curve.png')
        fig_pr.savefig(pr_path); plt.close(fig_pr)
        logger.info(f"PR curve plot saved to {pr_path}")

    logger.info("Testing finished.")

if __name__ == '__main__':
    main()
