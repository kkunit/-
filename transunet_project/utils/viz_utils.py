import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_training_curves(history, metrics=['loss', 'accuracy', 'dice', 'iou']):
    """
    Plots training and validation loss and other specified metrics.
    history: A dictionary like {'train_loss': [...], 'val_loss': [...],
                               'train_acc': [...], 'val_acc': [...], ...}
    metrics: List of metric prefixes to plot (e.g., 'loss', 'accuracy').
             Will look for 'train_{metric}' and 'val_{metric}' in history.
    """
    num_metrics = len(metrics)
    if num_metrics == 0:
        print("No metrics specified for plotting.")
        return

    fig, axes = plt.subplots(1, num_metrics, figsize=(5 * num_metrics, 4))
    if num_metrics == 1: # Make axes iterable if only one metric
        axes = [axes]

    for i, metric_name in enumerate(metrics):
        train_metric_key = f'train_{metric_name}'
        val_metric_key = f'val_{metric_name}'

        ax = axes[i]
        epochs = range(1, len(history.get(train_metric_key, [])) + 1)

        if train_metric_key in history and len(history[train_metric_key]) > 0:
            ax.plot(epochs, history[train_metric_key], 'bo-', label=f'Training {metric_name}')
        if val_metric_key in history and len(history[val_metric_key]) > 0:
            ax.plot(epochs, history[val_metric_key], 'ro-', label=f'Validation {metric_name}')

        ax.set_title(f'Training and Validation {metric_name.capitalize()}')
        ax.set_xlabel('Epochs')
        ax.set_ylabel(metric_name.capitalize())
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    # Instead of plt.show(), we might want to return the figure object
    # for embedding in a Tkinter/PyQt GUI.
    # For now, let's assume it will be saved or shown directly.
    # plt.show()
    return fig


def display_segmentation_results(images, true_masks, pred_masks, num_samples=3, threshold=0.5, title_suffix=""):
    """
    Displays a few sample images, their true masks, and predicted masks.
    images: Batch of images (B, C, H, W), PyTorch tensor, normalized.
    true_masks: Batch of true masks (B, 1 or C, H, W), PyTorch tensor.
    pred_masks: Batch of predicted masks (B, 1 or C, H, W), PyTorch tensor (logits or probabilities).
    num_samples: Number of samples to display from the batch.
    threshold: Threshold for converting probabilities to binary mask (for binary segmentation).
    title_suffix: Optional suffix for plot titles.
    """
    if images.ndim != 4 or true_masks.ndim not in [3,4] or pred_masks.ndim != 4:
        print("Invalid input dimensions for display_segmentation_results.")
        print(f"Images: {images.shape}, True Masks: {true_masks.shape}, Pred Masks: {pred_masks.shape}")
        return

    num_samples = min(num_samples, images.shape[0])
    if num_samples == 0:
        print("No samples to display.")
        return

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    if num_samples == 1: # Make axes a 2D array like for consistency
        axes = np.array([axes])

    # Inverse normalization for display (assuming ImageNet stats)
    inv_normalize = plt.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                                  std=[1/0.229, 1/0.224, 1/0.225])

    for i in range(num_samples):
        img = images[i].cpu().permute(1, 2, 0).numpy() # C,H,W -> H,W,C
        # img = inv_normalize(img) # This needs careful application if using torchvision.transforms.Normalize
        # A simpler way if normalized: clip to [0,1] if necessary after reversing scaling
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean # Unnormalize
        img = np.clip(img, 0, 1)


        # True Mask Processing
        tm = true_masks[i].cpu()
        if tm.ndim == 3 and tm.shape[0] > 1 : # Multi-class mask (C, H, W), show argmax
            tm = torch.argmax(tm, dim=0) # H, W
        elif tm.ndim == 3 and tm.shape[0] == 1: # Binary mask (1, H, W)
            tm = tm.squeeze(0) # H, W
        elif tm.ndim == 2: # Already (H,W)
            pass
        tm = tm.numpy()

        # Predicted Mask Processing
        pm = pred_masks[i].cpu()
        if pm.shape[0] == 1: # Binary segmentation (1, H, W)
            pm_probs = torch.sigmoid(pm)
            pm_labels = (pm_probs > threshold).byte().squeeze(0) # H, W
        else: # Multi-class segmentation (C, H, W)
            pm_probs = torch.softmax(pm, dim=0)
            pm_labels = torch.argmax(pm_probs, dim=0) # H, W
        pm_labels = pm_labels.numpy()

        # Plotting
        ax_img = axes[i, 0]
        ax_img.imshow(img)
        ax_img.set_title(f"Image {i+1} {title_suffix}")
        ax_img.axis('off')

        ax_true_mask = axes[i, 1]
        ax_true_mask.imshow(tm, cmap='gray') # Or a specific cmap for multi-class
        ax_true_mask.set_title(f"True Mask {i+1} {title_suffix}")
        ax_true_mask.axis('off')

        ax_pred_mask = axes[i, 2]
        ax_pred_mask.imshow(pm_labels, cmap='gray') # Or a specific cmap for multi-class
        ax_pred_mask.set_title(f"Predicted Mask {i+1} {title_suffix}")
        ax_pred_mask.axis('off')

    plt.tight_layout()
    # plt.show()
    return fig

def save_single_segmentation_sample(image_tensor, true_mask_tensor, pred_mask_tensor,
                                   output_dir, sample_idx, threshold=0.5,
                                   mean=np.array([0.485, 0.456, 0.406]),
                                   std=np.array([0.229, 0.224, 0.225])):
    """
    Saves individual components (original image, true mask, predicted mask) of a single segmentation sample.
    image_tensor: Single image (C, H, W), PyTorch tensor, normalized.
    true_mask_tensor: Single true mask (1 or C, H, W), PyTorch tensor.
    pred_mask_tensor: Single predicted mask (1 or C, H, W), PyTorch tensor (logits or probabilities).
    output_dir: Directory to save the images.
    sample_idx: Index of the sample (for filename).
    threshold: Threshold for binarizing predicted mask.
    mean, std: For unnormalizing the original image.
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- Process Original Image ---
    img = image_tensor.cpu().permute(1, 2, 0).numpy() # C,H,W -> H,W,C
    img = std * img + mean # Unnormalize
    img = np.clip(img, 0, 1)
    plt.imsave(os.path.join(output_dir, f"sample_{sample_idx}_orig.png"), img)

    # --- Process True Mask ---
    tm = true_mask_tensor.cpu()
    if tm.ndim == 3 and tm.shape[0] > 1 : # Multi-class mask (C, H, W), show argmax
        tm = torch.argmax(tm, dim=0)
    elif tm.ndim == 3 and tm.shape[0] == 1: # Binary mask (1, H, W)
        tm = tm.squeeze(0)
    # if tm is already (H,W), it's fine
    tm_np = tm.numpy()
    plt.imsave(os.path.join(output_dir, f"sample_{sample_idx}_gt.png"), tm_np, cmap='gray')

    # --- Process Predicted Mask ---
    pm = pred_mask_tensor.cpu()
    if pm.shape[0] == 1: # Binary segmentation (1, H, W) - logits assumed
        pm_probs = torch.sigmoid(pm)
        pm_labels = (pm_probs > threshold).byte().squeeze(0)
    else: # Multi-class segmentation (C, H, W) - logits assumed
        pm_probs = torch.softmax(pm, dim=0)
        pm_labels = torch.argmax(pm_probs, dim=0)
    pm_labels_np = pm_labels.numpy()
    plt.imsave(os.path.join(output_dir, f"sample_{sample_idx}_pred.png"), pm_labels_np, cmap='gray')


def plot_roc_curve(fpr, tpr, auc_score, title='ROC Curve'):
    """Plots a single ROC curve."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc_score:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True)
    plt.tight_layout()
    return fig

def plot_pr_curve(recall, precision, title='Precision-Recall Curve'):
    """Plots a single Precision-Recall curve."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    # The PR curve from sklearn might have precision for recall=0 at the end, handle this.
    ax.plot(recall, precision, color='blue', lw=2, label='PR curve')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.0])
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True)
    plt.tight_layout()
    return fig

def plot_confusion_matrix(cm, class_names, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    Plots a confusion matrix.
    cm: Confusion matrix (numpy array).
    class_names: List of class names for labels.
    """
    fig, ax = plt.subplots(1, 1, figsize=(len(class_names)*0.8 + 2, len(class_names)*0.8 + 1)) # Adjust size
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names)

    # Loop over data dimensions and create text annotations.
    fmt = 'd' # Integer format
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    plt.tight_layout()
    return fig


if __name__ == '__main__':
    # --- Test plot_training_curves ---
    print("Testing plot_training_curves...")
    dummy_history = {
        'train_loss': [0.5, 0.3, 0.2, 0.15],
        'val_loss': [0.6, 0.4, 0.3, 0.25],
        'train_dice': [0.7, 0.8, 0.85, 0.9],
        'val_dice': [0.65, 0.75, 0.8, 0.82]
    }
    fig_train = plot_training_curves(dummy_history, metrics=['loss', 'dice'])
    # In a script, you might do: fig_train.savefig("training_curves.png"); plt.close(fig_train)
    print("Training curve plot generated (not shown in non-interactive).")
    plt.close(fig_train) # Close to free memory

    # --- Test display_segmentation_results ---
    print("\nTesting display_segmentation_results...")
    # Create dummy data (normalized image, binary mask, binary prediction logits)
    dummy_images = torch.rand(3, 3, 64, 64) * 0.2 + 0.4 # Approx normalized range
    dummy_true_masks = torch.randint(0, 2, (3, 1, 64, 64)).float()
    dummy_pred_logits = torch.randn(3, 1, 64, 64) # Logits

    fig_seg = display_segmentation_results(dummy_images, dummy_true_masks, dummy_pred_logits, num_samples=2)
    print("Segmentation results plot generated.")
    plt.close(fig_seg)

    # Multi-class example for display
    num_c = 3
    dummy_true_masks_mc = torch.randint(0, num_c, (3, 64, 64)).long() # B, H, W (indices)
    # Convert to one-hot like for display function if it expects (B,C,H,W) or (B,1,H,W)
    # Or modify display function to handle (B,H,W) integer masks directly.
    # For now, let's assume true_masks can be (B,H,W) for multi-class.
    dummy_pred_logits_mc = torch.randn(3, num_c, 64, 64)

    # The display_segmentation_results needs true_masks as (B,C,H,W) if C>1 for argmax logic, or (B,H,W)
    # Let's test with (B,H,W) true masks for multi-class:
    # Need to adjust the display_segmentation_results to handle this if tm.ndim == 2 for true masks.
    # Current code: if tm.ndim == 3 and tm.shape[0] > 1 : # Multi-class mask (C, H, W), show argmax
    # This means it expects one-hot like true masks if C > 1.
    # Let's make a one-hot version for testing this path:
    dummy_true_masks_mc_onehot = torch.nn.functional.one_hot(dummy_true_masks_mc, num_classes=num_c).permute(0,3,1,2).float()

    fig_seg_mc = display_segmentation_results(dummy_images, dummy_true_masks_mc_onehot, dummy_pred_logits_mc, num_samples=2, title_suffix="MC")
    print("Multi-class segmentation results plot generated.")
    plt.close(fig_seg_mc)


    # --- Test plot_roc_curve ---
    print("\nTesting plot_roc_curve...")
    dummy_fpr = np.array([0, 0.1, 0.2, 0.5, 1])
    dummy_tpr = np.array([0, 0.5, 0.7, 0.9, 1])
    dummy_auc = 0.85
    fig_roc = plot_roc_curve(dummy_fpr, dummy_tpr, dummy_auc)
    print("ROC curve plot generated.")
    plt.close(fig_roc)

    # --- Test plot_pr_curve ---
    print("\nTesting plot_pr_curve...")
    dummy_recall = np.array([0, 0.2, 0.5, 0.8, 1])
    dummy_precision = np.array([1, 0.9, 0.8, 0.7, 0.6])
    fig_pr = plot_pr_curve(dummy_recall, dummy_precision)
    print("PR curve plot generated.")
    plt.close(fig_pr)

    # --- Test plot_confusion_matrix ---
    print("\nTesting plot_confusion_matrix...")
    dummy_cm = np.array([[10, 2], [3, 15]])
    class_names = ['Class A', 'Class B']
    fig_cm = plot_confusion_matrix(dummy_cm, class_names)
    print("Confusion matrix plot generated.")
    plt.close(fig_cm)

    print("\nBasic viz_utils.py implemented and tested.")
    # These functions return matplotlib figures, which can then be saved to files
    # or embedded into a GUI canvas (e.g., Tkinter FigureCanvasTkAgg).

    # Test save_single_segmentation_sample
    print("\nTesting save_single_segmentation_sample...")
    if not os.path.exists("temp_viz_output"):
        os.makedirs("temp_viz_output")

    save_single_segmentation_sample(
        dummy_images[0],
        dummy_true_masks[0],
        dummy_pred_logits[0],
        output_dir="temp_viz_output",
        sample_idx=0
    )
    print("Saved single sample components to temp_viz_output (orig, gt, pred).")
    # import shutil; shutil.rmtree("temp_viz_output") # Optional cleanup
    pass
