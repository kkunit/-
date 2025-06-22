import torch
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve, classification_report

# Helper to process inputs/targets for metric calculation
def _prepare_input_targets(preds, targets, threshold=0.5, num_classes=None):
    """
    Prepares predictions and targets for metric calculation.
    - Applies sigmoid/softmax to predictions.
    - Converts predictions to binary/class labels.
    - Flattens spatial dimensions.
    - Handles one-hot encoding for targets if necessary for some metrics.
    """
    if preds.ndim == 4: # B, C, H, W
        if preds.shape[1] == 1: # Binary segmentation
            probs = torch.sigmoid(preds)
            # For binary, predicted labels are based on threshold
            predicted_labels = (probs > threshold).byte()
        else: # Multi-class segmentation
            probs = torch.softmax(preds, dim=1)
            # Predicted labels are argmax over class dimension
            predicted_labels = torch.argmax(probs, dim=1) # Shape: B, H, W
            if num_classes is None:
                num_classes = preds.shape[1]
    elif preds.ndim == 2: # B, NumClasses (already processed for classification)
        if preds.shape[1] == 1: # Binary classification (single logit output)
            probs = torch.sigmoid(preds)
            predicted_labels = (probs > threshold).byte()
        else: # Multi-class classification
            probs = torch.softmax(preds, dim=1)
            predicted_labels = torch.argmax(probs, dim=1)
            if num_classes is None:
                num_classes = preds.shape[1]
    else:
        raise ValueError(f"Unsupported prediction shape: {preds.shape}")

    # Flatten spatial dimensions if they exist
    if predicted_labels.ndim == 3: # B, H, W (from multi-class segmentation)
        predicted_labels_flat = predicted_labels.view(-1)
    elif predicted_labels.ndim == 4: # B, 1, H, W (from binary segmentation)
        predicted_labels_flat = predicted_labels.view(-1)
    else: # B or B, C (from classification or already flat)
        predicted_labels_flat = predicted_labels.view(-1)

    # Prepare targets
    # Targets for segmentation are often B, H, W (multi-class) or B, 1, H, W (binary)
    # Targets for classification are often B
    targets_flat = targets.view(-1)

    # Ensure targets are also binary (0/1) for binary case if they were float
    if preds.shape[1] == 1 and targets.dtype == torch.float:
        targets_flat = (targets_flat > 0.5).byte() # Assuming targets are 0 or 1

    return probs, predicted_labels_flat.cpu().numpy(), targets_flat.cpu().numpy()


# --- Pixel/Voxel Level Metrics for Segmentation ---
def dice_coefficient(preds, targets, smooth=1e-6, threshold=0.5):
    """Calculates Dice Coefficient for binary or multi-class segmentation.
    preds: Raw logits (B, C, H, W)
    targets: Ground truth (B, H, W) for multi-class or (B, 1, H, W) for binary {0,1}
    """
    num_classes = preds.shape[1]

    if num_classes == 1: # Binary
        probs = torch.sigmoid(preds)
        predicted = (probs > threshold).byte() # B, 1, H, W
        targets = targets.byte()

        intersection = (predicted & targets).sum(dim=(0, 2, 3)) # Sum over B, H, W, keep C=1
        total_sum = predicted.sum(dim=(0, 2, 3)) + targets.sum(dim=(0, 2, 3))
        dice = (2. * intersection + smooth) / (total_sum + smooth)
        return dice.mean().item() # Mean over classes (only 1 class here)
    else: # Multi-class
        probs = torch.softmax(preds, dim=1) # B, C, H, W
        predicted_labels = torch.argmax(probs, dim=1) # B, H, W

        dice_scores = []
        for cls_idx in range(num_classes):
            pred_cls = (predicted_labels == cls_idx).byte() # B, H, W
            target_cls = (targets == cls_idx).byte()      # B, H, W

            intersection = (pred_cls & target_cls).sum()
            total_sum = pred_cls.sum() + target_cls.sum()
            dice = (2. * intersection + smooth) / (total_sum + smooth)
            dice_scores.append(dice.item())
        return np.mean(dice_scores) # Mean Dice over all classes

def jaccard_index(preds, targets, smooth=1e-6, threshold=0.5): # Also IoU
    """Calculates Jaccard Index (IoU) for binary or multi-class segmentation."""
    num_classes = preds.shape[1]

    if num_classes == 1: # Binary
        probs = torch.sigmoid(preds)
        predicted = (probs > threshold).byte()
        targets = targets.byte()

        intersection = (predicted & targets).sum(dim=(0, 2, 3))
        union = (predicted | targets).sum(dim=(0, 2, 3))
        iou = (intersection + smooth) / (union + smooth)
        return iou.mean().item()
    else: # Multi-class
        probs = torch.softmax(preds, dim=1)
        predicted_labels = torch.argmax(probs, dim=1)

        iou_scores = []
        for cls_idx in range(num_classes):
            pred_cls = (predicted_labels == cls_idx).byte()
            target_cls = (targets == cls_idx).byte()

            intersection = (pred_cls & target_cls).sum()
            union = (pred_cls | target_cls).sum()
            iou = (intersection + smooth) / (union + smooth)
            iou_scores.append(iou.item())
        return np.mean(iou_scores)

# --- Classification Metrics (can be adapted for segmentation by flattening) ---
def calculate_classification_metrics(preds, targets, threshold=0.5, average='binary', num_classes_for_roc_auc=None):
    """
    Calculates a suite of classification metrics.
    preds: Raw logits. For segmentation (B,C,H,W), for classification (B,C).
    targets: Ground truth. For segmentation (B,H,W) or (B,1,H,W), for classification (B).
    average: For precision, recall, f1. 'binary' for binary case.
             'micro', 'macro', 'weighted' for multi-class.
    num_classes_for_roc_auc: specify number of classes for roc_auc_score in multi-class 'ovr'
    """

    # Determine if it's segmentation (4D preds) or classification (2D preds)
    is_segmentation = preds.ndim == 4
    num_pred_outputs = preds.shape[1] # Number of channels or classes

    # Prepare for metric calculation
    # For segmentation, probs will be (B,C,H,W) or (B,1,H,W)
    # For classification, probs will be (B,C) or (B,1)
    # y_true and y_pred are flattened numpy arrays of labels
    probs, y_pred_labels, y_true_labels = _prepare_input_targets(preds, targets, threshold, num_classes=num_pred_outputs)

    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true_labels, y_pred_labels)

    # Precision, Recall, F1
    # For multi-class segmentation, 'macro' or 'weighted' average is common for pixel-wise metrics
    # If it's binary segmentation (num_pred_outputs == 1), use average='binary'
    clf_average_mode = average
    if num_pred_outputs == 1 and is_segmentation: # Binary segmentation
        clf_average_mode = 'binary'
    elif num_pred_outputs > 1 and is_segmentation: # Multi-class segmentation
        # For pixel-wise classification metrics, 'macro' or 'weighted' are common
        if average == 'binary': # Default for binary might not be suitable for multi-class seg
            clf_average_mode = 'macro'

    # Handle zero_division for precision, recall, f1
    # Can be 0, 1, or 'warn'. Setting to 0 returns 0.0 if all predictions/targets for a class are zero.
    zero_division_behavior = 0

    metrics['precision'] = precision_score(y_true_labels, y_pred_labels, average=clf_average_mode, zero_division=zero_division_behavior)
    metrics['recall'] = recall_score(y_true_labels, y_pred_labels, average=clf_average_mode, zero_division=zero_division_behavior)
    metrics['f1_score'] = f1_score(y_true_labels, y_pred_labels, average=clf_average_mode, zero_division=zero_division_behavior)

    # Confusion Matrix
    # For segmentation, this will be a large matrix if not careful (num_pixels x num_pixels)
    # Usually, for segmentation, confusion matrix is per-class (e.g., 2x2 for binary)
    # The y_true_labels and y_pred_labels are already class indices (0, 1, ... C-1)
    # So, labels for confusion_matrix should be list(range(num_classes))
    labels_for_cm = None
    if num_pred_outputs > 1 : # Multi-class
        labels_for_cm = list(range(num_pred_outputs))
    # For binary, it's implicitly [0,1] if data contains both.
    # Explicitly setting labels=[0,1] can be safer if one class is missing in a batch.
    elif num_pred_outputs == 1:
        labels_for_cm = [0,1]

    try:
        # Ensure y_true_labels and y_pred_labels are not empty and have valid values for CM
        if len(y_true_labels) > 0 and len(y_pred_labels) > 0:
             # Check if all labels in y_true/y_pred are within the expected range for labels_for_cm
            if labels_for_cm:
                valid_true = all(l in labels_for_cm for l in np.unique(y_true_labels))
                valid_pred = all(l in labels_for_cm for l in np.unique(y_pred_labels))
                if not (valid_true and valid_pred) and len(np.unique(y_true_labels))==1 and len(np.unique(y_pred_labels))==1 and y_true_labels[0]==y_pred_labels[0]:
                     # Handle case where only one class is present and correctly predicted (e.g. all background)
                     # CM would be [[N,0],[0,0]] or [[0,0],[0,N]]. sklearn might struggle.
                     # For now, we'll let it try, or skip if it's problematic.
                     pass


            metrics['confusion_matrix'] = confusion_matrix(y_true_labels, y_pred_labels, labels=labels_for_cm).tolist()
        else:
            metrics['confusion_matrix'] = "Not enough samples for CM"
    except ValueError as e:
        metrics['confusion_matrix'] = f"Error calculating CM: {e}"


    # ROC AUC, PR Curve, ROC Curve points
    # These require probability scores, not just labels.
    # For binary case (num_pred_outputs == 1):
    if num_pred_outputs == 1:
        # probs_flat should be (N_samples,) for roc_auc_score and curve functions
        probs_flat = probs.view(-1).cpu().numpy()
        try:
            if len(np.unique(y_true_labels)) > 1 : # Needs at least two classes in y_true for ROC AUC
                metrics['roc_auc'] = roc_auc_score(y_true_labels, probs_flat)
                fpr, tpr, _ = roc_curve(y_true_labels, probs_flat)
                metrics['roc_curve'] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist()}

                precision_vals, recall_vals, _ = precision_recall_curve(y_true_labels, probs_flat)
                metrics['pr_curve'] = {'precision': precision_vals.tolist(), 'recall': recall_vals.tolist()}
            else:
                metrics['roc_auc'] = "Not enough classes in y_true for ROC AUC"
                metrics['roc_curve'] = "Not enough classes for ROC Curve"
                metrics['pr_curve'] = "Not enough classes for PR Curve"
        except ValueError as e:
            metrics['roc_auc'] = f"Error: {e}"
            metrics['roc_curve'] = f"Error: {e}"
            metrics['pr_curve'] = f"Error: {e}"


    # For multi-class case:
    elif num_pred_outputs > 1:
        # probs here is (B,C,H,W) for seg or (B,C) for clf
        # Need to reshape to (N_samples, N_classes) for roc_auc_score multi-class
        if is_segmentation:
            probs_mc = probs.permute(0, 2, 3, 1).contiguous().view(-1, num_pred_outputs).cpu().numpy() # (B*H*W, C)
        else: # Classification
            probs_mc = probs.cpu().numpy() # (B, C)

        # Targets for multi-class ROC AUC should be labels, not one-hot. y_true_labels is already this.
        try:
            # Check if y_true_labels contains multiple classes needed for 'ovr'/'ovo'
            if len(np.unique(y_true_labels)) > 1:
                 # 'ovo' (one-vs-one) or 'ovr' (one-vs-rest)
                roc_auc_multi_strategy = 'ovr'
                # num_classes_for_roc_auc should be num_pred_outputs
                metrics['roc_auc'] = roc_auc_score(y_true_labels, probs_mc, multi_class=roc_auc_multi_strategy, average='macro', labels=labels_for_cm)
                # ROC and PR curves for multi-class are typically plotted per class or by micro/macro averaging.
                # sklearn's roc_curve and precision_recall_curve are for binary.
                # To get per-class curves: iterate, treat one class as positive, rest as negative.
                # This can be added later if needed. For now, just the AUC score.
                metrics['roc_curve'] = "Multi-class ROC curve (per-class) not implemented yet."
                metrics['pr_curve'] = "Multi-class PR curve (per-class) not implemented yet."

            else:
                metrics['roc_auc'] = "Not enough classes in y_true for multi-class ROC AUC"
                metrics['roc_curve'] = "Not enough classes for ROC Curve"
                metrics['pr_curve'] = "Not enough classes for PR Curve"

        except ValueError as e:
            metrics['roc_auc'] = f"Error: {e}"
            metrics['roc_curve'] = f"Error: {e}" # Placeholder
            metrics['pr_curve'] = f"Error: {e}"  # Placeholder

    # Classification Report (provides precision, recall, f1 per class)
    try:
        if len(y_true_labels) > 0 and len(y_pred_labels) > 0:
            # Ensure target_names match the labels if provided
            target_names_clf_report = [f"Class {i}" for i in labels_for_cm] if labels_for_cm else None
            metrics['classification_report'] = classification_report(y_true_labels, y_pred_labels, labels=labels_for_cm, target_names=target_names_clf_report, output_dict=True, zero_division=zero_division_behavior)
        else:
            metrics['classification_report'] = "Not enough samples for classification report"
    except ValueError as e:
        metrics['classification_report'] = f"Error: {e}"

    return metrics

# Hausdorff Distance (often used in medical image segmentation)
# Requires scipy, which might not be a default dependency.
try:
    from scipy.spatial.distance import directed_hausdorff

    def hausdorff_distance(preds_coords, targets_coords):
        """
        Calculates the Hausdorff Distance between two sets of points.
        preds_coords: Numpy array of shape (N, D) for N points in D dimensions (predicted boundary points).
        targets_coords: Numpy array of shape (M, D) for M points in D dimensions (true boundary points).
        Returns: max(h(A,B), h(B,A))
        This is a placeholder. Actual extraction of boundary points from segmentation masks is needed.
        For segmentation masks, one would typically find contours first.
        """
        if preds_coords.shape[0] == 0 or targets_coords.shape[0] == 0:
            return np.inf # Or handle as appropriate, e.g. if one set is empty
        if preds_coords.ndim != 2 or targets_coords.ndim != 2 or preds_coords.shape[1]!=targets_coords.shape[1]:
            raise ValueError("Inputs must be 2D arrays of coordinates with same number of dimensions.")

        hd1 = directed_hausdorff(preds_coords, targets_coords)[0]
        hd2 = directed_hausdorff(targets_coords, preds_coords)[0]
        return max(hd1, hd2)

except ImportError:
    def hausdorff_distance(preds_coords, targets_coords):
        print("SciPy not installed. Hausdorff distance calculation is unavailable.")
        return -1 # Or some other indicator

# --- Example Usage ---
if __name__ == '__main__':
    # --- Binary Segmentation Example ---
    print("--- Binary Segmentation Metrics ---")
    dummy_preds_seg_bin = torch.randn(2, 1, 64, 64) # B, C=1, H, W (logits)
    dummy_targets_seg_bin = torch.randint(0, 2, (2, 1, 64, 64)).float() # B, C=1, H, W {0,1}

    dice_val = dice_coefficient(dummy_preds_seg_bin, dummy_targets_seg_bin)
    print(f"Dice Coefficient (binary seg): {dice_val:.4f}")

    iou_val = jaccard_index(dummy_preds_seg_bin, dummy_targets_seg_bin)
    print(f"Jaccard/IoU (binary seg): {iou_val:.4f}")

    # Using calculate_classification_metrics for pixel-wise binary classification
    # Here, 'average' should be 'binary'
    print("\nClassification metrics for binary segmentation (pixel-wise):")
    clf_metrics_seg_bin = calculate_classification_metrics(dummy_preds_seg_bin, dummy_targets_seg_bin, average='binary')
    for k, v in clf_metrics_seg_bin.items():
        if k not in ['roc_curve', 'pr_curve', 'confusion_matrix', 'classification_report']:
            print(f"  {k}: {v}")
        elif k in ['confusion_matrix', 'classification_report'] and isinstance(v, str) and "Error" in v:
             print(f"  {k}: {v}") # Print error string
        elif k == 'confusion_matrix':
             print(f"  {k}: \n{np.array(v)}")
        # else: print(f"  {k}: (data for curve/report)")


    # --- Multi-class Segmentation Example ---
    print("\n--- Multi-class Segmentation Metrics ---")
    num_classes_mc = 3
    dummy_preds_seg_mc = torch.randn(2, num_classes_mc, 32, 32) # B, C=3, H, W
    dummy_targets_seg_mc = torch.randint(0, num_classes_mc, (2, 32, 32)).long() # B, H, W (indices)

    dice_val_mc = dice_coefficient(dummy_preds_seg_mc, dummy_targets_seg_mc)
    print(f"Dice Coefficient (multi-class seg, mean over classes): {dice_val_mc:.4f}")

    iou_val_mc = jaccard_index(dummy_preds_seg_mc, dummy_targets_seg_mc)
    print(f"Jaccard/IoU (multi-class seg, mean over classes): {iou_val_mc:.4f}")

    print("\nClassification metrics for multi-class segmentation (pixel-wise):")
    # For multi-class, average can be 'macro', 'micro', or 'weighted'
    clf_metrics_seg_mc = calculate_classification_metrics(dummy_preds_seg_mc, dummy_targets_seg_mc, average='macro', num_classes_for_roc_auc=num_classes_mc)
    for k, v in clf_metrics_seg_mc.items():
        if k not in ['roc_curve', 'pr_curve', 'confusion_matrix', 'classification_report']:
            print(f"  {k}: {v}")
        elif k in ['confusion_matrix', 'classification_report'] and isinstance(v, str) and "Error" in v:
             print(f"  {k}: {v}")
        elif k == 'confusion_matrix':
             print(f"  {k}: \n{np.array(v)}")
        # else: print(f"  {k}: (data for curve/report)")

    # --- Hausdorff Distance Example (conceptual) ---
    # This requires extracting contour points from masks first.
    # For demonstration, using random points.
    print("\n--- Hausdorff Distance (Conceptual) ---")
    try:
        # Dummy contour points (replace with actual contour extraction)
        points_pred = np.random.rand(50, 2) # 50 points in 2D
        points_true = np.random.rand(60, 2) # 60 points in 2D
        hd = hausdorff_distance(points_pred, points_true)
        print(f"Hausdorff Distance (dummy points): {hd:.4f}")

        # Test empty case
        empty_points = np.array([]).reshape(0,2)
        hd_empty = hausdorff_distance(points_pred, empty_points)
        print(f"Hausdorff Distance (one empty set): {hd_empty}")


    except NameError: # If scipy not installed, hausdorff_distance might not be defined
        print("Hausdorff distance test skipped (scipy likely not installed).")
    except Exception as e:
        print(f"Error in Hausdorff test: {e}")


    print("\nBasic eval_utils.py implemented and tested.")
    # Note: For segmentation, many of these "classification" metrics are applied pixel-wise.
    # The GUI will need to present these results clearly.
    # ROC/PR curves for multi-class segmentation are often done per-class.
    # The current implementation of ROC/PR for multi-class is basic (AUC score only).
    pass
