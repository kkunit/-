# Utility functions
import tkinter as tk
from tkinter import scrolledtext
import numpy as np # Added for np.array in plot_evaluation_metrics

# This is a placeholder. In a real scenario, you might parse the model
# or use a library to visualize, or have a more structured way to get this.
from .resnet_builder import get_resnet_architecture_summary as get_arch_summary_from_builder

def display_network_architecture_in_gui(model_name, text_widget):
    """
    Fetches and displays the network architecture summary in the GUI.
    """
    summary = get_arch_summary_from_builder(model_name)

    text_widget.config(state=tk.NORMAL)
    text_widget.delete(1.0, tk.END)
    text_widget.insert(tk.END, summary)
    text_widget.config(state=tk.DISABLED)

# Placeholder for plotting functions that can be embedded in Tkinter
# For example, using Matplotlib FigureCanvasTkAgg

def plot_training_curves(figure_canvas, history):
    """
    Plots training curves (loss, accuracy) on a Tkinter canvas.
    'history' is a dict like {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    """
    if not history or not history.get('train_loss'): # Check if history is empty or lacks data
        print("Plotting training curves: No history data provided.")
        return

    fig = figure_canvas.figure
    fig.clear() # Clear previous plots

    epochs_range = range(1, len(history['train_loss']) + 1)

    # Plot Loss
    ax1 = fig.add_subplot(121)
    ax1.plot(epochs_range, history['train_loss'], label='训练损失 (Train Loss)')
    if history.get('val_loss') and any(history['val_loss']): # Check if val_loss exists and is not all zeros/placeholders
        ax1.plot(epochs_range, history['val_loss'], label='验证损失 (Validation Loss)')
    ax1.set_title('损失曲线 (Loss Curves)')
    ax1.set_xlabel('轮次 (Epoch)')
    ax1.set_ylabel('损失 (Loss)')
    ax1.legend()
    ax1.grid(True)

    # Plot Accuracy
    ax2 = fig.add_subplot(122)
    ax2.plot(epochs_range, history['train_acc'], label='训练准确率 (Train Accuracy)')
    if history.get('val_acc') and any(history['val_acc']): # Check if val_acc exists and is not all zeros/placeholders
        ax2.plot(epochs_range, history['val_acc'], label='验证准确率 (Validation Accuracy)')
    ax2.set_title('准确率曲线 (Accuracy Curves)')
    ax2.set_xlabel('轮次 (Epoch)')
    ax2.set_ylabel('准确率 (Accuracy)')
    ax2.legend()
    ax2.grid(True)

    fig.tight_layout()
    figure_canvas.draw()
    print("Training curves plotted.")

def plot_evaluation_metrics(figure_canvas, metrics):
    """
    Plots performance metrics like Confusion Matrix, ROC, and PR curves on Tkinter canvas.
    'metrics' is a dict containing 'confusion_matrix', 'roc_curve_data_per_class',
    'pr_curve_data_per_class', 'roc_auc_per_class', 'pr_auc_per_class', 'class_names'.
    """
    if not metrics:
        print("Plotting evaluation metrics: No metrics data provided.")
        return

    fig = figure_canvas.figure
    fig.clear()

    class_names = metrics.get("class_names", [])
    num_classes = len(class_names)

    # 1. Plot Confusion Matrix
    ax1 = fig.add_subplot(131) # 1 row, 3 columns, 1st plot
    cm = metrics.get("confusion_matrix")
    if cm and num_classes > 0 :
        # Using imshow for heatmap effect
        cax = ax1.imshow(cm, interpolation='nearest', cmap='Blues')
        fig.colorbar(cax, ax=ax1)

        ax1.set_title('混淆矩阵 (Confusion Matrix)')
        ax1.set_xlabel('预测标签 (Predicted Label)')
        ax1.set_ylabel('真实标签 (True Label)')

        tick_marks = range(num_classes)
        ax1.set_xticks(tick_marks)
        ax1.set_xticklabels(class_names, rotation=45, ha="right")
        ax1.set_yticks(tick_marks)
        ax1.set_yticklabels(class_names)

        # Add text annotations for numbers in cells
        thresh = (np.array(cm).max() + np.array(cm).min()) / 2. # Threshold for text color
        for i in range(num_classes):
            for j in range(num_classes):
                ax1.text(j, i, format(cm[i][j], 'd'),
                         ha="center", va="center",
                         color="white" if cm[i][j] > thresh else "black")
    else:
        ax1.text(0.5, 0.5, '混淆矩阵数据不可用', ha='center', va='center', fontsize=9) # CM data not available
        ax1.set_title('混淆矩阵 (Confusion Matrix)')
    ax1.grid(False)


    # 2. Plot ROC Curves (per class)
    ax2 = fig.add_subplot(132)
    roc_curves_data = metrics.get("roc_curve_data_per_class", [])
    roc_aucs = metrics.get("roc_auc_per_class", [])
    if roc_curves_data and roc_aucs and len(roc_curves_data) == len(roc_aucs):
        for i, data in enumerate(roc_curves_data):
            fpr = data.get("fpr", [])
            tpr = data.get("tpr", [])
            class_name = data.get("class_name", f"Class {i}")
            roc_auc_val = roc_aucs[i] if i < len(roc_aucs) else float('nan')
            if fpr and tpr: # Ensure data is present
                 ax2.plot(fpr, tpr, lw=2, label=f'{class_name} (AUC = {roc_auc_val:.2f})')
        ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # Diagonal line
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('假阳性率 (FPR)')
        ax2.set_ylabel('真阳性率 (TPR)')
        ax2.set_title('ROC曲线 (ROC Curves)')
        ax2.legend(loc="lower right", fontsize='small')
        ax2.grid(True)
    else:
        ax2.text(0.5, 0.5, 'ROC曲线数据不可用', ha='center', va='center', fontsize=9) # ROC data not available
        ax2.set_title('ROC曲线 (ROC Curves)')


    # 3. Plot PR Curves (per class)
    ax3 = fig.add_subplot(133)
    pr_curves_data = metrics.get("pr_curve_data_per_class", [])
    pr_aucs = metrics.get("pr_auc_per_class", [])
    if pr_curves_data and pr_aucs and len(pr_curves_data) == len(pr_aucs):
        for i, data in enumerate(pr_curves_data):
            recall = data.get("recall", [])
            precision = data.get("precision", [])
            class_name = data.get("class_name", f"Class {i}")
            pr_auc_val = pr_aucs[i] if i < len(pr_aucs) else float('nan')
            if recall and precision: # Ensure data is present
                ax3.plot(recall, precision, lw=2, label=f'{class_name} (AUC = {pr_auc_val:.2f})')
        ax3.set_xlabel('召回率 (Recall)')
        ax3.set_ylabel('精确率 (Precision)')
        ax3.set_title('PR曲线 (PR Curves)')
        ax3.legend(loc="lower left", fontsize='small') # Use lower left or best
        ax3.grid(True)
        ax3.set_ylim([0.0, 1.05]) # Precision can be 1
        ax3.set_xlim([0.0, 1.0])  # Recall is between 0 and 1
    else:
        ax3.text(0.5, 0.5, 'PR曲线数据不可用', ha='center', va='center', fontsize=9) # PR data not available
        ax3.set_title('PR曲线 (PR Curves)')

    try:
        fig.tight_layout(pad=1.5) # Add some padding
    except Exception as e:
        print(f"Error during tight_layout: {e}") # Sometimes raises error with specific plot contents

    figure_canvas.draw()
    print("Evaluation metrics plots updated.")


# def plot_performance_metrics_on_canvas(figure_canvas, metrics): # Old name
#     """
#     Plots performance metrics like ROC and PR curves on Tkinter canvas.
#     'metrics' is a dict containing 'roc_curve_data': (fpr,tpr), 'pr_curve_data': (recall, precision), etc.
#     """
# This function is now effectively plot_evaluation_metrics
#     fig = figure_canvas.figure
#     fig.clear()

#     # Plot ROC Curve
#     ax1 = fig.add_subplot(121)
#     if metrics.get('roc_curve_data') and metrics.get('roc_auc') is not None:
#         fpr, tpr = metrics['roc_curve_data']
#         roc_auc = metrics['roc_auc']
#         ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
#         ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#         ax1.set_xlim([0.0, 1.0])
#         ax1.set_ylim([0.0, 1.05])
#         ax1.set_xlabel('False Positive Rate')
#         ax1.set_ylabel('True Positive Rate')
#         ax1.set_title('ROC Curve')
#         ax1.legend(loc="lower right")
#     else:
#         ax1.text(0.5, 0.5, 'ROC data not available', ha='center', va='center')


#     # Plot PR Curve
#     ax2 = fig.add_subplot(122)
#     if metrics.get('pr_curve_data') and metrics.get('pr_auc') is not None:
#         recall, precision = metrics['pr_curve_data']
#         pr_auc = metrics['pr_auc']
#         ax2.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
#         ax2.set_xlabel('Recall')
#         ax2.set_ylabel('Precision')
#         ax2.set_title('Precision-Recall Curve')
#         ax2.legend(loc="lower left")
#     else:
#         ax2.text(0.5, 0.5, 'PR data not available', ha='center', va='center')

#     fig.tight_layout()
#     figure_canvas.draw()
