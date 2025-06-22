# Utility functions
import tkinter as tk
from tkinter import scrolledtext

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

# def plot_curves_on_canvas(figure_canvas, history):
#     """
#     Plots training curves (loss, accuracy) on a Tkinter canvas.
#     'history' is a dict like {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
#     """
#     if not history or not history.get('train_loss'): # Check if history is empty or lacks data
#         return

#     fig = figure_canvas.figure
#     fig.clear() # Clear previous plots

#     ax1 = fig.add_subplot(121)
#     ax1.plot(history['train_loss'], label='Train Loss')
#     if history.get('val_loss'):
#         ax1.plot(history['val_loss'], label='Validation Loss')
#     ax1.set_title('Loss Curves')
#     ax1.set_xlabel('Epoch')
#     ax1.set_ylabel('Loss')
#     ax1.legend()

#     ax2 = fig.add_subplot(122)
#     ax2.plot(history['train_acc'], label='Train Accuracy')
#     if history.get('val_acc'):
#         ax2.plot(history['val_acc'], label='Validation Accuracy')
#     ax2.set_title('Accuracy Curves')
#     ax2.set_xlabel('Epoch')
#     ax2.set_ylabel('Accuracy')
#     ax2.legend()

#     fig.tight_layout()
#     figure_canvas.draw()

# def plot_performance_metrics_on_canvas(figure_canvas, metrics):
#     """
#     Plots performance metrics like ROC and PR curves on Tkinter canvas.
#     'metrics' is a dict containing 'roc_curve_data': (fpr,tpr), 'pr_curve_data': (recall, precision), etc.
#     """
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
