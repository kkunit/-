import torch
import numpy as np
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, auc, accuracy_score
)
from sklearn.preprocessing import label_binarize
# import matplotlib.pyplot as plt # Not needed for calculations, only for direct plotting

class ModelEvaluator:
    def __init__(self, model, test_loader, device, class_names=None):
        self.model = model
        self.test_loader = test_loader
        self.device = device

        # Try to infer class_names and num_classes if not provided
        if class_names:
            self.class_names = class_names
            self.num_classes = len(class_names)
        elif hasattr(test_loader, 'dataset') and hasattr(test_loader.dataset, 'classes'):
            self.class_names = test_loader.dataset.classes
            self.num_classes = len(self.class_names)
        else:
            # Fallback if class_names cannot be inferred (e.g. test_loader is None or custom dataset)
            # This might happen if evaluate() is called with dummy_metrics before loader is ready
            self.class_names = ["Class_0", "Class_1"] # Default assumption
            self.num_classes = 2
            print("Warning: class_names not provided or inferable, defaulting to basic binary classes for evaluator.")


    def evaluate(self):
        if not self.model or not self.test_loader:
            print("Error: Model or test_loader not provided for evaluation. Returning dummy metrics.")
            return self.dummy_metrics(message="Model or test_loader missing.")

        # Update class_names and num_classes if test_loader is now available and they weren't set properly
        if hasattr(self.test_loader, 'dataset') and hasattr(self.test_loader.dataset, 'classes'):
            if self.class_names != self.test_loader.dataset.classes : #If initial default was used
                 self.class_names = self.test_loader.dataset.classes
                 self.num_classes = len(self.class_names)
                 print(f"Evaluator class names updated to: {self.class_names}")

        self.model.eval()
        all_labels_list = []
        all_preds_list = []
        all_probs_list = [] # For ROC/AUC

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                probabilities = torch.softmax(outputs, dim=1)

                all_labels_list.extend(labels.cpu().numpy())
                all_preds_list.extend(preds.cpu().numpy())
                all_probs_list.extend(probabilities.cpu().numpy())

        all_labels = np.array(all_labels_list)
        all_preds = np.array(all_preds_list)
        all_probs = np.array(all_probs_list) # Shape (n_samples, n_classes)

        if len(all_labels) == 0:
            print("Warning: No samples found in test_loader during evaluation. Returning dummy metrics.")
            return self.dummy_metrics(message="No samples in test_loader.")

        # Ensure num_classes used for binarization matches the actual data from model output
        if all_probs.shape[1] != self.num_classes:
            print(f"Warning: Mismatch between inferred num_classes ({self.num_classes}) and model output probabilities ({all_probs.shape[1]}). Adjusting num_classes.")
            self.num_classes = all_probs.shape[1]
            # Attempt to generate generic class names if they don't match
            if len(self.class_names) != self.num_classes:
                self.class_names = [f"Class_{i}" for i in range(self.num_classes)]


        return self._calculate_metrics(all_labels, all_preds, all_probs)

    def _calculate_metrics(self, y_true, y_pred_class, y_pred_proba):
        metrics = {"class_names": self.class_names}

        # 1. Confusion Matrix
        try:
            metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred_class, labels=list(range(self.num_classes))).tolist()
        except Exception as e:
            print(f"Error calculating confusion matrix: {e}")
            metrics["confusion_matrix"] = [[0]*self.num_classes for _ in range(self.num_classes)]

        # 2. Classification Report (includes precision, recall, F1, support) & Accuracy
        try:
            # Ensure target_names matches the number of classes observed
            target_names_for_report = self.class_names
            if len(self.class_names) != self.num_classes:
                 target_names_for_report = [f"Class_{i}" for i in range(self.num_classes)]

            report_dict = classification_report(
                y_true, y_pred_class,
                labels=list(range(self.num_classes)),
                target_names=target_names_for_report,
                output_dict=True, zero_division=0
            )
            metrics["classification_report"] = report_dict
            metrics["accuracy"] = report_dict.get("accuracy", accuracy_score(y_true, y_pred_class))
        except Exception as e:
            print(f"Error calculating classification report: {e}")
            metrics["classification_report"] = {name: {"precision": 0, "recall": 0, "f1-score": 0, "support": 0} for name in self.class_names}
            metrics["classification_report"]["accuracy"] = accuracy_score(y_true, y_pred_class) if len(y_true)>0 else 0
            metrics["accuracy"] = metrics["classification_report"]["accuracy"]

        # For ROC, PR curves, and their AUCs
        y_true_binarized = label_binarize(y_true, classes=list(range(self.num_classes)))

        # Ensure y_true_binarized has self.num_classes columns, even if some classes are not in y_true
        if y_true_binarized.shape[1] < self.num_classes:
            print(f"Warning: y_true_binarized has {y_true_binarized.shape[1]} classes, expected {self.num_classes}. Padding with zeros.")
            padding = np.zeros((y_true_binarized.shape[0], self.num_classes - y_true_binarized.shape[1]))
            y_true_binarized = np.hstack((y_true_binarized, padding))


        metrics["roc_curve_data_per_class"] = []
        metrics["roc_auc_per_class"] = []
        metrics["pr_curve_data_per_class"] = []
        metrics["pr_auc_per_class"] = []

        # Handle binary case separately for roc_auc_score if y_true is not binarized for it.
        # However, scikit-learn's roc_auc_score with multi_class='ovr' handles binarized y_true well.

        for i in range(self.num_classes):
            class_name = self.class_names[i] if i < len(self.class_names) else f"Class_{i}"

            # ROC
            try:
                if y_true_binarized.shape[1] > i and y_pred_proba.shape[1] > i :
                    fpr, tpr, _ = roc_curve(y_true_binarized[:, i], y_pred_proba[:, i])
                    roc_auc = auc(fpr, tpr) # More direct than roc_auc_score here as we have fpr,tpr
                    # roc_auc = roc_auc_score(y_true_binarized[:, i], y_pred_proba[:, i]) # Alternative
                    metrics["roc_curve_data_per_class"].append({"fpr": fpr.tolist(), "tpr": tpr.tolist(), "class_name": class_name})
                    metrics["roc_auc_per_class"].append(roc_auc)
                else: # Should not happen if num_classes is consistent
                    metrics["roc_curve_data_per_class"].append({"fpr": [], "tpr": [], "class_name": class_name})
                    metrics["roc_auc_per_class"].append(float('nan'))
            except Exception as e_roc:
                print(f"Could not compute ROC for class {class_name}: {e_roc}")
                metrics["roc_curve_data_per_class"].append({"fpr": [], "tpr": [], "class_name": class_name})
                metrics["roc_auc_per_class"].append(float('nan'))

            # PR
            try:
                if y_true_binarized.shape[1] > i and y_pred_proba.shape[1] > i:
                    precision, recall, _ = precision_recall_curve(y_true_binarized[:, i], y_pred_proba[:, i])
                    pr_auc = auc(recall, precision)
                    metrics["pr_curve_data_per_class"].append({"precision": precision.tolist(), "recall": recall.tolist(), "class_name": class_name})
                    metrics["pr_auc_per_class"].append(pr_auc)
                else:
                    metrics["pr_curve_data_per_class"].append({"precision": [], "recall": [], "class_name": class_name})
                    metrics["pr_auc_per_class"].append(float('nan'))
            except Exception as e_pr:
                print(f"Could not compute PR for class {class_name}: {e_pr}")
                metrics["pr_curve_data_per_class"].append({"precision": [], "recall": [], "class_name": class_name})
                metrics["pr_auc_per_class"].append(float('nan'))

        # Macro averages for AUCs
        valid_roc_aucs = [val for val in metrics["roc_auc_per_class"] if not np.isnan(val)]
        metrics["roc_auc_macro"] = np.mean(valid_roc_aucs) if valid_roc_aucs else float('nan')

        valid_pr_aucs = [val for val in metrics["pr_auc_per_class"] if not np.isnan(val)]
        metrics["pr_auc_macro"] = np.mean(valid_pr_aucs) if valid_pr_aucs else float('nan')

        return metrics

    def dummy_metrics(self, message="Using dummy metrics"):
        print(message)
        # Use self.num_classes and self.class_names from __init__ if available
        num_dummy_classes = self.num_classes
        dummy_class_names = self.class_names

        cm = np.random.randint(0, 50, size=(num_dummy_classes, num_dummy_classes)).tolist()

        report = {name: {"precision": np.random.rand(), "recall": np.random.rand(), "f1-score": np.random.rand(), "support": np.random.randint(50,100)} for name in dummy_class_names}
        accuracy = np.random.rand()
        report["accuracy"] = accuracy
        report["macro avg"] = {"precision": np.random.rand(), "recall": np.random.rand(), "f1-score": np.random.rand(), "support": np.random.randint(100,sum(item['support'] for item in report.values() if isinstance(item,dict)) if report else 200)}
        report["weighted avg"] = {"precision": np.random.rand(), "recall": np.random.rand(), "f1-score": np.random.rand(), "support": np.random.randint(100,sum(item['support'] for item in report.values()if isinstance(item,dict)) if report else 200)}


        roc_auc_per_class = [np.random.rand() for _ in range(num_dummy_classes)]
        pr_auc_per_class = [np.random.rand() for _ in range(num_dummy_classes)]

        roc_curve_data_per_class = []
        pr_curve_data_per_class = []
        for i in range(num_dummy_classes):
            class_name = dummy_class_names[i]
            fpr = np.sort(np.concatenate(([0], np.random.rand(8), [1]))).tolist()
            tpr = np.sort(np.concatenate(([0], np.random.rand(8), [1]))).tolist()
            roc_curve_data_per_class.append({"fpr": fpr, "tpr": tpr, "class_name": class_name})

            recall = np.sort(np.concatenate(([0], np.random.rand(8), [1])))[::-1].tolist()
            precision = np.sort(np.concatenate(([1],np.random.rand(8),[0])))[::-1].tolist() # Precision starts high
            pr_curve_data_per_class.append({"precision": precision, "recall": recall, "class_name": class_name})

        return {
            "confusion_matrix": cm,
            "classification_report": report,
            "accuracy": accuracy,
            "roc_auc_per_class": roc_auc_per_class,
            "roc_auc_macro": np.mean(roc_auc_per_class) if roc_auc_per_class else float('nan'),
            "pr_auc_per_class": pr_auc_per_class,
            "pr_auc_macro": np.mean(pr_auc_per_class) if pr_auc_per_class else float('nan'),
            "roc_curve_data_per_class": roc_curve_data_per_class,
            "pr_curve_data_per_class": pr_curve_data_per_class,
            "class_names": dummy_class_names,
            "message": message
        }


    def get_metrics_summary(self, metrics):
        display_class_names = metrics.get("class_names", self.class_names)
        num_classes_disp = len(display_class_names)


        summary = []
        summary.append("--- 性能指标 ---") # Performance Metrics

        acc = metrics.get("accuracy") # Already calculated in _calculate_metrics
        if acc is not None:
            summary.append(f"准确率 (Accuracy): {acc:.4f}")
        else: # Fallback if not directly in metrics dict for some reason
            acc_report = metrics.get("classification_report", {}).get("accuracy")
            if acc_report is not None:
                 summary.append(f"准确率 (Accuracy): {acc_report:.4f}")
            else:
                 summary.append("准确率 (Accuracy): N/A")


        summary.append("\n混淆矩阵 (Confusion Matrix):")
        cm = metrics.get("confusion_matrix")
        if cm:
            # Header for CM
            header_cm = "\t" + "\t".join(display_class_names)
            summary.append(header_cm)
            for i, row in enumerate(cm):
                row_name = display_class_names[i] if i < len(display_class_names) else f"True_{i}"
                summary.append(f"{row_name}\t" + "\t".join(map(str, row)))
        else:
            summary.append("N/A")

        summary.append("\n分类报告 (Classification Report):")
        report = metrics.get("classification_report")
        if report:
            header_cr = f"{'':<15} {'精确率 (Prec)':<10} {'召回率 (Recall)':<10} {'F1值 (F1)':<10} {'样本数 (Support)':<10}"
            summary.append(header_cr)

            for i in range(num_classes_disp):
                class_name = display_class_names[i]
                if class_name in report and isinstance(report[class_name], dict):
                    r = report[class_name]
                    summary.append(f"{class_name:<15} {r.get('precision',0):<10.2f} {r.get('recall',0):<10.2f} {r.get('f1-score',0):<10.2f} {r.get('support',0):<10.0f}")

            for avg_type in ["macro avg", "weighted avg"]:
                if avg_type in report:
                    r = report[avg_type]
                    summary.append(f"{avg_type:<15} {r.get('precision',0):<10.2f} {r.get('recall',0):<10.2f} {r.get('f1-score',0):<10.2f} {r.get('support',0):<10.0f}")
        else:
            summary.append("N/A")

        roc_auc_macro = metrics.get('roc_auc_macro', float('nan'))
        pr_auc_macro = metrics.get('pr_auc_macro', float('nan'))

        summary.append(f"\nAUC (ROC) - Macro平均: {roc_auc_macro:.4f}" if not np.isnan(roc_auc_macro) else "\nAUC (ROC) - Macro平均: N/A")
        summary.append(f"AUC (PR) - Macro平均: {pr_auc_macro:.4f}" if not np.isnan(pr_auc_macro) else "AUC (PR) - Macro平均: N/A")

        if metrics.get("roc_auc_per_class") and display_class_names:
            summary.append("\nAUC (ROC) - 每类:")
            for i, auc_val in enumerate(metrics["roc_auc_per_class"]):
                class_name = display_class_names[i] if i < num_classes_disp else f"Class_{i}"
                summary.append(f"  {class_name}: {auc_val:.4f}" if not np.isnan(auc_val) else f"  {class_name}: N/A")

        if metrics.get("pr_auc_per_class") and display_class_names:
            summary.append("AUC (PR) - 每类:")
            for i, auc_val in enumerate(metrics["pr_auc_per_class"]):
                class_name = display_class_names[i] if i < num_classes_disp else f"Class_{i}"
                summary.append(f"  {class_name}: {auc_val:.4f}" if not np.isnan(auc_val) else f"  {class_name}: N/A")

        if "message" in metrics:
            summary.append(f"\nNote: {metrics['message']}")

        return "\n".join(summary)

    # def plot_roc_curve(self, fpr, tpr, roc_auc_val): # roc_auc_val instead of roc_auc
    #     plt.figure()
    #     plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_val:.2f})')
    #     plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    #     plt.xlim([0.0, 1.0])
    #     plt.ylim([0.0, 1.05])
    #     plt.xlabel('假阳性率 (False Positive Rate)') # Changed to Chinese
    #     plt.ylabel('真阳性率 (True Positive Rate)') # Changed to Chinese
    #     plt.title('ROC曲线 (Receiver Operating Characteristic Curve)') # Changed to Chinese
    #     plt.legend(loc="lower right")
    #     # Instead of plt.show(), save to a buffer or file to display in Tkinter
    #     # For now, just show, but this needs integration with Tkinter canvas
    #     plt.show()


    # def plot_pr_curve(self, recall, precision, pr_auc_val): # pr_auc_val instead of pr_auc
    #     plt.figure()
    #     plt.plot(recall, precision, color='blue', lw=2, label=f'PR曲线 (面积 = {pr_auc_val:.2f})') # Changed to Chinese
    #     plt.xlabel('召回率 (Recall)') # Changed to Chinese
    #     plt.ylabel('精确率 (Precision)') # Changed to Chinese
    #     plt.title('精确率-召回率曲线 (Precision-Recall Curve)') # Changed to Chinese
    #     plt.legend(loc="lower left")
    #     plt.show()
