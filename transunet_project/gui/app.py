import sys
import os
import subprocess # For running training/testing scripts
import threading # For running tasks in background
import queue # For communication between threads and GUI
import json

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QLabel, QLineEdit, QPushButton, QFileDialog, QComboBox,
    QSpinBox, QCheckBox, QTextEdit, QProgressBar, QScrollArea, QSizePolicy,
    QFormLayout, QGroupBox
)
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QPixmap, QImage # For displaying images

# Project specific imports for model
from models.transunet import TransUNet, VIT_CONFIG
try:
    from torchinfo import summary as torchinfo_summary
    TORCHINFO_AVAILABLE = True
except ImportError:
    TORCHINFO_AVAILABLE = False
    torchinfo_summary = None

from gui.widgets import PlotWidget # Matplotlib canvas widget

# Placeholder for worker threads that will run training/testing
class WorkerThread(QThread):
    log_message = Signal(str)
    progress_update = Signal(int) # For progress bar (0-100)
    # task_finished now emits: status (str), output_directory (str)
    task_finished = Signal(str, str)
    # plot_data_signal = Signal(dict, str) # This was a placeholder, direct file reading is better

    def __init__(self, command_list, output_dir_base): # Pass base output dir to form potential final path
        super().__init__()
        self.command_list = command_list
        self.output_dir_base = output_dir_base # Store base output dir
        self.process = None
        self.final_output_dir_from_log = None # To store parsed output dir

    def run(self):
        try:
            self.log_message.emit(f"Starting process: {' '.join(self.command_list)}")
            self.process = subprocess.Popen(
                self.command_list,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                cwd=os.getcwd() # Ensure it runs from the project root if paths are relative
            )

            if self.process.stdout:
                for line in iter(self.process.stdout.readline, ''):
                    line_stripped = line.strip()
                    self.log_message.emit(line_stripped)

                    # Parse epoch progress
                    if "Epoch [" in line_stripped and "/" in line_stripped:
                        try:
                            progress_part = line_stripped.split("Epoch [")[1].split("]")[0]
                            current_epoch, total_epochs = map(int, progress_part.split('/'))
                            self.progress_update.emit(int((current_epoch / total_epochs) * 100))
                        except Exception:
                            pass

                    # Parse final output directory from log (main_train.py now logs this)
                    if "All training outputs saved in:" in line_stripped:
                        self.final_output_dir_from_log = line_stripped.split("All training outputs saved in:")[1].strip()
                        self.log_message.emit(f"Detected final output directory: {self.final_output_dir_from_log}")


            self.process.wait()

            if self.process.returncode == 0:
                self.log_message.emit("Process finished successfully.")
                # Use parsed output_dir_final if available, otherwise construct from base (less reliable)
                output_dir_to_emit = self.final_output_dir_from_log if self.final_output_dir_from_log else self.output_dir_base
                self.task_finished.emit("Success", output_dir_to_emit)
            else:
                self.log_message.emit(f"Process failed with exit code {self.process.returncode}.")
                self.task_finished.emit(f"Error: Exit code {self.process.returncode}", self.output_dir_base)

        except Exception as e:
            self.log_message.emit(f"Exception in worker thread: {e}")
            self.task_finished.emit(f"Exception: {str(e)}", self.output_dir_base)
        finally:
            if self.process and self.process.stdout:
                 self.process.stdout.close()



class MainAppWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TransUNet GUI")
        self.setGeometry(100, 100, 1200, 800) # Increased default size

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.train_tab = QWidget()
        self.test_tab = QWidget()

        self.tabs.addTab(self.train_tab, "Train Model")
        self.tabs.addTab(self.test_tab, "Test Model")

        self.worker_thread = None

        self._setup_train_tab()
        self._setup_test_tab()

    def _create_file_input(self, label_text, line_edit_obj, is_folder=True):
        layout = QHBoxLayout()
        layout.addWidget(QLabel(label_text))
        layout.addWidget(line_edit_obj)
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(lambda: self._browse_path(line_edit_obj, is_folder))
        layout.addWidget(browse_button)
        return layout

    def _browse_path(self, line_edit_obj, is_folder):
        if is_folder:
            path = QFileDialog.getExistingDirectory(self, "Select Folder")
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Select File", filter="PyTorch Models (*.pth);;All Files (*)")
        if path:
            line_edit_obj.setText(path)

    # --- Train Tab Setup ---
    def _setup_train_tab(self):
        layout = QVBoxLayout(self.train_tab)

        # Scroll Area for parameters
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_area.setWidget(scroll_widget)
        form_layout = QFormLayout(scroll_widget)

        # Dataset Paths Group
        dataset_group = QGroupBox("Dataset Paths")
        dataset_layout = QFormLayout()
        self.train_img_dir_le = QLineEdit()
        self.train_mask_dir_le = QLineEdit()
        self.val_img_dir_le = QLineEdit()
        self.val_mask_dir_le = QLineEdit()
        dataset_layout.addRow("Train Image Dir:", self._create_file_input("", self.train_img_dir_le))
        dataset_layout.addRow("Train Mask Dir:", self._create_file_input("", self.train_mask_dir_le))
        dataset_layout.addRow("Validation Image Dir (Optional):", self._create_file_input("", self.val_img_dir_le))
        dataset_layout.addRow("Validation Mask Dir (Optional):", self._create_file_input("", self.val_mask_dir_le))
        dataset_group.setLayout(dataset_layout)
        form_layout.addRow(dataset_group)

        # Model Config Group
        model_config_group = QGroupBox("Model Configuration")
        model_config_layout = QFormLayout()
        self.cnn_backbone_combo = QComboBox()
        self.cnn_backbone_combo.addItems(['resnet18', 'resnet50'])
        model_config_layout.addRow("CNN Backbone:", self.cnn_backbone_combo)
        # Add more model params as needed (ViT config, etc.)
        self.num_classes_spin = QSpinBox(); self.num_classes_spin.setRange(1, 100); self.num_classes_spin.setValue(1)
        model_config_layout.addRow("Number of Classes:", self.num_classes_spin)
        self.img_h_spin = QSpinBox(); self.img_h_spin.setRange(32, 2048); self.img_h_spin.setValue(256); self.img_h_spin.setSingleStep(16)
        self.img_w_spin = QSpinBox(); self.img_w_spin.setRange(32, 2048); self.img_w_spin.setValue(256); self.img_w_spin.setSingleStep(16)
        model_config_layout.addRow("Image Size (H x W):", QHBoxLayout([self.img_h_spin, QLabel("x"), self.img_w_spin]))

        # Connect signals for architecture update
        self.cnn_backbone_combo.currentTextChanged.connect(self._update_architecture_display)
        self.num_classes_spin.valueChanged.connect(self._update_architecture_display)
        self.img_h_spin.valueChanged.connect(self._update_architecture_display)
        self.img_w_spin.valueChanged.connect(self._update_architecture_display)

        model_config_group.setLayout(model_config_layout)
        form_layout.addRow(model_config_group)

        # Architecture Display Group
        arch_display_group = QGroupBox("Model Architecture")
        arch_display_layout = QVBoxLayout()
        self.arch_display_text = QTextEdit()
        self.arch_display_text.setReadOnly(True)
        self.arch_display_text.setFixedHeight(150) # Adjust height as needed
        arch_display_layout.addWidget(self.arch_display_text)
        arch_display_group.setLayout(arch_display_layout)
        form_layout.addRow(arch_display_group)


        # Training Hyperparameters Group
        hyperparams_group = QGroupBox("Training Hyperparameters")
        hyperparams_layout = QFormLayout()
        self.epochs_spin = QSpinBox(); self.epochs_spin.setRange(1, 1000); self.epochs_spin.setValue(50)
        hyperparams_layout.addRow("Epochs:", self.epochs_spin)
        self.batch_size_spin = QSpinBox(); self.batch_size_spin.setRange(1, 128); self.batch_size_spin.setValue(4)
        hyperparams_layout.addRow("Batch Size:", self.batch_size_spin)
        self.lr_le = QLineEdit("1e-4")
        hyperparams_layout.addRow("Learning Rate:", self.lr_le)
        self.optimizer_combo = QComboBox(); self.optimizer_combo.addItems(["Adam", "AdamW", "SGD"])
        hyperparams_layout.addRow("Optimizer:", self.optimizer_combo)
        self.loss_combo = QComboBox(); self.loss_combo.addItems(["DiceBCE", "Dice", "Jaccard", "CrossEntropy"])
        hyperparams_layout.addRow("Loss Function:", self.loss_combo)
        self.no_aug_checkbox = QCheckBox("Disable Augmentation")
        hyperparams_layout.addRow(self.no_aug_checkbox)
        hyperparams_group.setLayout(hyperparams_layout)
        form_layout.addRow(hyperparams_group)

        # Output and Control Group
        output_control_group = QGroupBox("Output and Control")
        output_control_layout = QFormLayout()
        self.output_dir_train_le = QLineEdit("runs/gui_train_run")
        output_control_layout.addRow("Output Directory:", self._create_file_input("", self.output_dir_train_le))
        self.start_train_button = QPushButton("Start Training")
        self.start_train_button.clicked.connect(self.start_training)
        output_control_layout.addRow(self.start_train_button)
        output_control_group.setLayout(output_control_layout)
        form_layout.addRow(output_control_group)

        layout.addWidget(scroll_area)

        # --- Bottom section for logs and plots ---
        bottom_splitter = QWidget() # Using QWidget as a container for QHBoxLayout
        bottom_layout = QHBoxLayout(bottom_splitter)

        # Log display and progress bar (Left side of bottom_splitter)
        self.train_log_display = QTextEdit()
        self.train_log_display.setReadOnly(True)
        self.train_progress_bar = QProgressBar()
        self.train_progress_bar.setRange(0,100)

        log_status_widget = QWidget()
        status_layout = QVBoxLayout(log_status_widget)
        status_layout.addWidget(QLabel("Training Log:"))
        status_layout.addWidget(self.train_log_display, stretch=1)
        status_layout.addWidget(QLabel("Training Progress:"))
        status_layout.addWidget(self.train_progress_bar)

        bottom_layout.addWidget(log_status_widget, 1) # Log area takes 1 part

        # Plotting area (Right side of bottom_splitter)
        self.train_plot_widget = PlotWidget()
        self.train_plot_widget.get_canvas().axes.set_title("Training Curves (Loss, Dice)") # Initial title
        self.train_plot_widget.get_canvas().draw()
        bottom_layout.addWidget(self.train_plot_widget, 1) # Plot area takes 1 part

        layout.addWidget(bottom_splitter) # Add the splitter to the main VBox layout

        self._update_architecture_display() # Initial display


    def _update_architecture_display(self):
        try:
            cnn_backbone = self.cnn_backbone_combo.currentText()
            num_classes = self.num_classes_spin.value()
            img_h = self.img_h_spin.value()
            img_w = self.img_w_spin.value()

            # Ensure img_size is divisible by 32 for default TransUNet (common for ResNet backbone)
            # Or handle this within TransUNet model if it supports dynamic padding/cropping.
            # For summary, exact divisibility might not crash torchinfo but good for model itself.
            if img_h % 32 != 0 or img_w % 32 != 0:
                self.arch_display_text.setText(f"Warning: Image height ({img_h}) and width ({img_w}) should ideally be divisible by 32 for standard ResNet backbones in TransUNet.")
                # return # Or proceed with a warning

            temp_model = TransUNet(
                img_size=(img_h, img_w),
                num_classes=num_classes,
                cnn_backbone_type=cnn_backbone,
                vit_config=VIT_CONFIG # Using default ViT config for summary
            )

            # Device for summary, CPU is fine
            device = torch.device("cpu")
            temp_model.to(device)

            if TORCHINFO_AVAILABLE:
                # Batch size 1, 3 channels (standard for RGB images)
                summary_str = str(torchinfo_summary(temp_model, input_size=(1, 3, img_h, img_w), verbose=0, device=device))
                self.arch_display_text.setText(summary_str)
            else:
                self.arch_display_text.setText(
                    f"torchinfo library not found. Cannot display detailed model summary.\n"
                    f"Model: TransUNet\n"
                    f"CNN Backbone: {cnn_backbone}\n"
                    f"Num Classes: {num_classes}\n"
                    f"Image Size: ({img_h}x{img_w})"
                )
        except Exception as e:
            self.arch_display_text.setText(f"Error generating model summary: {e}")
            if hasattr(self, 'train_log_display') and self.train_log_display:
                 self.train_log_display.append(f"Error for arch display: {e}")
            else: # Fallback if log display not ready
                print(f"Error for arch display (log not ready): {e}")


    def start_training(self):
        if self.worker_thread and self.worker_thread.isRunning():
            self.train_log_display.append("A task is already running.")
            return

        self.train_log_display.clear()
        self.train_progress_bar.setValue(0)
        # Clear previous plots
        train_canvas = self.train_plot_widget.get_canvas()
        train_canvas.axes.cla()
        train_canvas.axes.set_title("Training Curves (Loss, Dice)")
        train_canvas.draw()


        # Construct command for main_train.py
        # Ensure paths are absolute or correctly relative to where main_train.py is run from
        # For simplicity, assume main_train.py handles relative paths from project root
        script_path = os.path.join(os.getcwd(), "transunet_project", "main_train.py")
        if not os.path.exists(script_path):
            self.train_log_display.append(f"Error: Training script not found at {script_path}")
            return

        cmd = ["python", script_path]
        cmd.extend(["--image_dir", self.train_img_dir_le.text()])
        cmd.extend(["--mask_dir", self.train_mask_dir_le.text()])
        if self.val_img_dir_le.text():
            cmd.extend(["--val_image_dir", self.val_img_dir_le.text()])
            cmd.extend(["--val_mask_dir", self.val_mask_dir_le.text()])

        cmd.extend(["--cnn_backbone", self.cnn_backbone_combo.currentText()])
        cmd.extend(["--num_classes", str(self.num_classes_spin.value())])
        cmd.extend(["--img_size_h", str(self.img_h_spin.value())])
        cmd.extend(["--img_size_w", str(self.img_w_spin.value())])

        cmd.extend(["--epochs", str(self.epochs_spin.value())])
        cmd.extend(["--batch_size", str(self.batch_size_spin.value())])
        cmd.extend(["--lr", self.lr_le.text()])
        cmd.extend(["--optimizer_name", self.optimizer_combo.currentText()])
        cmd.extend(["--loss_type", self.loss_combo.currentText()])
        if self.no_aug_checkbox.isChecked():
            cmd.append("--no_augmentation")

        cmd.extend(["--output_dir", self.output_dir_train_le.text()])

        self.worker_thread = WorkerThread(cmd, self.output_dir_train_le.text()) # Pass base output dir
        self.worker_thread.log_message.connect(self.train_log_display.append)
        self.worker_thread.progress_update.connect(self.train_progress_bar.setValue)
        self.worker_thread.task_finished.connect(self.on_training_finished)
        self.worker_thread.start()
        self.start_train_button.setEnabled(False)

    @Slot(str, str) # Receives status and output_directory
    def on_training_finished(self, status, output_directory):
        self.train_log_display.append(f"Training finished: {status}")
        self.start_train_button.setEnabled(True)
        self.train_progress_bar.setValue(100 if "Success" in status else 0)

        if "Success" in status and output_directory:
            self.train_log_display.append(f"Attempting to load training curves from: {output_directory}")
            history_file = os.path.join(output_directory, "history.json")
            if os.path.exists(history_file):
                try:
                    with open(history_file, 'r') as f:
                        history = json.load(f)
                    self._plot_gui_training_curves(history)
                    self.train_log_display.append("Training curves plotted in GUI.")
                except Exception as e:
                    self.train_log_display.append(f"Error loading or plotting history.json: {e}")
            else:
                self.train_log_display.append(f"history.json not found in {output_directory}. Cannot plot curves in GUI.")
                # Fallback: could try to load the training_curves.png if needed
                # png_file = os.path.join(output_directory, "training_curves.png")
                # if os.path.exists(png_file): ... display PNG ...

    def _plot_gui_training_curves(self, history):
        canvas = self.train_plot_widget.get_canvas()
        axes = canvas.axes
        axes.cla() # Clear previous plots

        epochs = range(1, len(history.get('train_loss', [])) + 1)

        # Plot Loss
        if history.get('train_loss'):
            axes.plot(epochs, history['train_loss'], 'bo-', label='Train Loss')
        if history.get('val_loss'):
            axes.plot(epochs, history['val_loss'], 'ro-', label='Validation Loss')

        ax2 = axes.twinx() # Second y-axis for Dice
        plotted_dice = False
        if history.get('val_dice'): # Assuming 'val_dice' is the key from history.json
            ax2.plot(epochs, history['val_dice'], 'gs-', label='Validation Dice')
            plotted_dice = True
        # Add train_dice if available in history
        # if history.get('train_dice'):
        #     ax2.plot(epochs, history['train_dice'], 'cs-', label='Train Dice')
        #     plotted_dice = True

        axes.set_xlabel("Epochs")
        axes.set_ylabel("Loss", color='tab:blue')
        axes.tick_params(axis='y', labelcolor='tab:blue')
        lines, labels = axes.get_legend_handles_labels()

        if plotted_dice:
            ax2.set_ylabel("Dice Score", color='tab:green')
            ax2.tick_params(axis='y', labelcolor='tab:green')
            lines2, labels2 = ax2.get_legend_handles_labels()
            axes.legend(lines + lines2, labels + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2) # Adjust legend position
        else: # Only loss was plotted
            axes.legend(loc='best')
            ax2.set_yticks([]) # Hide secondary y-axis if no dice data

        axes.set_title("Training Loss & Dice Curves")
        fig.subplots_adjust(bottom=0.2) # Adjust bottom margin for legend
        axes.grid(True)
        canvas.draw()

    # --- Test Tab Setup ---
    def _setup_test_tab(self):
        layout = QVBoxLayout(self.test_tab)
        # Similar structure: params, controls, log, display areas

        # Model Path and Config Group
        model_load_group = QGroupBox("Load Trained Model")
        model_load_layout = QFormLayout()
        self.test_model_path_le = QLineEdit()
        model_load_layout.addRow("Model Path (.pth):", self._create_file_input("", self.test_model_path_le, is_folder=False))
        # Critical: User must provide config matching the loaded model
        self.test_cnn_backbone_combo = QComboBox(); self.test_cnn_backbone_combo.addItems(['resnet18', 'resnet50'])
        model_load_layout.addRow("CNN Backbone (of model):", self.test_cnn_backbone_combo)
        self.test_num_classes_spin = QSpinBox(); self.test_num_classes_spin.setRange(1,100); self.test_num_classes_spin.setValue(1)
        model_load_layout.addRow("Num Classes (of model):", self.test_num_classes_spin)
        self.test_img_h_spin = QSpinBox(); self.test_img_h_spin.setRange(32,2048); self.test_img_h_spin.setValue(256)
        self.test_img_w_spin = QSpinBox(); self.test_img_w_spin.setRange(32,2048); self.test_img_w_spin.setValue(256)
        model_load_layout.addRow("Image Size (H x W, of model):", QHBoxLayout([self.test_img_h_spin, QLabel("x"), self.test_img_w_spin]))
        model_load_group.setLayout(model_load_layout)
        layout.addWidget(model_load_group)

        # Test Data Group
        test_data_group = QGroupBox("Test Dataset")
        test_data_layout = QFormLayout()
        self.test_img_dir_le = QLineEdit()
        self.test_mask_dir_le = QLineEdit() # For metrics calculation
        test_data_layout.addRow("Test Image Dir:", self._create_file_input("", self.test_img_dir_le))
        test_data_layout.addRow("Test Mask Dir (Optional):", self._create_file_input("", self.test_mask_dir_le))
        test_data_group.setLayout(test_data_layout)
        layout.addWidget(test_data_group)

        # Output and Control
        test_control_group = QGroupBox("Output and Control")
        test_control_layout = QFormLayout()
        self.test_output_dir_le = QLineEdit("test_results/gui_test_run")
        test_control_layout.addRow("Output Directory:", self._create_file_input("", self.test_output_dir_le))
        self.run_test_button = QPushButton("Run Segmentation and Evaluation")
        self.run_test_button.clicked.connect(self.start_testing)
        test_control_layout.addRow(self.run_test_button)
        test_control_group.setLayout(test_control_layout)
        layout.addWidget(test_control_group)

        # Log Display
        self.test_log_display = QTextEdit()
        self.test_log_display.setReadOnly(True)
        layout.addWidget(QLabel("Testing Log & Metrics:"))
        layout.addWidget(self.test_log_display, stretch=1)

        # --- Bottom section for image samples and metrics plots ---
        test_bottom_splitter = QWidget()
        test_bottom_layout = QHBoxLayout(test_bottom_splitter)

        # Image Display Area (Left side of bottom_splitter)
        image_display_group = QGroupBox("Segmentation Result Sample")
        self.image_display_layout = QHBoxLayout() # Will hold 3 QLabels
        self.orig_img_label = QLabel("Original Image"); self.orig_img_label.setAlignment(Qt.AlignCenter); self.orig_img_label.setFixedSize(200,200) # Slightly smaller
        self.gt_mask_label = QLabel("Ground Truth Mask"); self.gt_mask_label.setAlignment(Qt.AlignCenter); self.gt_mask_label.setFixedSize(200,200)
        self.pred_mask_label = QLabel("Predicted Mask"); self.pred_mask_label.setAlignment(Qt.AlignCenter); self.pred_mask_label.setFixedSize(200,200)
        self.image_display_layout.addWidget(self.orig_img_label)
        self.image_display_layout.addWidget(self.gt_mask_label)
        self.image_display_layout.addWidget(self.pred_mask_label)
        image_display_group.setLayout(self.image_display_layout)
        test_bottom_layout.addWidget(image_display_group, 1) # Image samples take 1 part

        # Metrics Plot Area (Right side of bottom_splitter)
        self.test_metrics_plot_widget = PlotWidget()
        self.test_metrics_plot_widget.get_canvas().axes.set_title("Evaluation Metrics (CM, ROC, PR)")
        self.test_metrics_plot_widget.get_canvas().draw()
        test_bottom_layout.addWidget(self.test_metrics_plot_widget, 1) # Metrics plot takes 1 part

        layout.addWidget(test_bottom_splitter) # Add this splitter to the main test tab layout
        layout.addStretch()

    def start_testing(self):
        if self.worker_thread and self.worker_thread.isRunning():
            self.test_log_display.append("A task is already running.")
            return

        self.test_log_display.clear()
        self.orig_img_label.clear(); self.gt_mask_label.clear(); self.pred_mask_label.clear()

        script_path_test = os.path.join(os.getcwd(), "transunet_project", "main_test.py")
        if not os.path.exists(script_path_test):
            self.test_log_display.append(f"Error: Testing script not found at {script_path_test}")
            return

        cmd = ["python", script_path_test]
        cmd.extend(["--image_dir", self.test_img_dir_le.text()])
        if self.test_mask_dir_le.text():
            cmd.extend(["--mask_dir", self.test_mask_dir_le.text()])

        cmd.extend(["--model_path", self.test_model_path_le.text()])
        cmd.extend(["--cnn_backbone", self.test_cnn_backbone_combo.currentText()])
        cmd.extend(["--num_classes", str(self.test_num_classes_spin.value())])
        cmd.extend(["--img_size_h", str(self.test_img_h_spin.value())])
        cmd.extend(["--img_size_w", str(self.test_img_w_spin.value())])
        cmd.extend(["--output_dir", self.test_output_dir_le.text()])
        cmd.extend(["--save_samples_count", "3"])

        self.worker_thread = WorkerThread(cmd, self.test_output_dir_le.text()) # Pass base output dir
        self.worker_thread.log_message.connect(self.test_log_display.append)
        self.worker_thread.task_finished.connect(self.on_testing_finished)
        self.worker_thread.start()
        self.run_test_button.setEnabled(False)

    @Slot(str, str) # Receives status and output_directory
    def on_testing_finished(self, status, output_directory):
        self.test_log_display.append(f"Testing finished: {status}")
        self.run_test_button.setEnabled(True)

        if "Success" in status and output_directory:
            self.test_log_display.append(f"Attempting to load test metrics from: {output_directory}")
            metrics_file = os.path.join(output_directory, "test_metrics.json")
            if os.path.exists(metrics_file):
                try:
                    with open(metrics_file, 'r') as f:
                        metrics_data = json.load(f)

                    # Display textual summary of metrics
                    summary_text = "--- Evaluation Metrics Summary ---\n"
                    summary_text += f"Dice Coefficient: {metrics_data.get('dice_coefficient', 'N/A'):.4f}\n"
                    summary_text += f"Jaccard Index (IoU): {metrics_data.get('jaccard_index', 'N/A'):.4f}\n"
                    if 'accuracy' in metrics_data: # From calculate_classification_metrics
                        summary_text += f"Pixel-wise Accuracy: {metrics_data.get('accuracy', 'N/A'):.4f}\n"
                    if 'f1_score' in metrics_data:
                        summary_text += f"Pixel-wise F1-score ({metrics_data.get('classification_report',{}).get('accuracy',{}).get('note','macro')} avg): {metrics_data.get('f1_score', 'N/A'):.4f}\n" # a bit verbose way to get avg mode
                    if 'roc_auc' in metrics_data:
                         summary_text += f"ROC AUC: {metrics_data.get('roc_auc', 'N/A')}\n" # Value might be string if error
                    if 'hausdorff_distance' in metrics_data:
                        hd_val = metrics_data.get('hausdorff_distance')
                        if isinstance(hd_val, float) and not np.isnan(hd_val):
                            summary_text += f"Hausdorff Distance: {hd_val:.4f}\n"
                        else:
                            summary_text += f"Hausdorff Distance: {hd_val if isinstance(hd_val, str) else 'N/A or not calculated'}\n"

                    # Add more key metrics as desired
                    self.test_log_display.append(summary_text)

                    # Plot graphical metrics
                    self._plot_gui_test_metrics(metrics_data)
                    self.test_log_display.append("Evaluation metrics plots updated in GUI.")

                except Exception as e:
                    self.test_log_display.append(f"Error loading or processing test_metrics.json: {e}")
            else:
                self.test_log_display.append(f"test_metrics.json not found in {output_directory}.")

            self.load_and_display_test_samples(output_directory) # Call to load images
        else:
            self.test_log_display.append("Testing failed or output directory not found.")

    def load_and_display_test_samples(self, output_dir):
        self.test_log_display.append(f"Attempting to load sample images from {output_dir}...")

        sample_idx = 0 # We are saving sample '0' from main_test.py
        orig_img_path = os.path.join(output_dir, f"sample_{sample_idx}_orig.png")
        gt_mask_path = os.path.join(output_dir, f"sample_{sample_idx}_gt.png")
        pred_mask_path = os.path.join(output_dir, f"sample_{sample_idx}_pred.png")

        loaded_any = False
        if os.path.exists(orig_img_path):
            pixmap_orig = QPixmap(orig_img_path)
            self.orig_img_label.setPixmap(pixmap_orig.scaled(self.orig_img_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            loaded_any = True
        else:
            self.orig_img_label.setText("Orig Img N/A")

        if os.path.exists(gt_mask_path):
            pixmap_gt = QPixmap(gt_mask_path)
            self.gt_mask_label.setPixmap(pixmap_gt.scaled(self.gt_mask_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            loaded_any = True
        else:
            self.gt_mask_label.setText("GT Mask N/A")

        if os.path.exists(pred_mask_path):
            pixmap_pred = QPixmap(pred_mask_path)
            self.pred_mask_label.setPixmap(pixmap_pred.scaled(self.pred_mask_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            loaded_any = True
        else:
            self.pred_mask_label.setText("Pred Mask N/A")

        if loaded_any:
            self.test_log_display.append("Sample images displayed.")
        else:
            self.test_log_display.append("No sample images found or loaded.")

    def _plot_gui_test_metrics(self, metrics_data):
        canvas = self.test_metrics_plot_widget.get_canvas()
        fig = canvas.fig
        fig.clear() # Clear entire figure

        # Define class names, default if not in metrics_data (e.g. for older test_metrics.json)
        num_classes_from_cm = len(metrics_data.get("confusion_matrix", [[]]))
        class_names = metrics_data.get("class_names")
        if not class_names and 'classification_report' in metrics_data:
            # Try to get from classification_report keys (excluding 'accuracy', 'macro avg', 'weighted avg')
            report = metrics_data['classification_report']
            class_names = [k for k in report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
        if not class_names: # Fallback if still not found
            class_names = [f"Class {i}" for i in range(num_classes_from_cm if num_classes_from_cm > 0 else 1)]


        # 1. Confusion Matrix
        ax1 = fig.add_subplot(131)
        cm_data = metrics_data.get("confusion_matrix")
        if cm_data and isinstance(cm_data, list):
            cm_np = np.array(cm_data)
            cax = ax1.imshow(cm_np, interpolation='nearest', cmap='Blues')
            fig.colorbar(cax, ax=ax1, shrink=0.8)
            ax1.set_title('Confusion Matrix', fontsize=10)
            ax1.set_xlabel('Predicted Label', fontsize=8)
            ax1.set_ylabel('True Label', fontsize=8)
            if class_names and len(class_names) == cm_np.shape[0]:
                tick_marks = np.arange(len(class_names))
                ax1.set_xticks(tick_marks)
                ax1.set_xticklabels(class_names, rotation=45, ha="right", fontsize=7)
                ax1.set_yticks(tick_marks)
                ax1.set_yticklabels(class_names, fontsize=7)

            thresh = cm_np.max() / 2. if cm_np.size > 0 else 0.
            for i in range(cm_np.shape[0]):
                for j in range(cm_np.shape[1]):
                    ax1.text(j, i, format(cm_np[i, j], 'd'),
                             ha="center", va="center", fontsize=7,
                             color="white" if cm_np[i, j] > thresh else "black")
            ax1.grid(False)
        else:
            ax1.text(0.5, 0.5, 'CM data N/A', ha='center', va='center')
            ax1.set_title('Confusion Matrix', fontsize=10)

        # 2. ROC Curve (only if binary data is available in metrics_data['roc_curve'])
        ax2 = fig.add_subplot(132)
        roc_curve_data = metrics_data.get("roc_curve") # Expects {'fpr': [...], 'tpr': [...]}
        roc_auc = metrics_data.get("roc_auc", "N/A") # Expects float or "N/A" string
        if isinstance(roc_curve_data, dict) and 'fpr' in roc_curve_data and 'tpr' in roc_curve_data:
            fpr = roc_curve_data['fpr']
            tpr = roc_curve_data['tpr']
            auc_label = f"AUC = {roc_auc:.2f}" if isinstance(roc_auc, float) else f"AUC = {roc_auc}"
            ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC ({auc_label})')
            ax2.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
            ax2.set_xlim([0.0, 1.0])
            ax2.set_ylim([0.0, 1.05])
            ax2.set_xlabel('False Positive Rate (FPR)', fontsize=8)
            ax2.set_ylabel('True Positive Rate (TPR)', fontsize=8)
            ax2.set_title('ROC Curve', fontsize=10)
            ax2.legend(loc="lower right", fontsize='x-small')
            ax2.grid(True)
        else:
            ax2.text(0.5, 0.5, 'ROC data N/A', ha='center', va='center')
            ax2.set_title('ROC Curve', fontsize=10)

        # 3. PR Curve (only if binary data is available in metrics_data['pr_curve'])
        ax3 = fig.add_subplot(133)
        pr_curve_data = metrics_data.get("pr_curve")
        if isinstance(pr_curve_data, dict) and 'precision' in pr_curve_data and 'recall' in pr_curve_data:
            precision = pr_curve_data['precision']
            recall = pr_curve_data['recall']
            ax3.plot(recall, precision, color='blue', lw=2, label=f'PR Curve')
            ax3.set_xlabel('Recall', fontsize=8)
            ax3.set_ylabel('Precision', fontsize=8)
            ax3.set_ylim([0.0, 1.05])
            ax3.set_xlim([0.0, 1.0])
            ax3.set_title('Precision-Recall Curve', fontsize=10)
            ax3.legend(loc="best", fontsize='x-small') # 'best' can sometimes overlap, consider 'lower left' or similar
            ax3.grid(True)
        else:
            ax3.text(0.5, 0.5, 'PR data N/A', ha='center', va='center')
            ax3.set_title('Precision-Recall Curve', fontsize=10)

        try:
            fig.tight_layout(pad=1.0, h_pad=0.5, w_pad=0.5) # Adjust padding
        except Exception:
             pass # Sometimes tight_layout fails with specific content
        canvas.draw()


    def closeEvent(self, event):
        # Ensure worker thread is properly terminated if running
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.log_message.emit("Attempting to terminate worker process...")
            if self.worker_thread.process:
                self.worker_thread.process.terminate() # Try to terminate gracefully
                try:
                    self.worker_thread.process.wait(timeout=5) # Wait a bit
                except subprocess.TimeoutExpired:
                    self.worker_thread.process.kill() # Force kill if necessary
                    self.worker_thread.log_message.emit("Worker process force killed.")
            self.worker_thread.quit()
            self.worker_thread.wait()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    # Apply a style if desired (optional)
    # app.setStyle('Fusion')
    window = MainAppWindow()
    window.show()
    sys.exit(app.exec())
