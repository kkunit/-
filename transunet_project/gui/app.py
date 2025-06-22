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

from gui.widgets import PlotWidget # Matplotlib canvas widget

# Placeholder for worker threads that will run training/testing
class WorkerThread(QThread):
    log_message = Signal(str)
    progress_update = Signal(int) # For progress bar (0-100)
    task_finished = Signal(str)   # Path to results or error message
    plot_data_signal = Signal(dict, str) # history dict, type (e.g. 'loss', 'dice')

    def __init__(self, command_list):
        super().__init__()
        self.command_list = command_list
        self.process = None

    def run(self):
        try:
            self.log_message.emit(f"Starting process: {' '.join(self.command_list)}")
            # Use Popen for non-blocking I/O and to get live output
            self.process = subprocess.Popen(
                self.command_list,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, # Redirect stderr to stdout
                text=True,
                bufsize=1, # Line buffered
                universal_newlines=True
            )

            # TODO: Implement robust progress parsing if possible from script output
            # For now, just stream output. Progress bar might be manually updated or by epoch.
            if self.process.stdout:
                for line in iter(self.process.stdout.readline, ''):
                    self.log_message.emit(line.strip())
                    # Example parsing for epoch progress (very basic, assumes specific output format)
                    if "Epoch [" in line and "/" in line:
                        try:
                            # "Epoch [1/100]" -> parts[0]="1", parts[1]="100]"
                            progress_part = line.split("Epoch [")[1].split("]")[0]
                            current_epoch, total_epochs = map(int, progress_part.split('/'))
                            self.progress_update.emit(int((current_epoch / total_epochs) * 100))
                        except Exception:
                            pass # Ignore parsing errors for progress

            self.process.wait() # Wait for the process to complete

            if self.process.returncode == 0:
                self.log_message.emit("Process finished successfully.")
                # Assuming the script saves results and we might know the output_dir_final
                # This part needs coordination with how main_train/main_test report their output dir
                self.task_finished.emit("Success") # Or pass a path to results
            else:
                self.log_message.emit(f"Process failed with exit code {self.process.returncode}.")
                self.task_finished.emit(f"Error: Exit code {self.process.returncode}")

        except Exception as e:
            self.log_message.emit(f"Exception in worker thread: {e}")
            self.task_finished.emit(f"Exception: {e}")
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
        model_config_group.setLayout(model_config_layout)
        form_layout.addRow(model_config_group)

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

        layout.addWidget(scroll_area) # Add scroll area to main VBox

        # Log display and progress bar
        self.train_log_display = QTextEdit()
        self.train_log_display.setReadOnly(True)
        self.train_progress_bar = QProgressBar()
        self.train_progress_bar.setRange(0,100)

        status_layout = QVBoxLayout()
        status_layout.addWidget(QLabel("Training Log:"))
        status_layout.addWidget(self.train_log_display, stretch=1) # Give more stretch
        status_layout.addWidget(QLabel("Training Progress:"))
        status_layout.addWidget(self.train_progress_bar)

        # Plotting area (placeholder, real plots need more work)
        self.train_plot_widget = PlotWidget() # from gui.widgets
        # self.train_plot_widget.get_canvas().plot([0],[0], title="Training Curves") # Initial empty plot

        # Splitter for logs and plots
        # main_content_layout = QHBoxLayout()
        # main_content_layout.addLayout(status_layout, 1) # Log area takes 1 part
        # main_content_layout.addWidget(self.train_plot_widget, 1) # Plot area takes 1 part
        # layout.addLayout(main_content_layout)
        layout.addLayout(status_layout) # Simpler: logs and progress bar below params
        layout.addWidget(self.train_plot_widget) # Plot below logs


    def start_training(self):
        if self.worker_thread and self.worker_thread.isRunning():
            self.train_log_display.append("A task is already running.")
            return

        self.train_log_display.clear()
        self.train_progress_bar.setValue(0)

        cmd = ["python", "transunet_project/main_train.py"]
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
        # Add other args like scheduler, weights paths, freeze options later

        self.worker_thread = WorkerThread(cmd)
        self.worker_thread.log_message.connect(self.train_log_display.append)
        self.worker_thread.progress_update.connect(self.train_progress_bar.setValue)
        self.worker_thread.task_finished.connect(self.on_training_finished)
        # self.worker_thread.plot_data_signal.connect(self.update_training_plot)
        self.worker_thread.start()
        self.start_train_button.setEnabled(False)

    @Slot(str)
    def on_training_finished(self, result_message):
        self.train_log_display.append(f"Training finished: {result_message}")
        self.start_train_button.setEnabled(True)
        self.train_progress_bar.setValue(100 if "Success" in result_message else 0)
        # After training, attempt to load and display the training_curves.png
        # This assumes output_dir_final is known or can be inferred.
        # For simplicity, let's assume user checks the output folder for now.
        # A more robust way would be for WorkerThread to emit the final output_dir_final path.
        # Or parse it from the log.
        # self.load_and_display_training_curves(output_dir_final)

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

        # Image Display Area (Horizontal: Original, GT Mask, Pred Mask)
        image_display_group = QGroupBox("Segmentation Result Sample")
        self.image_display_layout = QHBoxLayout()
        self.orig_img_label = QLabel("Original Image"); self.orig_img_label.setAlignment(Qt.AlignCenter); self.orig_img_label.setFixedSize(256,256)
        self.gt_mask_label = QLabel("Ground Truth Mask"); self.gt_mask_label.setAlignment(Qt.AlignCenter); self.gt_mask_label.setFixedSize(256,256)
        self.pred_mask_label = QLabel("Predicted Mask"); self.pred_mask_label.setAlignment(Qt.AlignCenter); self.pred_mask_label.setFixedSize(256,256)
        self.image_display_layout.addWidget(self.orig_img_label)
        self.image_display_layout.addWidget(self.gt_mask_label)
        self.image_display_layout.addWidget(self.pred_mask_label)
        image_display_group.setLayout(self.image_display_layout)
        layout.addWidget(image_display_group)

        # Placeholder for metrics plots (CM, ROC, PR)
        self.test_metrics_plot_widget = PlotWidget() # Placeholder
        # layout.addWidget(self.test_metrics_plot_widget) # Add when ready to implement

        layout.addStretch() # Push everything up

    def start_testing(self):
        if self.worker_thread and self.worker_thread.isRunning():
            self.test_log_display.append("A task is already running.")
            return

        self.test_log_display.clear()
        self.orig_img_label.clear(); self.gt_mask_label.clear(); self.pred_mask_label.clear()

        cmd = ["python", "transunet_project/main_test.py"]
        cmd.extend(["--image_dir", self.test_img_dir_le.text()])
        if self.test_mask_dir_le.text(): # Mask dir is optional for just prediction, required for metrics
            cmd.extend(["--mask_dir", self.test_mask_dir_le.text()])

        cmd.extend(["--model_path", self.test_model_path_le.text()])
        cmd.extend(["--cnn_backbone", self.test_cnn_backbone_combo.currentText()])
        cmd.extend(["--num_classes", str(self.test_num_classes_spin.value())])
        cmd.extend(["--img_size_h", str(self.test_img_h_spin.value())])
        cmd.extend(["--img_size_w", str(self.test_img_w_spin.value())])
        cmd.extend(["--output_dir", self.test_output_dir_le.text()])
        cmd.extend(["--save_samples_count", "3"]) # Save a few samples by default

        self.worker_thread = WorkerThread(cmd)
        self.worker_thread.log_message.connect(self.test_log_display.append)
        self.worker_thread.task_finished.connect(self.on_testing_finished)
        self.worker_thread.start()
        self.run_test_button.setEnabled(False)

    @Slot(str)
    def on_testing_finished(self, result_message):
        self.test_log_display.append(f"Testing finished: {result_message}")
        self.run_test_button.setEnabled(True)
        # Attempt to load and display a sample image and metrics plots
        # This requires knowing the output_dir_final from the test script
        # For now, user needs to check the output folder.
        # A more advanced version would parse main_test.py's output_dir_final or have it write to a known temp file.
        self.test_log_display.append("Please check the output directory for saved images and detailed metrics files.")
        # Example: self.load_and_display_test_results(parsed_output_dir_final)


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
