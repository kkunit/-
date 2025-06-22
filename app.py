import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
from tkinter import messagebox
import torch # For device detection

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk

# Import modules from src
from src import utils as app_utils
from src import data_loader as app_data_loader
from src import trainer as app_trainer
from src import evaluator as app_evaluator
from src import resnet_builder as app_resnet_builder # Import the actual ResNet builder

class CVApp:
    def __init__(self, root):
        self.root = root
        self.root.title("图像分类与分割工具") # Changed to Chinese
        self.root.geometry("1000x800") # Reduced height slightly, will manage internal layout

        # Model related attributes
        self.current_model_instance = None # Will hold the instantiated model
        self.train_loader = None
        self.val_loader = None
        self.test_loader_proper = None
        self.class_names = []
        self.num_classes = 0
        self.dataset_sizes = {}
        self.adhoc_test_image_paths = [] # For storing paths of individually selected images

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("CUDA is available. Using GPU.")
        else:
            self.device = torch.device("cpu")
            print("CUDA not available. Using CPU.")


        # --- Classification Tab ---
        self.classification_tab = ttk.Frame(root)
        self.setup_classification_ui(self.classification_tab)
        self.classification_tab.pack(expand=True, fill='both', padx=10, pady=10)

        self.display_network_architecture() # Display for default model at startup


    def setup_classification_ui(self, tab):
        # --- Top Frame for Data and Model Selection ---
        top_frame = ttk.LabelFrame(tab, text="设置") # Changed
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        ttk.Label(top_frame, text="数据集路径:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W) # Changed
        self.dataset_path_var = tk.StringVar()
        self.dataset_path_entry = ttk.Entry(top_frame, textvariable=self.dataset_path_var, width=50)
        self.dataset_path_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
        self.browse_dataset_button = ttk.Button(top_frame, text="浏览...", command=self.browse_and_load_dataset) # Changed
        self.browse_dataset_button.grid(row=0, column=2, padx=5, pady=5)

        self.dataset_info_label = ttk.Label(top_frame, text="数据集信息：未加载", wraplength=300, justify=tk.LEFT) # Changed
        self.dataset_info_label.grid(row=0, column=3, rowspan=2, padx=10, pady=5, sticky=tk.NW)

        ttk.Label(top_frame, text="选择模型:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W) # Changed
        self.model_var = tk.StringVar()
        # For now, ResNet_Baseline will map to resnet18. ResNet_Improved is a placeholder.
        self.model_options = ["ResNet18_Baseline", "ResNet34_Baseline", "ResNet_Improved"] # Kept model names technical
        self.model_dropdown = ttk.Combobox(top_frame, textvariable=self.model_var,
                                           values=self.model_options, state="readonly")
        self.model_dropdown.grid(row=1, column=1, padx=5, pady=5, sticky=tk.EW)
        self.model_dropdown.set(self.model_options[0]) # Default to ResNet18_Baseline
        self.model_dropdown.bind("<<ComboboxSelected>>", self.display_network_architecture)

        top_frame.columnconfigure(1, weight=1)
        top_frame.columnconfigure(3, weight=1) # Allow dataset info label to expand a bit

        # --- Middle Frame for Architecture and Training Params ---
        middle_frame = ttk.Frame(tab)
        middle_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        arch_frame = ttk.LabelFrame(middle_frame, text="网络架构") # Changed
        arch_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5) # Keep this expandable as summary can be long
        self.arch_display = scrolledtext.ScrolledText(arch_frame, height=8, width=45, wrap=tk.WORD, state=tk.DISABLED) # Reduced height
        self.arch_display.pack(expand=True, fill=tk.BOTH, padx=5, pady=5)

        params_frame = ttk.LabelFrame(middle_frame, text="训练参数")
        params_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)

        ttk.Label(params_frame, text="优化器:").grid(row=0, column=0, padx=5, pady=3, sticky=tk.W) # Changed
        self.optimizer_var = tk.StringVar(value="Adam")
        self.optimizer_dropdown = ttk.Combobox(params_frame, textvariable=self.optimizer_var,
                                               values=["Adam", "SGD"], state="readonly", width=12)
        self.optimizer_dropdown.grid(row=0, column=1, padx=5, pady=3, sticky=tk.EW)

        ttk.Label(params_frame, text="学习率:").grid(row=1, column=0, padx=5, pady=3, sticky=tk.W) # Changed
        self.lr_var = tk.DoubleVar(value=0.001)
        self.lr_entry = ttk.Entry(params_frame, textvariable=self.lr_var, width=15)
        self.lr_entry.grid(row=1, column=1, padx=5, pady=3, sticky=tk.EW)

        ttk.Label(params_frame, text="批次大小:").grid(row=2, column=0, padx=5, pady=3, sticky=tk.W) # Changed
        self.batch_size_var = tk.IntVar(value=32)
        self.batch_size_entry = ttk.Entry(params_frame, textvariable=self.batch_size_var, width=15)
        self.batch_size_entry.grid(row=2, column=1, padx=5, pady=3, sticky=tk.EW)

        ttk.Label(params_frame, text="训练轮次:").grid(row=3, column=0, padx=5, pady=3, sticky=tk.W) # Changed
        self.epochs_var = tk.IntVar(value=10)
        self.epochs_entry = ttk.Entry(params_frame, textvariable=self.epochs_var, width=15)
        self.epochs_entry.grid(row=3, column=1, padx=5, pady=3, sticky=tk.EW)

        self.train_button = ttk.Button(params_frame, text="训练模型", command=self.train_model, state=tk.DISABLED) # Changed
        self.train_button.grid(row=4, column=0, columnspan=2, padx=5, pady=10, sticky=tk.EW)

        self.save_model_button = ttk.Button(params_frame, text="保存已训练模型", command=self.save_model, state=tk.DISABLED) # Changed
        self.save_model_button.grid(row=5, column=0, columnspan=2, padx=5, pady=5, sticky=tk.EW)

        progress_frame = ttk.LabelFrame(tab, text="训练过程与日志") # Changed
        progress_frame.pack(fill=tk.X, padx=10, pady=5)
        self.progress_display = scrolledtext.ScrolledText(progress_frame, height=5, wrap=tk.WORD, state=tk.DISABLED) # Aggressively Reduced height
        self.progress_display.pack(expand=True, fill=tk.X, padx=5, pady=5)

        # --- Training Curves Frame ---
        training_curves_frame = ttk.LabelFrame(tab, text="Training Curves (Epoch vs. Metric)") # English
        training_curves_frame.pack(fill=tk.X, expand=False, padx=10, pady=5) # fill=tk.X

        fig_train_curves = Figure(figsize=(7, 2), dpi=75) # Aggressively Reduced size & DPI
        self.training_curves_canvas = FigureCanvasTkAgg(fig_train_curves, master=training_curves_frame)
        self.training_curves_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.X, expand=False)
        train_ax1 = fig_train_curves.add_subplot(121)
        train_ax1.set_title("Loss")
        train_ax1.set_xlabel("Epoch")
        train_ax1.set_ylabel("Loss")
        train_ax2 = fig_train_curves.add_subplot(122)
        train_ax2.set_title("Accuracy") # Placeholder
        train_ax2.set_xlabel("Epoch")
        train_ax2.set_ylabel("Accuracy")
        fig_train_curves.tight_layout()
        self.training_curves_canvas.draw()


        # --- Testing and Evaluation Frame (Combined) ---
        # This frame should not expand excessively to push buttons off.
        testing_main_frame = ttk.LabelFrame(tab, text="模型测试与评估")
        testing_main_frame.pack(fill=tk.X, expand=False, padx=10, pady=10) # fill=tk.X, expand=False

        # Controls for testing
        test_controls_frame = ttk.Frame(testing_main_frame) # This frame holds buttons
        test_controls_frame.pack(fill=tk.X, pady=2)
        self.load_model_button = ttk.Button(test_controls_frame, text="加载已训练模型", command=self.load_trained_model)
        self.load_model_button.pack(side=tk.LEFT, padx=5, pady=2)

        self.select_adhoc_images_button = ttk.Button(test_controls_frame, text="加载单张/多张图片测试", command=self.select_test_images, state=tk.DISABLED) # Renamed
        self.select_adhoc_images_button.pack(side=tk.LEFT, padx=5, pady=2)

        self.predict_selected_button = ttk.Button(test_controls_frame, text="对选中图片进行预测", command=self.predict_on_selected_images, state=tk.DISABLED) # New button
        self.predict_selected_button.pack(side=tk.LEFT, padx=5, pady=2)

        self.recognize_button = ttk.Button(test_controls_frame, text="运行整体评估(用测试集)", command=self.recognize_images, state=tk.DISABLED) # Clarified name
        self.recognize_button.pack(side=tk.LEFT, padx=5, pady=2)

        # Frame for single image display and prediction
        single_image_results_frame = ttk.Frame(testing_main_frame)
        single_image_results_frame.pack(fill=tk.X, padx=5, pady=5)

        self.single_image_display_label = ttk.Label(single_image_results_frame, text="Selected Image Appears Here")
        self.single_image_display_label.pack(side=tk.LEFT, padx=5, pady=2)
        # Set a fixed size for the image display area or it might resize unpredictably
        # self.single_image_display_label.config(width=200, height=200) # Example, adjust as needed

        self.single_image_prediction_text = scrolledtext.ScrolledText(single_image_results_frame, height=3, width=40, wrap=tk.WORD, state=tk.DISABLED)
        self.single_image_prediction_text.pack(side=tk.LEFT, padx=5, pady=2, fill=tk.BOTH, expand=True)


        # Test results text display (for overall evaluation)
        self.test_results_display = scrolledtext.ScrolledText(testing_main_frame, height=4, wrap=tk.WORD, state=tk.DISABLED)
        self.test_results_display.pack(expand=False, fill=tk.X, padx=5, pady=5)

        # --- Evaluation Metrics Plot Frame ---
        eval_metrics_plot_frame = ttk.LabelFrame(testing_main_frame, text="Evaluation Metric Charts") # English
        eval_metrics_plot_frame.pack(fill=tk.X, expand=False, padx=5, pady=5)

        fig_eval_metrics = Figure(figsize=(7, 2.5), dpi=75) # Aggressively Reduced size & DPI
        self.evaluation_metrics_canvas = FigureCanvasTkAgg(fig_eval_metrics, master=eval_metrics_plot_frame)
        self.evaluation_metrics_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.X, expand=False)

        eval_ax1 = fig_eval_metrics.add_subplot(131)
        eval_ax1.set_title("CM")
        eval_ax2 = fig_eval_metrics.add_subplot(132)
        eval_ax2.set_title("ROC") # Placeholder
        eval_ax3 = fig_eval_metrics.add_subplot(133)
        eval_ax3.set_title("PR") # Placeholder
        fig_eval_metrics.tight_layout()
        self.evaluation_metrics_canvas.draw()

    def browse_and_load_dataset(self):
        path = filedialog.askdirectory()
        if path:
            self.dataset_path_var.set(path)
            self.log_message(f"Selected dataset path: {path}")
            try:
                info = app_data_loader.load_dataset_info(path)

                info_text = f"Path: {path}\n"
                data_load_mode = info.get("data_load_mode", "N/A")
                # Log the full info dictionary for debugging from GUI log panel
                self.log_message(f"Data loader info: {info}")

                info_text = f"Path: {path}\n"
                data_load_mode = info.get("data_load_mode", "N/A")
                info_text += f"Data Loading Mode: {data_load_mode}\n"
                info_text += f"Classes: {info.get('classes', 'N/A')}\n"

                if data_load_mode == "auto_split":
                    info_text += f"Total Samples (to be auto-split): {info.get('train_samples', 0)}\n"
                    info_text += f"Distribution (total): {info.get('train_distribution', 'N/A')}\n"
                    info_text += "Data will be automatically split into Train/Validation/Test sets.\n"
                elif data_load_mode == "standard_split":
                    info_text += f"Train Samples: {info.get('train_samples', 0)} ({info.get('train_distribution', 'N/A')})\n"
                    info_text += f"Validation Samples: {info.get('val_samples', 0)} ({info.get('val_distribution', 'N/A')})\n"
                    info_text += f"Test Samples: {info.get('test_samples', 0)} ({info.get('test_distribution', 'N/A')})\n"
                # No specific message for "unsupported" here, as it's handled by allow_training check and issues list

                issues = info.get('issues', [])
                if issues:
                    info_text += "\nIssues Found:\n" + "\n".join(issues)

                self.dataset_info_label.config(text=info_text)

                allow_training = data_load_mode in ["standard_split", "auto_split"] and info.get('classes') and len(info.get('classes', [])) > 0

                if allow_training:
                    self.class_names = info['classes']
                    self.num_classes = len(self.class_names)
                    self.train_button.config(state=tk.NORMAL)
                    self.log_message(f"Dataset information loaded. Mode: {data_load_mode}. Classes: {self.class_names}. Num classes: {self.num_classes}. Training enabled.")
                    self.display_network_architecture()
                else:
                    self.class_names = []
                    self.num_classes = 0
                    self.train_button.config(state=tk.DISABLED)

                    error_summary = "Training disabled. Reasons:\n"
                    if not info.get('classes') or len(info.get('classes', [])) == 0:
                        error_summary += "- No classes found or classes could not be determined.\n"
                    if data_load_mode not in ["standard_split", "auto_split"]:
                        error_summary += f"- Data structure is unsupported (mode: '{data_load_mode}').\n"
                    if issues:
                        error_summary += "\nSpecific issues from data loader:\n" + "\n".join([f"  - {issue}" for issue in issues])

                    self.log_message(error_summary.replace("\n", " ")) # Log a one-liner version
                    messagebox.showerror("Dataset Error", error_summary + "\nPlease check dataset structure and logs for details.")

            except Exception as e:
                messagebox.showerror("Dataset Processing Error", f"Failed to process dataset: {e}")
                self.log_message(f"Error processing dataset: {e}")
                self.log_message(f"处理数据集时出错: {e}") # Changed
                self.train_button.config(state=tk.DISABLED)

    def display_network_architecture(self, event=None):
        model_choice = self.model_var.get()
        num_display_classes = self.num_classes if self.num_classes > 0 else 2 # Default to 2 if dataset not loaded

        temp_model_instance = None
        if model_choice == "ResNet18_Baseline":
            temp_model_instance = app_resnet_builder.resnet18(num_classes=num_display_classes, pretrained=False)
        elif model_choice == "ResNet34_Baseline":
            temp_model_instance = app_resnet_builder.resnet34(num_classes=num_display_classes, pretrained=False)
        elif model_choice == "ResNet_Improved":
            # Placeholder: When improved model exists, instantiate it here
            # For now, just show a message or generic summary
            pass

        summary = app_resnet_builder.get_resnet_architecture_summary(
            model_name=model_choice,
            num_classes=num_display_classes,
            model_instance=temp_model_instance
        )

        self.arch_display.config(state=tk.NORMAL)
        self.arch_display.delete(1.0, tk.END)
        self.arch_display.insert(tk.END, summary)
        self.arch_display.config(state=tk.DISABLED)
        self.log_message(f"Displaying architecture for {model_choice} with {num_display_classes} classes.")


    def train_model(self):
        dataset_path = self.dataset_path_var.get()
        if not dataset_path or self.num_classes == 0:
            messagebox.showerror("Error", "Please select a valid dataset and ensure classes are loaded.")
            return

        try:
            self.log_progress("Preparing DataLoaders...")
            # Use actual create_dataloaders
            batch_size = self.batch_size_var.get()
            # Consider adding num_workers to GUI or as a config
            self.train_loader, self.val_loader, self.test_loader_proper, _, self.dataset_sizes = \
                app_data_loader.create_dataloaders(dataset_path, batch_size=batch_size, val_split=0.15, num_workers=0) # num_workers=0 for GUI stability in some cases

            if not self.train_loader:
                messagebox.showerror("数据错误", "创建训练数据加载器失败。请检查数据集路径和结构。") # Changed
                return
            self.log_progress(f"数据加载器已创建。训练批次: {len(self.train_loader)}, 验证批次: {len(self.val_loader) if self.val_loader else '无'}") # Changed

            # Instantiate the selected model
            model_choice = self.model_var.get()
            self.log_progress(f"Instantiating model: {model_choice} for {self.num_classes} classes.")
            if model_choice == "ResNet18_Baseline":
                self.current_model_instance = app_resnet_builder.resnet18(num_classes=self.num_classes, pretrained=True) # Using pretrained for transfer learning
            elif model_choice == "ResNet34_Baseline":
                self.current_model_instance = app_resnet_builder.resnet34(num_classes=self.num_classes, pretrained=True)
            elif model_choice == "ResNet_Improved":
                messagebox.showinfo("未实现", "ResNet_Improved 模型训练尚未实现。") # Changed
                # self.current_model_instance = app_resnet_builder.resnet_improved(...) # When available
                return
            else:
                messagebox.showerror("模型错误", f"选择了未知模型: {model_choice}") # Changed
                return

            self.current_model_instance.to(self.device)
            self.log_progress(f"Model {model_choice} instantiated and moved to {self.device}.")

            # Get training parameters from GUI
            epochs = self.epochs_var.get()
            lr = self.lr_var.get()
            optimizer_name = self.optimizer_var.get()

            self.log_progress(f"Starting training with: Optimizer={optimizer_name}, LR={lr}, Epochs={epochs}, Batch Size={batch_size}")

            # Instantiate the actual ModelTrainer
            trainer = app_trainer.ModelTrainer(
                model=self.current_model_instance,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                device=self.device,
                epochs=epochs,
                lr=lr,
                optimizer_name=optimizer_name
            )

            # Start the actual training process
            # The trainer.start_training method will use self.log_progress for GUI updates
            history = trainer.start_training(self.log_progress)

            if "error" in history:
                self.log_message(f"训练因错误停止: {history['error']}") # Changed
                messagebox.showerror("训练错误", f"训练失败: {history['error']}") # Changed
            else:
                self.log_progress("实际训练过程已完成。") # Changed
                # Log summary of history if needed, e.g., best validation accuracy
                if history['val_acc'] and any(history['val_acc']): # Check if val_acc has any non-zero/non-placeholder entries
                    best_val_acc = max(history['val_acc'])
                    self.log_progress(f"Best validation accuracy during training: {best_val_acc:.4f}")

                # Plot training curves
                if history and not history.get("error"):
                    self.log_progress("Plotting training curves...")
                    app_utils.plot_training_curves(self.training_curves_canvas, history)
                    self.log_progress("Training curves displayed.")
                else:
                    self.log_progress("Skipping training curve plotting due to training error or no history.")


                self.save_model_button.config(state=tk.NORMAL)
                self.recognize_button.config(state=tk.NORMAL)

            self.select_test_images_button.config(state=tk.NORMAL)

        except Exception as e:
            self.log_message(f"训练设置或过程中发生错误: {e}") # Changed
            messagebox.showerror("训练错误", f"发生错误: {e}") # Changed


    def save_model(self):
        if not self.current_model_instance:
            messagebox.showerror("错误", "尚未训练或加载模型。") # Changed
            return
        filepath = filedialog.asksaveasfilename(
            defaultextension=".pth",
            filetypes=[("PyTorch 模型文件", "*.pth"), ("所有文件", "*.*")], # Changed
            title="保存已训练模型" # Changed
        )
        if filepath:
            try:
                torch.save(self.current_model_instance.state_dict(), filepath)
                self.log_message(f"模型已保存至: {filepath}") # Changed
                messagebox.showinfo("保存模型", f"模型成功保存至 {filepath}") # Changed
            except Exception as e:
                self.log_message(f"保存模型时出错: {e}") # Changed
                messagebox.showerror("保存错误", f"无法保存模型: {e}") # Changed


    def load_trained_model(self):
        filepath = filedialog.askopenfilename(
            filetypes=[("PyTorch 模型文件", "*.pth"), ("所有文件", "*.*")], # Changed
            title="加载已训练模型" # Changed
        )
        if filepath:
            try:
                # Determine num_classes. If dataset loaded, use that. Otherwise, prompt or use a default.
                # For simplicity, assume we need to know num_classes for the model architecture.
                # This might require saving model config alongside weights or inferring from filename.
                # For now, if dataset is loaded, use its num_classes.

                num_model_classes = self.num_classes
                if num_model_classes == 0: # If no dataset loaded yet to define num_classes
                     # Basic prompt or default. A real app might need more robust handling.
                    try:
                        # Attempt to get num_classes from a simple prompt
                        num_prompted_classes = simpledialog.askinteger("类别数量", "请输入加载模型的类别数量:", initialvalue=2) # Changed
                        if num_prompted_classes is not None and num_prompted_classes > 0:
                            num_model_classes = num_prompted_classes
                        else: # User cancelled or entered invalid
                            messagebox.showwarning("加载模型", "未能确定类别数量。假设模型结构为2个类别。") # Changed
                            num_model_classes = 2 # Fallback
                    except Exception: # In case simpledialog is not available or other issues
                         messagebox.showwarning("加载模型", "无法确定类别数量。假设模型结构为2个类别。") # Changed
                         num_model_classes = 2 # Fallback

                model_choice = self.model_var.get() # Use current selection in combobox
                self.log_message(f"尝试从 {filepath} 加载具有 {num_model_classes} 个类别的模型 {model_choice}") # Changed

                if model_choice == "ResNet18_Baseline":
                    self.current_model_instance = app_resnet_builder.resnet18(num_classes=num_model_classes, pretrained=False)
                elif model_choice == "ResNet34_Baseline":
                    self.current_model_instance = app_resnet_builder.resnet34(num_classes=num_model_classes, pretrained=False)
                # Add ResNet_Improved here when available
                else:
                    messagebox.showerror("加载错误", f"无法识别用于加载的模型类型 {model_choice}。") # Changed
                    return

                self.current_model_instance.load_state_dict(torch.load(filepath, map_location=self.device))
                self.current_model_instance.to(self.device)
                self.current_model_instance.eval() # Set to evaluation mode

                self.log_message(f"模型已从 {filepath} 加载并设置为评估模式。") # Changed
                messagebox.showinfo("加载模型", f"模型已成功从 {filepath} 加载。") # Changed
                self.recognize_button.config(state=tk.NORMAL)
                self.select_test_images_button.config(state=tk.NORMAL)
                # Update architecture display for the loaded model config
                self.display_network_architecture()

            except Exception as e:
                self.log_message(f"加载模型时出错: {e}") # Changed
                messagebox.showerror("加载错误", f"无法加载模型: {e}") # Changed


    def select_test_images(self): # Now for ad-hoc image selection for single prediction
        filepaths = filedialog.askopenfilenames(
            title="选择单张或多张图片进行即时预测",
            filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp"), ("所有文件", "*.*")]
        )
        if filepaths:
            self.adhoc_test_image_paths = list(filepaths)
            self.log_message(f"Loaded {len(self.adhoc_test_image_paths)} image(s) for single prediction. First: {self.adhoc_test_image_paths[0]}")
            self.predict_selected_button.config(state=tk.NORMAL)

            # Display the first selected image
            try:
                img_path = self.adhoc_test_image_paths[0]
                img = Image.open(img_path)
                # Resize for display label, keeping aspect ratio
                img.thumbnail((200, 200)) # Resize to fit in a 200x200 box
                photo = ImageTk.PhotoImage(img)

                self.single_image_display_label.config(image=photo)
                self.single_image_display_label.image = photo # Keep a reference!
            except Exception as e:
                self.log_message(f"Error displaying selected image: {e}")
                self.single_image_display_label.config(image=None, text="Error displaying image.")
                self.single_image_display_label.image = None

            # Clear previous single prediction text
            self.single_image_prediction_text.config(state=tk.NORMAL)
            self.single_image_prediction_text.delete(1.0, tk.END)
            self.single_image_prediction_text.insert(tk.END, "Image(s) loaded. Click 'Predict on Selected Image(s)'.")
            self.single_image_prediction_text.config(state=tk.DISABLED)
        else:
            self.adhoc_test_image_paths = []
            self.predict_selected_button.config(state=tk.DISABLED)
            self.single_image_display_label.config(image=None, text="No image selected.")
            self.single_image_display_label.image = None
            self.single_image_prediction_text.config(state=tk.NORMAL)
            self.single_image_prediction_text.delete(1.0, tk.END)
            self.single_image_prediction_text.config(state=tk.DISABLED)

    def predict_on_selected_images(self):
        if not self.current_model_instance:
            messagebox.showerror("Error", "No model loaded or trained. Please load or train a model first.")
            self.log_message("Prediction failed: Model not available.")
            return
        if not self.adhoc_test_image_paths:
            messagebox.showinfo("Info", "No images selected for prediction. Please load images first.")
            self.log_message("Prediction skipped: No ad-hoc images loaded.")
            return
        if not self.class_names:
            messagebox.showerror("Error", "Class names not available. Please load a dataset first to define classes.")
            self.log_message("Prediction failed: Class names not defined (load dataset).")
            return

        self.log_message(f"Starting prediction for {len(self.adhoc_test_image_paths)} selected image(s)...")

        # Prepare model and transforms
        self.current_model_instance.eval()
        try:
            test_transforms = app_data_loader.get_data_transforms()['test']
        except Exception as e:
            messagebox.showerror("Error", f"Failed to get data transformations: {e}")
            self.log_message(f"Prediction failed: Could not get test transformations: {e}")
            return

        results_summary = []

        for i, img_path in enumerate(self.adhoc_test_image_paths):
            try:
                image = Image.open(img_path).convert('RGB')

                # Display the current image being processed if it's the first one,
                # or if we enhance UI later to show multiple.
                if i == 0:
                    img_display = image.copy()
                    img_display.thumbnail((200, 200)) # Resize for display label
                    photo = ImageTk.PhotoImage(img_display)
                    self.single_image_display_label.config(image=photo)
                    self.single_image_display_label.image = photo # Keep reference

                input_tensor = test_transforms(image)
                input_batch = input_tensor.unsqueeze(0).to(self.device)

                with torch.no_grad():
                    outputs = self.current_model_instance(input_batch)
                    probabilities = torch.softmax(outputs, dim=1)
                    confidence, predicted_idx = torch.max(probabilities, 1)

                predicted_class = self.class_names[predicted_idx.item()]
                conf_score = confidence.item()

                result_str = f"Image: {os.path.basename(img_path)}\nPredicted: {predicted_class} (Confidence: {conf_score:.4f})"
                results_summary.append(result_str)
                self.log_message(result_str) # Log each prediction to main log

                if i == 0: # Update dedicated single image prediction text area for the first image
                    self.single_image_prediction_text.config(state=tk.NORMAL)
                    self.single_image_prediction_text.delete(1.0, tk.END)
                    self.single_image_prediction_text.insert(tk.END, result_str)
                    self.single_image_prediction_text.config(state=tk.DISABLED)

            except Exception as e:
                err_msg = f"Error predicting on image {img_path}: {e}"
                self.log_message(err_msg)
                results_summary.append(err_msg)
                if i == 0: # Show error for the first image in its dedicated text area
                    self.single_image_prediction_text.config(state=tk.NORMAL)
                    self.single_image_prediction_text.delete(1.0, tk.END)
                    self.single_image_prediction_text.insert(tk.END, err_msg)
                    self.single_image_prediction_text.config(state=tk.DISABLED)

        # Log all results to the main test results display as well
        self.log_test_result("\n--- Single Image(s) Prediction Results ---")
        for res in results_summary:
            self.log_test_result(res)
        self.log_test_result("--- End of Single Image(s) Prediction ---")

        messagebox.showinfo("Prediction Complete", f"Finished predicting on {len(self.adhoc_test_image_paths)} image(s). Results logged.")

    def recognize_images(self): # This is the "Run Evaluation" button for the whole test set
        if not self.current_model_instance:
             messagebox.showerror("错误", "尚未加载或训练模型。") # Changed
             return

        if not self.test_loader_proper and self.dataset_path_var.get():
            # Try to create test_loader if dataset path is available but loader wasn't made (e.g. model loaded separately)
            self.log_message("未找到测试加载器，尝试从数据集路径创建...") # Changed
            try:
                # Need batch_size, could get from GUI or default
                _, _, self.test_loader_proper, _, _ = \
                    app_data_loader.create_dataloaders(self.dataset_path_var.get(),
                                                       batch_size=self.batch_size_var.get(),
                                                       val_split=0) # No val split needed here
                if not self.test_loader_proper:
                    messagebox.showerror("评估错误", "无法从数据集路径创建测试数据加载器。请确保 'test' 子文件夹存在且结构正确。") # Changed
                    return
                self.log_message(f"测试加载器已创建。批次数: {len(self.test_loader_proper)}") # Changed
            except Exception as e:
                messagebox.showerror("评估错误", f"创建测试数据加载器失败: {e}") # Changed
                self.log_message(f"创建测试加载器时出错: {e}") # Changed
                return
        elif not self.test_loader_proper:
            messagebox.showerror("评估错误", "无可用测试数据。请加载包含测试集的数据集。") # Changed
            return

        self.log_test_result("开始在测试集上进行评估...") # Changed

        # Instantiate ModelEvaluator with class_names
        evaluator = app_evaluator.ModelEvaluator(
            self.current_model_instance,
            self.test_loader_proper,
            self.device,
            class_names=self.class_names # Pass class names
        )

        self.log_test_result("正在计算评估指标...") # Calculating evaluation metrics...
        metrics = evaluator.evaluate() # This is the actual call

        self.log_test_result("评估指标计算完成。") # Evaluation metrics calculation complete.

        summary = evaluator.get_metrics_summary(metrics) # get_metrics_summary now uses class_names from metrics
        self.log_test_result(summary)

        # Plot evaluation metrics
        self.log_test_result("正在绘制评估图表...") # Plotting evaluation charts...
        app_utils.plot_evaluation_metrics(self.evaluation_metrics_canvas, metrics)
        self.log_test_result("评估图表已显示。") # Evaluation charts displayed.

        self.log_test_result("\n评估完成。") # Evaluation finished.


    def log_message(self, message):
        print(message)
        self.log_progress(f"[INFO] {message}")

    def log_progress(self, message):
        self.progress_display.config(state=tk.NORMAL)
        self.progress_display.insert(tk.END, message + "\n")
        self.progress_display.see(tk.END)
        self.progress_display.config(state=tk.DISABLED)

    def log_test_result(self, message):
        self.test_results_display.config(state=tk.NORMAL)
        self.test_results_display.insert(tk.END, message + "\n")
        self.test_results_display.see(tk.END)
        self.test_results_display.config(state=tk.DISABLED)

if __name__ == '__main__':
    from tkinter import simpledialog # Add this for the load_model prompt, ensure it's handled if not available
    root = tk.Tk()
    app = CVApp(root)
    root.mainloop()
