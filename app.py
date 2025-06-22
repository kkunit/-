import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
from tkinter import messagebox
import torch # For device detection

# Import modules from src
from src import utils as app_utils
from src import data_loader as app_data_loader
from src import trainer as app_trainer
from src import evaluator as app_evaluator
from src import resnet_builder as app_resnet_builder # Import the actual ResNet builder

class CVApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Classification Tool")
        self.root.geometry("1000x850") # Slightly increased height for logs

        # Model related attributes
        self.current_model_instance = None # Will hold the instantiated model
        self.train_loader = None
        self.val_loader = None
        self.test_loader_proper = None
        self.class_names = []
        self.num_classes = 0
        self.dataset_sizes = {}

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
        top_frame = ttk.LabelFrame(tab, text="Setup")
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        ttk.Label(top_frame, text="Dataset Path:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.dataset_path_var = tk.StringVar()
        self.dataset_path_entry = ttk.Entry(top_frame, textvariable=self.dataset_path_var, width=50)
        self.dataset_path_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
        self.browse_dataset_button = ttk.Button(top_frame, text="Browse...", command=self.browse_and_load_dataset)
        self.browse_dataset_button.grid(row=0, column=2, padx=5, pady=5)

        self.dataset_info_label = ttk.Label(top_frame, text="Dataset info: Not loaded", wraplength=300, justify=tk.LEFT)
        self.dataset_info_label.grid(row=0, column=3, rowspan=2, padx=10, pady=5, sticky=tk.NW)

        ttk.Label(top_frame, text="Select Model:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.model_var = tk.StringVar()
        # For now, ResNet_Baseline will map to resnet18. ResNet_Improved is a placeholder.
        self.model_options = ["ResNet18_Baseline", "ResNet34_Baseline", "ResNet_Improved"]
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

        arch_frame = ttk.LabelFrame(middle_frame, text="Network Architecture")
        arch_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.arch_display = scrolledtext.ScrolledText(arch_frame, height=15, width=45, wrap=tk.WORD, state=tk.DISABLED)
        self.arch_display.pack(expand=True, fill=tk.BOTH, padx=5, pady=5)

        params_frame = ttk.LabelFrame(middle_frame, text="Training Parameters")
        params_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)

        ttk.Label(params_frame, text="Optimizer:").grid(row=0, column=0, padx=5, pady=3, sticky=tk.W)
        self.optimizer_var = tk.StringVar(value="Adam")
        self.optimizer_dropdown = ttk.Combobox(params_frame, textvariable=self.optimizer_var,
                                               values=["Adam", "SGD"], state="readonly", width=12)
        self.optimizer_dropdown.grid(row=0, column=1, padx=5, pady=3, sticky=tk.EW)

        ttk.Label(params_frame, text="Learning Rate:").grid(row=1, column=0, padx=5, pady=3, sticky=tk.W)
        self.lr_var = tk.DoubleVar(value=0.001)
        self.lr_entry = ttk.Entry(params_frame, textvariable=self.lr_var, width=15)
        self.lr_entry.grid(row=1, column=1, padx=5, pady=3, sticky=tk.EW)

        ttk.Label(params_frame, text="Batch Size:").grid(row=2, column=0, padx=5, pady=3, sticky=tk.W)
        self.batch_size_var = tk.IntVar(value=32)
        self.batch_size_entry = ttk.Entry(params_frame, textvariable=self.batch_size_var, width=15)
        self.batch_size_entry.grid(row=2, column=1, padx=5, pady=3, sticky=tk.EW)

        ttk.Label(params_frame, text="Epochs:").grid(row=3, column=0, padx=5, pady=3, sticky=tk.W)
        self.epochs_var = tk.IntVar(value=10)
        self.epochs_entry = ttk.Entry(params_frame, textvariable=self.epochs_var, width=15)
        self.epochs_entry.grid(row=3, column=1, padx=5, pady=3, sticky=tk.EW)

        self.train_button = ttk.Button(params_frame, text="Train Model", command=self.train_model, state=tk.DISABLED)
        self.train_button.grid(row=4, column=0, columnspan=2, padx=5, pady=10, sticky=tk.EW)

        self.save_model_button = ttk.Button(params_frame, text="Save Trained Model", command=self.save_model, state=tk.DISABLED)
        self.save_model_button.grid(row=5, column=0, columnspan=2, padx=5, pady=5, sticky=tk.EW)

        progress_frame = ttk.LabelFrame(tab, text="Training Progress & Logs")
        progress_frame.pack(fill=tk.X, padx=10, pady=10)
        self.progress_display = scrolledtext.ScrolledText(progress_frame, height=10, wrap=tk.WORD, state=tk.DISABLED)
        self.progress_display.pack(expand=True, fill=tk.X, padx=5, pady=5)

        testing_frame = ttk.LabelFrame(tab, text="Model Testing & Evaluation")
        testing_frame.pack(fill=tk.X, padx=10, pady=10)
        test_controls_frame = ttk.Frame(testing_frame)
        test_controls_frame.pack(fill=tk.X, pady=5)
        self.load_model_button = ttk.Button(test_controls_frame, text="Load Trained Model", command=self.load_trained_model)
        self.load_model_button.pack(side=tk.LEFT, padx=5)
        self.select_test_images_button = ttk.Button(test_controls_frame, text="Select Test Image(s)", command=self.select_test_images, state=tk.DISABLED)
        self.select_test_images_button.pack(side=tk.LEFT, padx=5)
        self.recognize_button = ttk.Button(test_controls_frame, text="Run Evaluation", command=self.recognize_images, state=tk.DISABLED)
        self.recognize_button.pack(side=tk.LEFT, padx=5)
        self.test_results_display = scrolledtext.ScrolledText(testing_frame, height=12, wrap=tk.WORD, state=tk.DISABLED)
        self.test_results_display.pack(expand=True, fill=tk.X, padx=5, pady=5)

    def browse_and_load_dataset(self):
        path = filedialog.askdirectory()
        if path:
            self.dataset_path_var.set(path)
            self.log_message(f"Selected dataset path: {path}")
            try:
                info = app_data_loader.load_dataset_info(path)

                info_text = f"Path: {path}\n"
                info_text += f"Classes: {info.get('classes', 'N/A')}\n"
                info_text += f"Train: {info.get('train_samples', 0)} ({info.get('train_distribution', 'N/A')})\n"
                info_text += f"Val: {info.get('val_samples', 0)} ({info.get('val_distribution', 'N/A')})\n"
                info_text += f"Test: {info.get('test_samples', 0)} ({info.get('test_distribution', 'N/A')})\n"

                if info.get('issues'):
                    info_text += "\nIssues Found:\n" + "\n".join(info['issues'])

                self.dataset_info_label.config(text=info_text)

                if not info.get('issues') and info.get('classes'):
                    self.class_names = info['classes']
                    self.num_classes = len(self.class_names)
                    if self.num_classes > 0 :
                        self.train_button.config(state=tk.NORMAL)
                        self.log_message(f"Dataset loaded. Classes: {self.class_names}. Num classes: {self.num_classes}")
                        # Automatically update architecture display for new num_classes
                        self.display_network_architecture()
                    else:
                        messagebox.showwarning("Dataset Warning", "No classes found in the dataset. Training disabled.")
                        self.train_button.config(state=tk.DISABLED)
                else:
                    messagebox.showerror("Dataset Error", "Could not properly load dataset info or no classes found. Check logs.")
                    self.train_button.config(state=tk.DISABLED)

            except Exception as e:
                messagebox.showerror("Dataset Error", f"Failed to process dataset: {e}")
                self.log_message(f"Error processing dataset: {e}")
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
                messagebox.showerror("Data Error", "Failed to create training data loader. Check dataset path and structure.")
                return
            self.log_progress(f"DataLoaders created. Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader) if self.val_loader else 'N/A'}")

            # Instantiate the selected model
            model_choice = self.model_var.get()
            self.log_progress(f"Instantiating model: {model_choice} for {self.num_classes} classes.")
            if model_choice == "ResNet18_Baseline":
                self.current_model_instance = app_resnet_builder.resnet18(num_classes=self.num_classes, pretrained=True) # Using pretrained for transfer learning
            elif model_choice == "ResNet34_Baseline":
                self.current_model_instance = app_resnet_builder.resnet34(num_classes=self.num_classes, pretrained=True)
            elif model_choice == "ResNet_Improved":
                messagebox.showinfo("Not Implemented", "ResNet_Improved model training is not yet implemented.")
                # self.current_model_instance = app_resnet_builder.resnet_improved(...) # When available
                return
            else:
                messagebox.showerror("Model Error", f"Unknown model selected: {model_choice}")
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
                self.log_message(f"Training stopped due to an error: {history['error']}")
                messagebox.showerror("Training Error", f"Training failed: {history['error']}")
            else:
                self.log_progress("Actual training process finished.")
                # Log summary of history if needed, e.g., best validation accuracy
                if history['val_acc'] and any(history['val_acc']): # Check if val_acc has any non-zero/non-placeholder entries
                    best_val_acc = max(history['val_acc'])
                    self.log_progress(f"Best validation accuracy during training: {best_val_acc:.4f}")

                self.save_model_button.config(state=tk.NORMAL)
                self.recognize_button.config(state=tk.NORMAL)

            self.select_test_images_button.config(state=tk.NORMAL)

        except Exception as e:
            self.log_message(f"ERROR during training setup or process: {e}")
            messagebox.showerror("Training Error", f"An error occurred: {e}")


    def save_model(self):
        if not self.current_model_instance:
            messagebox.showerror("Error", "No model has been trained or loaded yet.")
            return
        filepath = filedialog.asksaveasfilename(
            defaultextension=".pth",
            filetypes=[("PyTorch Model Files", "*.pth"), ("All Files", "*.*")],
            title="Save Trained Model"
        )
        if filepath:
            try:
                torch.save(self.current_model_instance.state_dict(), filepath)
                self.log_message(f"Model saved to: {filepath}")
                messagebox.showinfo("Save Model", f"Model successfully saved to {filepath}")
            except Exception as e:
                self.log_message(f"Error saving model: {e}")
                messagebox.showerror("Save Error", f"Could not save model: {e}")


    def load_trained_model(self):
        filepath = filedialog.askopenfilename(
            filetypes=[("PyTorch Model Files", "*.pth"), ("All Files", "*.*")],
            title="Load Trained Model"
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
                        num_prompted_classes = simpledialog.askinteger("Number of Classes", "Enter number of classes for the loaded model:", initialvalue=2)
                        if num_prompted_classes is not None and num_prompted_classes > 0:
                            num_model_classes = num_prompted_classes
                        else: # User cancelled or entered invalid
                            messagebox.showwarning("Load Model", "Number of classes not determined. Assuming 2 classes for model structure.")
                            num_model_classes = 2 # Fallback
                    except Exception: # In case simpledialog is not available or other issues
                         messagebox.showwarning("Load Model", "Could not determine number of classes. Assuming 2 classes for model structure.")
                         num_model_classes = 2 # Fallback

                model_choice = self.model_var.get() # Use current selection in combobox
                self.log_message(f"Attempting to load model {model_choice} with {num_model_classes} classes from {filepath}")

                if model_choice == "ResNet18_Baseline":
                    self.current_model_instance = app_resnet_builder.resnet18(num_classes=num_model_classes, pretrained=False)
                elif model_choice == "ResNet34_Baseline":
                    self.current_model_instance = app_resnet_builder.resnet34(num_classes=num_model_classes, pretrained=False)
                # Add ResNet_Improved here when available
                else:
                    messagebox.showerror("Load Error", f"Model type {model_choice} not recognized for loading.")
                    return

                self.current_model_instance.load_state_dict(torch.load(filepath, map_location=self.device))
                self.current_model_instance.to(self.device)
                self.current_model_instance.eval() # Set to evaluation mode

                self.log_message(f"Model loaded from: {filepath} and set to evaluation mode.")
                messagebox.showinfo("Load Model", f"Model loaded successfully from {filepath}")
                self.recognize_button.config(state=tk.NORMAL)
                self.select_test_images_button.config(state=tk.NORMAL)
                # Update architecture display for the loaded model config
                self.display_network_architecture()

            except Exception as e:
                self.log_message(f"Error loading model: {e}")
                messagebox.showerror("Load Error", f"Could not load model: {e}")


    def select_test_images(self):
        # This function is more for selecting individual images for ad-hoc testing.
        # The main evaluation will use the test set from the loaded dataset path.
        filepaths = filedialog.askopenfilenames(
            title="Select Test Image(s)",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp"), ("All Files", "*.*")]
        )
        if filepaths:
            self.log_test_result(f"Selected {len(filepaths)} image(s) for ad-hoc testing (simulated): {filepaths[0]}...")
            # Store these filepaths if you plan to implement single/multi-image prediction display
            # self.adhoc_test_image_paths = filepaths
            messagebox.showinfo("Images Selected", f"{len(filepaths)} images selected. Ad-hoc prediction for these is not fully implemented in this step, evaluation uses the test set.")


    def recognize_images(self): # This is the "Run Evaluation" button
        if not self.current_model_instance:
             messagebox.showerror("Error", "No model loaded or trained.")
             return

        if not self.test_loader_proper and self.dataset_path_var.get():
            # Try to create test_loader if dataset path is available but loader wasn't made (e.g. model loaded separately)
            self.log_message("Test loader not found, attempting to create from dataset path...")
            try:
                # Need batch_size, could get from GUI or default
                _, _, self.test_loader_proper, _, _ = \
                    app_data_loader.create_dataloaders(self.dataset_path_var.get(),
                                                       batch_size=self.batch_size_var.get(),
                                                       val_split=0) # No val split needed here
                if not self.test_loader_proper:
                    messagebox.showerror("Evaluation Error", "Could not create test data loader from the dataset path. Please ensure the 'test' subfolder exists and is structured correctly.")
                    return
                self.log_message(f"Test loader created. Batches: {len(self.test_loader_proper)}")
            except Exception as e:
                messagebox.showerror("Evaluation Error", f"Failed to create test data loader: {e}")
                self.log_message(f"Error creating test loader: {e}")
                return
        elif not self.test_loader_proper:
            messagebox.showerror("Evaluation Error", "No test data available. Please load a dataset with a test set.")
            return

        self.log_test_result("Starting evaluation on the test set...")

        # Actual evaluation (using placeholders for now, will be fully implemented later)
        evaluator = app_evaluator.ModelEvaluator(self.current_model_instance, self.test_loader_proper, self.device)
        # metrics = evaluator.evaluate() # This will be the actual call

        # Using dummy metrics for now from evaluator.py
        metrics = evaluator.dummy_metrics()
        summary = evaluator.get_metrics_summary(metrics)
        self.log_test_result(summary)

        # Placeholder for plotting curves - will require matplotlib canvas integration
        # app_utils.plot_performance_metrics_on_canvas(self.test_figure_canvas, metrics)
        self.log_test_result("\nEvaluation finished (using simulated metrics for now).")
        self.log_test_result("Actual metrics and curve plotting will be implemented in later steps.")


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
