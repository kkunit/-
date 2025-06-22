import torch
import torch.optim as optim
import torch.nn as nn
import time # For simulating work and timing epochs

class ModelTrainer:
    def __init__(self, model, train_loader, val_loader, device, epochs, lr, optimizer_name):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader # This can be None if no validation set is used
        self.device = device
        self.epochs = epochs
        self.lr = lr
        self.optimizer_name = optimizer_name

        # Initialize loss function
        self.criterion = nn.CrossEntropyLoss()

        # Initialize optimizer
        if self.model: # Ensure model is provided before setting up optimizer
            if optimizer_name.lower() == "adam":
                self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
            elif optimizer_name.lower() == "sgd":
                self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
            else:
                # Default to Adam if optimizer_name is unknown or add error handling
                print(f"Warning: Optimizer '{optimizer_name}' not recognized. Defaulting to Adam.")
                self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        else:
            # This case is for when the dummy_trainer was called in app.py
            # In a real scenario, the model should always be present.
            self.optimizer = None
            print("Warning: ModelTrainer initialized without a model. Optimizer not set.")


    def train_epoch(self):
        self.model.train()  # Set model to training mode
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        if not self.train_loader:
            print("Error: Train loader not available for train_epoch.")
            return 0.0, 0.0 # Or raise an error

        for inputs, labels in self.train_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # Zero the parameter gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels.data)
            total_samples += labels.size(0) # Use labels.size(0) for robustness

        if total_samples == 0: # Avoid division by zero if train_loader is empty
            epoch_loss = 0.0
            epoch_acc = 0.0
        else:
            epoch_loss = running_loss / total_samples
            epoch_acc = correct_predictions.double() / total_samples

        return epoch_loss, epoch_acc.item()

    def validate_epoch(self):
        self.model.eval()  # Set model to evaluation mode
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        if not self.val_loader:
            print("Warning: Validation loader not available. Skipping validation epoch.")
            return 0.0, 0.0 # Or some indicator that validation was skipped

        with torch.no_grad():  # Disable gradient calculations
            for inputs, labels in self.val_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct_predictions += torch.sum(preds == labels.data)
                total_samples += labels.size(0)

        if total_samples == 0:
            epoch_loss = 0.0
            epoch_acc = 0.0
        else:
            epoch_loss = running_loss / total_samples
            epoch_acc = correct_predictions.double() / total_samples

        return epoch_loss, epoch_acc.item()

    def start_training(self, progress_callback_gui):
        if not self.model or not self.optimizer or not self.train_loader:
            message = "Model, Optimizer, or Train Loader not properly initialized. Cannot start training."
            if progress_callback_gui:
                progress_callback_gui(f"[ERROR] {message}")
            print(f"[ERROR] {message}")
            return {"error": message, 'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

        if progress_callback_gui:
            progress_callback_gui(f"Starting training for {self.epochs} epochs on device: {self.device}")

        for epoch in range(self.epochs):
            epoch_start_time = time.time()

            # Train for one epoch
            train_loss, train_acc = self.train_epoch()

            # Validate for one epoch
            if self.val_loader:
                val_loss, val_acc = self.validate_epoch()
            else:
                val_loss, val_acc = 0.0, 0.0 # Or use None or specific flags

            epoch_duration = time.time() - epoch_start_time

            log_message = f"Epoch {epoch+1}/{self.epochs} | " \
                          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            if self.val_loader:
                log_message += f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
            log_message += f"Duration: {epoch_duration:.2f}s"

            if progress_callback_gui:
                progress_callback_gui(log_message)
            else: # Fallback print if no GUI callback
                print(log_message)

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            if self.val_loader:
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
            else: # Append placeholders if no validation
                history['val_loss'].append(0.0) # Or float('nan') or None
                history['val_acc'].append(0.0)

        final_message = "Training finished."
        if progress_callback_gui:
            progress_callback_gui(final_message)
        else:
            print(final_message)

        return history
