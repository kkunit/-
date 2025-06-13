# Required dependencies:
# pip install torch torchvision torchaudio
# pip install medmnist
# pip install matplotlib
# pip install numpy
# pip install scikit-learn (MedMNIST evaluator might use this for metrics like AUC)

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import medmnist
from medmnist import PneumoniaMNIST # Specifically import PneumoniaMNIST
import matplotlib.pyplot as plt
import numpy as np


# Function to display images
def show_images(images, labels, title):
    plt.figure(figsize=(10, 10))
    for i in range(min(25, len(images))):  # Display up to 25 images
        plt.subplot(5, 5, i + 1)
        # MedMNIST images are single channel, squeeze if necessary and convert to uint8
        img = images[i].squeeze()
        if img.is_floating_point(): # Check if tensor is float
            img = (img * 255).byte() # Scale to 0-255 and convert to byte
        plt.imshow(img.numpy(), cmap='gray')
        plt.title(f"Label: {labels[i].item()}")
        plt.axis('off')
    plt.suptitle(title)
    plt.show()


if __name__ == '__main__':
    print("Script execution started.") # <<< New print
    # Define transformations
    # For custom AlexNet expecting 1-channel 28x28 images
    data_transform = transforms.Compose([
        transforms.Resize((28, 28)),
        # PneumoniaMNIST is already grayscale, ToTensor will handle channel dimension
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    # Download and load the PneumoniaMNIST dataset
    # Specify the MedMNIST data root directory, e.g. './medmnist_data/'
    # If the directory does not exist, it will be created automatically
    # and the dataset will be downloaded into it.
    data_root = './medmnist_data/'

    # Ensure the data_root directory exists
    import os
    os.makedirs(data_root, exist_ok=True)

    train_dataset = PneumoniaMNIST(split='train', transform=data_transform, download=True, root=data_root)
    val_dataset_full = PneumoniaMNIST(split='val', transform=data_transform, download=True, root=data_root)
    test_dataset_full = PneumoniaMNIST(split='test', transform=data_transform, download=True, root=data_root)

    # Reduce dataset size for faster execution in sandbox
    # NOTE: The dataset is currently using a subset for quick testing.
    # For full training, use the complete dataset:
    # train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
    # val_loader = DataLoader(dataset=val_dataset_full, batch_size=32, shuffle=False)
    # test_loader = DataLoader(dataset=test_dataset_full, batch_size=32, shuffle=False)
    # Or adjust subset_fraction to 1.0
    subset_fraction = 0.05 # Further reduced subset_fraction
    train_subset_indices = np.random.choice(len(train_dataset), int(len(train_dataset) * subset_fraction), replace=False)
    val_subset_indices = np.random.choice(len(val_dataset_full), int(len(val_dataset_full) * subset_fraction), replace=False)
    test_subset_indices = np.random.choice(len(test_dataset_full), int(len(test_dataset_full) * subset_fraction), replace=False)

    train_dataset_subset = torch.utils.data.Subset(train_dataset, train_subset_indices)
    val_dataset_subset = torch.utils.data.Subset(val_dataset_full, val_subset_indices)
    test_dataset_subset = torch.utils.data.Subset(test_dataset_full, test_subset_indices)

    # Create DataLoaders with subsets
    train_loader = DataLoader(dataset=train_dataset_subset, batch_size=32, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset_subset, batch_size=32, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset_subset, batch_size=32, shuffle=False)

    print(f"Using {subset_fraction*100}% of the data for speed.")
    print(f"Train dataset samples: {len(train_dataset_subset)} (original: {len(train_dataset)})")
    print(f"Validation dataset samples: {len(val_dataset_subset)} (original: {len(val_dataset_full)})")
    print(f"Test dataset samples: {len(test_dataset_subset)} (original: {len(test_dataset_full)})")
    print(f"MedMNIST info: {train_dataset.info}")

    # Display some images from the training set
    # Get a batch of training data
    try:
        images, labels = next(iter(train_loader))
        # Make sure labels are suitable for display (e.g., not one-hot encoded if not needed for title)
        if labels.ndim > 1 and labels.shape[1] > 1: # Basic check for one-hot encoding
             _, labels_for_display = torch.max(labels, 1) # Convert one-hot to class indices
        else:
             labels_for_display = labels.squeeze() # Squeeze if it's [batch_size, 1]

        # Attempt to show images, but handle display errors gracefully in environments without GUI
        show_images(images, labels_for_display, "Sample Training Images")
        print("Sample images displayed (if GUI is available).")
    except Exception as e:
        print(f"Could not display images (matplotlib error or no GUI): {e}")
        print("Continuing without displaying images.")

    # Define the model, optimizer, and loss function
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define AlexNet model from scratch
    class AlexNet(nn.Module):
        def __init__(self, num_classes=1): # Adjusted for binary classification with BCEWithLogitsLoss
            super(AlexNet, self).__init__()
            self.features = nn.Sequential(
                # Input: 28x28 grayscale
                # Layer 1
                nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1), # Output: 28x28x32
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2), # Output: 14x14x32

                # Layer 2
                nn.Conv2d(32, 64, kernel_size=3, padding=1), # Output: 14x14x64
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2), # Output: 7x7x64

                # Layer 3
                nn.Conv2d(64, 128, kernel_size=3, padding=1), # Output: 7x7x128
                nn.ReLU(inplace=True),

                # Layer 4
                nn.Conv2d(128, 256, kernel_size=3, padding=1), # Output: 7x7x256
                nn.ReLU(inplace=True),

                # Layer 5
                nn.Conv2d(256, 256, kernel_size=3, padding=1), # Output: 7x7x256
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2), # Output: 3x3x256
            )
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 3 * 3, 1024),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(1024, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, num_classes),
            )

        def forward(self, x):
            x = self.features(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x

    model = AlexNet(num_classes=1).to(device)

    criterion = nn.BCEWithLogitsLoss() # Suitable for binary classification
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training function
    def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0

            for i, (inputs, labels) in enumerate(train_loader):
                # labels from PneumoniaMNIST are expected to be [batch_size, 1] already.
                # Ensure they are float for BCEWithLogitsLoss.
                inputs, labels = inputs.to(device), labels.to(device).float()

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                # For BCEWithLogitsLoss, output > 0 means class 1, else class 0
                predicted = (outputs > 0).float()
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

            epoch_train_loss = running_loss / len(train_loader.dataset)
            epoch_train_acc = correct_train / total_train
            train_losses.append(epoch_train_loss)
            train_accuracies.append(epoch_train_acc)

            model.eval()
            running_val_loss = 0.0
            correct_val = 0
            total_val = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device).float()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    running_val_loss += loss.item() * inputs.size(0)
                    predicted = (outputs > 0).float()
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()

            epoch_val_loss = running_val_loss / len(val_loader.dataset)
            epoch_val_acc = correct_val / total_val
            val_losses.append(epoch_val_loss)
            val_accuracies.append(epoch_val_acc)

            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} - "
                  f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")

        return train_losses, val_losses, train_accuracies, val_accuracies

    # Evaluation function
    def evaluate_model(model, test_loader, criterion):
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device).float()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)

                predicted = (outputs > 0).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        test_loss = running_loss / len(test_loader.dataset)
        test_acc = correct / total
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

        # Calculate AUC using MedMNIST's evaluator
        evaluator = medmnist.Evaluator('pneumoniamnist', 'test')
        auc = evaluator.evaluate({'y_true': np.array(all_labels).squeeze(), 'y_pred': np.array(all_preds).squeeze()})
        print(f"Test AUC: {auc:.4f}")

        return test_loss, test_acc, auc

    # Start training
    # NOTE: num_epochs is set low for quick testing.
    # Increase for full training (e.g., num_epochs = 20 or more).
    num_epochs = 1 # Further reduced num_epochs
    print(f"\nStarting training for {num_epochs} epochs...")
    train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model, train_loader, val_loader, criterion, optimizer, num_epochs=num_epochs
    )

    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss over Epochs')

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy over Epochs')

    # Save the figure before trying to show it
    try:
        plt.savefig('training_history_plots.png')
        print("Training history plots saved to 'training_history_plots.png'")
    except Exception as e:
        print(f"Could not save training plots: {e}")

    try:
        plt.show()
        print("Training history plots displayed (if GUI is available).")
    except Exception as e:
        print(f"Could not display training plots (matplotlib error or no GUI): {e}")
        print("Continuing without displaying plots.")


    # Evaluate the model on the test set
    print("\nEvaluating on the test set...")
    test_loss, test_acc, test_auc = evaluate_model(model, test_loader, criterion)
    print("Script execution completed.") # <<< New print
