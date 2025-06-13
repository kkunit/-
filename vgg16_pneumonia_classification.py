import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from medmnist import PneumoniaMNIST
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Data transformations
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # VGG16 expects 224x224 input
    transforms.Grayscale(num_output_channels=3), # Convert to 3 channels if grayscale, VGG16 needs 3 channels
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Imagenet normalization
])

# Download and load the datasets
BATCH_SIZE = 32 # Can be adjusted based on system VRAM

print("Downloading MedMNIST Pneumonia dataset...")
try:
    train_dataset = PneumoniaMNIST(split='train', transform=data_transform, download=True)
    val_dataset = PneumoniaMNIST(split='val', transform=data_transform, download=True)
    test_dataset = PneumoniaMNIST(split='test', transform=data_transform, download=True)
except Exception as e:
    print(f"Error downloading or loading MedMNIST dataset: {e}")
    print("Please check your internet connection and if the MedMNIST servers are accessible.")
    # Depending on the execution environment, you might want to sys.exit() here
    # For now, we'll let it proceed, but DataLoader creation will fail.
    train_dataset, val_dataset, test_dataset = None, None, None


if train_dataset:
    # Create DataLoaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Training dataset: {len(train_dataset)} samples")
    print(f"Validation dataset: {len(val_dataset)} samples")
    print(f"Test dataset: {len(test_dataset)} samples")
    print(f"Number of batches in train_loader: {len(train_loader)}")
    print(f"Number of batches in val_loader: {len(val_loader)}")
    print(f"Number of batches in test_loader: {len(test_loader)}")

    # Example: Get some sample images and labels (optional, for verification)
    # try:
    #     sample_images, sample_labels = next(iter(train_loader))
    #     print(f"Sample batch images shape: {sample_images.shape}") # Should be [BATCH_SIZE, 3, 224, 224]
    #     print(f"Sample batch labels shape: {sample_labels.shape}") # Should be [BATCH_SIZE, 1] for PneumoniaMNIST
    #     print(f"Sample labels: {sample_labels.squeeze().tolist()}")
    # except StopIteration:
    #     print("Could not retrieve a sample batch, train_loader might be empty.")
    # except Exception as e:
    #     print(f"Error retrieving sample batch: {e}")

else:
    print("Dataset loading failed. Cannot create DataLoaders.")
    # Ensure loaders are None so subsequent code doesn't try to use them
    train_loader, val_loader, test_loader = None, None, None


# --- Model Definition ---
# Load pre-trained VGG16 model
from torchvision import models

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load VGG16 with pre-trained weights
model_vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

# Freeze parameters of the pre-trained layers
for param in model_vgg16.features.parameters():
    param.requires_grad = False

# PneumoniaMNIST is a binary classification task (Pneumonia vs. Normal)
# The output of PneumoniaMNIST is a single value (0 or 1)
# We need to replace the VGG16 classifier with a new one appropriate for this.
# VGG16's classifier:
# (classifier): Sequential(
#    (0): Linear(in_features=25088, out_features=4096, bias=True)
#    (1): ReLU(inplace=True)
#    (2): Dropout(p=0.5, inplace=False)
#    (3): Linear(in_features=4096, out_features=4096, bias=True)
#    (4): ReLU(inplace=True)
#    (5): Dropout(p=0.5, inplace=False)
#    (6): Linear(in_features=4096, out_features=1000, bias=True)
#  )
# We need out_features=1 for binary classification with BCEWithLogitsLoss.

num_ftrs = model_vgg16.classifier[6].in_features # Get number of features from the original last layer
model_vgg16.classifier[6] = nn.Linear(num_ftrs, 1) # Replace the last layer

# Move the model to the designated device
model_vgg16 = model_vgg16.to(device)

# --- Loss Function and Optimizer ---
# For binary classification, BCEWithLogitsLoss is suitable as it combines Sigmoid and BCELoss.
# It expects raw logits from the model.
criterion = nn.BCEWithLogitsLoss()

# Optimizer - Adam is a common choice.
# We only want to train the parameters of the modified classifier.
optimizer = optim.Adam(model_vgg16.classifier.parameters(), lr=0.001)


# Print model summary (optional, can be verbose for VGG16)
# print(model_vgg16)

print("Model defined, VGG16 classifier modified for binary classification.")
print("Loss function and optimizer are set up.")

# --- Training and Validation ---
NUM_EPOCHS = 10 # Can be adjusted

# Lists to store performance history
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# Check if DataLoaders are available
if not (train_loader and val_loader and test_loader):
    print("DataLoaders not initialized properly. Skipping training and evaluation.")
else:
    print(f"\nStarting training for {NUM_EPOCHS} epochs on {device}...")
    for epoch in range(NUM_EPOCHS):
        # --- Training Phase ---
        model_vgg16.train()  # Set model to training mode
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device).float() # Ensure labels are float for BCEWithLogitsLoss

            optimizer.zero_grad()  # Zero the parameter gradients
            outputs = model_vgg16(images)  # Forward pass
            loss = criterion(outputs, labels)  # Calculate loss
            loss.backward()  # Backward pass
            optimizer.step()  # Optimize

            running_loss += loss.item() * images.size(0)

            # Calculate accuracy
            predicted = torch.sigmoid(outputs) > 0.5 # Apply sigmoid and threshold for binary prediction
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            if (i + 1) % 100 == 0: # Print progress every 100 batches
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Batch [{i+1}/{len(train_loader)}], Train Loss: {loss.item():.4f}")

        epoch_train_loss = running_loss / len(train_dataset)
        epoch_train_accuracy = correct_train / total_train
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_accuracy)

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Training Loss: {epoch_train_loss:.4f}, Training Accuracy: {epoch_train_accuracy:.4f}")

        # --- Validation Phase ---
        model_vgg16.eval()  # Set model to evaluation mode
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():  # Disable gradient calculation
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device).float()
                outputs = model_vgg16(images)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * images.size(0)

                predicted = torch.sigmoid(outputs) > 0.5
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        epoch_val_loss = running_val_loss / len(val_dataset)
        epoch_val_accuracy = correct_val / total_val
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_accuracy)

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Validation Loss: {epoch_val_loss:.4f}, Validation Accuracy: {epoch_val_accuracy:.4f}")

    print("\nTraining finished.")

    # --- Plotting Training History ---
    if train_losses and val_losses and train_accuracies and val_accuracies:
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(range(1, NUM_EPOCHS + 1), train_losses, label='Train Loss')
        plt.plot(range(1, NUM_EPOCHS + 1), val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(range(1, NUM_EPOCHS + 1), train_accuracies, label='Train Accuracy')
        plt.plot(range(1, NUM_EPOCHS + 1), val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig("training_validation_curves.png")
        print("Training and validation curves saved to 'training_validation_curves.png'")
        # plt.show() # Uncomment if running in an environment that can display plots
    else:
        print("Not enough data to plot training curves.")

    # --- Test Phase ---
    if test_loader and model_vgg16: # Ensure test_loader is available and model is defined
        print("\nStarting evaluation on the test set...")
        model_vgg16.eval()  # Set model to evaluation mode
        correct_test = 0
        total_test = 0
        all_labels_test = []
        all_predictions_test = []

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device).float()
                outputs = model_vgg16(images)

                predicted_probs = torch.sigmoid(outputs)
                predicted_classes = predicted_probs > 0.5

                total_test += labels.size(0)
                correct_test += (predicted_classes == labels).sum().item()

                all_labels_test.extend(labels.cpu().numpy())
                all_predictions_test.extend(predicted_classes.cpu().numpy())

        if total_test > 0:
            test_accuracy = correct_test / total_test
            print(f"Test Accuracy: {test_accuracy:.4f}")

            # For more detailed metrics like precision, recall, F1, confusion matrix,
            # you would typically use scikit-learn:
            # from sklearn.metrics import classification_report, confusion_matrix
            # print("\nClassification Report:")
            # print(classification_report(all_labels_test, all_predictions_test, target_names=['Normal', 'Pneumonia']))
            # print("\nConfusion Matrix:")
            # conf_matrix = confusion_matrix(all_labels_test, all_predictions_test)
            # print(conf_matrix)
            # # Code to plot confusion matrix could be added here using matplotlib/seaborn
        else:
            print("No samples found in the test set to evaluate.")
    else:
        print("Test loader not available or model not trained. Skipping test phase.")


    # --- Save the Model ---
    if model_vgg16:
        model_save_path = "vgg16_pneumonia_medmnist_pytorch.pth"
        torch.save(model_vgg16.state_dict(), model_save_path)
        print(f"\nTrained model state_dict saved to '{model_save_path}'")

# End of script
print("\nScript execution finished.")
