import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.transforms import Compose, Resize, ToTensor
import numpy as np
import matplotlib.pyplot as plt

# Assuming model.py and dataset.py are in unet_pet_project/src/
from src.model import UNet
from src.dataset import PetDataset

# --- Configuration ---
BASE_DATA_DIR = "data"
IMAGE_DIR = os.path.join(BASE_DATA_DIR, "images")
MASK_DIR = os.path.join(BASE_DATA_DIR, "annotations", "trimaps")

LEARNING_RATE = 1e-4
BATCH_SIZE = 4
EPOCHS = 10
IMG_SIZE = (128, 128)
VALIDATION_SPLIT = 0.1

# --- Helper Functions ---
def check_data_paths(image_dir, mask_dir):
    """Checks if the provided data paths exist and contain data."""
    if not os.path.isdir(image_dir):
        print(f"Error: Image directory not found at {image_dir}")
        print("Please ensure you have downloaded the Oxford-IIIT Pet dataset and extracted it.")
        print(f"Expected image path: {os.path.abspath(image_dir)}")
        return False
    if not os.path.isdir(mask_dir):
        print(f"Error: Mask directory not found at {mask_dir}")
        print(f"Expected mask path: {os.path.abspath(mask_dir)}")
        return False
    # Check if directories are empty only if they exist
    if os.path.isdir(image_dir) and not os.listdir(image_dir):
        print(f"Error: No files found in image directory {image_dir}")
        return False
    if os.path.isdir(mask_dir) and not os.listdir(mask_dir):
        print(f"Error: No files found in mask directory {mask_dir}")
        return False
    return True

def calculate_accuracy(preds, targets, threshold=0.5):
    """Calculates pixel accuracy for binary segmentation."""
    preds_binary = (torch.sigmoid(preds) > threshold).float()
    correct = (preds_binary == targets).sum().item()
    total_pixels = targets.nelement()
    accuracy = correct / total_pixels
    return accuracy

def plot_training_curves(history, epochs, save_dir="."):
    """Plots training and validation loss and accuracy curves."""

    if not history or not all(k in history for k in ['train_loss', 'val_loss', 'train_acc', 'val_acc']):
        print("History object is incomplete or missing. Skipping plotting.")
        return

    epoch_range = range(1, epochs + 1)

    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epoch_range, history['train_loss'], label='Training Loss')
    plt.plot(epoch_range, history['val_loss'], label='Validation Loss')
    plt.title('Loss vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epoch_range, history['train_acc'], label='Training Accuracy')
    plt.plot(epoch_range, history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    loss_curve_path = os.path.join(save_dir, "training_loss_accuracy_curves.png")
    try:
        plt.savefig(loss_curve_path)
        print(f"Training curves saved to {os.path.abspath(loss_curve_path)}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    plt.close() # Close the plot to free memory

# --- Main Training Script ---
def main():
    print("Starting U-Net training for Oxford-IIIT Pet Dataset...")

    project_root = os.path.dirname(os.path.abspath(__file__))
    abs_image_dir = os.path.join(project_root, IMAGE_DIR)
    abs_mask_dir = os.path.join(project_root, MASK_DIR)

    if not check_data_paths(abs_image_dir, abs_mask_dir):
        print("Exiting due to data path issues.")
        return None # Return None if paths are bad

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = PetDataset(image_dir=abs_image_dir, mask_dir=abs_mask_dir, target_size=IMG_SIZE)

    if len(dataset) == 0:
        print("Dataset is empty. Please check data paths and dataset integrity.")
        return None

    n_val = int(len(dataset) * VALIDATION_SPLIT)
    n_train = len(dataset) - n_val

    if n_train <= 0 or n_val <= 0: # Ensure both are positive
        print(f"Dataset is too small for the current validation split of {VALIDATION_SPLIT*100}%. Needs at least {int(1/VALIDATION_SPLIT) if VALIDATION_SPLIT > 0 else 'N/A'} samples for a split.")
        print("Using all data for training and validation (not recommended for proper evaluation).")
        train_dataset = dataset
        val_dataset = dataset
        # Update n_train and n_val for dataloader checks if we proceed this way
        n_train = len(train_dataset)
        n_val = len(val_dataset)
        if n_train == 0 : # Still possible if dataset was empty initially, though checked above.
            print("Dataset is confirmed empty or became empty. Cannot proceed.")
            return None
    else:
        train_dataset, val_dataset = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    # Check if datasets for loaders are empty
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("Resulting train or validation dataset is empty after split. Cannot create DataLoaders.")
        return None

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    print(f"Dataset loaded: {len(dataset)} samples. Training: {len(train_dataset)} samples, Validation: {len(val_dataset)} samples.")

    model = UNet(n_channels=3, n_classes=1, bilinear=False).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    print(f"Training for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        model.train()
        running_train_loss = 0.0
        running_train_acc = 0.0

        for i, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * images.size(0)
            running_train_acc += calculate_accuracy(outputs, masks) * images.size(0)

            if (i + 1) % 20 == 0:
                 print(f"Epoch [{epoch+1}/{EPOCHS}], Batch [{i+1}/{len(train_loader)}], Train Loss: {loss.item():.4f}")

        epoch_train_loss = running_train_loss / len(train_loader.dataset) # Use len of dataset subset
        epoch_train_acc = running_train_acc / len(train_loader.dataset)
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)

        model.eval()
        running_val_loss = 0.0
        running_val_acc = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                running_val_loss += loss.item() * images.size(0)
                running_val_acc += calculate_accuracy(outputs, masks) * images.size(0)

        epoch_val_loss = running_val_loss / len(val_loader.dataset) # Use len of dataset subset
        epoch_val_acc = running_val_acc / len(val_loader.dataset)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)

        print(f"Epoch [{epoch+1}/{EPOCHS}]:")
        print(f"  Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}")
        print(f"  Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")

    print("Training finished.")

    print("Plotting training curves...")
    plot_training_curves(history, EPOCHS, save_dir=project_root) # Pass actual EPOCHS and save_dir

    model_save_path = os.path.join(project_root, "unet_pet_model.pth")
    try:
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {os.path.abspath(model_save_path)}")
    except Exception as e:
        print(f"Error saving model: {e}")

    return history

if __name__ == "__main__":
    project_root_main = os.path.dirname(os.path.abspath(__file__))
    test_image_dir = os.path.join(project_root_main, IMAGE_DIR)
    test_mask_dir = os.path.join(project_root_main, MASK_DIR)

    # Simplified dummy data check/creation for dry run
    if not check_data_paths(test_image_dir, test_mask_dir):
        print("Attempting to create dummy data for a dry run as dataset was not found...")

        if not os.path.exists(test_image_dir): os.makedirs(test_image_dir, exist_ok=True)
        if not os.path.exists(test_mask_dir): os.makedirs(test_mask_dir, exist_ok=True)

        dummy_files_created = 0
        try:
            # Create a few dummy files to make PetDataset load something
            for i in range(BATCH_SIZE * 2 if BATCH_SIZE > 0 else 2): # Ensure enough for a couple of batches if possible
                img_path = os.path.join(test_image_dir, f"dummy_{i}.jpg")
                mask_path = os.path.join(test_mask_dir, f"dummy_{i}.png")
                if not os.path.exists(img_path):
                    Image.new('RGB', (IMG_SIZE[0], IMG_SIZE[1]), color = 'red').save(img_path)
                if not os.path.exists(mask_path):
                    # Make mask values 1, 2, or 3 for PetDataset compatibility
                    mask_array = np.random.randint(1, 4, size=(IMG_SIZE[0], IMG_SIZE[1]), dtype=np.uint8)
                    Image.fromarray(mask_array, mode='L').save(mask_path)
                dummy_files_created +=1
            if dummy_files_created > 0:
                 print(f"Created/verified {dummy_files_created} dummy image/mask pairs for testing.")
                 print("NOTE: This is a dry run with dummy data. For actual training, please use the Oxford-IIIT Pet dataset.")
            else:
                print("Failed to create dummy files. The script might fail if it can't find any data.")

        except Exception as e:
            print(f"Could not create dummy files: {e}.")

    main_history = main()

    if main_history:
        print("\n--- Training History Summary ---")
        # Check if history has the expected keys before trying to access them
        if all(k in main_history for k in ['train_loss', 'val_loss', 'train_acc', 'val_acc']):
            num_epochs_recorded = len(main_history['train_loss'])
            for epoch_idx in range(num_epochs_recorded):
                print(f"Epoch {epoch_idx+1}: Train Loss: {main_history['train_loss'][epoch_idx]:.4f}, Val Loss: {main_history['val_loss'][epoch_idx]:.4f}, Train Acc: {main_history['train_acc'][epoch_idx]:.4f}, Val Acc: {main_history['val_acc'][epoch_idx]:.4f}")
        else:
            print("Training history is missing expected keys. Cannot print summary.")
    else:
        print("Training did not complete successfully or history was not generated.")
