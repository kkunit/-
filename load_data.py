import torch
import torchvision.transforms as transforms
from medmnist import PneumoniaMNIST

def main():
    # 1. Define the transformation
    print("Defining transformation...")
    transform = transforms.Compose([
        transforms.ToTensor() # Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    ])
    print("Transformation defined.")

    # 2. Load the datasets
    print("\nLoading datasets...")
    try:
        train_dataset = PneumoniaMNIST(split='train', transform=transform, download=True, as_rgb=False) # as_rgb=False for 1-channel
        val_dataset = PneumoniaMNIST(split='val', transform=transform, download=True, as_rgb=False)
        test_dataset = PneumoniaMNIST(split='test', transform=transform, download=True, as_rgb=False)
        print("Datasets loaded successfully.")
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return

    # 3. Print the lengths of each dataset
    print("\nDataset lengths:")
    len_train = len(train_dataset)
    len_val = len(val_dataset)
    len_test = len(test_dataset)
    print(f"Length of training dataset: {len_train}")
    print(f"Length of validation dataset: {len_val}")
    print(f"Length of test dataset: {len_test}")

    # Expected lengths (as per MedMNIST documentation for PneumoniaMNIST)
    # Train: 4708, Val: 524, Test: 624
    if len_train == 4708 and len_val == 524 and len_test == 624:
        print("Dataset lengths match expected values.")
    else:
        print("Warning: Dataset lengths do NOT match expected values. Expected: Train=4708, Val=524, Test=624.")

    # 4. Check the shape and data type of a sample from each dataset
    print("\nChecking sample data from training dataset:")
    if len_train > 0:
        train_sample_image, train_sample_label = train_dataset[0]
        print(f"Shape of a training sample image: {train_sample_image.shape}")
        print(f"Data type of a training sample image: {train_sample_image.dtype}")
        print(f"Label of the first training sample: {train_sample_label}")
        if train_sample_image.shape == torch.Size([1, 28, 28]):
            print("Training sample image shape is correct (1, 28, 28).")
        else:
            print(f"Warning: Training sample image shape is {train_sample_image.shape}, expected torch.Size([1, 28, 28]).")
        if train_sample_image.dtype == torch.float32:
            print("Training sample image data type is correct (torch.float32).")
        else:
            print(f"Warning: Training sample image data type is {train_sample_image.dtype}, expected torch.float32.")

    else:
        print("Training dataset is empty, cannot check sample.")

    print("\nChecking sample data from validation dataset:")
    if len_val > 0:
        val_sample_image, val_sample_label = val_dataset[0]
        print(f"Shape of a validation sample image: {val_sample_image.shape}")
        print(f"Data type of a validation sample image: {val_sample_image.dtype}")
        print(f"Label of the first validation sample: {val_sample_label}")
        if val_sample_image.shape == torch.Size([1, 28, 28]):
            print("Validation sample image shape is correct (1, 28, 28).")
        else:
            print(f"Warning: Validation sample image shape is {val_sample_image.shape}, expected torch.Size([1, 28, 28]).")

    else:
        print("Validation dataset is empty, cannot check sample.")


    print("\nChecking sample data from test dataset:")
    if len_test > 0:
        test_sample_image, test_sample_label = test_dataset[0]
        print(f"Shape of a test sample image: {test_sample_image.shape}")
        print(f"Data type of a test sample image: {test_sample_image.dtype}")
        print(f"Label of the first test sample: {test_sample_label}")
        if test_sample_image.shape == torch.Size([1, 28, 28]):
            print("Test sample image shape is correct (1, 28, 28).")
        else:
            print(f"Warning: Test sample image shape is {test_sample_image.shape}, expected torch.Size([1, 28, 28]).")
    else:
        print("Test dataset is empty, cannot check sample.")

if __name__ == '__main__':
    main()
