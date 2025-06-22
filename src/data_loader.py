import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

# Standard ImageNet normalization
IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]

def get_data_transforms(image_size=(224, 224)):
    """
    Returns a dictionary of data transformations for training, validation, and testing.
    """
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(image_size), # Resize shorter side to image_size[0]
            transforms.CenterCrop(image_size), # Crop to image_size
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            # transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)), # More augmentations
            transforms.ToTensor(),
            transforms.Normalize(IMAGE_NET_MEAN, IMAGE_NET_STD)
        ]),
        'val': transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGE_NET_MEAN, IMAGE_NET_STD)
        ]),
        'test': transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGE_NET_MEAN, IMAGE_NET_STD)
        ]),
    }
    return data_transforms

def create_dataloaders(dataset_path, batch_size, image_size=(224, 224), val_split=0.15, num_workers=4):
    """
    Creates PyTorch DataLoaders for train, validation, and test sets.
    If a 'val' directory doesn't exist, it splits the training set to create a validation set.

    Args:
        dataset_path (str): Root path to the dataset.
                            Expected structure: dataset_path/{train/test/val}/{class_name}/image.jpg
        batch_size (int): Batch size for DataLoaders.
        image_size (tuple): Target image size (height, width).
        val_split (float): Proportion of training data to use for validation if 'val' dir is missing.
        num_workers (int): Number of worker processes for DataLoader.

    Returns:
        tuple: (train_loader, val_loader, test_loader, class_names)
               Returns None for a loader if the respective directory (train, test) is not found.
               val_loader might be derived from train_loader.
    """
    data_transforms = get_data_transforms(image_size)

    image_datasets = {}
    dataloaders = {'train': None, 'val': None, 'test': None}
    class_names = None

    # Check for train, val, test directories
    train_dir = os.path.join(dataset_path, 'train')
    val_dir = os.path.join(dataset_path, 'val')
    test_dir = os.path.join(dataset_path, 'test')

    if not os.path.exists(train_dir):
        print(f"Training directory not found: {train_dir}")
        # Depending on strictness, could raise error or return Nones
        # For now, we'll allow continuing if only test set is needed for evaluation later

    if not os.path.exists(test_dir):
        print(f"Test directory not found: {test_dir}")

    # Load Train dataset
    if os.path.exists(train_dir):
        image_datasets['train_full'] = datasets.ImageFolder(train_dir, data_transforms['train'])
        class_names = image_datasets['train_full'].classes

        if os.path.exists(val_dir):
            print(f"Using existing validation directory: {val_dir}")
            image_datasets['val'] = datasets.ImageFolder(val_dir, data_transforms['val'])
            image_datasets['train'] = image_datasets['train_full'] # Use the full train set
            # Apply train transform to the training part
            image_datasets['train'].transform = data_transforms['train']
        elif val_split > 0:
            print(f"No 'val' directory found. Splitting 'train' data. Validation split: {val_split*100}%.")
            train_indices, val_indices = train_test_split(
                list(range(len(image_datasets['train_full']))),
                test_size=val_split,
                stratify=image_datasets['train_full'].targets # Stratify by class
            )
            image_datasets['train'] = Subset(image_datasets['train_full'], train_indices)
            image_datasets['val'] = Subset(image_datasets['train_full'], val_indices)

            # Important: Apply the correct transforms to the subsets
            # Create new datasets with correct transforms for train and val subsets
            # This is a bit tricky with Subset. A common way is to wrap Subset or handle transform in __getitem__
            # For simplicity here, we'll assign the base dataset's transform, then override for val_subset.
            # A cleaner way is to have custom Dataset wrappers if Subset becomes cumbersome with transforms.

            # Let's ensure 'train_full' has train transforms, then 'val' subset gets 'val' transforms.
            # This requires careful handling. A simpler approach for subsets is to make them new ImageFolder-like objects
            # or ensure the transform is applied correctly when items are fetched.
            # For now, we'll create new Dataset objects for the split parts with correct transforms.

            # Re-creating datasets for split with correct transforms:
            # This is inefficient as it loads data again. Better to handle transforms in a custom Subset or Dataset.
            # However, for this project's scope, let's assume a simpler path or that user provides train/val.
            # A common pattern:
            temp_train_dataset = datasets.ImageFolder(train_dir,transform=None) # Load once without specific transform for splitting

            train_subset_dataset = Subset(temp_train_dataset, train_indices)
            val_subset_dataset = Subset(temp_train_dataset, val_indices)

            # Now, assign transforms to these Subsets by wrapping them or having a transform attribute
            # This is a common pain point. For torchvision.datasets.ImageFolder, the transform is part of the dataset object.
            # A direct way:
            image_datasets['train'] = train_subset_dataset
            image_datasets['train'].dataset.transform = data_transforms['train'] # Apply to underlying full dataset for this subset

            image_datasets['val'] = val_subset_dataset
            image_datasets['val'].dataset.transform = data_transforms['val'] # Apply to underlying full dataset for this subset
            # This approach of modifying .dataset.transform might be problematic if both subsets point to the same .dataset object
            # A safer, but more verbose way, is to create custom Dataset wrappers for the subsets.
            # For now, let's assume this simplified assignment works for typical use or recommend separate val folder.
            # A more robust way for Subset transforms:
            class TransformedSubset(torch.utils.data.Dataset):
                def __init__(self, subset, transform):
                    self.subset = subset
                    self.transform = transform
                def __getitem__(self, index):
                    x, y = self.subset[index]
                    if self.transform:
                        x = self.transform(x) # ImageFolder already returns PIL, ready for transform
                    return x, y
                def __len__(self):
                    return len(self.subset)

            image_datasets['train'] = TransformedSubset(Subset(datasets.ImageFolder(train_dir), train_indices), data_transforms['train'])
            image_datasets['val'] = TransformedSubset(Subset(datasets.ImageFolder(train_dir), val_indices), data_transforms['val'])
            print(f"Train samples: {len(image_datasets['train'])}, Validation samples: {len(image_datasets['val'])}")

        else: # No val_dir and no val_split
            print("No validation set will be used as 'val' directory is missing and val_split is 0.")
            image_datasets['train'] = image_datasets['train_full']
            image_datasets['val'] = None # No validation loader

        dataloaders['train'] = DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        if image_datasets['val']:
            dataloaders['val'] = DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Load Test dataset
    if os.path.exists(test_dir):
        image_datasets['test'] = datasets.ImageFolder(test_dir, data_transforms['test'])
        if not class_names: # If train_dir didn't exist
            class_names = image_datasets['test'].classes
        dataloaders['test'] = DataLoader(image_datasets['test'], batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    if not class_names:
        print("Could not determine class names (e.g. no train or test data found).")
        # Potentially load from a config or raise error

    dataset_sizes = {x: len(image_datasets[x]) for x in image_datasets if image_datasets[x] is not None}
    print(f"Dataset sizes: {dataset_sizes}")

    return dataloaders['train'], dataloaders['val'], dataloaders['test'], class_names, dataset_sizes


def load_dataset_info(dataset_path):
    """
    Scans dataset path to provide information for the GUI.
    Assumes dataset_path contains 'train', 'test' (and optionally 'val') subdirectories,
    which in turn contain class name subdirectories.
    """
    info = {
        "path": dataset_path,
        "train_samples": 0, "val_samples": 0, "test_samples": 0,
        "classes": [],
        "train_distribution": "N/A", "val_distribution": "N/A", "test_distribution": "N/A",
        "issues": []
    }

    found_classes = set()

    for phase in ['train', 'val', 'test']:
        phase_path = os.path.join(dataset_path, phase)
        if not os.path.exists(phase_path) or not os.path.isdir(phase_path):
            info['issues'].append(f"Directory not found or not a directory: {phase_path}")
            continue

        phase_samples = 0
        phase_distribution = {}

        try:
            current_classes = sorted([d for d in os.listdir(phase_path) if os.path.isdir(os.path.join(phase_path, d))])
            if not current_classes:
                info['issues'].append(f"No class subdirectories found in {phase_path}")
                continue

            for class_name in current_classes:
                found_classes.add(class_name)
                class_path = os.path.join(phase_path, class_name)
                try:
                    num_images = len([name for name in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, name))])
                    phase_distribution[class_name] = num_images
                    phase_samples += num_images
                except OSError as e:
                    info['issues'].append(f"Could not read class directory {class_path}: {e}")


            info[f'{phase}_samples'] = phase_samples
            info[f'{phase}_distribution'] = ", ".join([f"{k}: {v}" for k,v in phase_distribution.items()]) if phase_distribution else "No images found"

        except OSError as e:
            info['issues'].append(f"Could not read phase directory {phase_path}: {e}")

    info['classes'] = sorted(list(found_classes))
    if not info['classes']:
         info['issues'].append(f"No classes found across train, val, test directories.")

    # If 'val' was not found but 'train' was, indicate that it might be split from train
    if info['val_samples'] == 0 and info['train_samples'] > 0 :
        info['val_distribution'] = "Will be split from train if val_split > 0"

    return info

if __name__ == '__main__':
    # Example Usage:
    # Create dummy dataset structure for testing
    # ./dummy_dataset/
    #   train/
    #     NORMAL/ (10 images)
    #     PNEUMONIA/ (10 images)
    #   test/
    #     NORMAL/ (5 images)
    #     PNEUMONIA/ (5 images)
    #   val/ (optional)
    #     NORMAL/ (3 images)
    #     PNEUMONIA/ (3 images)

    print("Testing data_loader.py...")

    dummy_path = "dummy_chest_xray"
    if not os.path.exists(dummy_path):
        os.makedirs(dummy_path, exist_ok=True)
        for phase in ['train', 'test', 'val']:
            os.makedirs(os.path.join(dummy_path, phase), exist_ok=True)
            for class_name in ['NORMAL', 'PNEUMONIA']:
                class_dir = os.path.join(dummy_path, phase, class_name)
                os.makedirs(class_dir, exist_ok=True)
                num_files = {'train': 10, 'test': 5, 'val': 3}[phase]
                for i in range(num_files):
                    # Create tiny dummy png files
                    try:
                        from PIL import Image
                        img = Image.new('RGB', (10, 10), color = 'red')
                        img.save(os.path.join(class_dir, f"dummy_{i+1}.png"))
                    except ImportError:
                         with open(os.path.join(class_dir, f"dummy_{i+1}.jpeg"), 'w') as f:
                            f.write("dummy") # Simpler fallback if PIL not handy for test script
        print(f"Created dummy dataset at {dummy_path}")

    print("\n--- Testing load_dataset_info ---")
    dataset_info = load_dataset_info(dummy_path)
    print(f"Dataset Info: {dataset_info}")
    assert dataset_info['train_samples'] == 20
    assert dataset_info['test_samples'] == 10
    assert dataset_info['val_samples'] == 6
    assert "NORMAL" in dataset_info['classes'] and "PNEUMONIA" in dataset_info['classes']

    print("\n--- Testing create_dataloaders (with val folder) ---")
    train_loader, val_loader, test_loader, class_names, dset_sizes = create_dataloaders(dummy_path, batch_size=4, num_workers=0) # num_workers=0 for easier debugging

    if train_loader: print(f"Train loader: {len(train_loader.dataset)} samples, {len(train_loader)} batches")
    if val_loader: print(f"Val loader: {len(val_loader.dataset)} samples, {len(val_loader)} batches")
    if test_loader: print(f"Test loader: {len(test_loader.dataset)} samples, {len(test_loader)} batches")
    print(f"Class names: {class_names}")
    print(f"Dataset sizes from create_dataloaders: {dset_sizes}")

    # Test fetching a batch
    if train_loader:
        try:
            inputs, classes = next(iter(train_loader))
            print(f"Sample batch input shape: {inputs.shape}, classes: {classes}")
            assert inputs.shape[0] <= 4 # batch_size
            assert inputs.shape[1] == 3 # channels
            assert inputs.shape[2] == 224 and inputs.shape[3] == 224 # image_size
        except Exception as e:
            print(f"Error fetching batch from train_loader: {e}")
            if "num_workers > 0" in str(e) and os.name == 'nt':
                 print("Hint: On Windows, DataLoader issues can occur with num_workers > 0 in __main__. Set num_workers=0 for testing in this script.")


    # Test splitting if val folder is missing
    print("\n--- Testing create_dataloaders (splitting train for val) ---")
    dummy_path_no_val = "dummy_chest_xray_no_val"
    if not os.path.exists(dummy_path_no_val):
        os.makedirs(dummy_path_no_val, exist_ok=True)
        for phase in ['train', 'test']: # No 'val' phase
            os.makedirs(os.path.join(dummy_path_no_val, phase), exist_ok=True)
            for class_name in ['NORMAL', 'PNEUMONIA']:
                class_dir = os.path.join(dummy_path_no_val, phase, class_name)
                os.makedirs(class_dir, exist_ok=True)
                num_files = {'train': 20, 'test': 8}[phase] # More train samples for splitting
                for i in range(num_files):
                    try:
                        from PIL import Image
                        img = Image.new('RGB', (10, 10), color = 'blue')
                        img.save(os.path.join(class_dir, f"dummy_split_{i+1}.png"))
                    except ImportError:
                        with open(os.path.join(class_dir, f"dummy_split_{i+1}.jpeg"), 'w') as f:
                            f.write("dummy_split")
        print(f"Created dummy dataset (no val folder) at {dummy_path_no_val}")

    info_no_val = load_dataset_info(dummy_path_no_val)
    print(f"Dataset Info (no val folder): {info_no_val}")
    assert info_no_val['val_samples'] == 0
    assert info_no_val['train_samples'] == 40 # 20 NORMAL + 20 PNEUMONIA

    train_loader_split, val_loader_split, _, _, dset_sizes_split = create_dataloaders(
        dummy_path_no_val, batch_size=4, val_split=0.25, num_workers=0
    ) # 25% of 40 is 10 for val

    if train_loader_split: print(f"Split Train loader: {len(train_loader_split.dataset)} samples, {len(train_loader_split)} batches") # Should be 30
    if val_loader_split: print(f"Split Val loader: {len(val_loader_split.dataset)} samples, {len(val_loader_split)} batches") # Should be 10

    assert len(train_loader_split.dataset) == 30
    assert len(val_loader_split.dataset) == 10

    print("\nData loader tests completed.")
    # Consider cleaning up dummy datasets:
    # import shutil
    # shutil.rmtree(dummy_path)
    # shutil.rmtree(dummy_path_no_val)
    # print("Cleaned up dummy datasets.")
