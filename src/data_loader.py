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
    class_names = []
    dataset_sizes = {}

    # Define TransformedSubset class here
    class TransformedSubset(torch.utils.data.Dataset):
        def __init__(self, subset, transform=None):
            self.subset = subset
            self.transform = transform

        def __getitem__(self, index):
            x, y = self.subset[index] # The subset's underlying dataset (ImageFolder) returns PIL Image
            if self.transform:
                x = self.transform(x)
            return x, y

        def __len__(self):
            return len(self.subset)

    # Mode detection
    print(f"[create_dataloaders] Received dataset_path: {dataset_path}, batch_size: {batch_size}, val_split: {val_split}")
    train_dir = os.path.join(dataset_path, 'train')
    test_dir = os.path.join(dataset_path, 'test')
    is_standard_mode = os.path.isdir(train_dir) and os.path.isdir(test_dir)
    print(f"[create_dataloaders] Is standard mode (train/test dirs exist): {is_standard_mode}")

    if is_standard_mode:
        print(f"[create_dataloaders] Operating in Standard Mode: Using 'train', 'val', 'test' subfolders from {dataset_path}")
        # Load Train dataset
        if os.path.exists(train_dir):
            train_full_for_split = datasets.ImageFolder(train_dir)
            class_names = train_full_for_split.classes
            print(f"[create_dataloaders] Standard Mode: Found classes in train_dir: {class_names}")

            val_dir = os.path.join(dataset_path, 'val')
            if os.path.exists(val_dir) and os.path.isdir(val_dir):
                print(f"[create_dataloaders] Standard Mode: Using existing validation directory: {val_dir}")
                image_datasets['train'] = TransformedSubset(datasets.ImageFolder(train_dir), data_transforms['train'])
                image_datasets['val'] = TransformedSubset(datasets.ImageFolder(val_dir), data_transforms['val'])
            elif val_split > 0:
                print(f"[create_dataloaders] Standard Mode: No 'val' directory. Splitting 'train' data. Validation split: {val_split*100}%.")
                train_indices, val_indices = train_test_split(
                    list(range(len(train_full_for_split))),
                    test_size=val_split,
                    stratify=train_full_for_split.targets
                )
                train_subset_orig = Subset(train_full_for_split, train_indices)
                val_subset_orig = Subset(train_full_for_split, val_indices)

                image_datasets['train'] = TransformedSubset(train_subset_orig, data_transforms['train'])
                image_datasets['val'] = TransformedSubset(val_subset_orig, data_transforms['val'])
                print(f"[create_dataloaders] Standard Mode: Train subset size: {len(image_datasets['train'])}, Val subset size: {len(image_datasets['val'])}")
            else:
                print("[create_dataloaders] Standard Mode: No validation set will be used (no 'val' dir, val_split is 0).")
                image_datasets['train'] = TransformedSubset(datasets.ImageFolder(train_dir), data_transforms['train'])
                image_datasets['val'] = None
        else:
            print(f"[create_dataloaders] Standard Mode: Training directory not found: {train_dir}")

        # Load Test dataset
        if os.path.exists(test_dir):
            image_datasets['test'] = TransformedSubset(datasets.ImageFolder(test_dir), data_transforms['test'])
            if not class_names:
                class_names = image_datasets['test'].subset.dataset.classes if hasattr(image_datasets['test'], 'subset') else []
                print(f"[create_dataloaders] Standard Mode: Classes inferred from test_dir: {class_names}")
        else:
            print(f"[create_dataloaders] Standard Mode: Test directory not found: {test_dir}")
            image_datasets['test'] = None

        # Create DataLoaders for standard mode
        if image_datasets.get('train'):
            dataloaders['train'] = DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        if image_datasets.get('val'):
            dataloaders['val'] = DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        if image_datasets.get('test'):
            dataloaders['test'] = DataLoader(image_datasets['test'], batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    else: # Auto-split mode
        print(f"[create_dataloaders] Operating in Auto-Split Mode for path: {dataset_path}")
        try:
            full_dataset = datasets.ImageFolder(dataset_path)
            if not full_dataset.classes:
                raise ValueError("No classes found in the selected directory for auto-split mode.")
            class_names = full_dataset.classes
            print(f"[create_dataloaders] Auto-Split Mode: Found classes: {class_names}. Total samples: {len(full_dataset)}")

            test_set_size = 0.20
            indices = list(range(len(full_dataset)))
            targets = full_dataset.targets
            print(f"[create_dataloaders] Auto-Split Mode: Test set size: {test_set_size*100}%, Val split (from remaining): {val_split*100}%")


            if len(set(targets)) < 2 :
                print("[create_dataloaders] Auto-Split Mode: Warning: Only one class found or samples for only one class. Using non-stratified split.")
                trainval_idx, test_idx = train_test_split(indices, test_size=test_set_size, shuffle=True, random_state=42)
                if val_split > 0 and len(trainval_idx) > 1 :
                     train_idx, val_idx = train_test_split(trainval_idx, test_size=val_split, shuffle=True, random_state=42)
                else:
                     train_idx = trainval_idx
                     val_idx = []
            else:
                trainval_idx, test_idx = train_test_split(indices, test_size=test_set_size, stratify=targets, shuffle=True, random_state=42)
                print(f"[create_dataloaders] Auto-Split Mode: After test split: trainval_idx size {len(trainval_idx)}, test_idx size {len(test_idx)}")

                if val_split > 0 and len(trainval_idx) > 1 :
                    trainval_targets = [targets[i] for i in trainval_idx]
                    if len(set(trainval_targets)) < 2 or len(trainval_idx) < 2 :
                        print("[create_dataloaders] Auto-Split Mode: Warning: Not enough samples or classes in train/val portion for stratification. Using non-stratified train/val split.")
                        train_idx, val_idx = train_test_split(trainval_idx, test_size=val_split, shuffle=True, random_state=42)
                    else:
                        train_idx, val_idx = train_test_split(trainval_idx, test_size=val_split, stratify=trainval_targets, shuffle=True, random_state=42)
                else:
                    train_idx = trainval_idx
                    val_idx = []
                print(f"[create_dataloaders] Auto-Split Mode: After val split: train_idx size {len(train_idx)}, val_idx size {len(val_idx)}")


            image_datasets['train'] = TransformedSubset(Subset(full_dataset, train_idx), data_transforms['train'])
            if val_idx:
                image_datasets['val'] = TransformedSubset(Subset(full_dataset, val_idx), data_transforms['val'])
            else:
                image_datasets['val'] = None
            image_datasets['test'] = TransformedSubset(Subset(full_dataset, test_idx), data_transforms['test'])
            print(f"[create_dataloaders] Auto-Split Mode: Train subset type: {type(image_datasets['train'])}, Test subset type: {type(image_datasets['test'])}")


            dataloaders['train'] = DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
            if image_datasets['val']:
                dataloaders['val'] = DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
            dataloaders['test'] = DataLoader(image_datasets['test'], batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

            print(f"[create_dataloaders] Auto-split complete. Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)} samples.")

        except Exception as e:
            print(f"[create_dataloaders] Error during auto-split data loading: {e}")
            return None, None, None, [], {}


    if not class_names:
        print("[create_dataloaders] Could not determine class names.")

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test'] if image_datasets.get(x) is not None}
    print(f"[create_dataloaders] Final dataset sizes: {dataset_sizes}")
    print(f"[create_dataloaders] Final class names: {class_names}")

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
        "issues": [],
        "auto_split_mode": False,
        "data_load_mode": "N/A"
    }
    print(f"[load_dataset_info] Received dataset_path: {dataset_path}")

    train_dir = os.path.join(dataset_path, 'train')
    test_dir = os.path.join(dataset_path, 'test')
    is_standard_mode = os.path.isdir(train_dir) and os.path.isdir(test_dir)

    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')
    print(f"[load_dataset_info] Checking for standard mode. train_dir exists: {os.path.isdir(train_dir)}, test_dir exists: {os.path.isdir(test_dir)}. Is standard mode: {is_standard_mode}")

    if is_standard_mode:
        print("[load_dataset_info] Operating in Standard Mode (train/test/val subfolders).")
        info["data_load_mode"] = "standard_split"
        found_classes = set()
        for phase in ['train', 'val', 'test']:
            phase_path = os.path.join(dataset_path, phase)
            print(f"[load_dataset_info] Standard Mode: Checking phase '{phase}' at path '{phase_path}'")
            if not os.path.exists(phase_path) or not os.path.isdir(phase_path):
                if phase == 'val':
                    info['issues'].append(f"Optional directory not found: {phase_path}")
                    print(f"[load_dataset_info] Standard Mode: Optional phase '{phase}' dir not found.")
                else:
                    info['issues'].append(f"Required directory not found or not a directory: {phase_path}")
                    print(f"[load_dataset_info] Standard Mode: Required phase '{phase}' dir not found.")
                continue

            phase_samples = 0
            phase_distribution = {}
            try:
                current_classes = sorted([d for d in os.listdir(phase_path) if os.path.isdir(os.path.join(phase_path, d))])
                print(f"[load_dataset_info] Standard Mode: Phase '{phase}', found potential class folders: {current_classes}")
                if not current_classes:
                    info['issues'].append(f"No class subdirectories found in {phase_path}")
                    continue

                for class_name in current_classes:
                    found_classes.add(class_name)
                    class_path = os.path.join(phase_path, class_name)
                    try:
                        num_images = len([name for name in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, name)) and name.lower().endswith(image_extensions)])
                        if num_images > 0:
                            phase_distribution[class_name] = num_images
                            phase_samples += num_images
                        else:
                            info['issues'].append(f"No images found in class directory {class_path}")
                            print(f"[load_dataset_info] Standard Mode: No images in {class_path}")
                    except OSError as e:
                        info['issues'].append(f"Could not read class directory {class_path}: {e}")

                info[f'{phase}_samples'] = phase_samples
                info[f'{phase}_distribution'] = ", ".join([f"{k}: {v}" for k,v in phase_distribution.items()]) if phase_distribution else "No images found"
            except OSError as e:
                info['issues'].append(f"Could not read phase directory {phase_path}: {e}")

        info['classes'] = sorted(list(found_classes))
        if not info['classes']:
             info['issues'].append(f"No classes found across train, val, test directories in standard mode.")
        if info['val_samples'] == 0 and info['train_samples'] > 0 :
            info['val_distribution'] = "Will be split from train if val_split > 0 during dataloader creation."
        print(f"[load_dataset_info] Standard Mode: Final classes: {info['classes']}, Train samples: {info['train_samples']}, Val samples: {info['val_samples']}, Test samples: {info['test_samples']}")

    else:
        print("[load_dataset_info] Standard mode not detected. Attempting Auto-Split Mode.")
        info["auto_split_mode"] = True
        info["data_load_mode"] = "auto_split"

        potential_classes = []
        try:
            potential_classes = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
        except OSError as e:
            info['issues'].append(f"Error reading directory {dataset_path} for auto-split: {e}")
            info["data_load_mode"] = "unsupported"

        print(f"[load_dataset_info] Auto-Split Mode: Found potential class subdirectories: {potential_classes}")
        found_classes_flexible = set()
        total_samples_flexible = 0
        flexible_distribution = {}

        if not potential_classes and info["data_load_mode"] != "unsupported":
            info['issues'].append(f"No subdirectories found in {dataset_path} to be used as classes for auto-split mode.")
            info["data_load_mode"] = "unsupported"
        elif info["data_load_mode"] != "unsupported":
            for class_name in potential_classes:
                class_path = os.path.join(dataset_path, class_name)
                try:
                    num_images = len([name for name in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, name)) and name.lower().endswith(image_extensions)])
                    if num_images > 0:
                        found_classes_flexible.add(class_name)
                        flexible_distribution[class_name] = num_images
                        total_samples_flexible += num_images
                        print(f"[load_dataset_info] Auto-Split Mode: Class '{class_name}' has {num_images} images.")
                    else:
                        info['issues'].append(f"No images found in potential class directory {class_path} for auto-split.")
                        print(f"[load_dataset_info] Auto-Split Mode: No images in {class_path}")
                except OSError as e:
                    info['issues'].append(f"Could not read potential class directory {class_path}: {e}")

            if found_classes_flexible:
                info['classes'] = sorted(list(found_classes_flexible))
                info['train_samples'] = total_samples_flexible
                info['train_distribution'] = ", ".join([f"{k}: {v}" for k,v in flexible_distribution.items()])
                info['val_samples'] = 0
                info['test_samples'] = 0
                info['val_distribution'] = "To be auto-split from total data."
                info['test_distribution'] = "To be auto-split from total data."
                print(f"[load_dataset_info] Auto-Split Mode: Final classes: {info['classes']}, Total samples: {info['train_samples']}")
            else:
                info['issues'].append(f"No images found in any subdirectories of {dataset_path} for auto-split mode.")
                info["data_load_mode"] = "unsupported"

    if not info['classes'] and info["data_load_mode"] != "unsupported":
        info['issues'].append(f"No classes with images found in the dataset at {dataset_path}.")
        info["data_load_mode"] = "unsupported"

    if info["data_load_mode"] == "unsupported":
        print(f"[load_dataset_info] Mode determined as 'unsupported'. Checking for flat image structure.")
        try:
            flat_images_count = len([name for name in os.listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, name)) and name.lower().endswith(image_extensions)])
            if flat_images_count > 0:
                info['issues'].append(f"Found {flat_images_count} images directly in the root folder. This flat structure is not supported for training. Please organize images into class subfolders, or use train/test/val subfolders.")
                print(f"[load_dataset_info] Found {flat_images_count} flat images. Not supported for training.")
            elif not potential_classes and not is_standard_mode :
                 info['issues'].append(f"The selected folder does not contain train/test/val subdirectories, nor does it contain class subdirectories with images. It also does not contain images directly for training to be possible.")
                 print(f"[load_dataset_info] Folder empty or no valid structure/images found.")
        except OSError as e:
            info['issues'].append(f"Could not read the selected directory {dataset_path}: {e}")

    print(f"[load_dataset_info] Final info before return: {info}")
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
