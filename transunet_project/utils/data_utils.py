import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as F_vision # Renamed for clarity
import os
from PIL import Image
import numpy as np
import random # For choosing rotation angle

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir,
                 h_flip_prob=0.0,
                 v_flip_prob=0.0,
                 rotation_prob=0.0,
                 rotation_angles=None, # e.g., [0, 90, 180, 270] or a range for random.uniform
                 resize_size=None, # e.g., (256, 256)
                 image_normalization_mean=None, # Default to ImageNet stats
                 image_normalization_std=None,  # Default to ImageNet stats
                 mask_target_type='binary_float' # 'binary_float' ([0,1] float), 'multiclass_long' (long indices)
                ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_filenames = sorted([f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))])

        self.h_flip_prob = h_flip_prob
        self.v_flip_prob = v_flip_prob
        self.rotation_prob = rotation_prob
        self.rotation_angles = rotation_angles if rotation_angles is not None else [0, 90, 180, 270]

        self.resize_size = resize_size
        self.image_normalization_mean = image_normalization_mean if image_normalization_mean is not None else [0.485, 0.456, 0.406]
        self.image_normalization_std = image_normalization_std if image_normalization_std is not None else [0.229, 0.224, 0.225]
        self.mask_target_type = mask_target_type

    def __len__(self):
        return len(self.image_filenames)

    def _find_mask_path(self, image_filename):
        base_name, img_ext = os.path.splitext(image_filename)
        # Common naming conventions for masks
        possible_mask_names = [
            image_filename,  # Mask has exact same name
            f"{base_name}_mask{img_ext}",
            f"{base_name}_segmentation{img_ext}",
            f"{base_name}_seg{img_ext}",
            f"{base_name}_gt{img_ext}",
            # Try with common mask extensions like .png if image is .jpg, etc.
            f"{base_name}.png",
            f"{base_name}_mask.png",
            f"{base_name}_segmentation.png",
            f"{base_name}_seg.png",
            f"{base_name}_gt.png",
        ]
        # Also consider if mask dir mirrors image dir structure (e.g. subfolders)
        # For now, assume flat mask_dir.

        for m_name in possible_mask_names:
            potential_path = os.path.join(self.mask_dir, m_name)
            if os.path.exists(potential_path):
                return potential_path
        return None

    def __getitem__(self, idx):
        img_name_short = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, img_name_short)

        mask_path = self._find_mask_path(img_name_short)
        if mask_path is None:
            raise FileNotFoundError(f"Mask not found for image {img_name_short} in {self.mask_dir}.")

        try:
            image = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path) # Open mask, convert later based on needs
        except Exception as e:
            raise IOError(f"Error opening image/mask: {img_path} or {mask_path}. Error: {e}")


        # Apply augmentations that affect geometry (image and mask together)
        image, mask = self.apply_geometric_augmentations(image, mask)

        # Apply transforms like resize (after geometric, before to_tensor)
        if self.resize_size:
            image = F_vision.resize(image, self.resize_size, interpolation=Image.BILINEAR if hasattr(Image, 'BILINEAR') else F_vision.InterpolationMode.BILINEAR)
            # For masks, NEAREST is crucial to preserve class labels without interpolation artifacts.
            mask = F_vision.resize(mask, self.resize_size, interpolation=Image.NEAREST if hasattr(Image, 'NEAREST') else F_vision.InterpolationMode.NEAREST)

        # Convert image to tensor and normalize
        image = F_vision.to_tensor(image) # Converts PIL image (H, W, C) [0,255] to (C, H, W) [0,1]
        image = F_vision.normalize(image, mean=self.image_normalization_mean, std=self.image_normalization_std)

        # Convert mask to tensor based on its type and required output format
        if self.mask_target_type == 'binary_float':
            if mask.mode != 'L' and mask.mode != '1': # Ensure grayscale or binary before to_tensor for simple scaling
                mask = mask.convert('L')
            mask_tensor = F_vision.to_tensor(mask) # Converts L mode [0, 255] to (1, H, W) [0, 1] FloatTensor
            mask_tensor = (mask_tensor > 0.5).float() # Binarize to ensure strict 0 or 1
        elif self.mask_target_type == 'multiclass_long':
            if mask.mode == 'RGB' or mask.mode == 'RGBA': # If mask is RGB, it might encode classes in one channel
                 # This part is tricky: if mask is RGB, how are classes encoded?
                 # Assuming for now multiclass masks are 'L' or 'P' with direct class indices.
                 # If mask is, e.g., (0,0,0), (0,0,1), (0,0,2) for classes, needs specific parsing.
                 # For now, assume 'L' mode where pixel values are class indices.
                mask = mask.convert('L')
            mask_np = np.array(mask, dtype=np.int64) # Convert PIL to numpy array of int64
            mask_tensor = torch.from_numpy(mask_np) # Convert to LongTensor (H, W)
            # CrossEntropyLoss expects (N, H, W) and target (N, H, W) with class indices.
            # No channel dimension for mask indices.
        else:
            raise ValueError(f"Unsupported mask_target_type: {self.mask_target_type}")

        return image, mask_tensor

    def apply_geometric_augmentations(self, image, mask):
        # Random Horizontal Flip
        if torch.rand(1) < self.h_flip_prob:
            image = F_vision.hflip(image)
            mask = F_vision.hflip(mask)

        # Random Vertical Flip
        if torch.rand(1) < self.v_flip_prob:
            image = F_vision.vflip(image)
            mask = F_vision.vflip(mask)

        # Random Rotation
        # torchvision.transforms.functional.rotate handles PIL mask interpolation correctly (nearest for 'L', 'P', '1')
        if torch.rand(1) < self.rotation_prob and self.rotation_angles:
            angle = random.choice(self.rotation_angles)
            image = F_vision.rotate(image, angle, interpolation=F_vision.InterpolationMode.BILINEAR, fill=0) # fill for image
            mask = F_vision.rotate(mask, angle, interpolation=F_vision.InterpolationMode.NEAREST, fill=0) # fill for mask (0 is often background)

        # Placeholder for other geometric augmentations like RandomResizedCrop, Affine, Elastic
        # Example: RandomResizedCrop would need careful implementation:
        # if torch.rand(1) < self.random_resized_crop_prob:
        #     i, j, h, w = transforms.RandomResizedCrop.get_params(image, scale=..., ratio=...)
        #     image = F_vision.crop(image, i, j, h, w)
        #     mask = F_vision.crop(mask, i, j, h, w)
        #     # And then resize back to target size if needed
        #     image = F_vision.resize(image, self.resize_size, ...)
        #     mask = F_vision.resize(mask, self.resize_size, ...)

        return image, mask

# Example of how to get a configured dataset instance:
def get_augmented_dataset(image_dir, mask_dir, img_size=(256, 256), augment=True, mask_type='binary_float'):
    if augment:
        dataset = SegmentationDataset(
            image_dir=image_dir,
            mask_dir=mask_dir,
            h_flip_prob=0.5,
            v_flip_prob=0.5,
            rotation_prob=0.5,
            rotation_angles=[0, 90, 180, 270], # Or e.g. lambda: random.uniform(-30, 30) for continuous if rotate supports it
            resize_size=img_size,
            mask_target_type=mask_type
        )
    else: # No augmentation, just resize and normalize
        dataset = SegmentationDataset(
            image_dir=image_dir,
            mask_dir=mask_dir,
            resize_size=img_size,
            mask_target_type=mask_type
        )
    return dataset


if __name__ == '__main__':
    # Create dummy data for testing
    IMG_SIZE = (128, 128)
    DUMMY_DATA_ROOT = 'dummy_dataset_aug'
    DUMMY_IMG_DIR = os.path.join(DUMMY_DATA_ROOT, 'images')
    DUMMY_MASK_DIR = os.path.join(DUMMY_DATA_ROOT, 'masks')

    if not os.path.exists(DUMMY_IMG_DIR): os.makedirs(DUMMY_IMG_DIR)
    if not os.path.exists(DUMMY_MASK_DIR): os.makedirs(DUMMY_MASK_DIR)

    try:
        # Create 2 dummy images and masks
        for i in range(2):
            dummy_img_pil = Image.new('RGB', (200, 200), color = 'red')
            dummy_img_pil.save(os.path.join(DUMMY_IMG_DIR, f'img{i}.png'))

            # Binary mask (0 and 255)
            dummy_mask_array_bin = np.zeros((200, 200), dtype=np.uint8)
            dummy_mask_array_bin[50:150, 50:150] = 255 # A white square
            dummy_mask_pil_bin = Image.fromarray(dummy_mask_array_bin, mode='L')
            dummy_mask_pil_bin.save(os.path.join(DUMMY_MASK_DIR, f'img{i}_mask.png')) # Suffix _mask

            # Multi-class mask (0, 1, 2)
            dummy_mask_array_mc = np.zeros((200, 200), dtype=np.uint8)
            dummy_mask_array_mc[0:50, 0:50] = 0     # Class 0 (background)
            dummy_mask_array_mc[50:100, 50:100] = 1 # Class 1
            dummy_mask_array_mc[100:150, 100:150] = 2 # Class 2
            dummy_mask_pil_mc = Image.fromarray(dummy_mask_array_mc, mode='L')
            dummy_mask_pil_mc.save(os.path.join(DUMMY_MASK_DIR, f'img{i}_multimask.png'))


        print("--- Testing Binary Float Masks with Augmentation ---")
        dataset_bin_aug = get_augmented_dataset(
            DUMMY_IMG_DIR, DUMMY_MASK_DIR, img_size=IMG_SIZE, augment=True, mask_type='binary_float'
        )
        # Manually set filenames to point to binary masks for this test run
        dataset_bin_aug.image_filenames = ['img0.png', 'img1.png'] # Ensure it uses masks named imgX_mask.png

        print(f"Binary Augmented Dataset size: {len(dataset_bin_aug)}")
        img_b, mask_b = dataset_bin_aug[0]
        print(f"Image shape: {img_b.shape}, type: {img_b.dtype}")
        print(f"Mask shape: {mask_b.shape}, type: {mask_b.dtype}, min: {mask_b.min()}, max: {mask_b.max()}")
        assert img_b.shape == (3, IMG_SIZE[0], IMG_SIZE[1])
        assert mask_b.shape == (1, IMG_SIZE[0], IMG_SIZE[1]) # ToTensor adds channel dim for L mode
        assert mask_b.dtype == torch.float32
        assert mask_b.min() == 0.0 and mask_b.max() == 1.0


        print("\n--- Testing Multi-class Long Masks with Augmentation ---")
        dataset_mc_aug = get_augmented_dataset(
            DUMMY_IMG_DIR, DUMMY_MASK_DIR, img_size=IMG_SIZE, augment=True, mask_type='multiclass_long'
        )
        # Manually set filenames to point to multi-class masks for this test run
        dataset_mc_aug.image_filenames = ['img0.png', 'img1.png']
        # And we need _find_mask_path to find 'imgX_multimask.png' when image is 'imgX.png'
        # For the test, let's temporarily adjust the mask finding or name for simplicity:
        # Or, more simply, ensure the _find_mask_path can find it.
        # The current _find_mask_path might not find 'img0_multimask.png' for 'img0.png' unless we add that pattern.
        # For this test, let's assume _find_mask_path is adapted or filenames match.
        # To make the test pass with current _find_mask_path, let's rename the expected mask in the test:
        # This is hacky for test; in real use, _find_mask_path needs to be robust or names must match.
        # A better test: use specific image names that _find_mask_path WILL resolve.
        # For now, let's assume img0.png should map to img0_multimask.png.
        # We can ensure _find_mask_path tries basename + "_multimask" + ext

        # Simplified: for this test, let's assume img0.png's mask is img0_multimask.png
        # and _find_mask_path is updated or we test with img0_multimask.png as the "image name"
        # to find its corresponding mask.
        # For this specific test, let's assume the dataset is smart enough or we test one file:

        # Test with an image name that will lead to the multi-class mask
        idx_for_mc_test = 0
        dataset_mc_aug.image_filenames = [f'img{idx_for_mc_test}.png'] # Focus on one image
        # We need _find_mask_path to find 'img0_multimask.png'
        # Let's make a small modification to _find_mask_path for the test or ensure it works.
        # The current _find_mask_path has f"{base_name}_mask.png". If we save MC mask as imgX_mask.png, it'd work.
        # Let's assume the dummy MC mask was saved as imgX_mask.png for this test section for simplicity,
        # or that _find_mask_path is robust.
        # For the dummy data, we saved "imgX_multimask.png".
        # A quick fix for the test:
        original_find_mask_path = dataset_mc_aug._find_mask_path
        def _mock_find_mask_path_for_mc_test(self, image_filename):
            base, ext = os.path.splitext(image_filename)
            return os.path.join(self.mask_dir, f"{base}_multimask.png")
        dataset_mc_aug._find_mask_path = _mock_find_mask_path_for_mc_test.__get__(dataset_mc_aug, SegmentationDataset)


        print(f"Multi-class Augmented Dataset size: {len(dataset_mc_aug)}")
        img_m, mask_m = dataset_mc_aug[0] # Will use img0.png and find img0_multimask.png
        dataset_mc_aug._find_mask_path = original_find_mask_path # Restore

        print(f"Image shape: {img_m.shape}, type: {img_m.dtype}")
        print(f"Mask shape: {mask_m.shape}, type: {mask_m.dtype}, unique vals: {torch.unique(mask_m)}")
        assert img_m.shape == (3, IMG_SIZE[0], IMG_SIZE[1])
        assert mask_m.ndim == 2 and mask_m.shape == (IMG_SIZE[0], IMG_SIZE[1]) # (H, W)
        assert mask_m.dtype == torch.int64
        assert all(v in [0,1,2] for v in torch.unique(mask_m))


        print("\n--- Testing No Augmentation ---")
        dataset_no_aug = get_augmented_dataset(
            DUMMY_IMG_DIR, DUMMY_MASK_DIR, img_size=IMG_SIZE, augment=False, mask_type='binary_float'
        )
        dataset_no_aug.image_filenames = ['img0.png', 'img1.png']
        img_n, mask_n = dataset_no_aug[0]
        print(f"Image shape (no aug): {img_n.shape}")
        print(f"Mask shape (no aug): {mask_n.shape}")
        assert img_n.shape == (3, IMG_SIZE[0], IMG_SIZE[1])
        assert mask_n.shape == (1, IMG_SIZE[0], IMG_SIZE[1])

    finally:
        # Clean up dummy data
        import shutil
        if os.path.exists(DUMMY_DATA_ROOT):
            shutil.rmtree(DUMMY_DATA_ROOT)

    print("\nData augmentation basics implemented in data_utils.py and tested.")
    print("Includes: H/V Flip, Rotation. Resize and ToTensor/Normalize also handled.")
    print("Supports binary float masks and multi-class long masks.")
    # Note: More advanced augmentations like RandomResizedCrop, Affine, Elastic Deformations, ColorJitter
    # would require further implementation or a library like Albumentations.
    pass
