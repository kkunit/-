import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import numpy as np

class PetDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, target_size=(256, 256)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.target_size = target_size

        self.image_filenames = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        # Filter out problematic files like '.DS_Store' or those without a valid mask
        self.valid_filenames = []
        for img_name in self.image_filenames:
            base_name = os.path.splitext(img_name)[0]
            mask_name = base_name + ".png" # Masks are typically .png
            if os.path.exists(os.path.join(mask_dir, mask_name)):
                self.valid_filenames.append(img_name)
            # else:
            #     print(f"Warning: Mask not found for image {img_name}") # Optional: for debugging

        if not self.valid_filenames:
            raise RuntimeError(f"No images found in {image_dir} with corresponding masks in {mask_dir}. Check paths and ensure masks end with .png and share base names with images.")


        # Define standard transformations if none are provided
        if self.transform is None:
            self.transform = T.Compose([
                T.Resize(self.target_size),
                T.ToTensor(), # Converts to [0, 1] and CxHxW
            ])

        self.mask_transform = T.Compose([
            T.Resize(self.target_size, interpolation=T.InterpolationMode.NEAREST), # Use NEAREST for masks
        ])

    def __len__(self):
        return len(self.valid_filenames)

    def __getitem__(self, idx):
        img_name = self.valid_filenames[idx]
        img_path = os.path.join(self.image_dir, img_name)

        base_name = os.path.splitext(img_name)[0]
        mask_name = base_name + ".png" # Masks are .png
        mask_path = os.path.join(self.mask_dir, mask_name)

        try:
            image = Image.open(img_path).convert("RGB")
            mask_pil = Image.open(mask_path).convert("L") # Load mask as grayscale
        except FileNotFoundError:
            # This should ideally not happen if valid_filenames is correctly populated
            raise FileNotFoundError(f"Image or mask not found for index {idx}. Img: {img_path}, Mask: {mask_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading image/mask {img_name}: {e}")


        # Apply image transformations (Resize, ToTensor)
        if self.transform:
            image = self.transform(image) # This should take care of ToTensor for image

        # Apply mask transformations (Resize) and then convert to tensor manually
        if self.mask_transform:
            mask_pil = self.mask_transform(mask_pil)

        mask_np = np.array(mask_pil) # H, W

        # Convert trimap to binary mask:
        # Class 1 (Pet) -> 1
        # Class 2 (Background) -> 0
        # Class 3 (Border) -> 0
        binary_mask = np.zeros_like(mask_np, dtype=np.float32) # Initialize with 0 (background)
        binary_mask[mask_np == 1] = 1.0 # Pet
        # Pixels with value 2 (background) or 3 (border) remain 0 or are explicitly set to 0

        # Convert numpy mask to tensor (add channel dimension)
        mask_tensor = torch.from_numpy(binary_mask).unsqueeze(0) # Becomes 1xHxW

        return image, mask_tensor

if __name__ == '__main__':
    # This is a placeholder for where you'd put your data path
    # You need to download the Oxford-IIIT Pet Dataset and extract it
    # For example, if you extracted it to './unet_pet_project/data/'
    # then image_dir would be './unet_pet_project/data/images'
    # and mask_dir would be './unet_pet_project/data/annotations/trimaps'

    print("Attempting to test PetDataset...")
    # Create dummy directories and files for testing if data is not present
    test_image_dir = "temp_test_images"
    test_mask_dir = "temp_test_masks" # Corresponds to annotations/trimaps usually

    if not os.path.exists(test_image_dir):
        os.makedirs(test_image_dir)
    if not os.path.exists(test_mask_dir):
        os.makedirs(test_mask_dir)

    try:
        # Create a few dummy image and mask files
        for i in range(3):
            try:
                # Dummy Image (RGB)
                img = Image.new('RGB', (60, 30), color = 'red')
                img.save(os.path.join(test_image_dir, f"test_img_{i}.jpg"))

                # Dummy Mask (Grayscale, with values 1, 2, 3)
                mask_array = np.random.randint(1, 4, size=(40, 20), dtype=np.uint8)
                mask_img = Image.fromarray(mask_array, mode='L')
                mask_img.save(os.path.join(test_mask_dir, f"test_img_{i}.png")) # Save with .png extension
            except Exception as e:
                print(f"Error creating dummy file {i}: {e}")
                continue # Skip if one file fails

        print(f"Dummy files created in {test_image_dir} and {test_mask_dir}")

        # Check if files were actually created
        print(f"Files in {test_image_dir}: {os.listdir(test_image_dir)}")
        print(f"Files in {test_mask_dir}: {os.listdir(test_mask_dir)}")

        if not os.listdir(test_image_dir) or not os.listdir(test_mask_dir):
             print("Failed to create dummy files for testing. Skipping PetDataset instantiation.")
        else:
            pet_dataset = PetDataset(image_dir=test_image_dir, mask_dir=test_mask_dir)
            print(f"Successfully instantiated PetDataset with {len(pet_dataset)} samples.")

            if len(pet_dataset) > 0:
                img_tensor, mask_tensor = pet_dataset[0]
                print("Image tensor shape:", img_tensor.shape) # Expected: [3, target_size_H, target_size_W]
                print("Mask tensor shape:", mask_tensor.shape)  # Expected: [1, target_size_H, target_size_W]
                print("Mask tensor unique values:", torch.unique(mask_tensor)) # Expected: 0. and 1.
                print("Image tensor type:", img_tensor.dtype)
                print("Mask tensor type:", mask_tensor.dtype)

    except Exception as e:
        print(f"Error during PetDataset test: {e}")
        print("Please ensure you have manually downloaded the Oxford-IIIT Pet dataset and placed it in a directory structure like:")
        print("- unet_pet_project/data/images/")
        print("- unet_pet_project/data/annotations/trimaps/")
        print("Then update the image_dir and mask_dir paths in a test script accordingly.")
    finally:
        # Clean up dummy directories and files
        if os.path.exists(test_image_dir):
            for f in os.listdir(test_image_dir):
                os.remove(os.path.join(test_image_dir, f))
            os.rmdir(test_image_dir)
        if os.path.exists(test_mask_dir):
            for f in os.listdir(test_mask_dir):
                os.remove(os.path.join(test_mask_dir, f))
            os.rmdir(test_mask_dir)
        print("Cleaned up dummy test files and directories.")
