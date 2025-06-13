# U-Net for Oxford-IIIT Pet Dataset Segmentation

This project implements a U-Net model using PyTorch for segmenting pets from the background in the Oxford-IIIT Pet dataset.

## Directory Structure

The project expects the following directory structure:

```
unet_pet_project/
├── data/
│   ├── images/               # Extracted images (e.g., Abyssinian_1.jpg)
│   │   └── ...
│   └── annotations/
│       └── trimaps/          # Extracted trimap masks (e.g., Abyssinian_1.png)
│           └── ...
├── src/
│   ├── __init__.py           # Optional: makes src a package
│   ├── dataset.py            # PyTorch Dataset class for data loading
│   └── model.py              # U-Net model implementation
├── train.py                  # Main script to train the U-Net model
├── unet_pet_model.pth        # Saved model (after training)
└── training_loss_accuracy_curves.png # Saved training curves (after training)
```

## Prerequisites

*   Python 3.7+
*   PyTorch (>=1.8)
*   Torchvision (>=0.9)
*   Pillow (PIL)
*   NumPy
*   Matplotlib

You can install the required Python packages using pip:

```bash
pip install torch torchvision torchaudio
pip install Pillow numpy matplotlib
```

## Dataset Setup

1.  **Download the Dataset**:
    The Oxford-IIIT Pet Dataset is required. You can download it from the official website:
    [https://www.robots.ox.ac.uk/~vgg/data/pets/](https://www.robots.ox.ac.uk/~vgg/data/pets/)
    You will need:
    *   `images.tar.gz`
    *   `annotations.tar.gz`

2.  **Place and Extract**:
    *   Create the `unet_pet_project/data/` directory if it doesn't exist.
    *   Place the downloaded `images.tar.gz` and `annotations.tar.gz` into the `unet_pet_project/data/` directory.
    *   Extract the archives. For example, on Linux/macOS:
        ```bash
        cd unet_pet_project/data/
        tar -xzf images.tar.gz
        tar -xzf annotations.tar.gz
        cd ../..
        ```
    This should result in `unet_pet_project/data/images/` and `unet_pet_project/data/annotations/` (which contains `trimaps/`) directories. The `train.py` script specifically looks for masks in `unet_pet_project/data/annotations/trimaps/`.

## Running the Experiment

1.  **Navigate to the project directory**:
    ```bash
    cd path/to/your/unet_pet_project
    ```

2.  **Run the training script**:
    ```bash
    python train.py
    ```

3.  **Adjust Hyperparameters (Optional)**:
    You can modify hyperparameters such as `LEARNING_RATE`, `BATCH_SIZE`, `EPOCHS`, and `IMG_SIZE` directly in the `train.py` script.

## Output

After running `train.py`, you should expect:

*   **Console Output**: Training progress, including loss and accuracy for each epoch for both training and validation sets.
*   **Saved Model**: The trained model weights will be saved as `unet_pet_model.pth` in the `unet_pet_project/` directory.
*   **Training Curves**: A plot named `training_loss_accuracy_curves.png` showing the training and validation loss and accuracy over epochs will be saved in the `unet_pet_project/` directory.
