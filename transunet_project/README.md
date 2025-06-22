# TransUNet Project for Image Segmentation

This project implements the TransUNet model and provides a framework for training, testing, and evaluating image segmentation tasks, with a focus on medical image datasets. The implementation allows for using a base TransUNet model or a modified version.

## Project Structure

```
transunet_project/
|-- models/                       # Core model implementations
|   |-- __init__.py
|   |-- transunet.py              # Main TransUNet model
|   |-- vision_transformer.py     # Vision Transformer (ViT) components
|   |-- cnn_backbone.py           # CNN backbone (e.g., ResNet)
|   |-- blocks.py                 # Basic neural network building blocks
|   |-- attention_modules.py      # (Optional) Advanced attention mechanisms
|-- utils/                        # Utility functions
|   |-- __init__.py
|   |-- data_utils.py             # Data loading, preprocessing, augmentation
|   |-- training_utils.py         # Training loops, loss functions, optimizers
|   |-- eval_utils.py             # Evaluation metrics and calculation
|   |-- viz_utils.py              # Visualization tools for results and curves
|-- gui/                          # Graphical User Interface (Tkinter/PyQt)
|   |-- __init__.py
|   |-- app.py                    # Main GUI application
|   |-- widgets.py                # Custom GUI widgets (if any)
|-- main_train.py                 # Script for model training (command-line)
|-- main_test.py                  # Script for model testing (command-line)
|-- requirements.txt              # Python package dependencies
|-- README.md                     # This file
|-- datasets/                     # (Placeholder) Directory for datasets
|   |-- dataset1/
|   |   |-- images/
|   |   |-- masks/
|   |-- ...
|-- saved_models/                 # (Placeholder) Directory for trained models
```

## Features

*   **TransUNet Model**: Implementation of the TransUNet architecture from scratch, combining a CNN backbone with a Vision Transformer encoder and a cascaded upsampler (CUP) decoder.
*   **Customizable Backbone**: Supports different CNN backbones (e.g., ResNet18, ResNet50-like).
*   **Customizable ViT**: Configurable Vision Transformer parameters (depth, heads, embed dimension).
*   **Data Handling**: Includes utilities for dataset loading, preprocessing, and (planned) data augmentation.
*   **Training and Evaluation**: Scripts and utilities for training models, evaluating performance with various metrics (Dice, Jaccard/IoU, accuracy, precision, recall, F1, ROC/PR curves, confusion matrix), and visualizing results.
*   **GUI (Planned)**: A Tkinter or PyQt based graphical user interface for:
    *   Selecting datasets and models.
    *   Configuring training parameters.
    *   Initiating training and monitoring progress (loss/metric curves).
    *   Loading trained models.
    *   Testing models on new images and viewing segmentation results.
    *   Displaying performance metrics.
*   **Modularity**: Code is organized into modules for models, utilities, and GUI for better maintainability.

## Setup

1.  **Clone the repository (if applicable)**
    ```bash
    # git clone ...
    # cd transunet_project
    ```
2.  **Create a virtual environment (recommended)**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```
    If `scipy` is needed for Hausdorff distance and not included by default:
    ```bash
    pip install scipy
    ```

## Datasets
The project is designed to work with various image segmentation datasets. The `utils/data_utils.py` provides a `SegmentationDataset` class that can be adapted. Expected dataset structure:
```
datasets/your_dataset_name/
  images/
    img1.png
    img2.jpg
    ...
  masks/
    img1_mask.png  # Mask names should correspond to image names
    img2_mask.png
    ...
```
Modify `data_utils.py` if your dataset has a different naming convention or structure. Links to example datasets are provided in the course design document.

## Usage (Command Line - Basic Examples)

(These scripts `main_train.py` and `main_test.py` will be developed in later steps)

**Training:**
```bash
python main_train.py --dataset_path path/to/your_dataset \
                     --model_type TransUNet_ResNet18 \
                     --epochs 50 \
                     --batch_size 4 \
                     --lr 1e-4 \
                     --output_dir saved_models/my_transunet_run
```

**Testing:**
```bash
python main_test.py --dataset_path path/to/your_test_dataset \
                    --model_path saved_models/my_transunet_run/best_model.pth \
                    --output_dir test_results/my_transunet_run
```

## GUI Usage (Planned)

Once the GUI is implemented (`gui/app.py`), it can be launched with:
```bash
python gui/app.py
```
The GUI will provide an interactive way to perform training, testing, and result visualization.

## Model Implementation Details

*   **CNN Backbone (`models/cnn_backbone.py`)**:
    *   Currently implements a ResNet-like structure using `ResidualBlock` (BasicBlock from ResNet).
    *   Provides feature maps at different scales (e.g., H/4, H/8, H/16, H/32) for skip connections and ViT input.
    *   To match the original TransUNet paper's ResNet50 (which uses Bottleneck blocks and has larger channel dimensions), the `ResNetBackbone` and `ResidualBlock` would need to be updated to include a Bottleneck implementation.
*   **Vision Transformer (`models/vision_transformer.py`)**:
    *   Implements a standard ViT encoder with multi-head self-attention and MLP blocks.
    *   Takes the flattened feature map from the CNN backbone (e.g., H/32 x W/32) as input patches (patch size typically 1x1 for TransUNet).
    *   Includes learnable positional embeddings.
*   **TransUNet (`models/transunet.py`)**:
    *   Integrates the CNN backbone and ViT encoder.
    *   The ViT output is reshaped into a 2D feature map.
    *   A decoder (Cascaded Upsampler - CUP) uses `DecoderBlock` modules to upsample the features and merge them with skip connections from the CNN backbone at corresponding resolutions.
    *   The final layer is a segmentation head to produce class logits.

## TODO / Future Work (aligned with plan)

*   Implement full data augmentation pipeline in `utils/data_utils.py` (Step 4).
*   Implement迁移学习 (transfer learning) and fine-tuning capabilities (Step 5).
*   Develop comprehensive training (`main_train.py`) and testing (`main_test.py`) scripts (Steps 6, 7).
*   Design and implement the Tkinter/PyQt GUI (`gui/app.py`) (Step 8).
*   Refine the ResNet50 backbone to use Bottleneck blocks for closer replication of the original TransUNet paper, if required for performance.
*   Implement and test specific TransUNet modifications as per the user's request (Step 2 & 3).

## Requirements File
The `requirements.txt` lists the necessary Python packages. Ensure they are installed in your environment.
```
torch>=1.8.0
torchvision>=0.9.0
numpy>=1.19.0
matplotlib>=3.3.0
scikit-learn>=0.24.0
Pillow>=8.0.0
einops>=0.3.0
# scipy (optional)
```
This README provides a good overview of the project structure and components created so far.
The core model files (`transunet.py`, `vision_transformer.py`, `cnn_backbone.py`, `blocks.py`) and the basic utility files (`data_utils.py`, `training_utils.py`, `eval_utils.py`, `viz_utils.py`) have been created with foundational code.
The `requirements.txt` and this `README.md` complete the initial file setup.
This concludes the "搭建基础的TransUNet模型" (Build basic TransUNet model) step.
The next steps will involve researching and implementing improvements, developing data augmentation, training/testing scripts, and the GUI.
