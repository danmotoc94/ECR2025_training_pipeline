# 🧠 ECR2025_training_pipeline: 3D Medical Image Processing

This repository contains a foundational pipeline for loading, preprocessing, and visualizing 3D medical image data, such as MRI scans in NIfTI format (`.nii.gz`). It leverages **PyTorch** for data handling and **TorchIO** for robust medical image-specific transformations and augmentations.

---

## 💻 Prerequisites

To run this pipeline, you'll need the following Python libraries. It's recommended to use a virtual environment.

```bash
pip install torch numpy nibabel matplotlib nilearn torchio
```

---

## 📂 Data Structure

This pipeline assumes your data is structured as follows, containing NIfTI files for both images and their corresponding segmentation labels:

```
.
└── Task01_BrainTumour/
    ├── imagesTr/
    │   ├── BRATS_001.nii.gz
    │   ├── BRATS_002.nii.gz
    │   └── ...
    └── labelsTr/
        ├── BRATS_001.nii.gz
        ├── BRATS_002.nii.gz
        └── ...
```

The script is currently configured to look for data in:

```python
images_dir = '/home/rob/Desktop/abstract/Task01_BrainTumour/imagesTr/'
labels_dir = '/home/rob/Desktop/abstract/Task01_BrainTumour/labelsTr/'
```

**Note:** You must update these paths in the script to match your local data directory.

---

## 🛠️ Pipeline Components

The script is divided into three main logical parts: a **Visualization Utility**, a **Custom Dataset Class**, and the **Main Execution Block**.

### 1. Visualization Utility (`plot_tensor`)

This function is critical for debugging and validating the data both before and after transformations.

**Function:** `plot_tensor(tensor, channel_index=0, affine=None)`

**Purpose:** Takes a PyTorch tensor (expected shape `C×D×H×W` or `D×H×W`) and converts it into a NiBabel NIfTI image object. It then uses Nilearn to plot an orthographic view of the 3D volume, showing axial, coronal, and sagittal slices.

**Key Detail:** It correctly handles 4D tensors by extracting a specific channel via `channel_index` before plotting the 3D volume.

---

### 2. Custom Dataset (`MedicalImageDataset`)

This class inherits from `torch.utils.data.Dataset` and manages the data loading process.

**Initialization (`__init__`):**
- Finds and sorts corresponding image and label file paths.
- Includes a `limit` parameter for quickly testing the pipeline with a subset of files.

**Item Retrieval (`__getitem__`):**
- Loads raw data using NiBabel (`nib.load().get_fdata()`).
- Converts NumPy arrays to PyTorch tensors.
- **Crucial Step for Labels:** It handles 3D label tensors by unsqueezing and repeating the label mask to match the expected number of output classes (e.g., 4 classes for a multi-class segmentation task like BraTS).

```python
if label.ndim == 3:
    label = label.unsqueeze(-1)  # (D, H, W, 1)
    label = label.repeat(1, 1, 1, 4)  # (D, H, W, 4) - assuming 4 classes
```

**TorchIO Integration:**
- Wraps the image and label tensors in `tio.ScalarImage` and `tio.LabelMap`, respectively, into a `tio.Subject`. This is essential because TorchIO operations are designed to work on the Subject object, ensuring that geometric transforms applied to the image are identically applied to the label map.

**Permutation:**
- After transformations, the data is permuted from the TorchIO default `(D,H,W,C)` to the PyTorch standard `(C,D,H,W)` before returning.

---

### 3. TorchIO Transformations (`train_transforms`)

This `tio.Compose` sequence defines the preprocessing and data augmentation steps applied to the data.

| **Step**         | **Function**                          | **Purpose**                                                                                                      |
|------------------|---------------------------------------|------------------------------------------------------------------------------------------------------------------|
| **Preprocessing** | `tio.RescaleIntensity`               | Normalizes voxel intensities to a range of (0, 1).                                                              |
|                  | `tio.Resample`                        | Resamples the spatial dimensions (D, H, W) to a target spacing, e.g., (2, 2, 2) mm. This helps standardize input sizes. |
| **Augmentation**  | `tio.RandomAffine`                   | Applies random rotations, scaling, and translations to introduce variance and improve model generalization.     |
|                  | `tio.RandomFlip`                      | Randomly flips the image and label along any of the three axes.                                                 |
| **Tensor Shaping** | `tio.Lambda(lambda x: x.permute(...))` | These are used to move the channel dimension to the front before `tio.Resample` and move it back after it. This is a workaround to ensure `tio.Resample` only operates on spatial dimensions (D, H, W) and not the channel dimension. |

---

## 🚀 How to Run the Script

1. Set up your environment and install dependencies (see **Prerequisites**).
2. Update the `images_dir` and `labels_dir` variables in the script to point to your dataset.
3. Run the Python script:

```bash
python your_script_name.py
```

### Expected Output

The script first loads the first raw image and label, prints their shapes, and plots them using `plot_tensor()`.

```
torch.Size([4, 240, 240, 155])  # Example raw image shape (C, D, H, W)
torch.Size([4, 240, 240, 155])  # Example raw label shape (C, D, H, W)
# (Nilearn plot 1: Image)
# (Nilearn plot 2: Label)
```

It then initializes the DataLoader (currently with `batch_size=1`) and iterates over the first batch to confirm the final shape:

```
Batch 1:
Image shape: torch.Size([1, 4, 240, 240, 155])  # (B, C, D, H, W)
Label shape: torch.Size([1, 4, 240, 240, 155])  # (B, C, D, H, W)
```

The output confirms that the data is correctly loaded and prepared in the standard format for 3D CNNs: `(Batch, Channels, Depth, Height, Width)`.

---

## ▶️ Next Steps

1. **Implement the UNet Model:** Define your 3D deep learning model (e.g., 3D UNet) and integrate it into a training loop. (A suggested implementation is available in `unet_model.py`).

2. **Add More Augmentations:** Explore other TorchIO transforms like `RandomGhosting`, `RandomSpike`, or `RandomNoise` to further enhance data robustness.

3. **Cross-Validation:** Refactor the main block to include a robust cross-validation scheme for training.

4. **Training Loop:** Implement a complete training pipeline with loss functions (e.g., Dice Loss, Cross-Entropy), optimizers, and validation metrics.

---

## 📝 License

This project is provided as-is for educational and research purposes.

---

## 🙏 Acknowledgments

- **TorchIO** for medical image preprocessing and augmentation
- **NiBabel** for NIfTI file handling
- **Nilearn** for visualization
- **BraTS Dataset** for brain tumor segmentation data
