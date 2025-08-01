import nibabel as nib
import os
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torch
import numpy as np
from nilearn import plotting
import torchio as tio
import torch.nn as nn

def plot_tensor(tensor, channel_index=0, affine=None):
    """
    Converts a PyTorch tensor to a NIfTI image object, selecting a specific channel,
    and plots the resulting image.

    Parameters:
    - tensor (torch.Tensor): Input tensor with shape (C, D, H, W) where C is the number of channels.
    - channel_index (int): Index of the channel or time point to extract. Default is 0.
    - affine (np.ndarray, optional): Affine transformation matrix for the NIfTI image. If None, an identity matrix is used.

    Returns:
    - None: Displays the NIfTI image using Nilearn's plotting functions.
    """
    # Convert the PyTorch tensor to a NumPy array
    numpy_array = tensor.numpy()

    # Ensure the tensor is 4D, i.e., (C, D, H, W)
    if numpy_array.ndim == 4:
        # Check if channel_index is within bounds
        if channel_index >= numpy_array.shape[0]:  # Channels are now in the 0th dimension
            raise ValueError(f"Channel index {channel_index} out of range. Tensor has {numpy_array.shape[0]} channels.")
        
        # Extract the specified channel by selecting the first dimension (channels)
        numpy_array = numpy_array[channel_index, :, :, :]
    elif numpy_array.ndim == 3:
        # For 3D tensor, use it directly
        pass
    else:
        raise ValueError("Tensor must be 3D or 4D.")

    # Create an affine matrix if not provided
    if affine is None:
        affine = np.eye(4)

    # Create a NIfTI image
    nifti_image = nib.Nifti1Image(numpy_array, affine)
    
    # Plot the NIfTI image
    plotting.plot_img(nifti_image, display_mode='ortho', title='MRI Image Visualization')
    plotting.show()



train_transforms = tio.Compose([
    tio.RescaleIntensity(out_min_max=(0, 1)),  # Rescale intensity
    tio.Lambda(lambda x: x.permute(3, 0, 1, 2)),  # Move the channel dimension to the front
    tio.Resample((2, 2, 2)),  # Apply resampling to the spatial dimensions only
    tio.Lambda(lambda x: x.permute(1, 2, 3, 0)),  # Move the channel dimension back to the last position
    tio.RandomAffine(
        scales=(0.8, 1.4),
        degrees=(30, 30, 30),
        translation=(10, 15, 5),
        isotropic=False
    ),
    tio.RandomFlip(axes=(0, 1, 2), flip_probability=0.5)
])


class MedicalImageDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None, limit=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform

        # Retrieve and sort paths
        self.image_paths = sorted([os.path.join(images_dir, fname) for fname in os.listdir(images_dir) if fname.endswith('.nii.gz')])
        self.label_paths = sorted([os.path.join(labels_dir, fname) for fname in os.listdir(labels_dir) if fname.endswith('.nii.gz')])

        # Limit to the first `limit` files if specified
        if limit:
            self.image_paths = self.image_paths[:limit]
            self.label_paths = self.label_paths[:limit]

        # Verify the dataset length
        if len(self.image_paths) == 0:
            raise ValueError("No image files found.")
        if len(self.image_paths) != len(self.label_paths):
            raise ValueError("Number of images and labels do not match.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = nib.load(self.image_paths[idx]).get_fdata()
        label = nib.load(self.label_paths[idx]).get_fdata()

        image = torch.tensor(image, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)  # For segmentation masks
        
        if label.ndim == 3:  # If label is 3D, unsqueeze and repeat it to make it 4D
            label = label.unsqueeze(-1)  # Add a new dimension at the end, shape becomes (240, 240, 155, 1)
            label = label.repeat(1, 1, 1, 4)  # Repeat the label 4 times along the last dimension


        # Create a TorchIO Subject with both image and label
        subject = tio.Subject(
            image=tio.ScalarImage(tensor=image),  # Image is wrapped in tio.ScalarImage
            label=tio.LabelMap(tensor=label)  # Label wrapped in tio.LabelMap
        )


        # Apply the transform to both the image and the label
        if self.transform:
            subject = self.transform(subject)

        image = subject['image'].data.permute(3, 0, 1, 2)  # Ensure channels are in the first dimension
        label = subject['label'].data.permute(3, 0, 1, 2)  # Do the same for the label
        # Return the transformed image and label
        return image, label


# Define directories
images_dir = '/home/rob/Desktop/abstract/Task01_BrainTumour/imagesTr/'
labels_dir = '/home/rob/Desktop/abstract/Task01_BrainTumour/labelsTr/'

# Instantiate dataset and dataloader with limit of 10 files
dataset = MedicalImageDataset(images_dir, labels_dir, limit=10, transform= None)
image, label = dataset[0]
print(image.shape)
print(label.shape)
plot_tensor(image)
plot_tensor(label)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)  # Set batch_size to 1
for i, (image, label) in enumerate(dataloader):
    print(f"Batch {i+1}:")
    print(f"Image shape: {image.shape}")
    print(f"Label shape: {label.shape}")
    
    # Break after first batch if you only want to check the first batch
    break
# print(type(image))
# image = nib.load('/home/rob/Desktop/abstract/Task01_BrainTumour/imagesTr/BRATS_004.nii.gz').get_fdata()
# label = nib.load('/home/rob/Desktop/abstract/Task01_BrainTumour/labelsTr/BRATS_004.nii.gz').get_fdata()
# print(image.shape)
# print(label.shape)
# image, label = dataset[0]
# print(image.dim())
# print(image.shape)
# dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
# plot_tensor(label, channel_index=0)
