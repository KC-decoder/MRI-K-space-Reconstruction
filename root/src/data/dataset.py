# IMPORT LIBRARIES
# general imports
import os
import yaml
import random
import logging
import h5py
from datetime import datetime


# python math imports
import numpy as np
import pandas as pd

# plotting imports
import matplotlib.pyplot as plt
import seaborn as sns

# META fastmri imports
import fastmri
from fastmri.data import transforms as T
from fastmri.data.mri_data import fetch_dir
from fastmri.data.transforms import UnetDataTransform
from fastmri.pl_modules import FastMriDataModule
from fastmri.pl_modules import UnetModule

# pytorch imports
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import torch.nn as nn
import torch.nn.functional as F

# adapted torch imports
from torchsummary import summary

# scipy imports
from scipy.ndimage import rotate


# distributed training imports
import torch.distributed as dist
import torch.multiprocessing as mp
 
 
# # load model 
from models.models import UNet
 
# # load training module 
from training.training import train_model 
from training.distributed_training import train_model_distributed
 
from utils.config_loader import load_config
from utils.data_utils import analyze_directory , extract_central_slices , filter_valid_files , nmse
from utils.checkpoint_utils import  load_checkpoint , save_checkpoint 
from utils.model_utils import double_conv , crop_tensor 
from utils.logger_utils import setup_logger
from utils.config_loader import get_config_path , load_config



class AxialMRIDataset(Dataset):
    def __init__(self, valid_files, root_dir, num_central_slices=300, max_height=368, max_width=50):
        """
        Initialize the dataset by loading the list of valid .h5 files and setting parameters.
        :param valid_files: List of valid .h5 files.
        :param num_central_slices: Number of central axial slices to extract (default is 300).
        :param max_height: Maximum height to filter slices (default is 368).
        :param max_width: Maximum width for padding/truncating (default is 50).
        """
        
        self.valid_files = valid_files
        self.root_dir =  root_dir
        self.num_central_slices = num_central_slices
        self.max_height = max_height  # We will filter based on this height
        self.max_width = max_width

    def extract_central_slices(self, axial_slices):
        """
        Extract the central 'n' slices along the axial direction.
        :param axial_slices: Tensor of axial slices (shape: slice', height', width').
        """
        num_slices = axial_slices.shape[0]  # This is the slice' dimension (axial slices)
        center_slice = num_slices // 2
        start_idx = center_slice - (self.num_central_slices // 2)
        end_idx = start_idx + self.num_central_slices
        return axial_slices[start_idx:end_idx, :, :]

    def visualize_axial_image(self, image):
        """
        Visualize the axial image sample.
        :param image: The image data tensor with shape (1, 368, 50).
        """

        # Convert to numpy for visualization
        axial_image = image.numpy()
        print(axial_image.shape)

        # Plot the axial image
        plt.figure(figsize=(8, 8))
        plt.imshow(axial_image[0], cmap='gray')
        plt.title('Axial Image')
        plt.axis('off')
        plt.show()

    def pad_to_final_shape(self, data, target_shape):
        """
        Pad/truncate the input data (image or k-space) to the desired shape.
        :param data: Input tensor of shape (channels, height, width).
        :param target_shape: The desired shape, e.g., (channels, 368, 50).
        """
        current_shape = data.shape
        pad_height = max(0, target_shape[1] - current_shape[1])  # Difference in height
        pad_width = max(0, target_shape[2] - current_shape[2])    # Difference in width

        # Apply symmetric padding
        padding = (pad_width // 2, pad_width - (pad_width // 2),  # Left-Right padding
                   pad_height // 2, pad_height - (pad_height // 2))  # Top-Bottom padding

        # Apply padding or trimming
        padded_data = F.pad(data, padding, mode='constant', value=0)

        return padded_data

    def __len__(self):
        return len(self.valid_files) * self.num_central_slices

    def __getitem__(self, idx):
        # Find which file this index corresponds to
        file_idx = idx // self.num_central_slices
        slice_idx = idx % self.num_central_slices
        
        file_name = self.valid_files[file_idx]
        file_path = os.path.join(self.root_dir, file_name)

        # Load the k-space data from the H5 file
        with h5py.File(file_path, 'r') as f:
            kspace_data = f['kspace'][:]  # Shape: (slice, height, width)
        
        kspace2 = T.to_tensor(kspace_data)
        image = fastmri.ifft2c(kspace2)
        image_abs = fastmri.complex_abs(image)
        
        # Transpose the k-space data to get axial slices: new shape (height', width', slice')
        # axial_slices = np.transpose(image_abs, (1, 2, 0))
        
        # Extract central 'n' axial slices (along height dimension, now treated as slice')
        central_axial_slices = self.extract_central_slices(axial_slices)
        
        # Get the specific axial slice for this sample
        axial_sample = central_axial_slices[slice_idx]
        
        # Step 2: Normalize the image
        image_mean = axial_sample.mean()
        image_std = axial_sample.std()
        image_normalized = (axial_sample - image_mean) / image_std
        
        # Pad or truncate the normalized image to (368, 50)
        target_image = self.pad_to_final_shape(image_normalized.unsqueeze(0), (1, self.max_height, self.max_width))
        
        # Step 3: Convert the normalized image back to k-space
        image_complex = torch.stack([image_normalized, torch.zeros_like(image_normalized)], dim=-1)
        input_kspace = fastmri.fft2c(image_complex)  # Forward Fourier transform back to k-space

        # **Pad or truncate k-space**
        input_kspace_padded = self.pad_to_final_shape(input_kspace.permute(2, 0, 1), (2, self.max_height, self.max_width))

        # Return final sample: target (normalized and padded k-space) and input (padded image)
        return input_kspace_padded, target_image
    
    
    
    
class SagittalMRIDataset(Dataset):
    def __init__(self, valid_files, root_dir, num_slices=15, max_height=450, max_width=500):
        """
        Initialize the dataset by loading the list of valid .h5 files and setting parameters.
        :param valid_files: List of valid .h5 files.
        :param num_slices: Number of sagittal slices to extract (default is 37).
        :param max_height: Maximum height for padding/truncating (default is 450).
        :param max_width: Maximum width for padding/truncating (default is 500).
        """
        
        self.valid_files = valid_files
        self.root_dir = root_dir
        self.num_slices = num_slices
        self.max_height = max_height
        self.max_width = max_width
        
        
        
        

    # Function to extract central slices from the second dimension (width 640)
    def extract_central_slices(self, kspace_data, num_central_slices = 15):
        """
        Extract central 'n' slices along the width dimension (640).
        :param kspace_data: The k-space data with shape (num_slices, height, width).
        :param num_central_slices_width: The number of central width slices to extract (300 in this case).
        :return: The k-space data with central width slices extracted.
        """
        # Extract slice dimension
        num_width = kspace_data.shape[0]  # slice
        center_width = num_width // 2  # Center of slice dimension

        # Start and end indices for central width slices
        start_idx = center_width - (num_central_slices // 2)
        end_idx = start_idx + num_central_slices

        # Extract the central width slices
        return kspace_data[start_idx:end_idx,:,: ]  # Apply along the first dimension
    
    
    
    

    def pad_to_final_shape(self, data, target_shape):
        """
        Pad or truncate the input data (image or k-space) to the desired shape.
        :param data: Input tensor of shape (channels, height, width).
        :param target_shape: The desired shape, e.g., (channels, 450, 500).
        """
        current_shape = data.shape
        pad_height = max(0, target_shape[1] - current_shape[1])
        pad_width = max(0, target_shape[2] - current_shape[2])

        # Apply symmetric padding
        padding = (pad_width // 2, pad_width - (pad_width // 2),  # Left-Right padding
                pad_height // 2, pad_height - (pad_height // 2))  # Top-Bottom padding

        # Apply padding or trimming as needed
        padded_data = F.pad(data, padding, mode='constant', value=0)
        
        # If the height or width of `padded_data` exceeds the target, truncate it.
        if padded_data.size(1) > target_shape[1]:  # Truncate height if needed
            start_idx = (padded_data.size(1) - target_shape[1]) // 2
            padded_data = padded_data[:, start_idx:start_idx + target_shape[1], :]
            
        if padded_data.size(2) > target_shape[2]:  # Truncate width if needed
            start_idx = (padded_data.size(2) - target_shape[2]) // 2
            padded_data = padded_data[:, :, start_idx:start_idx + target_shape[2]]

        return padded_data
    
    
    
    def __len__(self):
        return len(self.valid_files) * self.num_slices
    
    
    

    def __getitem__(self, idx):
        # Find which file and slice this index corresponds to
        file_idx = idx // self.num_slices
        slice_idx = idx % self.num_slices
        
        file_name = self.valid_files[file_idx]
        file_path = os.path.join(self.root_dir, file_name)

        # Load the k-space data from the H5 file
        with h5py.File(file_path, 'r') as f:
            kspace_data = f['kspace'][:]  # Shape: (depth, height, width)

        # Check that the volume has enough slices for the intended number of sagittal slices
        if kspace_data.shape[0] < self.num_slices:
            raise ValueError(f"File {file_name} does not contain enough slices. Found {kspace_data.shape[0]}, expected at least {self.num_slices}.")

        # Convert the full k-space volume to the image domain
        kspace_tensor = T.to_tensor(kspace_data)  # Convert to torch tensor
        image_volume = fastmri.ifft2c(kspace_tensor)  # Convert k-space to image domain
        image_abs_volume = fastmri.complex_abs(image_volume)  # Get the magnitude images
        image_abs_volume_np = image_abs_volume.numpy()# Convert to numpy for slicing
        
        #print(image_abs_volume_np.shape)

        # Extract central slices from the image domain
        sagittal_slices = self.extract_central_slices(image_abs_volume_np)  # Extract central slices

        # Ensure slice_idx is within bounds after extracting central slices
        if slice_idx >= sagittal_slices.shape[0]:
            raise IndexError(f"slice_idx {slice_idx} is out of bounds for sagittal_slices with shape {sagittal_slices.shape}")

        # Get the specific sagittal slice for this sample
        sagittal_sample = sagittal_slices[slice_idx]  # Shape should be (height, width)
        sagittal_sample = torch.tensor(sagittal_sample)  # Convert back to tensor

        # Step 1: Normalize the image
        image_mean = sagittal_sample.mean()
        image_std = sagittal_sample.std()
        image_normalized = (sagittal_sample - image_mean) / image_std

        # Pad or truncate the normalized image to (450, 500)
        output_image = self.pad_to_final_shape(image_normalized.unsqueeze(0), (1, self.max_height, self.max_width))
        
        # Step 2: Convert the normalized image back to k-space with separate real and imaginary channels
        image_complex = torch.stack([image_normalized, torch.zeros_like(image_normalized)], dim=-1)
        input_kspace = fastmri.fft2c(image_complex.unsqueeze(0))  # Forward Fourier transform to k-space
        
        # Pad/truncate k-space to the target shape (2, 450, 500)
        input_kspace_padded = self.pad_to_final_shape(input_kspace.squeeze(0).permute(2, 0, 1), (2, self.max_height, self.max_width))

        # Return final sample: input image and target k-space with desired shapes
        return input_kspace_padded, output_image