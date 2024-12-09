
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
import torch.distributed as dist

# adapted torch imports
from torchsummary import summary

# scipy imports
from scipy.ndimage import rotate


# distributed training imports
import torch.distributed as dist
import torch.multiprocessing as mp
 
 
# # load model 
# from models.models import UNet
 
# # load training module 
# # from training.training import train_model 
# from training.distributed_training import train_model_distributed
 
from utils.model_utils import double_conv , crop_tensor 
# from utils.logger_utils import setup_logger
from utils.config_loader import get_config_path , load_config

# Function to analyze files in a directory

config = load_config()
def analyze_directory(directory_path):
    file_count = 0
    total_size = 0
    file_types = {}
    
    for root, _, files in os.walk(directory_path):
        for file in files:
            file_count += 1
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path)
            total_size += file_size
            
            file_extension = os.path.splitext(file)[1]
            if file_extension in file_types:
                file_types[file_extension] += 1
            else:
                file_types[file_extension] = 1
    
    return file_count, total_size, file_types




def seed_everything():
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    random.seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['seed'])
        torch.cuda.manual_seed_all(config['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False





# Function to extract central slices from the second dimension (width 640)
def extract_central_slices(kspace_data, num_central_slices):
    """
    Extract central 'n' slices along the width dimension (640).
    :param kspace_data: The k-space data with shape (num_slices, height, width).
    :param num_central_slices_width: The number of central width slices to extract (300 in this case).
    :return: The k-space data with central width slices extracted.
    """
    # Extract width (640 dimension)
    num_width = kspace_data.shape[1]  # 640
    center_width = num_width // 2  # Center of 640

    # Start and end indices for central width slices
    start_idx = center_width - (num_central_slices // 2)
    end_idx = start_idx + num_central_slices

    # Extract the central width slices
    return kspace_data[start_idx:end_idx,:,: ]  # Apply along the first dimension





def filter_valid_files(root_dir, max_height=368, CFG=None):
    """
    Preprocess and filter valid files based on the height' dimension, with an optional debug flag.
    :param root_dir: Path to the directory containing .h5 files.
    :param max_height: Maximum allowable height' (default is 368).
    :param CFG: Configuration dictionary. Expects 'debug' and 'debug_input_size' keys if debugging is enabled.
    :return: List of valid file paths where height' <= max_height.
    """
    valid_files = []
    for file_name in os.listdir(root_dir):
        if file_name.endswith('.h5'):
            file_path = os.path.join(root_dir, file_name)
            with h5py.File(file_path, 'r') as f:
                kspace_data = f['kspace'][:]
                # Transpose to axial orientation and check height'
                axial_slices = np.transpose(kspace_data, (1, 2, 0))
                height_prime = axial_slices.shape[1]  # height' is in the second dimension
                if height_prime <= max_height:
                    valid_files.append(file_name)
    
    # If debug mode is enabled, sample from the valid files
    if config["debug"]:
        valid_files_df = pd.DataFrame(valid_files, columns=['file_name'])  # Convert list to DataFrame for sampling
        
        if 0 < config["debug_input_size"] <= 1:  # If debug_input_size is a percentage
            sample_size = int(config["debug_input_size"] * len(valid_files_df))  # Calculate the number of files to sample
        else:
            sample_size = config["debug_input_size"]  # If it's a fixed number
        
        # Ensure the sample size does not exceed the available files
        sample_size = min(sample_size, len(valid_files_df))
        
        valid_files_sampled = valid_files_df.sample(sample_size)
        valid_files = valid_files_sampled['file_name'].tolist()  # Convert back to a list
    return valid_files




# Define NMSE function
def nmse(predicted, target):
    """
    Calculate the Normalized Mean Squared Error (NMSE) between the predicted and target k-space data.
    :param predicted: The predicted k-space data (output of the model).
    :param target: The ground truth k-space data (actual values).
    :return: NMSE value.
    """
    mse = torch.mean((predicted - target) ** 2)
    norm_factor = torch.mean(target ** 2)
    nmse_value = mse / norm_factor
    return nmse_value.item()



def gather_tensor(tensor):
    """
    Helper function to gather tensors from all ranks.
    """
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_tensors, tensor)
    return gathered_tensors