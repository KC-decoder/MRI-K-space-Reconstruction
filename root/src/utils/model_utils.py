
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
# from models.models import UNet
 
# # load training module 
# from training.training import train_model 
# from training.distributed_training import train_model_distributed
 
# from utils.data_utils import analyze_directory , extract_central_slices , filter_valid_files , nmse , setup_training , cleanup_training, load_checkpoint , save_checkpoint , run_distributed_training
# from utils.logger_utils import setup_logger
# from utils.config_loader import get_config_path , load_config

def double_conv(self, in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  # Use padding=1
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),  # Use padding=1
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

def crop_tensor(input_tensor, target_tensor):
    """
    Crops the input_tensor to match the size of the target_tensor.
    Both tensors are assumed to have shape (batch_size, channels, height, width).
    """
    _, _, target_h, target_w = target_tensor.shape
    _, _, input_h, input_w = input_tensor.shape

    # Calculate cropping for height and width
    crop_h = (input_h - target_h) // 2
    crop_w = (input_w - target_w) // 2

    # Adjust the height if the dimensions do not match
    if input_h > target_h:
        input_tensor = input_tensor[:, :, crop_h:crop_h + target_h, :]
    elif input_h < target_h:
        # Pad the tensor if it's smaller
        padding_h = (target_h - input_h) // 2
        input_tensor = F.pad(input_tensor, (0, 0, padding_h, target_h - input_h - padding_h), mode='constant', value=0)

    # Adjust the width if the dimensions do not match
    if input_w > target_w:
        input_tensor = input_tensor[:, :, :, crop_w:crop_w + target_w]
    elif input_w < target_w:
        # Pad the tensor if it's smaller
        padding_w = (target_w - input_w) // 2
        input_tensor = F.pad(input_tensor, (padding_w, target_w - input_w - padding_w, 0, 0), mode='constant', value=0)

    return input_tensor