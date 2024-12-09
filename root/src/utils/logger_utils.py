
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
 
 
# load model 
from models.models import UNet
 
# # load training module 
# from training.training import train_model 
# from training.distributed_training import train_model_distributed
 
from utils.config_loader import load_config , get_config_path
# from utils.data_utils import analyze_directory , extract_central_slices , filter_valid_files , nmse , setup_training , cleanup_training, load_checkpoint , save_checkpoint , run_distributed_training
# from utils.model_utils import double_conv , crop_tensor 

cfg = load_config()
log_directory = cfg['log_directory']


def setup_logger(log_directory, resume=False, rank=0): 
    """
    Set up a logger that writes logs to a .txt file in the logs directory.

    Args:
        log_directory (str): Directory where the checkpoint is saved.
        resume (bool): Whether training is resuming from an existing checkpoint.
        rank (int): Rank of the process. Only rank 0 will log to the file.

    Returns:
        logger (logging.Logger): Configured logger instance.
    """
    # Ensure logs directory exists
    logs_dir = os.path.join(os.path.dirname(log_directory), 'logs/CUNet')
    os.makedirs(logs_dir, exist_ok=True)

    # Determine log file name based on whether we are resuming training or not
    if resume:
        log_file_name = os.path.basename(log_directory).split('.')[0] + '.txt'
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file_name = f'log_MR_Reconstruction_UNet_CDC{timestamp}.txt'
    
    log_file_path = os.path.join(logs_dir, log_file_name)

    # Configure the logger
    logger = logging.getLogger(f'UNet_{rank}')
    logger.setLevel(logging.INFO)

    # Console handler to output logs to the console for all ranks
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter for both handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    
    # Add console handler to the logger
    if not logger.hasHandlers():  # To prevent adding handlers multiple times
        logger.addHandler(ch)

    # Only add file handler for rank 0 (the main process)
    if rank == 0:
        fh = logging.FileHandler(log_file_path)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger