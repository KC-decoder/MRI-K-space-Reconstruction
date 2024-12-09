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
checkpoint_dir = config['checkpoint_dir']


# Load checkpoint function
def load_checkpoint(checkpoint_filename, model, optimizer, logger,checkpoint_dir = checkpoint_dir, new_checkpoint=True, rank=0, distributed=False):
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
    
    if not new_checkpoint and os.path.isfile(checkpoint_path):
        if rank == 0:
            logger.info(f"Loading checkpoint '{checkpoint_path}'")
        if distributed:
            # Use `torch.distributed` compatible load
            map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}  # Map model to the correct device
            checkpoint = torch.load(checkpoint_path, map_location=map_location)
        else:
            # Normal single GPU or CPU loading
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda', torch.cuda.current_device()))
            logger.info(f"Loading checkpoint '{checkpoint_path}'")

        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        train_losses = checkpoint['train_losses']
        valid_losses = checkpoint['valid_losses']
        train_accuracies = checkpoint['train_accuracies']
        valid_accuracies = checkpoint['valid_accuracies']

        if rank == 0:
            logger.info(f"Loaded checkpoint from epoch {start_epoch}")
    else:
        if rank == 0:
            if new_checkpoint:
                logger.info(f"Creating a new checkpoint at '{checkpoint_path}'")
            else:
                logger.info(f"No checkpoint found at '{checkpoint_path}', starting fresh.")
        start_epoch = 0
        train_losses = []
        valid_losses = []
        train_accuracies = []
        valid_accuracies = []

    return start_epoch, train_losses, valid_losses, train_accuracies, valid_accuracies


# Save checkpoint function
def save_checkpoint(state, checkpoint_dir, checkpoint_filename, logger, rank=0, distributed=False):
    """
    Save the model checkpoint based on whether distributed training is used.
    - If distributed is True, only rank 0 saves the checkpoint.
    - If distributed is False, saving happens as usual on single GPU.
    """
    if not distributed or (distributed and rank == 0):  # Save checkpoint only for rank 0 in distributed mode
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
        torch.save(state, checkpoint_path)
        logger.info(f"Checkpoint saved at {checkpoint_dir}/{checkpoint_filename}")