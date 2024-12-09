
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
 
 
# # load training module 
# #from training.training import train_model 
# from training.distributed_training import train_model_distributed

# from utils.data_utils import analyze_directory , extract_central_slices , filter_valid_files , nmse , setup_training , cleanup_training, load_checkpoint , save_checkpoint , run_distributed_training
# from utils.model_utils import double_conv , crop_tensor 
# from utils.logger_utils import setup_logger

def get_config_path():
    # Get the absolute path to the src directory
    src_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Traverse upwards until you reach the root directory that contains 'config'
    while not os.path.exists(os.path.join(src_dir, 'config/config.yml')):
        src_dir = os.path.dirname(src_dir)
        
    # Construct the absolute path to the config file
    return os.path.join(src_dir, 'config/config.yml')


def load_config():
    config_path = get_config_path()
    with open(config_path, 'r') as file:
        cfg = yaml.safe_load(file)

    # Get the actual root_dir from the config
    root_dir = cfg.get('root_dir')

    # Replace ${root_dir} in all paths with the actual root directory path
    for key, value in cfg.items():
        if isinstance(value, str) and "${root_dir}" in value:
            cfg[key] = value.replace("${root_dir}", root_dir)

    return cfg