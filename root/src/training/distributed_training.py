
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
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
import torch.nn.functional as F

# adapted torch imports
from torchsummary import summary

# scipy imports
from scipy.ndimage import rotate


# distributed training imports
import torch.distributed as dist
import torch.multiprocessing as mp

from utils.data_utils import  nmse , gather_tensor
from utils.logger_utils import setup_logger
from utils.config_loader import get_config_path , load_config
from utils.run_utils import run_distributed , test_distributed , setup, cleanup
from utils.checkpoint_utils import load_checkpoint , save_checkpoint

config = load_config()
log_directory = config['log_directory']
checkpoint_dir = config['checkpoint_dir']


def train_model_distributed(rank, world_size, model, train_loader, valid_loader, logger, epochs=10, lr=1e-4, device='cuda', log_directory=log_directory, resume=False, checkpoint_dir=checkpoint_dir, checkpoint_filename="distributed_model_checkpoint_DDP.pth.tar", new_checkpoint=False):
    print(f"Starting distributed training on GPU {rank}")

    # Setup Distributed Training
    setup(rank, world_size)

    # Send model to the appropriate device and wrap with DistributedDataParallel
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Load checkpoint if available
    logger = setup_logger(log_directory, resume=resume, rank=rank)
    start_epoch, train_losses, valid_losses, train_accuracies, valid_accuracies = load_checkpoint(
        checkpoint_filename, model, optimizer, logger, checkpoint_dir=checkpoint_dir, new_checkpoint=new_checkpoint, rank=rank)
    
    # Use DistributedSampler to ensure each process gets a subset of the data
    train_sampler = DistributedSampler(train_loader, num_replicas=world_size, rank=rank)
    valid_sampler = DistributedSampler(valid_loader, num_replicas=world_size, rank=rank)
    
    if rank == 0:
        logger.info("Starting Training")
    
    # Track losses and accuracies
    for epoch in range(start_epoch, epochs):
        model.train()
        running_train_loss = 0.0
        running_train_nmse = 0.0
        train_sampler.set_epoch(epoch)
        
        for i, (image_batch, kspace_batch) in enumerate(train_loader):
            kspace_batch = kspace_batch.to(rank)
            image_batch = image_batch.to(rank)
            
            optimizer.zero_grad()
            outputs = model(image_batch)
            
            loss = criterion(outputs, kspace_batch)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item()
            batch_nmse = nmse(outputs, kspace_batch)
            running_train_nmse += batch_nmse
            
            if rank == 0 and (i + 1) % 100 == 0:
                logger.info(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], "
                            f"Training Loss: {loss.item():.4f}, NMSE: {batch_nmse:.4f}")
        
        # Validation
        model.eval()
        running_valid_loss = 0.0
        running_valid_nmse = 0.0
        with torch.no_grad():
            for image_batch, kspace_batch in valid_loader:
                kspace_batch = kspace_batch.to(rank)
                image_batch = image_batch.to(rank)

                outputs = model(image_batch)
                loss = criterion(outputs, kspace_batch)
                running_valid_loss += loss.item()
                batch_nmse = nmse(outputs, kspace_batch)
                running_valid_nmse += batch_nmse

        # Gather and average metrics across GPUs
        train_loss_tensor = torch.tensor(running_train_loss).to(rank)
        valid_loss_tensor = torch.tensor(running_valid_loss).to(rank)
        train_nmse_tensor = torch.tensor(running_train_nmse).to(rank)
        valid_nmse_tensor = torch.tensor(running_valid_nmse).to(rank)

        # Use all_reduce to sum the metrics from all ranks
        torch.distributed.all_reduce(train_loss_tensor, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(valid_loss_tensor, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(train_nmse_tensor, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(valid_nmse_tensor, op=torch.distributed.ReduceOp.SUM)

        # Average metrics
        avg_train_loss = train_loss_tensor.item() / world_size
        avg_valid_loss = valid_loss_tensor.item() / world_size
        avg_train_nmse = train_nmse_tensor.item() / world_size
        avg_valid_nmse = valid_nmse_tensor.item() / world_size

        if rank == 0:
            logger.info(f"Epoch [{epoch+1}/{epochs}] completed. "
                        f"Avg Train Loss: {avg_train_loss:.4f}, Train NMSE: {avg_train_nmse:.4f}, "
                        f"Avg Valid Loss: {avg_valid_loss:.4f}, Valid NMSE: {avg_valid_nmse:.4f}")

            # Save checkpoint
            state = {
                'epoch': epoch + 1,
                'state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_losses': avg_train_loss,
                'valid_losses': avg_valid_loss,
                'train_accuracies': avg_train_nmse,
                'valid_accuracies': avg_valid_nmse
            }
            save_checkpoint(state, checkpoint_dir, checkpoint_filename, logger, rank)

    cleanup()
    return model, avg_train_loss, avg_valid_loss, avg_train_nmse, avg_valid_nmse