
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


from utils.data_utils import nmse , seed_everything
from utils.checkpoint_utils import load_checkpoint , save_checkpoint
from utils.config_loader import get_config_path , load_config
from utils.ComplexFunctions_utils import adjust_to_target , complex_ssim, clip_complex_gradients


config = load_config()
checkpoint_dir = config['checkpoint_dir']


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


from utils.data_utils import nmse , seed_everything
from utils.checkpoint_utils import load_checkpoint , save_checkpoint
from utils.config_loader import get_config_path , load_config
from utils.model_utils import crop_tensor




config = load_config()
checkpoint_dir = config['checkpoint_dir']

def train_model(
    model,
    criterion,
    optimizer,
    train_loader,
    valid_loader,
    logger = 'logs.txt',
    epochs=10,
    lr=1e-4,
    device='cuda',
    log_directory= "train_logs",
    checkpoint_dir = checkpoint_dir,
    checkpoint_filename = "MR_Reconstruction_CUNet_Training_1.pth.tar"
):
    """
    Training loop for the UNet_ImageReconstruction model.
    """
    # Set up checkpoint loading
    seed_everything()
    start_epoch, train_losses, valid_losses, train_accuracies, valid_accuracies = load_checkpoint(
        checkpoint_filename, model, optimizer, logger, checkpoint_dir=checkpoint_dir, new_checkpoint = True
    )
    
    # Move model to the device (GPU/CPU)
    model = model.to(device)
    
    for epoch in range(start_epoch, epochs):
        logger.info(f"Starting Training for Epoch {epoch + 1}/{epochs}")
        
        # Training phase
        model.train()
        running_train_loss = 0.0
        running_train_nmse = 0.0
        
        for i, (input_k_space_batch, target_image_batch) in enumerate(train_loader):
            
            
             # Log the shape of the input k-space batch
            #logger.info(f"Input k-space batch shape: {input_k_space_batch.shape}")
            # Move data to the appropriate device
            input_k_space_batch = input_k_space_batch.to(device)
            target_image_batch = target_image_batch.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            predicted_image = model(input_k_space_batch)
            
            # Compute the loss
            loss = criterion(predicted_image, target_image_batch)
            
            # Backward pass
            loss.backward()
            
            
            # Update model parameters
            optimizer.step()
            
            # Accumulate training loss
            running_train_loss += loss.item()
            
            # Compute NMSE accuracy for this batch
            batch_nmse = nmse(predicted_image, target_image_batch)
            running_train_nmse += batch_nmse
            
            # Log training progress every 100 steps
            if (i + 1) % 100 == 0:
                logger.info(
                    f"Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}], "
                    f"Training Loss: {loss.item():.4f}, NMSE: {batch_nmse:.4f}"
                )
        
        # Compute average training loss and NMSE for the epoch
        avg_train_loss = running_train_loss / len(train_loader)
        avg_train_nmse = running_train_nmse / len(train_loader)
        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_nmse)
        
        logger.info(
            f"Epoch [{epoch + 1}/{epochs}] Training Complete. "
            f"Average Training Loss: {avg_train_loss:.4f}, "
            f"Average Training NMSE: {avg_train_nmse:.4f}"
        )
        
        # Validation phase
        logger.info(f"Starting Validation for Epoch {epoch + 1}/{epochs}")
        model.eval()
        running_valid_loss = 0.0
        running_valid_nmse = 0.0
        
        with torch.no_grad():  # No gradient computation during validation
            for input_k_space_batch, target_image_batch in valid_loader:
                # Move data to the device
                input_k_space_batch = input_k_space_batch.to(device)
                target_image_batch = target_image_batch.to(device)
                
                # Forward pass
                predicted_image = model(input_k_space_batch)
                
                # Compute validation loss
                loss = criterion(predicted_image, target_image_batch)
                
                # Compute NMSE accuracy
                batch_nmse = nmse(predicted_image, target_image_batch)
                
                # Accumulate validation loss and NMSE
                running_valid_loss += loss.item()
                running_valid_nmse += batch_nmse
        
        # Compute average validation loss and NMSE
        avg_valid_loss = running_valid_loss / len(valid_loader)
        avg_valid_nmse = running_valid_nmse / len(valid_loader)
        valid_losses.append(avg_valid_loss)
        valid_accuracies.append(avg_valid_nmse)
        
        logger.info(
            f"Epoch [{epoch + 1}/{epochs}] Validation Complete. "
            f"Average Validation Loss: {avg_valid_loss:.4f}, "
            f"Average Validation NMSE: {avg_valid_nmse:.4f}"
        )
        
        # Save the model checkpoint
        state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'train_losses': train_losses,
            'valid_losses': valid_losses,
            'train_accuracies': train_accuracies,
            'valid_accuracies': valid_accuracies,
        }
        save_checkpoint(state, checkpoint_dir, checkpoint_filename, logger)
    
    logger.info("Training Complete.")
    return model, train_losses, valid_losses, train_accuracies, valid_accuracies





def train_CUNet_model(
    model,
    criterion,
    optimizer,
    train_loader,
    valid_loader,
    logger,
    epochs=10,
    lr=1e-4,
    device='cuda',
    log_directory="train_logs",
    checkpoint_dir="checkpoint_dir",
    checkpoint_filename="MR_Reconstruction_CUNet_Training_1.pth.tar",
    accumulation_steps=4
        ):
    # Set up checkpoint loading
    seed_everything()
    start_epoch, train_losses, valid_losses, train_accuracies, valid_accuracies = load_checkpoint(
        checkpoint_filename, model, optimizer, logger, checkpoint_dir=checkpoint_dir, new_checkpoint=True
    )

    # Move model to the device (GPU/CPU)
    model = model.to(device)

    for epoch in range(start_epoch, epochs):
        logger.info(f"Starting Training for Epoch {epoch + 1}/{epochs}")

        # Training phase
        model.train()
        running_train_loss = 0.0
        running_train_ssim = 0.0

        optimizer.zero_grad()

        for i, (input_kspace, target_image) in enumerate(train_loader):
            
            input_kspace = input_kspace.to(device).to(torch.complex64)
            target_image = target_image.to(device)
            
            output = model(input_kspace)
            
            # Adjust output shape to match target
            output = adjust_to_target(output, target_image.shape)
            
            
            # Normalize the output using magnitude clamping for complex types
            output_magnitude = torch.abs(output)
            if not torch.isfinite(output_magnitude).all():
                logger.warning("Non-finite output detected!")

            # Normalize safely
            output_normalized = torch.where(output_magnitude > 1, output / (output_magnitude + 1e-8), output)

            # Normalize the target image for real-valued tensors
            target_normalized = torch.clamp(target_image, 0, 1)

            # Compute loss
            loss = criterion(output_normalized, target_normalized)
            
            # Normalize loss to account for accumulation
            loss = loss / accumulation_steps
            loss.backward()
            
            
            # Apply gradient clipping
            if any(p.grad is not None for p in model.parameters()):
                clip_complex_gradients(model, max_norm=1.0)

            running_train_loss += loss.item() * accumulation_steps
            # Training phase metric calculation
            batch_ssim = complex_ssim(output, target_image).item()
            running_train_ssim += batch_ssim 
        
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            if (i + 1) % 100 == 0:
                logger.info(
                    f"Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}], "
                    f"Training Loss: {running_train_loss / (i + 1):.4f}, Complex SSIM: {running_train_ssim / (i + 1):.4f}"
                )


        avg_train_loss = running_train_loss / len(train_loader)
        avg_train_ssim = running_train_ssim / len(train_loader)
        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_ssim)

        logger.info(
            f"Epoch [{epoch + 1}/{epochs}] Training Complete. "
            f"Average phase regularized MSE Training Loss: {avg_train_loss:.4f}, "
            f"Average Training Complex SSIM: {avg_train_ssim:.4f}"
        )

        # Validation phase
        logger.info(f"Starting Validation for Epoch {epoch + 1}/{epochs}")
        model.eval()
        running_valid_loss = 0.0
        running_valid_ssim = 0.0

        with torch.no_grad():
            for input_kspace, target_image in valid_loader:
                input_kspace = input_kspace.to(device).to(torch.complex64)
                target_image = target_image.to(device)
                output = model(input_kspace)
                output = adjust_to_target(output, target_image.shape)
                
                
                
               # Normalize the output using magnitude clamping for complex types
                output_magnitude = torch.abs(output)
                if torch.isnan(output_magnitude).any() or torch.isinf(output_magnitude).any():
                    print("NaN or Inf detected in output magnitude!")

                # Normalize safely
                output_normalized = torch.where(output_magnitude > 1, output / (output_magnitude + 1e-8), output)

                # Normalize the target image for real-valued tensors
                target_normalized = torch.clamp(target_image, 0, 1)

                # Compute loss
                loss = criterion(output_normalized, target_normalized)
                
                
                # Validation phase metric calculation
                batch_ssim = complex_ssim(output, target_image).item()
   
                running_valid_loss += loss.item()
                running_valid_ssim += batch_ssim 

        avg_valid_loss = running_valid_loss / len(valid_loader)
        avg_valid_ssim = running_valid_ssim / len(valid_loader)
        valid_losses.append(avg_valid_loss)
        valid_accuracies.append(avg_valid_ssim)

        logger.info(
            f"Epoch [{epoch + 1}/{epochs}] Validation Complete. "
            f"Average phase regularized MSE Validation Loss: {avg_valid_loss:.4f}, "
            f"Average Validation Complex SSIM: {avg_valid_ssim:.4f}"
        )

        # Save the model checkpoint
        state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'train_losses': train_losses,
            'valid_losses': valid_losses,
            'train_accuracies': train_accuracies,
            'valid_accuracies': valid_accuracies,
        }
        save_checkpoint(state, checkpoint_dir, checkpoint_filename, logger)

    logger.info("Training Complete.")
    return model, train_losses, valid_losses, train_accuracies, valid_accuracies
