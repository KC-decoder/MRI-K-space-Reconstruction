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
 
from utils.model_utils import double_conv , crop_tensor 
# from utils.logger_utils import setup_logger
from utils.config_loader import get_config_path , load_config

# Function to analyze files in a directory

config = load_config()





# Setup for Distributed Training
def setup(rank, world_size):
    dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:29500', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)



def cleanup():
    dist.destroy_process_group()
    
 
 
def test_distributed(rank, world_size):
    setup(rank, world_size)

    # Create a tensor with the value equal to the rank
    tensor = torch.tensor([rank], dtype=torch.float32).to(rank)

    print(f"\nBefore All-Reduce, Rank {rank}: {tensor}, device='cuda:{rank}'", flush = True)

    # Perform an All-Reduce operation (sum up all tensors across ranks)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    print(f"\nAfter All-Reduce, Rank {rank}: {tensor}, device='cuda:{rank}'", flush = True)

    cleanup()




def run_distributed(world_size):
    processes = []
    for rank in range(world_size):
        p = mp.Process(target=test_distributed, args=(rank, world_size))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()



def run_distributed_training(model, train_loader, valid_loader, world_size, logger, epochs=10, lr=1e-4, device='cuda', log_directory="log_dir", resume=False, checkpoint_dir="checkpoints", checkpoint_filename="model_checkpoint.pth", new_checkpoint=False):
    logger.info(f"Distributed Training")
    checkpoint_filename = "distributed_model_checkpoint_DDP.pth.tar"
    from training.distributed_training import train_model_distributed
    mp.spawn(train_model_distributed,
             args=(world_size, model, train_loader, valid_loader,logger, epochs, lr, device, log_directory, resume, checkpoint_dir, checkpoint_filename, new_checkpoint),
             nprocs=world_size,
             join=True)
    
    
    
    


def visualize_reconstruction(model, dataloader, device, save_dir, sample_idx):
    model.eval()  # Set model to evaluation mode

    # Calculate the batch index and index within the batch
    batch_size = dataloader.batch_size
    batch_idx = sample_idx // batch_size  # Determine which batch
    sample_within_batch_idx = sample_idx % batch_size  # Position within that batch

    # Iterate through the dataloader to locate the desired batch
    for current_batch_idx, (kspace_batch, image_batch) in enumerate(dataloader):
        if current_batch_idx == batch_idx:
            # Move the k-space and image data to the appropriate device (GPU or CPU)
            kspace_batch = kspace_batch.to(device)
            image_batch = image_batch.to(device)

            # Select the specific sample from the current batch
            kspace_sample = kspace_batch[sample_within_batch_idx].unsqueeze(0)  # Add batch dimension
            ground_truth = image_batch[sample_within_batch_idx].squeeze().cpu().numpy()  # Convert to numpy for visualization

            # Run inference to get the reconstructed image
            with torch.no_grad():
                reconstructed_image = model(kspace_sample).squeeze().cpu().numpy()  # Convert to numpy for visualization

            # Visualize the ground truth and reconstructed image
            plt.figure(figsize=(12, 6))

            # Ground Truth Image
            plt.subplot(1, 2, 1)
            plt.imshow(ground_truth, cmap='gray')
            plt.title('Ground Truth Image')
            plt.axis('off')

            # Reconstructed Image
            plt.subplot(1, 2, 2)
            plt.imshow(reconstructed_image, cmap='gray')
            plt.title('Reconstructed Image')
            plt.axis('off')

            # Save the figure as an image file
            output_path = os.path.join(save_dir, f'ground_truth_vs_reconstructed_{sample_idx}.png')
            plt.savefig(output_path, bbox_inches='tight', dpi=150)  # Save the figure
            plt.close()  # Close to free memory

            print(f"Comparison image saved at: {output_path}")
            return  # Exit after visualizing the desired sample
    print(f"Sample index {sample_idx} exceeds available data in the dataloader.")
    
    
def visualize_output_image(dataloader, sample_idx, plot_path):
    """
    Visualize a specific sample from the DataLoader.
    
    :param dataloader: DataLoader instance
    :param sample_idx: Index of the sample to visualize
    """
    # Calculate the batch index and index within the batch
    batch_size = dataloader.batch_size
    batch_idx = sample_idx // batch_size  # Determine which batch
    sample_within_batch_idx = sample_idx % batch_size  # Position within that batch

    # Iterate through the DataLoader to locate the desired batch
    for current_batch_idx, (input_kspace, output_image) in enumerate(dataloader):
        if current_batch_idx == batch_idx:
            # Extract the specific sample from the batch
            input_kspace_sample = input_kspace[sample_within_batch_idx]
            output_image_sample = output_image[sample_within_batch_idx]
            
            # Convert the output image to numpy for visualization
            output_image_np = output_image_sample.squeeze().numpy()

            # Visualize the output image
            plt.figure(figsize=(6, 6))
            plt.imshow(output_image_np, cmap='gray')
            plt.title(f'Sample {sample_idx} from DataLoader')
            plt.axis('off')
            plt.savefig(plot_path)  # Save the plot
            plt.close()
            plt.show()
            return  # Exit after visualizing the desired sample

    print(f"Sample index {sample_idx} exceeds the available data in the DataLoader.")