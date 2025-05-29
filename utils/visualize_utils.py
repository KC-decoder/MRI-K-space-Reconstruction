import math
import copy
import torch
from torch import nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial

from torch.utils import data
from pathlib import Path
from torch.optim import Adam
from torchvision import transforms, utils

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from einops import rearrange
import fastmri

import glob
import os
from PIL import Image
from net.u_net_diffusion import cycle, EMA, loss_backwards
from utils.evaluation_utils import calc_nmse_tensor, calc_psnr_tensor, calc_ssim_tensor
import os
from fastmri import complex_abs
import errno
from collections import OrderedDict




def visualize_data_sample(dataloader, sample_idx, title, save_path):
    """
    Visualizes:
    1. Undersampled Image
    2. Fully Sampled Target
    3. Binary Mask in k-space

    Args:
        dataloader: PyTorch DataLoader
        sample_idx: index within the first batch to visualize
        title: Plot title
        save_path: Path to save output PNG
    """
    # Get one batch
    sample = next(iter(dataloader))
    image_masked, target, mask = sample  # [B, 1, H, W]

    # Extract one sample from the batch
    image_masked = image_masked[sample_idx].squeeze(0).cpu().numpy()  # [H, W]
    target = target[sample_idx].squeeze(0).cpu().numpy()              # [H, W]
    mask = mask[sample_idx].squeeze(0).cpu().numpy()                  # [H, W]

    # Plot
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    axs[0].imshow(image_masked, cmap='gray')
    axs[0].set_title("Undersampled Image")
    axs[0].axis('off')

    axs[1].imshow(target, cmap='gray')
    axs[1].set_title("Fully Sampled Target")
    axs[1].axis('off')

    axs[2].imshow(mask, cmap='gray', vmin=0, vmax=1)  # Binary mask visualization
    axs[2].set_title("Binary Mask")
    axs[2].axis('off')

    plt.suptitle(title)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    print(f"Saved visualization to: {save_path}")

def save_image_from_kspace(kspace, i, save_path, filename="image_from_kspace.png"):
    """
    Convert k-space to image space, plot, and save the image.

    Args:
        kspace (torch.Tensor): K-space tensor of shape [B, Nc, H, W, 2].
        save_path (str): Directory where the image will be saved.
        filename (str): Filename for the saved image.
    """
    # Ensure save directory exists
    os.makedirs(save_path, exist_ok=True)

    # Check if the k-space tensor is in the expected format
    if kspace.dim() != 5 or kspace.shape[-1] != 2:
        raise ValueError(f"Expected k-space shape [B, Nc, H, W, 2], but got {kspace.shape}")

    # Convert the first k-space sample to image space
    kspace_sample = kspace[0]  # Select the first batch
    img_space = fastmri.ifft2c(kspace_sample)  # [Nc, H, W, 2]
    img_abs = fastmri.complex_abs(img_space)  # [Nc, H, W]

    # Combine multi-coil images (RSS)
    if img_abs.shape[0] > 1:  # If multi-coil, combine using RSS
        img_combined = fastmri.rss(img_abs, dim=0)  # [H, W]
    else:
        img_combined = img_abs[0]  # Single coil case [H, W]

    # Convert to NumPy for plotting
    img_combined_np = img_combined.cpu().numpy()

    # Plot the image
    plt.figure(figsize=(8, 8))
    plt.imshow(img_combined_np, cmap="gray")
    plt.axis("off")
    plt.title("Image from K-space")

    # Save the image
    filename = f"image_from_kspace_test_{i}.png"
    full_save_path = os.path.join(save_path, filename)
    plt.savefig(full_save_path, bbox_inches="tight")
    plt.close()
    print(f"Image from k-space saved at: {full_save_path}")

class Visualizer_Kspace_ColdDiffusion(object):
    def __init__(
            self,
            diffusion_model,
            *,
            ema_decay=0.995,
            dataloader_test=None,
            load_path=None,
            device = "cuda",
            output_dir="./kspace_analysis/"
    ):
        """
        Visualizer class to extract and store intermediate k-space representations.

        Args:
            diffusion_model: Trained diffusion model.
            ema_decay: Exponential moving average decay factor.
            dataloader_test: Test dataset for evaluation.
            load_path: Path to pre-trained model checkpoint.
            output_dir: Directory where .npy files will be saved.
        """
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.dl_test = dataloader_test
        self.output_dir = output_dir
        self.device = device

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        self.reset_parameters()
        self.load(load_path)
        

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def load(self, load_path):
        print("Loading : ", load_path)
        data = torch.load(load_path)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])

    def show_intermediate_kspace_cold_diffusion(self, t, idx_case):
        for i_case, data in enumerate(self.dl_test):
            if i_case != idx_case:
                continue
            kspace, mask, mask_fold = data  # [B,Nc(1),H,W,2]
            kspace = kspace.to(self.device)
            mask = mask.to(self.device)
            mask_fold = mask_fold.to(self.device)
            B, Nc, H, W, C = kspace.shape
            gt_imgs = fastmri.ifft2c(kspace)  # [B,Nc,H,W,2]

            # [B,Nc,H,W,2]
            xt, direct_recons, sample_imgs = self.ema_model.sample(kspace, mask, mask_fold, t=t)
            kspacet = fastmri.fft2c(xt)

            gt_imgs_abs = fastmri.complex_abs(gt_imgs)  # [B,Nc,H,W]
            direct_recons_abs = fastmri.complex_abs(direct_recons)  # [B,Nc,H,W]
            sample_imgs_abs = fastmri.complex_abs(sample_imgs)  # [B,Nc,H,W]

            xt = xt[0, 0]  # [H,W,2]
            kspacet = kspacet[0, 0]
            kspace = kspace[0, 0]
            gt_imgs_abs = gt_imgs_abs[0, 0]  # [H,W]
            direct_recons_abs = direct_recons_abs[0, 0]
            sample_imgs_abs = sample_imgs_abs[0, 0]

            return {
                "timestep": t,
                "xt": xt.cpu().numpy(),
                "kspacet": kspacet.cpu().numpy(),
                "kspace": kspace.cpu().numpy(),
                "gt_imgs_abs": gt_imgs_abs.cpu().numpy(),
                "direct_recons_abs": direct_recons_abs.cpu().numpy(),
                "sample_imgs_abs": sample_imgs_abs.cpu().numpy(),
            }
            
    
    def save_intermediate_kspace_npy(self, idx_case, t, filename):
        """
        Extracts and saves intermediate k-space representations for a given timestep.

        Args:
            idx_case (int): Index of test sample.
            t (int): Timestep to extract.
            filename (str): Name of the output .npy file.
        """
        print(f"Processing timestep {t} for sample {idx_case}...")

        # Extract data at specific timestep
        data = self.show_intermediate_kspace_cold_diffusion(t, idx_case)

        # Save as .npy file
        save_path = os.path.join(self.output_dir, filename)
        np.save(save_path, data)
        print(f"Saved k-space data for sample {idx_case}, timestep {t} to {save_path}")


def show_intermediate_kspace_Gmask(dl_test, idx_case, t, t_all):
    '''
    Show intermediate kspace mask given a time step number.
    '''
    for i_case, data in enumerate(dl_test):
        if i_case != idx_case:
            continue
        kspace, mask, mask_fold = data  # [B,Nc(1),H,W,2]
        B, Nc, H, W, C = kspace.shape
        gt_imgs = fastmri.ifft2c(kspace)  # [B,Nc,H,W,2]

        mask = mask_fold[0, 0]  # [B,1,H,W] to [H,W]
        mask_t = mask.clone()

        # mask_inv = torch.ones_like(mask) - mask
        mask_inv_idx = torch.argwhere(abs(mask) < 1e-6)
        mask_inv_idx = mask_inv_idx[torch.randperm(mask_inv_idx.shape[0])]
        len_keep = int(np.floor((t_all - t) / t_all * mask_inv_idx.shape[0]))
        mask_inv_idx_keep = mask_inv_idx[:len_keep]

        for _, data in enumerate(mask_inv_idx_keep):
            [ix, iy] = data
            mask_t[ix, iy] = 1

        mask = mask[None, None, ...]
        mask_t = mask_t[None, None, ...]
        up_sample = nn.Upsample(scale_factor=(kspace.shape[-2]/mask_fold.shape[-1]), mode='nearest')
        mask = up_sample(mask)
        mask_t = up_sample(mask_t)

        # get image space
        mask_t = mask_t[..., None]
        kspace_t = mask_t * kspace
        imgs_t = fastmri.ifft2c(kspace_t)

        gt_imgs_abs = fastmri.complex_abs(gt_imgs)  # [B,Nc,H,W]
        imgs_t_abs = fastmri.complex_abs(imgs_t)  # [B,Nc,H,W]
        mask_t = mask_t[..., 0]

        return mask_t, mask, imgs_t_abs, gt_imgs_abs


def recon_unet(
        dataloader,
        net,
        device,
):
    '''
    Reconstruct image from the dataloader.
    '''
    net = net.to(device)
    net.eval()

    zf_list = []
    tg_list = []
    recon_list = []
    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            X, y, mask = data

            X = X.to(device).float()
            y = y.to(device).float()

            # network forward
            y_pred = net(X)

            # evaluation metrics
            tg = y.detach().squeeze(1)  # [B,H,W]
            pred = y_pred.detach().squeeze(1)
            max = torch.amax(X, dim=(1, 2, 3)).detach()
            scale_coeff = 1. / max  # [B,]
            tg = torch.einsum('ijk, i -> ijk', tg, scale_coeff)
            pred = torch.einsum('ijk, i -> ijk', pred, scale_coeff)

            zf_list.append(X.detach().cpu().squeeze(1))
            tg_list.append(tg.cpu())
            recon_list.append(pred.cpu())
    return recon_list, zf_list, tg_list


def recon_slice_unet(
        dataloader,
        net,
        device,
        idx_case,
):
    '''
    Reconstruct image from the dataloader.
    '''
    net = net.to(device)
    net.eval()

    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            if idx != idx_case:
                continue
            X, y, mask = data

            X = X.to(device).float()
            y = y.to(device).float()

            # network forward
            y_pred = net(X)

            # evaluation metrics
            tg = y.detach().squeeze(1)  # [B,H,W]
            pred = y_pred.detach().squeeze(1)
            max = torch.amax(X, dim=(1, 2, 3)).detach()
            scale_coeff = 1. / max  # [B,]
            tg = torch.einsum('ijk, i -> ijk', tg, scale_coeff)
            pred = torch.einsum('ijk, i -> ijk', pred, scale_coeff)

            zf = X.detach().cpu().squeeze(1)
            tg = tg.cpu()
            pred = pred.cpu()
            break
    return pred, zf, tg


def recon_wnet(
        dataloader,
        net,
        device,
):
    '''
    Reconstruct image from the dataloader.
    '''
    net = net.to(device)
    net.eval()

    zf_list = []
    tg_list = []
    recon_list = []
    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            Xk, yk, mask, y = data

            Xk = Xk.to(device).float()
            yk = yk.to(device).float()
            mask = mask.to(device).float()
            y = y.to(device).float()

            # network forward
            y_pred, k_pred_mid = net(Xk, mask)

            # evaluation metrics
            tg = y.detach().squeeze(1)  # [B,H,W]
            pred = y_pred.detach().squeeze(1)
            X = fastmri.complex_abs(fastmri.ifft2c(Xk.detach()))  # [B,Nc,H,W]
            max = torch.amax(X, dim=(1, 2, 3)).detach()
            scale_coeff = 1. / max  # [B,]
            tg = torch.einsum('ijk, i -> ijk', tg, scale_coeff)
            pred = torch.einsum('ijk, i -> ijk', pred, scale_coeff)

            zf_list.append(X.detach().cpu().squeeze(1))
            tg_list.append(tg.cpu())
            recon_list.append(pred.cpu())
    return recon_list, zf_list, tg_list


def recon_slice_wnet(
        dataloader,
        net,
        device,
        idx_case,
):
    '''
    Reconstruct image from the dataloader.
    '''
    net = net.to(device)
    net.eval()

    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            if idx != idx_case:
                continue
            Xk, yk, mask, y = data

            Xk = Xk.to(device).float()
            yk = yk.to(device).float()
            mask = mask.to(device).float()
            y = y.to(device).float()

            # network forward
            y_pred, k_pred_mid = net(Xk, mask)

            # evaluation metrics
            tg = y.detach().squeeze(1)  # [B,H,W]
            pred = y_pred.detach().squeeze(1)
            X = fastmri.complex_abs(fastmri.ifft2c(Xk.detach()))  # [B,Nc,H,W]
            max = torch.amax(X, dim=(1, 2, 3)).detach()
            scale_coeff = 1. / max  # [B,]
            tg = torch.einsum('ijk, i -> ijk', tg, scale_coeff)
            pred = torch.einsum('ijk, i -> ijk', pred, scale_coeff)

            zf = X.detach().cpu().squeeze(1)
            tg = tg.cpu()
            pred = pred.cpu()
            break
    return pred, zf, tg


def recon_varnet(
        dataloader,
        net,
        device,
):
    '''
    Reconstruct image from the dataloader.
    '''
    net = net.to(device)
    net.eval()

    zf_list = []
    tg_list = []
    recon_list = []
    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            Xk, yk, mask, y = data

            Xk = Xk.to(device).float()
            mask = mask.to(device).float()
            y = y.to(device).float()

            # network forward
            y_pred = net(Xk, mask)

            # evaluation metrics
            tg = y.detach()  # [B,H,W]
            pred = y_pred.detach()
            X = fastmri.complex_abs(fastmri.ifft2c(Xk.detach()))  # [B,Nc,H,W]
            max = torch.amax(X, dim=(1, 2, 3))
            scale_coeff = 1. / max  # [B,]
            tg = torch.einsum('ijk, i -> ijk', tg, scale_coeff)
            pred = torch.einsum('ijk, i -> ijk', pred, scale_coeff)

            zf_list.append(X.detach().cpu().squeeze(1))
            tg_list.append(tg.cpu())
            recon_list.append(pred.cpu())
    return recon_list, zf_list, tg_list


def recon_slice_varnet(
        dataloader,
        net,
        device,
        idx_case,
):
    '''
    Reconstruct image from the dataloader.
    '''
    net = net.to(device)
    net.eval()

    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            if idx != idx_case:
                continue
            Xk, yk, mask, y = data

            Xk = Xk.to(device).float()
            mask = mask.to(device).float()
            y = y.to(device).float()

            # network forward
            y_pred = net(Xk, mask)

            # evaluation metrics
            tg = y.detach()  # [B,H,W]
            pred = y_pred.detach()
            X = fastmri.complex_abs(fastmri.ifft2c(Xk.detach()))  # [B,Nc,H,W]
            max = torch.amax(X, dim=(1, 2, 3))
            scale_coeff = 1. / max  # [B,]
            tg = torch.einsum('ijk, i -> ijk', tg, scale_coeff)
            pred = torch.einsum('ijk, i -> ijk', pred, scale_coeff)

            zf = X.detach().cpu().squeeze(1)
            tg = tg.cpu()
            pred = pred.cpu()
            break
    return pred, zf, tg


def plot_reconstruction_results_from_npy(npy_path, save_path):
    """
    Loads data from an .npy file and plots the reconstruction process in a 2x2 grid:
    - Top-left: Ground Truth Image
    - Top-right: Final Reconstruction
    - Bottom-right: Intermediate Reconstruction
    - Bottom-left: Sampled Reconstruction
    
    Args:
        npy_path: Path to the saved .npy file containing reconstruction data.
        save_path: Path to save the resulting plot.
    """
    # Load the .npy file
    data = np.load(npy_path, allow_pickle=True).item()

    # Extract data from the loaded dictionary
    t = data["timestep"]
    gt_imgs_abs = data["gt_imgs_abs"]  # Ground Truth Image
    final_recons_abs = data["direct_recons_abs"]  # Final Reconstruction
    sample_imgs_abs = data["sample_imgs_abs"]  # Sampled Reconstruction
    xt = data["xt"]  # Intermediate k-space representation

    # Ensure save directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Create a 2x2 figure layout
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle("Reconstruction Process", fontsize=16)

    # Plot Ground Truth Image (Top-left)
    axes[0, 0].imshow(gt_imgs_abs, cmap='gray')
    axes[0, 0].set_title("Ground Truth Image")
    axes[0, 0].set_xlabel("Pixels")
    axes[0, 0].set_ylabel("Pixels")
    fig.colorbar(plt.cm.ScalarMappable(cmap="gray"), ax=axes[0, 0])

    # Plot Final Reconstruction (Top-right)
    axes[0, 1].imshow(final_recons_abs, cmap='gray')
    axes[0, 1].set_title("Final Reconstruction")
    axes[0, 1].set_xlabel("Pixels")
    axes[0, 1].set_ylabel("Pixels")
    fig.colorbar(plt.cm.ScalarMappable(cmap="gray"), ax=axes[0, 1])

    # Plot Sampled Reconstruction (Bottom-left)
    axes[1, 0].imshow(sample_imgs_abs, cmap='gray')
    axes[1, 0].set_title("Aliased reconstruction from undersampled data")
    axes[1, 0].set_xlabel("Pixels")
    axes[1, 0].set_ylabel("Pixels")
    fig.colorbar(plt.cm.ScalarMappable(cmap="gray"), ax=axes[1, 0])

    # Plot Intermediate Reconstruction (Bottom-right)
    magnitude_image = complex_abs(torch.tensor(xt)).numpy()
    axes[1, 1].imshow(magnitude_image.squeeze(), cmap='gray')
    axes[1, 1].set_title(f"Intermediate reconstruction at time step {t}")
    axes[1, 1].set_xlabel("Pixels")
    axes[1, 1].set_ylabel("Pixels")
    fig.colorbar(plt.cm.ScalarMappable(cmap="gray"), ax=axes[1, 1])

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save the figure
    plt.savefig(save_path)
    print(f"Plot saved at: {save_path}")
    plt.close()
    
    
    


class Visualizer_UNet_Reconstruction:
    def __init__(
            self,
            unet_model,
            *,
            ema_decay=0.995,
            dataloader_test=None,
            load_path=None,
            device="cuda",
            logger = './log.txt',
            output_dir="./unet_reconstruction/"
    ):
        """
        Visualizer for U-Net reconstructions.

        Args:
            unet_model: Trained U-Net model.
            ema_decay: Exponential moving average decay factor.
            dataloader_test: Test dataset for evaluation.
            load_path: Path to pre-trained model checkpoint.
            output_dir: Directory where .npy files will be saved.
        """
        super().__init__()
        self.model = unet_model.to(device)
        self.ema_decay = ema_decay
        self.ema_model = copy.deepcopy(self.model)
        self.dl_test = dataloader_test
        self.device = device
        self.logger = logger
        self.output_dir = output_dir

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        self.load(load_path)

    def load(self, load_path):
        """Load the trained U-Net model."""
        if load_path is not None:
            print("Loading model from:", load_path)
            checkpoint = torch.load(load_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.ema_model.load_state_dict(checkpoint["ema_model_state_dict"])

    def show_unet_reconstruction(self, idx_case):
        """
        Visualizes the U-Net reconstruction for a given test sample.

        Args:
            idx_case (int): Index of the test sample.

        Returns:
            Dictionary containing masked input, UNet reconstruction, and ground truth.
        """
        for i_case, data in enumerate(self.dl_test):
            if i_case != idx_case:
                continue

            image_masked, image_full, mask = data  # Extract input, ground truth, and mask
            image_masked = image_masked.to(self.device)
            image_full = image_full.to(self.device)
            # Ensure input tensor has the correct dtype (float32) before passing to model
            image_masked = image_masked.to(torch.float32)  # Convert to float32

            # Perform U-Net prediction
            with torch.no_grad():
                # Forward pass through the EMA model
                self.logger.log(f"Model Evaluation in progress...")
                pred_recon = self.ema_model(image_masked)

            # Convert tensors to NumPy
            image_masked = image_masked[0].cpu().numpy()  # Masked input
            pred_recon = pred_recon[0].cpu().numpy()  # U-Net reconstruction
            image_full = image_full[0].cpu().numpy()  # Ground truth

            return {
                "image_masked": image_masked,
                "pred_recon": pred_recon,
                "image_full": image_full,
                "mask": mask.cpu().numpy()[0]  # Mask for visualization
            }

    def save_unet_reconstruction_npy(self, idx_case, filename):
        """
        Extracts and saves the U-Net reconstruction for a given test sample.

        Args:
            idx_case (int): Index of test sample.
            filename (str): Name of the output .npy file.
        """
        self.logger.log(f"Processing U-Net reconstruction for sample {idx_case}...")

        # Extract data
        data = self.show_unet_reconstruction(idx_case)

        # Save as .npy file
        save_path = os.path.join(self.output_dir, filename)
        np.save(save_path, data)
        self.logger.log(f"Saved reconstruction for sample {idx_case} to {save_path}")

    def visualize_unet_reconstruction(self, idx_case, save_path=None):
        """
        Visualizes and optionally saves the U-Net reconstruction.

        Args:
            idx_case (int): Index of test sample.
            save_path (str, optional): Path to save the visualization.
        """
        data = self.show_unet_reconstruction(idx_case)

        pred_recon = data["pred_recon"]  # U-Net Reconstruction
        image_full = data["image_full"]  # Ground Truth

        # Plot only Ground Truth and UNet Reconstruction side-by-side
        fig, axs = plt.subplots(1, 2, figsize=(8, 6))

        axs[0].imshow(image_full.squeeze(), cmap="gray")
        axs[0].set_title("Ground Truth")
        axs[0].axis("off")

        axs[1].imshow(pred_recon.squeeze(), cmap="gray")
        axs[1].set_title("U-Net Reconstruction")
        axs[1].axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
            self.logger.log(f"Saved visualization to {save_path}")

        plt.show()