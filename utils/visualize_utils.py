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



def zscore_normalize(img):
    """Z-score normalization: (img - mean) / std."""
    img = img.astype(np.float32)
    mean = np.mean(img)
    std = np.std(img)
    if std < 1e-6:
        return np.zeros_like(img)  # Avoid divide-by-zero for flat images
    return (img - mean) / std

def plot_reconstruction_vs_ground_truth(pred, gt, ssim_value=None, save_path=None):
    """
    Plot the model's reconstruction and the ground truth side-by-side with z-score normalization.
    """
    # Convert to numpy if tensor
    if hasattr(pred, "detach"):
        pred = pred.detach().cpu().numpy()
    if hasattr(gt, "detach"):
        gt = gt.detach().cpu().numpy()

    if pred.ndim == 3:
        pred = pred[0]
    if gt.ndim == 3:
        gt = gt[0]

    # Apply z-score normalization
    # pred_norm = zscore_normalize(pred)
    # gt_norm = zscore_normalize(gt)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].imshow(gt, cmap='gray')
    axs[0].set_title("Ground Truth (unnormalized)")
    axs[0].axis("off")

    axs[1].imshow(pred, cmap='gray')
    title = "Reconstruction (unnormalized)"
    if ssim_value is not None:
        title += f"\nSSIM={ssim_value:.3f}"
    axs[1].set_title(title)
    axs[1].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved plot to: {save_path}")
    plt.show()


def plot_full_reconstruction_4panel(
    pred,
    zf,
    gt,
    mask,
    ssim_value=None,
    save_path=None
):
    def to_numpy(x):
        if hasattr(x, "detach"):
            return x.detach().cpu().squeeze().numpy()
        return np.squeeze(x)

    # Convert and normalize
    pred_np = zscore_normalize(to_numpy(pred))
    zf_np = zscore_normalize(to_numpy(zf))
    gt_np = zscore_normalize(to_numpy(gt))
    mask_np = to_numpy(mask)  # No normalization for binary mask

    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    axs[0].imshow(mask_np, cmap='gray')
    axs[0].set_title("K-space Mask (Ring)")
    axs[0].axis("off")

    axs[1].imshow(zf_np, cmap='gray')
    axs[1].set_title("Undersampled Input (ZF, z-score)")
    axs[1].axis("off")

    axs[2].imshow(gt_np, cmap='gray')
    axs[2].set_title("Ground Truth (z-score)")
    axs[2].axis("off")

    axs[3].imshow(pred_np, cmap='gray')
    title = "Reconstruction (z-score)"
    if ssim_value is not None:
        title += f"\nSSIM={ssim_value:.3f}"
    axs[3].set_title(title)
    axs[3].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"[Saved] {save_path}")
    plt.show()


def plot_gradcam_outputs(data_dict, save_path=None):
    print("=== [plot_gradcam_outputs] ===")
    print("Input Image stats: min", np.min(data_dict["input"]), "max", np.max(data_dict["input"]))
    print("GT Image stats: min", np.min(data_dict["gt"]), "max", np.max(data_dict["gt"]))
    print("Recon Image stats: min", np.min(data_dict["recon"]), "max", np.max(data_dict["recon"]))
    print("GradCAM Image stats: min", np.min(data_dict["gradcam_img"]), "max", np.max(data_dict["gradcam_img"]))

    def prepare_for_display(image, zscore=True):
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()

        # Handle batch or channel dimensions
        if image.ndim == 4:
            image = image[0, 0]
        elif image.ndim == 3:
            image = image[0]
        elif image.ndim != 2:
            raise ValueError(f"Unexpected image shape: {image.shape}")

        if zscore:
            mean = np.mean(image)
            std = np.std(image) + 1e-8  # avoid div by zero
            image = (image - mean) / std
            image = np.clip(image, -3, 3)  # optional but improves contrast
            image = (image + 3) / 6  # rescale to [0,1] for display
        return image

    # Apply normalization
    input_img = prepare_for_display(data_dict["input"])
    gt_img = prepare_for_display(data_dict["gt"])
    recon_img = prepare_for_display(data_dict["recon"])
    gradcam_img = prepare_for_display(data_dict["gradcam_img"])

    # Plotting
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    axs[0].imshow(input_img, cmap='gray')
    axs[0].set_title("Input (ZF)")
    axs[0].axis("off")

    axs[1].imshow(gt_img, cmap='gray')
    axs[1].set_title("Ground Truth")
    axs[1].axis("off")

    axs[2].imshow(recon_img, cmap='gray')
    axs[2].set_title("Reconstruction")
    axs[2].axis("off")

    axs[3].imshow(gradcam_img, cmap='jet')  # attention heatmap
    axs[3].set_title("Grad-CAM")
    axs[3].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"[Saved] {save_path}")
    plt.show()


   

    # fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    # axs[0, 0].imshow(prepare_for_display(data_dict["mask"]), cmap="gray")
    # axs[0, 0].set_title("k-space Binary Mask")
    # axs[0, 0].axis("off")

    # axs[0, 1].imshow(prepare_for_display(data_dict["input"]), cmap="gray")
    # axs[0, 1].set_title("Undersampled Input")
    # axs[0, 1].axis("off")

    # axs[0, 2].imshow(prepare_for_display(data_dict["gt"]), cmap="gray")
    # axs[0, 2].set_title("Ground Truth")
    # axs[0, 2].axis("off")

    # axs[1, 0].imshow(prepare_for_display(data_dict["recon"]), cmap="gray")
    # axs[1, 0].set_title("Reconstruction")
    # axs[1, 0].axis("off")

    # axs[1, 1].imshow(prepare_for_display(data_dict["gradcam_img"]), cmap="jet")
    # axs[1, 1].set_title("Grad-CAM (Image Space)")
    # axs[1, 1].axis("off")

    # axs[1, 2].imshow(prepare_for_display(data_dict["gradcam_k"]), cmap="jet")
    # axs[1, 2].set_title("Grad-CAM (K-space Proxy)")
    # axs[1, 2].axis("off")

    # plt.tight_layout()
    # if save_path:
    #     plt.savefig(save_path, bbox_inches="tight")
    #     print(f"Saved Grad-CAM visualization to {save_path}")
    # plt.show()


    
def generate_ring_masks(H=320, W=320, num_masks=5, max_radius_fraction=0.15, save_dir="./ring_masks"):
    """
    Generates thicker binary ring masks where the total diameter of the outermost ring
    does not exceed a fixed maximum radius, and all rings have equal radial thickness.

    Args:
        H, W: height and width of the mask.
        num_masks: number of concentric rings (including central disc).
        max_radius_fraction: fraction of H to set as the max radius.
        save_dir: path to save the masks.
    """
    os.makedirs(save_dir, exist_ok=True)
    center = (H // 2, W // 2)
    max_radius = H * max_radius_fraction  # set total limit (e.g., 0.15 * 320 = 48 pixels)
    step_r = max_radius / num_masks

    y, x = np.ogrid[:H, :W]
    distance = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)

    for i in range(num_masks):
        inner = i * step_r
        outer = (i + 1) * step_r
        ring_mask = ((distance >= inner) & (distance < outer)).astype(np.uint8)

        filename = os.path.join(save_dir, f"ring_mask_{i+1}.npy")
        np.save(filename, ring_mask)
        print(f"Saved: {filename} with shape {ring_mask.shape}")



def generate_ring_masks_fixed_step(
    H=320, 
    W=320, 
    total_rings=10, 
    step_r=9.6, 
    save_dir="./ring_masks", 
    existing_rings=5
):
    """
    Generate binary ring masks using fixed radial steps.
    Assumes the first `existing_rings` are already saved and unchanged.

    Args:
        H (int): Height of the mask.
        W (int): Width of the mask.
        total_rings (int): Total number of concentric rings to create.
        step_r (float): Radial thickness of each ring.
        save_dir (str): Directory to save .npy mask files.
        existing_rings (int): Number of rings already generated (will skip these).
    """
    os.makedirs(save_dir, exist_ok=True)
    center = (H // 2, W // 2)
    y, x = np.ogrid[:H, :W]
    distance = np.sqrt((x - center[1])**2 + (y - center[0])**2)

    for i in range(existing_rings, total_rings):
        inner = i * step_r
        outer = (i + 1) * step_r
        ring_mask = ((distance >= inner) & (distance < outer)).astype(np.uint8)

        filename = os.path.join(save_dir, f"ring_mask_{i + 1}.npy")
        np.save(filename, ring_mask)
        print(f"[✓] Saved: {filename}, shape={ring_mask.shape}")
        
        
        
def plot_ring_masks(save_dir, num_masks=10, output_path="ring_mask_grid.png"):
    """
    Plots the saved ring masks side-by-side in black and white.

    Args:
        save_dir (str): Directory containing ring_mask_*.npy files.
        num_masks (int): Number of masks to plot.
        output_path (str): Path to save the output plot.
    """
    fig, axs = plt.subplots(1, num_masks, figsize=(3 * num_masks, 3))

    for i in range(num_masks):
        mask_file = os.path.join(save_dir, f"ring_mask_{i+1}.npy")
        mask = np.load(mask_file)
        axs[i].imshow(mask, cmap='gray', vmin=0, vmax=1)
        axs[i].axis('off')
        axs[i].set_title(f"Ring {i+1}")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved ring mask visualization to: {output_path}")




def load_model(model_class, ckpt_path, device):
    model = model_class().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model

def visualize_multiple_models(model_paths, dataloaders, idx_case, device, save_path, model_class):
    fig, axs = plt.subplots(nrows=5, ncols=2, figsize=(10, 25))
    
    for i, (model_path, dataloader) in enumerate(zip(model_paths, dataloaders)):
        # Load model
        model = load_model(model_class, model_path, device)

        # Get sample
        sample = list(dataloader)[idx_case]
        X = sample['input'].to(device).unsqueeze(0)  # [1, C, H, W]
        y = sample['gt'].to(device).unsqueeze(0)

        # Forward pass
        with torch.no_grad():
            y_pred = model(X)  # [1, 1, H, W]

        # Process
        tg = y.detach().squeeze(1)         # [1, H, W]
        pred = y_pred.detach().squeeze(1)  # [1, H, W]
        max_vals = torch.amax(X, dim=(1, 2, 3)).detach()  # [1]
        scale_coeff = 1. / max_vals  # [1]

        # Apply scaling
        tg = torch.einsum('ijk, i -> ijk', tg, scale_coeff)
        pred = torch.einsum('ijk, i -> ijk', pred, scale_coeff)

        # Convert to numpy
        tg = tg.squeeze(0).cpu().numpy()
        pred = pred.squeeze(0).cpu().numpy()

        # Plot
        axs[i, 0].imshow(tg, cmap='gray')
        axs[i, 0].set_title(f"Ground Truth (Ring {i+1})")
        axs[i, 0].axis('off')

        axs[i, 1].imshow(pred, cmap='gray')
        axs[i, 1].set_title(f"Prediction (Ring {i+1})")
        axs[i, 1].axis('off')

    # Save plot
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    print(f"[Saved] 5x2 plot to: {save_path}")
    plt.close()
    


