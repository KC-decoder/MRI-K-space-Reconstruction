import numpy as np
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import fastmri
from fastmri.data import transforms

from diffusion.kspace_diffusion import KspaceDiffusion
from utils.evaluation_utils import *

from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm



def recon_unet(
        dataloader,
        net,
        device,
        idx_case,
        show_info=True,
):
    '''
    Reconstruct image from the dataloader
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
            if show_info:
                print('tg.shape:', tg.shape)

            i_nmse = calc_nmse_tensor(tg, pred)
            i_psnr = calc_psnr_tensor(tg, pred)
            i_ssim = calc_ssim_tensor(tg, pred)
            if show_info:
                print('NMSE: ' + str(i_nmse) + '|| PSNR: ' + str(i_psnr) + '|| SSIM: ' + str(i_ssim))

            # ZF-MRI
            zf = X.detach().squeeze(1)
            break
    return pred.cpu().numpy(), tg.cpu().numpy(), zf.cpu().numpy()


def recon_kspace_cold_diffusion(
        dataloader,
        net,
        timesteps,
        device,
        idx_case,
        show_info=True,
):
    assert isinstance(net, KspaceDiffusion), "Input net must be a KspaceDiffusion."
    net = net.to(device)
    net.eval()
    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            if idx != idx_case:
                continue
            
            kspace, mask, mask_fold = data  # [B,Nc,H,W,2]
            kspace = kspace.to(device)
            mask = mask.to(device)
            mask_fold = mask_fold.to(device)
            print(f"shape of kspace: {kspace.shape} ")
            B, Nc, H, W, C = kspace.shape
            gt_imgs = fastmri.ifft2c(kspace)  # [B,Nc,H,W,2]

            # network forward
            xt, direct_recons, sample_imgs = net.sample(kspace, mask, mask_fold, t=timesteps)
            gt_imgs_abs = fastmri.complex_abs(gt_imgs)  # [B,Nc,H,W]
            direct_recons_abs = fastmri.complex_abs(direct_recons)  # [B,Nc,H,W]
            sample_imgs_abs = fastmri.complex_abs(sample_imgs)  # [B,Nc,H,W]
            # combine coil
            gt_imgs_abs = fastmri.rss(gt_imgs_abs, dim=1)  # [B,H,W]
            direct_recons_abs = fastmri.rss(direct_recons_abs, dim=1)
            sample_imgs_abs = fastmri.rss(sample_imgs_abs, dim=1)
            #print(f"Shape of sample_imgs_abs: {sample_imgs_abs.shape}")

            # evaluation metrics
            tg = gt_imgs_abs.detach()  # [B,H,W]
            pred_dir = direct_recons_abs.detach()
            pred = sample_imgs_abs.detach()
            
            # if show_info:
            #     print('tg.shape:', tg.shape)

            i_nmse = calc_nmse_tensor(tg, pred)
            i_psnr = calc_psnr_tensor(tg, pred)
            i_ssim = calc_ssim_tensor(tg, pred)
            i_nmse_dir = calc_nmse_tensor(tg, pred_dir)
            i_psnr_dir = calc_psnr_tensor(tg, pred_dir)
            i_ssim_dir = calc_ssim_tensor(tg, pred_dir)
            if show_info:
                print('NMSE: ' + str(i_nmse) + '|| PSNR: ' + str(i_psnr) + '|| SSIM: ' + str(i_ssim))
                print('Direct Recon NMSE: ' + str(i_nmse_dir) + '|| PSNR: ' + str(i_psnr_dir) + '|| SSIM: ' + str(i_ssim_dir))

            break
    return pred.cpu().numpy(), tg.cpu().numpy(), pred_dir.cpu().numpy()



def recon_kspace_cold_diffusion_from_perturbed_data(
        kspace,
        mask,
        mask_fold,
        net,
        timesteps,
        device,
        show_info=True,
):
    assert isinstance(net, KspaceDiffusion), "Input net must be a KspaceDiffusion."
    net = net.to(device)
    net.eval()

    with torch.no_grad():
        kspace = kspace.to(device)
        mask = mask.to(device)
        mask_fold = mask_fold.to(device)
        B, Nc, H, W, C = kspace.shape

        gt_imgs = fastmri.ifft2c(kspace)  # Convert k-space to ground truth image
        xt, direct_recons, sample_imgs = net.sample(kspace, mask, mask_fold, t=timesteps)

        # Compute magnitudes
        gt_imgs_abs = fastmri.rss(fastmri.complex_abs(gt_imgs), dim=1)  
        direct_recons_abs = fastmri.rss(fastmri.complex_abs(direct_recons), dim=1)
        sample_imgs_abs = fastmri.rss(fastmri.complex_abs(sample_imgs), dim=1)

        tg = gt_imgs_abs.detach()  
        pred_dir = direct_recons_abs.detach()
        pred = sample_imgs_abs.detach()

        # Compute metrics for sample-based reconstruction
        i_nmse = calc_nmse_tensor(tg, pred)
        i_psnr = calc_psnr_tensor(tg, pred)
        i_ssim = calc_ssim_tensor(tg, pred)

        # Compute metrics for direct reconstruction
        i_nmse_dir = calc_nmse_tensor(tg, pred_dir)
        i_psnr_dir = calc_psnr_tensor(tg, pred_dir)
        i_ssim_dir = calc_ssim_tensor(tg, pred_dir)

        if show_info:
            print(f'NMSE: {i_nmse:.4f} || PSNR: {i_psnr:.2f} || SSIM: {i_ssim:.4f}')
            print(f'Direct Recon NMSE: {i_nmse_dir:.4f} || PSNR: {i_psnr_dir:.2f} || SSIM: {i_ssim_dir:.4f}')

    # Return both reconstructed images and computed metrics
    metrics = {
        "nmse": i_nmse.item(),
        "psnr": i_psnr.item(),
        "ssim": i_ssim.item(),
        "nmse_dir": i_nmse_dir.item(),
        "psnr_dir": i_psnr_dir.item(),
        "ssim_dir": i_ssim_dir.item(),
    }

    return pred.cpu().numpy(), tg.cpu().numpy(), pred_dir.cpu().numpy(), metrics




def reconstruct_multicoil(kspace):
    """
    Converts multicoil k-space data to an image using RSS.

    Args:
        kspace (torch.Tensor): Shape (C, H, W) where C is the number of coils.
    """
    # Convert to complex-valued tensor
    kspace_complex = torch.view_as_complex(kspace.unsqueeze(-1).contiguous())

    # Apply inverse FFT (2D)
    img_coilwise = fastmri.ifft2c(kspace_complex)  # Shape: (C, H, W)

    # Compute magnitude image per coil
    img_coilwise_abs = fastmri.complex_abs(img_coilwise)

    # Combine coils using Root Sum of Squares (RSS)
    img_rss = fastmri.rss(img_coilwise_abs, dim=0)

    # Normalize for visualization
    img_rss = img_rss / img_rss.max()

    # Plot the image
    plt.figure(figsize=(6, 6))
    plt.imshow(img_rss.numpy(), cmap="gray")
    plt.title("Reconstructed MRI Image (RSS)")
    plt.axis("off")
    plt.show()


def recon_slice_unet(
        dataloader,
        net,
        device,
        idx_case,
):
    '''
    Reconstruct image from the dataloader.
    '''

    []
    net = net.to(device)
    net.eval()

    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            print("Inside model evaluation")
            if idx != idx_case:
                continue
            X, y, mask = data

            X = X.to(device).float()
            y = y.to(device).float()

            print("Prediction")
            # network forward
            y_pred = net(X)

            # evaluation metrics
            tg = y.detach().squeeze(1)    # [B, H, W]
            pred = y_pred.detach().squeeze(1)
            print("pred: ", pred.shape)

            # --- Z-score normalization using input image stats ---
            mean = torch.mean(X, dim=(1, 2, 3)).detach()  # [B]
            std = torch.std(X, dim=(1, 2, 3)).detach()    # [B]
            std = std + 1e-8                              # avoid divide-by-zero

            tg = torch.einsum('ijk, i -> ijk', (tg - mean[:, None, None]), 1. / std)
            pred = torch.einsum('ijk, i -> ijk', (pred - mean[:, None, None]), 1. / std)

            zf = X.detach().cpu().squeeze(1)
            tg = tg.cpu()
            pred = pred.cpu()
            X_for_gradcam = X.detach().cpu()

            print("=== [recon_slice_unet] ===")
            print("X (input to model) stats: min:", X.min().item(), "max:", X.max().item(), "mean:", X.mean().item())

            print("y_pred (raw model output) stats: min:", y_pred.min().item(), "max:", y_pred.max().item())

            print("Z-score mean:", mean)
            print("Z-score std:", std)
            print("tg (after z-score) stats: min:", tg.min().item(), "max:", tg.max().item())
            print("pred (after z-score) stats: min:", pred.min().item(), "max:", pred.max().item())

            i_nmse = calc_nmse_tensor(tg, pred)
            i_psnr = calc_psnr_tensor(tg, pred)
            i_ssim = calc_ssim_tensor(tg, pred)
            print('NMSE: ' + str(i_nmse) + ' || PSNR: ' + str(i_psnr) + ' || SSIM: ' + str(i_ssim))
            break
    return pred, zf, tg, i_nmse, i_psnr, i_ssim, mask.cpu(), X_for_gradcam