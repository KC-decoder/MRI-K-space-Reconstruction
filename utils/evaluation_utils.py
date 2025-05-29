import argparse
import pathlib
from argparse import ArgumentParser
from typing import Optional

import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

import torch
import torch.nn.functional as F
from torchmetrics.functional import peak_signal_noise_ratio as psnr_tensor
from torchmetrics.functional import structural_similarity_index_measure as ssim_tensor


class L1MagSSIMLoss:
    """
    Combined L1 + SSIM loss for complex-valued MRI images represented as 2-channel real tensors.
    """

    def __init__(self, ssim_weight=0.1, epsilon=1e-8):
        self.ssim_weight = ssim_weight
        self.epsilon = epsilon

    def complex_to_mag(self, x):
        """
        Convert [B, 2, H, W] complex representation to magnitude [B, H, W]
        """
        return torch.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2 + self.epsilon)

    def normalize(self, x):
        """
        Normalize [B, H, W] to [0,1] per sample
        """
        B, H, W = x.shape
        x = x.view(B, -1)
        x_min = x.min(dim=1, keepdim=True)[0]
        x_max = x.max(dim=1, keepdim=True)[0]
        x = (x - x_min) / (x_max - x_min + self.epsilon)
        return x.view(B, H, W)

    def __call__(self, y_pred, y_true):
        """
        Args:
            y_pred: [B, 2, H, W] - Predicted complex-valued image
            y_true: [B, 2, H, W] - Ground truth complex-valued image
        Returns:
            Combined L1 + SSIM loss (scalar)
        """
        # Convert to magnitude
        pred_mag = self.complex_to_mag(y_pred)
        true_mag = self.complex_to_mag(y_true)

        # Normalize for SSIM
        pred_mag_norm = self.normalize(pred_mag).unsqueeze(1)  # [B, 1, H, W]
        true_mag_norm = self.normalize(true_mag).unsqueeze(1)

        # Compute individual losses
        l1 = F.l1_loss(pred_mag, true_mag)
        ssim = ssim_tensor(pred_mag_norm, true_mag_norm, data_range=1.0)

        # Combine
        total_loss = l1 + self.ssim_weight * (1 - ssim)
        return total_loss


def l1_image_loss(reconstructed_img, target_img):
    """
    Computes L1 loss (Mean Absolute Error) between reconstructed image and target image.
    
    Args:
        reconstructed_img: Predicted MRI image from U-Net (B, 1, 320, 320)
        target_img: Ground truth fully sampled MRI image (B, 1, 320, 320)
    
    Returns:
        L1 loss
    """
    return F.l1_loss(reconstructed_img, target_img)

def l2_image_loss(reconstructed_img, target_img):
    """
    Computes L2 loss (Mean Squared Error) between reconstructed image and target image.
    
    Args:
        reconstructed_img: Predicted MRI image from U-Net (B, 1, 320, 320)
        target_img: Ground truth fully sampled MRI image (B, 1, 320, 320)
    
    Returns:
        L2 loss
    """
    return F.mse_loss(reconstructed_img, target_img)

def get_error_map(target, pred):
    error = abs(target - pred)
    return error


def calc_nmse_tensor(gt, pred):
    """Compute Normalized Mean Squared Error (NMSE)"""
    '''
    tensor, [N,H,W]
    '''

    return torch.linalg.norm(gt - pred) ** 2 / torch.linalg.norm(gt) ** 2


def calc_psnr_tensor(gt, pred):
    """Compute Peak Signal to Noise Ratio metric (PSNR)"""
    '''
    tensor, [N,H,W]
    '''

    return psnr_tensor(pred, gt)


def calc_ssim_tensor(gt, pred):
    """Compute Structural Similarity Index Metric (SSIM)"""
    '''
    tensor, [N,H,W]
    '''
    if not gt.dim() == 3:
        raise ValueError("Unexpected number of dimensions in ground truth.")
    if not gt.dim() == pred.dim():
        raise ValueError("Ground truth dimensions does not match pred.")

    ssim = ssim_tensor(pred.unsqueeze(0), gt.unsqueeze(0))  # .unsqueeze(0) to [N,1,H,W]

    return ssim


def calc_complex_ssim_tensor(gt, pred, epsilon=1e-8):
    """
    Compute Structural Similarity Index Metric (SSIM)
    Inputs:
        gt, pred: torch.Tensor of shape [B, 2, H, W] — complex-valued
    Returns:
        Mean SSIM over batch
    """
    # Convert to magnitude: [B, H, W]
    gt_mag = torch.sqrt(gt[:, 0]**2 + gt[:, 1]**2 + epsilon)
    pred_mag = torch.sqrt(pred[:, 0]**2 + pred[:, 1]**2 + epsilon)

    # Normalize each image to [0,1]
    gt_mag = (gt_mag - gt_mag.amin(dim=(1, 2), keepdim=True)) / (gt_mag.amax(dim=(1, 2), keepdim=True) - gt_mag.amin(dim=(1, 2), keepdim=True) + epsilon)
    pred_mag = (pred_mag - pred_mag.amin(dim=(1, 2), keepdim=True)) / (pred_mag.amax(dim=(1, 2), keepdim=True) - pred_mag.amin(dim=(1, 2), keepdim=True) + epsilon)

    # Add channel dimension: [B, 1, H, W]
    gt_mag = gt_mag.unsqueeze(1)
    pred_mag = pred_mag.unsqueeze(1)

    # Compute SSIM per batch
    ssim_score = ssim_tensor(pred_mag, gt_mag, data_range=1.0)  # shape: [B]
    
    return ssim_score.mean()


def volume_nmse_tensor(gt, pred):
    """Volume Normalized Mean Squared Error (NMSE)"""
    '''
    tensor, [N,H,W]
    '''

    return torch.linalg.norm(gt - pred) ** 2 / torch.linalg.norm(gt) ** 2


def volume_psnr_tensor(gt, pred, maxval=None):
    """Volume Peak Signal to Noise Ratio metric (PSNR)"""
    '''
    tensor, [N,H,W]
    '''
    if maxval is None:
        maxval = gt.max() - gt.min()

    return psnr_tensor(pred, gt, data_range=maxval)


def volume_ssim_tensor(gt, pred, maxval=None):
    """Volume Structural Similarity Index Metric (SSIM)"""
    '''
    tensor, [N,H,W]
    '''
    if not gt.dim() == 3:
        raise ValueError("Unexpected number of dimensions in ground truth.")
    if not gt.dim() == pred.dim():
        raise ValueError("Ground truth dimensions does not match pred.")

    if maxval is None:
        maxval = gt.max() - gt.min()

    ssim = ssim_tensor(pred.unsqueeze(0), gt.unsqueeze(0), data_range=maxval)  # .unsqueeze(0) to [N,1,H,W]

    return ssim


def calc_psnr(
    gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None
) -> np.ndarray:
    """Compute Peak Signal to Noise Ratio metric (PSNR)"""
    if maxval is None:
        maxval = gt.max() - gt.min()
    return peak_signal_noise_ratio(gt, pred, data_range=maxval)


def calc_ssim(
    gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None
) -> np.ndarray:
    """Compute Structural Similarity Index Metric (SSIM)"""
    if not gt.ndim == 3:
        raise ValueError("Unexpected number of dimensions in ground truth.")
    if not gt.ndim == pred.ndim:
        raise ValueError("Ground truth dimensions does not match pred.")

    maxval = gt.max() - gt.min() if maxval is None else maxval

    ssim = np.array([0])
    for slice_num in range(gt.shape[0]):
        ssim = ssim + structural_similarity(
            gt[slice_num], pred[slice_num], data_range=maxval
        )

    return ssim / gt.shape[0]


def l1_mag_loss(y_pred, y_true):
    pred_mag = torch.sqrt(y_pred[:, 0]**2 + y_pred[:, 1]**2 + 1e-8)
    true_mag = torch.sqrt(y_true[:, 0]**2 + y_true[:, 1]**2 + 1e-8)
    return F.l1_loss(pred_mag, true_mag)
