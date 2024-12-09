import torch
import torch.nn as nn
import torch.nn.functional as F
from models.complex_layers import ComplexConv2d, ComplexConvTranspose2d, ComplexBatchNorm2d

def complex_init(module):
    if isinstance(module, (ComplexConv2d, ComplexConvTranspose2d)):
        nn.init.xavier_uniform_(module.conv_r.weight)
        nn.init.xavier_uniform_(module.conv_i.weight)
        if module.conv_r.bias is not None:
            nn.init.zeros_(module.conv_r.bias)
        if module.conv_i.bias is not None:
            nn.init.zeros_(module.conv_i.bias)
    elif isinstance(module, ComplexBatchNorm2d):
        if module.affine:
            nn.init.ones_(module.weight_real)
            nn.init.zeros_(module.weight_imag)
            nn.init.zeros_(module.bias_real)
            nn.init.zeros_(module.bias_imag)

def apply_complex_init(model):
    """
    Apply complex initialization to all convolutional layers in the model.
    """
    model.apply(complex_init)

def complex_ssim(pred, target, window_size=11, size_average=True):
    """
    Calculate Structural Similarity Index (SSIM) for complex-valued tensors.
    """
    pred_mag = torch.abs(pred)
    target_mag = torch.abs(target)
    
    C1, C2 = (0.01 * 255)**2, (0.03 * 255)**2
    
    window = torch.ones(1, 1, window_size, window_size) / (window_size * window_size)
    window = window.to(pred.device)
    
    mu1 = F.conv2d(pred_mag, window, padding=window_size//2, groups=1)
    mu2 = F.conv2d(target_mag, window, padding=window_size//2, groups=1)
    
    mu1_sq, mu2_sq = mu1.pow(2), mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(pred_mag * pred_mag, window, padding=window_size//2, groups=1) - mu1_sq
    sigma2_sq = F.conv2d(target_mag * target_mag, window, padding=window_size//2, groups=1) - mu2_sq
    sigma12 = F.conv2d(pred_mag * target_mag, window, padding=window_size//2, groups=1) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean() if size_average else ssim_map.mean(1).mean(1).mean(1)

class ComplexMSELoss(nn.Module):
    """
    Mean Squared Error (MSE) loss for complex-valued tensors.
    """
    def __init__(self):
        super(ComplexMSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        return self.mse(pred.real, target.real) + self.mse(pred.imag, target.imag)

class ComplexMSESSIMLoss(nn.Module):
    """
    Combined MSE and SSIM loss for complex-valued tensors.
    """
    def __init__(self, alpha=0.84):
        super(ComplexMSESSIMLoss, self).__init__()
        self.mse = ComplexMSELoss()
        self.alpha = alpha

    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        ssim_loss = 1 - complex_ssim(pred, target)
        return self.alpha * mse_loss + (1 - self.alpha) * ssim_loss

class ComplexNMSE(nn.Module):
    """
    Normalized Mean Squared Error (NMSE) for complex-valued tensors.
    """
    def __init__(self):
        super(ComplexNMSE, self).__init__()

    def forward(self, pred, target):
        mse = torch.mean(torch.abs(pred - target)**2)
        signal_power = torch.mean(torch.abs(target)**2)
        return mse / signal_power

class ComplexPSNR(nn.Module):
    """
    Peak Signal-to-Noise Ratio (PSNR) for complex-valued tensors.
    """
    def __init__(self):
        super(ComplexPSNR, self).__init__()

    def forward(self, pred, target):
        mse = torch.mean(torch.abs(pred - target)**2)
        max_val = torch.max(torch.abs(target))
        return 20 * torch.log10(max_val / torch.sqrt(mse))
