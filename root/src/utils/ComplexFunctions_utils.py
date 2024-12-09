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
    
    elif isinstance(module, ComplexBatchNorm2d) and module.affine:
        nn.init.ones_(module.weight_real)
        nn.init.zeros_(module.weight_imag)
        nn.init.zeros_(module.bias_real)
        nn.init.zeros_(module.bias_imag)

def apply_complex_init(model):
    """
    Apply complex initialization to all convolutional layers in the model.
    """
    model.apply(complex_init)
    
    
def clip_complex_gradients(model, max_norm=1.0):
    """
    Clips gradients of complex-valued parameters in the model.
    Applies norm-based clipping to real and imaginary parts separately.

    Args:
        model (nn.Module): The model with complex-valued parameters.
        max_norm (float): Maximum allowed gradient norm.
    """
    for param in model.parameters():
        if param.grad is not None and torch.is_complex(param.grad):
            # Clip real part
            real_grad = param.grad.real
            real_clipped = torch.clamp(real_grad, min=-max_norm, max=max_norm)
            param.grad.real.copy_(real_clipped)

            # Clip imaginary part
            imag_grad = param.grad.imag
            imag_clipped = torch.clamp(imag_grad, min=-max_norm, max=max_norm)
            param.grad.imag.copy_(imag_clipped)

def adjust_to_target(output, target_shape):
    """
    Adjusts the model's output to match the target shape by cropping or padding.
    """
    _, _, out_h, out_w = output.shape
    _, _, target_h, target_w = target_shape

    # Determine padding or cropping dimensions
    pad_h = max(0, target_h - out_h)
    pad_w = max(0, target_w - out_w)

    crop_h = max(0, out_h - target_h)
    crop_w = max(0, out_w - target_w)

    # Apply cropping if output is larger
    if crop_h > 0 or crop_w > 0:
        h_start = crop_h // 2
        w_start = crop_w // 2
        output = output[:, :, h_start:h_start + target_h, w_start:w_start + target_w]

    # Apply padding if output is smaller
    if pad_h > 0 or pad_w > 0:
        padding = (pad_w // 2, pad_w - (pad_w // 2), pad_h // 2, pad_h - (pad_h // 2))
        output = F.pad(output, padding, mode='constant', value=0)

    return output

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
    Supports both real-valued and complex-valued targets.
    """
    def __init__(self):
        super(ComplexMSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        if torch.is_complex(pred) and torch.is_complex(target):
            # Complex-to-complex comparison
            real_loss = self.mse(pred.real, target.real)
            imag_loss = self.mse(pred.imag, target.imag)
            return real_loss + imag_loss
        
        elif not torch.is_complex(pred) and not torch.is_complex(target):
            # Real-to-real comparison
            return self.mse(pred, target)
        
        elif torch.is_complex(pred) and not torch.is_complex(target):
            # Complex-to-real comparison (magnitude-based)
            pred_magnitude = torch.abs(pred)
            return self.mse(pred_magnitude, target)

        else:
            raise ValueError("Invalid input types for predictions and targets.")

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
    
    
    
    
# SSIM for magnitude comparison
# SSIM Magnitude
def ssim_magnitude(pred, target, window_size=11, C1=1e-4, C2=1e-4):
    mu_pred = F.avg_pool2d(pred, window_size, stride=1, padding=window_size // 2)
    mu_target = F.avg_pool2d(target, window_size, stride=1, padding=window_size // 2)

    sigma_pred = F.avg_pool2d(pred**2, window_size, stride=1, padding=window_size // 2) - mu_pred**2
    sigma_target = F.avg_pool2d(target**2, window_size, stride=1, padding=window_size // 2) - mu_target**2
    sigma_cross = F.avg_pool2d(pred * target, window_size, stride=1, padding=window_size // 2) - mu_pred * mu_target

    ssim_score = ((2 * mu_pred * mu_target + C1) * (2 * sigma_cross + C2)) / \
                 ((mu_pred**2 + mu_target**2 + C1) * (sigma_pred + sigma_target + C2))

    # Ensure SSIM stays in [0, 1]
    return torch.clamp(ssim_score.mean(), 0, 1)

# Phase Consistency
def phase_consistency(pred, target):
    pred_phase = torch.angle(pred)
    target_phase = torch.angle(target)

    # Normalize to [0, 1]
    phase_cons = (torch.cos(pred_phase - target_phase) + 1) / 2
    return phase_cons.mean()

# Combined Complex SSIM
def complex_ssim(pred, target, magnitude_weight=0.5, phase_weight=0.5):
    pred_mag = torch.abs(pred)
    target_mag = torch.abs(target)

    mag_ssim = ssim_magnitude(pred_mag, target_mag)
    phase_cons = phase_consistency(pred, target)

    # Compute the combined metric
    combined_ssim = magnitude_weight * mag_ssim + phase_weight * phase_cons
    return combined_ssim







def get_kspace(image):
    """
    Converts the output image to its k-space using 2D FFT.
    Args:
        image (torch.Tensor): The reconstructed image (real-valued).
    Returns:
        torch.Tensor: k-space representation (complex-valued).
    """
    kspace = torch.fft.fft2(image, dim=(-2, -1))
    return kspace


def extract_phase(kspace):
    """
    Extracts the phase from k-space.
    Args:
        kspace (torch.Tensor): Complex-valued k-space tensor.
    Returns:
        torch.Tensor: Phase tensor in radians.
    """
    return torch.angle(kspace)




class PhaseConsistencyLoss(nn.Module):
    """
    Calculates phase-consistency loss using cosine similarity.
    """

    def __init__(self, reduction='mean'):
        super(PhaseConsistencyLoss, self).__init__()
        self.reduction = reduction

    def forward(self, pred_image, target_image):
        # Convert to k-space using FFT
        pred_kspace = get_kspace(pred_image)
        target_kspace = get_kspace(target_image)

        # Extract phases from k-space
        pred_phase = extract_phase(pred_kspace)
        target_phase = extract_phase(target_kspace)

        # Calculate phase difference
        phase_diff = torch.cos(pred_phase - target_phase)

        # Phase consistency loss
        phase_loss = 1 - phase_diff

        if self.reduction == 'mean':
            return phase_loss.mean()
        elif self.reduction == 'sum':
            return phase_loss.sum()
        else:
            return phase_loss
        
        
class PhaseRegularizedMSELoss(nn.Module):
    """
    Combines MSE loss with Phase Consistency Loss.
    Args:
        alpha (float): Weight of phase regularization loss (default: 0.5).
    """

    def __init__(self, alpha=0.5):
        super(PhaseRegularizedMSELoss, self).__init__()
        self.mse = nn.MSELoss()
        self.phase_loss = PhaseConsistencyLoss()
        self.alpha = alpha

    def forward(self, pred_image, target_image):
        # Magnitude-based loss
        mag_loss = self.mse(pred_image, target_image)

        # Phase-based loss
        phase_reg_loss = self.phase_loss(pred_image, target_image)

        # Combined loss
        total_loss = mag_loss + self.alpha * phase_reg_loss

        return total_loss
