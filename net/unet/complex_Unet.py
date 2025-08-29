import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
import numpy as np
import traceback
import matplotlib.pyplot as plt 
from utils.kiki_helpers import (
    fft2, ifft2, DataConsist, complex_magnitude, 
    _to_complex, _to_2ch
)




class ComplexConv2d(nn.Module):
    """Complex 2D Convolution following Trabelsi et al."""
    def __init__(self, in_complex_channels, out_complex_channels, kernel_size, stride=1, padding=0, bias=True):
        super(ComplexConv2d, self).__init__()
        
        # For complex convolution:
        # in_complex_channels: number of complex input channels
        # out_complex_channels: number of complex output channels
        # Each complex channel is represented as 2 real channels (real + imaginary)
        
        self.in_complex_channels = in_complex_channels
        self.out_complex_channels = out_complex_channels
        
        self.conv_rr = nn.Conv2d(in_complex_channels, out_complex_channels, kernel_size, stride, padding, bias=bias)
        self.conv_ri = nn.Conv2d(in_complex_channels, out_complex_channels, kernel_size, stride, padding, bias=bias)
        self.conv_ir = nn.Conv2d(in_complex_channels, out_complex_channels, kernel_size, stride, padding, bias=bias)
        self.conv_ii = nn.Conv2d(in_complex_channels, out_complex_channels, kernel_size, stride, padding, bias=bias)
        
    def forward(self, x):
        # x shape: (batch, 2*in_complex_channels, H, W) 
        # where first half is real channels, second half is imaginary channels
        batch_size, total_channels, H, W = x.shape
        
        if total_channels != 2 * self.in_complex_channels:
            raise ValueError(f"Expected {2 * self.in_complex_channels} channels, got {total_channels}")
        
        # Split real and imaginary parts
        real = x[:, :self.in_complex_channels, :, :]      # (batch, in_complex_channels, H, W)
        imag = x[:, self.in_complex_channels:, :, :]      # (batch, in_complex_channels, H, W)
        
        # Complex multiplication: (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
        real_out = self.conv_rr(real) - self.conv_ii(imag)
        imag_out = self.conv_ri(real) + self.conv_ir(imag)
        
        # Concatenate real and imaginary parts
        return torch.cat([real_out, imag_out], dim=1)     # (batch, 2*out_complex_channels, H, W)

class ComplexBatchNorm2d(nn.Module):
    """Complex Batch Normalization following Trabelsi et al."""
    def __init__(self, complex_channels, eps=1e-5, momentum=0.1):
        super(ComplexBatchNorm2d, self).__init__()
        
        # complex_channels: number of complex channels
        # Total real channels = 2 * complex_channels
        self.complex_channels = complex_channels
        self.eps = eps
        self.momentum = momentum
        
        # Parameters for complex batch norm
        self.bn_r = nn.BatchNorm2d(complex_channels, eps=eps, momentum=momentum, affine=False)
        self.bn_i = nn.BatchNorm2d(complex_channels, eps=eps, momentum=momentum, affine=False)
        
        # Learnable parameters
        self.gamma_rr = nn.Parameter(torch.ones(complex_channels))
        self.gamma_ri = nn.Parameter(torch.zeros(complex_channels))
        self.gamma_ii = nn.Parameter(torch.ones(complex_channels))
        self.beta = nn.Parameter(torch.zeros(2 * complex_channels))
        
    def forward(self, x):
        batch_size, total_channels, H, W = x.shape
        
        if total_channels != 2 * self.complex_channels:
            raise ValueError(f"Expected {2 * self.complex_channels} channels, got {total_channels}")
        
        # Split real and imaginary parts
        real = x[:, :self.complex_channels, :, :]
        imag = x[:, self.complex_channels:, :, :]
        
        # Normalize real and imaginary parts
        real_norm = self.bn_r(real)
        imag_norm = self.bn_i(imag)
        
        # Apply complex scaling
        gamma_rr = self.gamma_rr.view(1, -1, 1, 1)
        gamma_ri = self.gamma_ri.view(1, -1, 1, 1)
        gamma_ii = self.gamma_ii.view(1, -1, 1, 1)
        
        real_out = gamma_rr * real_norm - gamma_ri * imag_norm
        imag_out = gamma_ri * real_norm + gamma_ii * imag_norm
        
        # Add bias
        beta_r = self.beta[:self.complex_channels].view(1, -1, 1, 1)
        beta_i = self.beta[self.complex_channels:].view(1, -1, 1, 1)
        
        real_out += beta_r
        imag_out += beta_i
        
        return torch.cat([real_out, imag_out], dim=1)

class ComplexReLU(nn.Module):
    """Complex ReLU activation"""
    def forward(self, x):
        # Apply ReLU to both real and imaginary parts independently
        return F.relu(x)

class ComplexLeakyReLU(nn.Module):
    """Complex Leaky ReLU activation"""
    def __init__(self, negative_slope=0.01):
        super(ComplexLeakyReLU, self).__init__()
        self.negative_slope = negative_slope
        
    def forward(self, x):
        return F.leaky_relu(x, self.negative_slope)

class ComplexMaxPool2d(nn.Module):
    """Complex Max Pooling - acts as low-pass filter in frequency domain"""
    def __init__(self, kernel_size, stride=None):
        super(ComplexMaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        
    def forward(self, x):
        batch_size, total_channels, H, W = x.shape
        channels = total_channels // 2
        
        real = x[:, :channels, :, :]
        imag = x[:, channels:, :, :]
        
        # For frequency domain, we use average pooling to act as low-pass filter
        real_out = F.avg_pool2d(real, self.kernel_size, self.stride)
        imag_out = F.avg_pool2d(imag, self.kernel_size, self.stride)
        
        return torch.cat([real_out, imag_out], dim=1)
    


# class ComplexUpsample(nn.Module):
#     """Complex Upsampling using Transpose Convolution"""
    
#     def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 2, stride: int = 2):
#         super().__init__()
#         self.upconv_r = nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride)
#         self.upconv_i = nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride)
        
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         real_up = self.upconv_r(x[:, 0])
#         imag_up = self.upconv_i(x[:, 1])
#         return torch.stack([real_up, imag_up], dim=1)
    

# ALSO NEED TO FIX ComplexUpsample - there might be an indexing issue
class ComplexUpsample(nn.Module):
    """Complex Upsampling using Transpose Convolution - FIXED VERSION"""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 2, stride: int = 2):
        super().__init__()
        # in_ch and out_ch are COMPLEX channel counts
        # Each complex channel = 2 real channels
        self.in_ch = in_ch
        self.out_ch = out_ch
        
        self.upconv_r = nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride)
        self.upconv_i = nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, 2 * in_ch, H, W) where first half=real, second half=imag
        B, C, H, W = x.shape
        
        expected_channels = 2 * self.in_ch
        if C != expected_channels:
            raise ValueError(f"Expected {expected_channels} channels (2 * {self.in_ch} complex channels), got {C}")
        
        # Split into real and imaginary parts
        real = x[:, :self.in_ch, :, :]   # (B, in_ch, H, W) - first half
        imag = x[:, self.in_ch:, :, :]   # (B, in_ch, H, W) - second half

        # Upsample each part
        real_up = self.upconv_r(real)    # (B, out_ch, H', W')
        imag_up = self.upconv_i(imag)    # (B, out_ch, H', W')

        # Concatenate: first out_ch channels = real, next out_ch channels = imag  
        return torch.cat([real_up, imag_up], dim=1)  # (B, 2*out_ch, H', W')
    
class ComplexDownsample(nn.Module):
    """Complex Downsampling using Max Pooling"""
    
    def __init__(self, kernel_size: int = 2):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply pooling to both real and imaginary parts"""
        real_pool = self.pool(x[:, 0])
        imag_pool = self.pool(x[:, 1])
        return torch.stack([real_pool, imag_pool], dim=1)


class ComplexAttentionBlock(nn.Module):
    """Complex Attention Block"""
    def __init__(self, complex_channels):
        super(ComplexAttentionBlock, self).__init__()
        
        # Ensure we have at least 1 complex channel for reduction
        reduction_complex_channels = max(1, complex_channels // 8)
            
        self.conv1 = ComplexConv2d(complex_channels, reduction_complex_channels, 1)
        self.conv2 = ComplexConv2d(reduction_complex_channels, complex_channels, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Global average pooling
        batch_size, channels, H, W = x.shape
        
        # Average pool to 1x1
        gap = F.adaptive_avg_pool2d(x, 1)
        
        # Attention mechanism
        att = self.conv1(gap)
        att = ComplexReLU()(att)
        att = self.conv2(att)
        att = self.sigmoid(att)
        
        return x * att

class ComplexResidualBlock(nn.Module):
    """Complex Residual Block with Attention"""
    def __init__(self, in_complex_channels, out_complex_channels, stride=1):
        super(ComplexResidualBlock, self).__init__()
        
        self.conv1 = ComplexConv2d(in_complex_channels, out_complex_channels, 3, stride, 1)
        self.bn1 = ComplexBatchNorm2d(out_complex_channels)
        self.conv2 = ComplexConv2d(out_complex_channels, out_complex_channels, 3, 1, 1)
        self.bn2 = ComplexBatchNorm2d(out_complex_channels)
        
        self.attention = ComplexAttentionBlock(out_complex_channels)
        
        # Skip connection
        if stride != 1 or in_complex_channels != out_complex_channels:
            self.skip = nn.Sequential(
                ComplexConv2d(in_complex_channels, out_complex_channels, 1, stride),
                ComplexBatchNorm2d(out_complex_channels)
            )
        else:
            self.skip = nn.Identity()
            
        self.relu = ComplexReLU()
        
    def forward(self, x):
        identity = self.skip(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = self.attention(out)
        
        out += identity
        out = self.relu(out)
        
        return out

class ComplexConvBlock(nn.Module):
    """Complex Convolutional Block with BatchNorm and Activation"""
    
    def __init__(self, in_ch: int, out_ch: int, activation: str = 'relu', dropout: float = 0.0):
        super().__init__()
        
        self.conv1 = ComplexConv2d(in_ch, out_ch, 3, 1, 1)
        self.bn1 = ComplexBatchNorm2d(out_ch)
        self.conv2 = ComplexConv2d(out_ch, out_ch, 3, 1, 1)
        self.bn2 = ComplexBatchNorm2d(out_ch)
        
        if activation == 'relu':
            self.act = ComplexReLU()
        elif activation == 'leaky_relu':
            self.act = ComplexLeakyReLU(0.2)
        else:
            raise ValueError(f"Unsupported activation: {activation}")
            
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.act(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        return x


class CUNet(nn.Module):
    """CUNet using KIKI helpers for better stability and mixed precision support"""
    
    def __init__(self, in_channels=2, out_channels=1, base_features=32, use_data_consistency=True):
        super(CUNet, self).__init__()
        self.use_data_consistency = use_data_consistency
        
        # Convert input channels to complex channels
        in_complex_channels = in_channels // 2  # 1 complex channel from 2 real channels
        
        # Channel progression for complex network
        f1 = base_features      # 32 complex channels = 64 real channels
        f2 = f1 * 2            # 64 complex channels = 128 real channels  
        f3 = f2 * 2            # 128 complex channels = 256 real channels
        f4 = f3 * 2            # 256 complex channels = 512 real channels
        f5 = f4 * 2            # 512 complex channels = 1024 real channels
        
        # Encoder (using existing complex layers from CUNet debug)
        self.encoder1 = ComplexResidualBlock(in_complex_channels, f1)
        self.pool1 = ComplexMaxPool2d(2)
        
        self.encoder2 = ComplexResidualBlock(f1, f2)
        self.pool2 = ComplexMaxPool2d(2)
        
        self.encoder3 = ComplexResidualBlock(f2, f3)
        self.pool3 = ComplexMaxPool2d(2)
        
        self.encoder4 = ComplexResidualBlock(f3, f4)
        self.pool4 = ComplexMaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = ComplexResidualBlock(f4, f5)
        
        # Decoder
        self.upconv4 = ComplexUpsample(f5, f4)
        self.decoder4 = ComplexResidualBlock(f4 + f4, f4)

        self.upconv3 = ComplexUpsample(f4, f3) 
        self.decoder3 = ComplexResidualBlock(f3 + f3, f3)

        self.upconv2 = ComplexUpsample(f3, f2)
        self.decoder2 = ComplexResidualBlock(f2 + f2, f2)

        self.upconv1 = ComplexUpsample(f2, f1)
        self.decoder1 = ComplexResidualBlock(f1 + f1, f1)
        
        # Output layer
        self.final_conv = ComplexConv2d(f1, in_complex_channels, 1)
        
    def forward(self, x, mask=None):
        """
        Forward pass using KIKI helpers for all operations
        
        Args:
            x: Input k-space (B, 2, H, W)
            mask: Sampling mask (B, 1, H, W), optional
            
        Returns:
            pred_magnitude: Reconstructed magnitude image (B, 1, H, W)
        """
        # Encoder path
        enc1 = self.encoder1(x)
        pool1 = self.pool1(enc1)
        
        enc2 = self.encoder2(pool1)
        pool2 = self.pool2(enc2)
        
        enc3 = self.encoder3(pool2)
        pool3 = self.pool3(enc3)
        
        enc4 = self.encoder4(pool3)
        pool4 = self.pool4(enc4)
        
        # Bottleneck
        bottleneck = self.bottleneck(pool4)
        
        # Decoder path
        up4 = self.upconv4(bottleneck)
        dec4 = self.decoder4(torch.cat([up4, enc4], dim=1))
        
        up3 = self.upconv3(dec4)
        dec3 = self.decoder3(torch.cat([up3, enc3], dim=1))
        
        up2 = self.upconv2(dec3)
        dec2 = self.decoder2(torch.cat([up2, enc2], dim=1))
        
        up1 = self.upconv1(dec2)
        dec1 = self.decoder1(torch.cat([up1, enc1], dim=1))
        
        # Predict k-space
        pred_k = self.final_conv(dec1)  # (B, 2, H, W)
        
        # Apply hard data consistency using KIKI helper
        if self.use_data_consistency and mask is not None:
            pred_k = DataConsist(pred_k, x, mask, is_k=True)
        
        # Convert to image domain using KIKI helper
        pred_img_2ch = ifft2(pred_k)  # (B, 2, H, W)
        
        # Convert to magnitude using KIKI helper  
        pred_magnitude = complex_magnitude(pred_img_2ch)  # (B, H, W)
        
        # Add channel dimension for output
        if pred_magnitude.dim() == 3:
            pred_magnitude = pred_magnitude.unsqueeze(1)  # (B, 1, H, W)
            
        return pred_magnitude
    
    def kspace_to_image(self, kspace):
        """Convert k-space to image using KIKI helper"""
        return ifft2(kspace)
    
    def image_to_kspace(self, image):
        """Convert image to k-space using KIKI helper"""  
        return fft2(image)
    
    def complex_to_magnitude(self, complex_img):
        """Convert complex image to magnitude using KIKI helper"""
        magnitude = complex_magnitude(complex_img)
        return magnitude.unsqueeze(1) if magnitude.dim() == 3 else magnitude

class CUNetLoss(nn.Module):
    """Loss function for CU-Net training"""
    def __init__(self, alpha=1.0, beta=0.1):
        super(CUNetLoss, self).__init__()
        self.alpha = alpha  # Weight for image domain loss
        self.beta = beta    # Weight for k-space domain loss
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        
    def forward(self, pred_img, target_img, pred_kspace=None, target_kspace=None):
        # Image domain loss
        img_loss = self.mse(pred_img, target_img) + 0.1 * self.l1(pred_img, target_img)
        
        total_loss = self.alpha * img_loss
        
        # Optional k-space domain loss
        if pred_kspace is not None and target_kspace is not None:
            kspace_loss = self.mse(pred_kspace, target_kspace)
            total_loss += self.beta * kspace_loss
            
        return total_loss


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Add these to your kiki_helpers.py or create a separate cunet_integration.py

# ===============================================
# STREAMLINED CUNET HELPERS (NO DUPLICATION)
# ===============================================

def create_cunet(device, base_features=32, use_data_consistency=True):
    """Create CUNet model"""
    model = CUNet(
        in_channels=2,
        out_channels=1,
        base_features=base_features,
        use_data_consistency=use_data_consistency
    ).to(device)
    
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"CUNet: {param_count:,} params, DC: {use_data_consistency}")
    return model

def test_cunet(device, test_size=64):
    """Test basic CUNet functionality"""
    model = create_cunet(device, base_features=8)
    
    k_space = torch.randn(1, 2, test_size, test_size).to(device)
    mask = torch.randint(0, 2, (1, 1, test_size, test_size)).float().to(device)
    
    with torch.no_grad():
        output = model(k_space, mask)
    
    success = output.shape == (1, 1, test_size, test_size)
    print(f"Test {test_size}x{test_size}: {'PASSED' if success else 'FAILED'}")
    return success

def prepare_cunet_data(X, y, mask):
    """Prepare data for CUNet"""
    # K-space input (B, 2, H, W)
    if X.shape[1] != 2:
        raise ValueError(f"Expected k-space (B, 2, H, W), got {X.shape}")
    
    # Target magnitude (B, 1, H, W)  
    if y.shape[1] == 1:
        target = y.float()
    elif y.shape[1] == 2:
        target = complex_magnitude(y).unsqueeze(1)
    else:
        raise ValueError(f"Expected target (B, 1, H, W) or (B, 2, H, W), got {y.shape}")
    
    # Mask (B, 1, H, W)
    if mask.dim() == 3:
        mask = mask.unsqueeze(1).float()
    elif mask.dim() == 4 and mask.shape[1] == 1:
        mask = mask.float()
    else:
        raise ValueError(f"Expected mask (B, H, W) or (B, 1, H, W), got {mask.shape}")
    
    return X.float(), target, mask


# Copy a directory from server to your remote machine
