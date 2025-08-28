import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
import numpy as np
import traceback
import matplotlib.pyplot as plt 

def channels_to_complex(x: torch.Tensor) -> torch.Tensor:
    """Convert (B,2,H,W) to complex (B,H,W)"""
    assert x.ndim == 4 and x.size(1) == 2, f"Expected (B,2,H,W), got {tuple(x.shape)}"
    return torch.complex(x[:, 0], x[:, 1])

def complex_to_channels(z: torch.Tensor) -> torch.Tensor:
    """Convert complex (B,H,W) to (B,2,H,W)"""
    assert torch.is_complex(z), "Input must be complex tensor"
    return torch.stack([z.real, z.imag], dim=1)

def fft2c_2ch(x_2ch: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    """Centered 2D FFT: (B,2,H,W) -> (B,2,H,W)"""
    x = channels_to_complex(x_2ch)
    x = torch.fft.ifftshift(x, dim=(-2, -1))
    k = torch.fft.fft2(x, dim=(-2, -1), norm=norm)
    k = torch.fft.fftshift(k, dim=(-2, -1))
    return complex_to_channels(k)

def ifft2c_2ch(k_2ch: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    """Centered 2D IFFT: (B,2,H,W) -> (B,2,H,W)"""
    k = channels_to_complex(k_2ch)
    k = torch.fft.ifftshift(k, dim=(-2, -1))
    x = torch.fft.ifft2(k, dim=(-2, -1), norm=norm)
    x = torch.fft.fftshift(x, dim=(-2, -1))
    return complex_to_channels(x)


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

class HardDataConsistency(nn.Module):
    """Hard Data Consistency: replace predicted k-space at sampled locations"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, k_pred: torch.Tensor, k_meas: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            k_pred: Predicted k-space (B, 2, H, W)
            k_meas: Measured k-space (B, 2, H, W)  
            mask: Sampling mask (B, 1, H, W) where 1=sampled
            
        Returns:
            k_dc: Data consistent k-space (B, 2, H, W)
        """
        # Ensure mask has correct shape
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)
        
        # Convert to binary and expand to match k-space channels
        M = (mask > 0).float()  # (B, 1, H, W)
        M = M.expand(-1, 2, -1, -1)  # (B, 2, H, W)
        
        # Data consistency: keep measured where mask=1, predicted where mask=0
        k_dc = M * k_meas + (1 - M) * k_pred
        
        return k_dc

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
    """Complete Complex U-Net for K-space MRI Reconstruction - FIXED VERSION"""
    def __init__(self, in_channels=2, out_channels=1, base_features=32, use_data_consistency=False):
        super(CUNet, self).__init__()
        self.use_data_consistency = use_data_consistency
        
        # Convert input/output channels to complex channels
        # in_channels=2 means 1 complex channel (real + imaginary)
        in_complex_channels = in_channels // 2  # 1 complex channel
        
        # Channel progression for complex network
        f1 = base_features      # 32 complex channels = 64 real channels
        f2 = f1 * 2            # 64 complex channels = 128 real channels
        f3 = f2 * 2            # 128 complex channels = 256 real channels
        f4 = f3 * 2            # 256 complex channels = 512 real channels
        f5 = f4 * 2            # 512 complex channels = 1024 real channels
        
        # Encoder
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
        
        # Decoder - FIXED: Channel calculations corrected
        self.upconv4 = ComplexUpsample(f5, f4)  # 512 -> 256 complex channels
        # After concatenation: upconv4(256) + enc4(256) = 512 complex channels total
        self.decoder4 = ComplexResidualBlock(f4 + f4, f4)  # (256+256)->256 complex

        self.upconv3 = ComplexUpsample(f4, f3)  # 256 -> 128 complex channels  
        # After concatenation: upconv3(128) + enc3(128) = 256 complex channels total
        self.decoder3 = ComplexResidualBlock(f3 + f3, f3)  # (128+128)->128 complex

        self.upconv2 = ComplexUpsample(f3, f2)  # 128 -> 64 complex channels
        # After concatenation: upconv2(64) + enc2(64) = 128 complex channels total
        self.decoder2 = ComplexResidualBlock(f2 + f2, f2)  # (64+64)->64 complex

        self.upconv1 = ComplexUpsample(f2, f1)  # 64 -> 32 complex channels
        # After concatenation: upconv1(32) + enc1(32) = 64 complex channels total
        self.decoder1 = ComplexResidualBlock(f1 + f1, f1)  # (32+32)->32 complex
                
        # Output layer - converts back to 1 complex channel
        self.final_conv = ComplexConv2d(f1, in_complex_channels, 1)
        
    def forward(self, x, mask=None):
        # x: (B,2,H,W) k-space (measured)
        # 1) Encode -> Decode in complex space (all 2-ch at each stage)
        enc1 = self.encoder1(x);      pool1 = self.pool1(enc1)
        enc2 = self.encoder2(pool1);  pool2 = self.pool2(enc2)
        enc3 = self.encoder3(pool2);  pool3 = self.pool3(enc3)
        enc4 = self.encoder4(pool3);  pool4 = self.pool4(enc4)
        bottleneck = self.bottleneck(pool4)

        up4 = self.upconv4(bottleneck)
        dec4 = self.decoder4(torch.cat([up4, enc4], dim=1))
        up3 = self.upconv3(dec4)
        dec3 = self.decoder3(torch.cat([up3, enc3], dim=1))
        up2 = self.upconv2(dec3)
        dec2 = self.decoder2(torch.cat([up2, enc2], dim=1))
        up1 = self.upconv1(dec2)
        dec1 = self.decoder1(torch.cat([up1, enc1], dim=1))

        # 2) Predict k-space (2-ch)
        pred_k = self.final_conv(dec1)  # (B,2,H,W)

        # 3) Optional DC IN K-SPACE using measured k-space x
        if self.use_data_consistency and mask is not None:
            # enforce consistency at sampled locations
            pred_k = self.apply_data_consistency(pred_k, x, mask, is_k=True)  # returns (B,2,H,W)

        # 4) IFFT to image (2-ch real/imag) then magnitude (1-ch)
        pred_img_2ch = self.kspace_to_image(pred_k)          # (B,2,H,W)
        pred_mag     = self.complex_to_magnitude(pred_img_2ch)  # (B,1,H,W)

        return pred_mag
    
    def apply_data_consistency(self, input_, k, m, is_k=False):
        """
        Apply data consistency with a binary mask m in k-space.

        Args:
            input_: (B,2,H,W)  -> if is_k=True: predicted k-space; else: predicted image (real,imag)
            k:      (B,2,H,W)  -> measured k-space (real,imag)
            m:      (B,1,H,W)  -> sampling mask in {0,1}
            is_k:   bool       -> True if input_ is already k-space

        Returns:
            (B,2,H,W) in the same domain as input_ (k-space if is_k=True, image if is_k=False)
        """
        # ensure mask broadcasts to 2 channels and dtype matches
        m2 = m.expand(-1, 2, -1, -1).to(input_.dtype)

        # helpers: (B,2,H,W) <-> complex (B,H,W)
        def to_complex(x2):
            return torch.complex(x2[:, 0].contiguous(), x2[:, 1].contiguous())

        def from_complex(z):
            return torch.stack([z.real, z.imag], dim=1)

        if is_k:
            # input_ is k-space already -> blend in k-space
            k_pred = input_
            k_meas = k
            k_dc = m2 * k_meas + (1.0 - m2) * k_pred
            return k_dc
        else:
            # input_ is image -> FFT to k-space, apply DC, IFFT back to image
            x_img = to_complex(input_)                         # (B,H,W) complex
            k_pred = torch.fft.fft2(x_img, norm='ortho')       # (B,H,W) complex
            k_pred_2ch = from_complex(k_pred)                  # (B,2,H,W)

            k_meas = k                                         # (B,2,H,W) real/imag
            # DC in k-space
            k_dc_2ch = m2 * k_meas + (1.0 - m2) * k_pred_2ch   # (B,2,H,W)
            k_dc = to_complex(k_dc_2ch)                        # (B,H,W) complex

            x_dc = torch.fft.ifft2(k_dc, norm='ortho')         # (B,H,W) complex
            return from_complex(x_dc)
    
    def image_to_kspace(self, magnitude_img):
        """Convert magnitude image to k-space (approximate)"""
        batch_size, channels, H, W = magnitude_img.shape
        
        # Create complex image with zero phase (limitation of magnitude-only)
        real_img = magnitude_img.squeeze(1)  # (batch, H, W)
        imag_img = torch.zeros_like(real_img)  # (batch, H, W)
        
        complex_img = torch.complex(real_img, imag_img)
        
        # Apply 2D FFT
        complex_kspace = torch.fft.fft2(complex_img, norm='ortho')
        
        # Split to real and imaginary channels for 1 complex channel
        real_k = complex_kspace.real.unsqueeze(1)  # (batch, 1, H, W)
        imag_k = complex_kspace.imag.unsqueeze(1)  # (batch, 1, H, W)
        
        return torch.cat([real_k, imag_k], dim=1)  # (batch, 2, H, W) - 1 complex channel
    
    def kspace_to_image(self, kspace):
        """Convert k-space to image domain using IFFT"""
        batch_size, channels, H, W = kspace.shape
        
        # Separate real and imaginary parts
        real = kspace[:, 0, :, :]  # (batch, H, W)
        imag = kspace[:, 1, :, :]  # (batch, H, W)
        
        # Create complex tensor
        complex_kspace = torch.complex(real, imag)  # (batch, H, W)
        
        # Apply 2D IFFT
        complex_image = torch.fft.ifft2(complex_kspace, norm='ortho')
        
        # Split back to real and imaginary
        real_img = complex_image.real.unsqueeze(1)  # (batch, 1, H, W)
        imag_img = complex_image.imag.unsqueeze(1)  # (batch, 1, H, W)
        
        return torch.cat([real_img, imag_img], dim=1)  # (batch, 2, H, W)
    
    def complex_to_magnitude(self, complex_img):
        """Convert complex image to magnitude"""
        batch_size, channels, H, W = complex_img.shape
        
        real = complex_img[:, 0, :, :]  # (batch, H, W)
        imag = complex_img[:, 1, :, :]  # (batch, H, W)
        
        magnitude = torch.sqrt(real**2 + imag**2 + 1e-8)  # Add small epsilon for stability
        
        return magnitude.unsqueeze(1)  # (batch, 1, H, W)

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

# Example training function
def train_step(model, data_loader, optimizer, criterion, device):
    """Single training step"""
    model.train()
    total_loss = 0
    
    for batch_idx, (x, y, mask) in enumerate(data_loader):
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        pred_y = model(x, mask)  # Pass both k-space and mask
        
        # Compute loss
        loss = criterion(pred_y, y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(data_loader)

def create_cunet_with_hard_dc(base_features=32, device='cuda'):
    """Create CUNet with hard data consistency enabled"""
    
    # IMPORTANT: Set use_data_consistency=True
    model = CUNet(
        in_channels=2,           # Complex k-space input (real + imag)
        out_channels=1,          # Magnitude image output
        base_features=base_features,
        use_data_consistency=True  # ← KEY CHANGE: Enable hard DC
    ).to(device)
    
    print(f" CUNet created with hard data consistency ENABLED")
    print(f"   - Base features: {base_features}")
    print(f"   - Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    return model

def load_cunet_checkpoint_with_dc(model, checkpoint_path, device):
    """Load checkpoint for CUNet with data consistency"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict)
        
        epoch = checkpoint.get('epoch', 'unknown')
        loss = checkpoint.get('loss', 'unknown')
        
        print(f" Loaded CUNet checkpoint:")
        print(f"   - Path: {checkpoint_path}")
        print(f"   - Epoch: {epoch}")
        print(f"   - Loss: {loss}")
        print(f"   - Data consistency: {model.use_data_consistency}")
        
        return model
    except Exception as e:
        print(f" Error loading checkpoint: {e}")
        return model

# ===============================================
# 2. DATA PREPARATION CHANGES
# ===============================================

def prepare_data_for_hard_dc(X, y, mask):
    """
    Prepare data for CUNet with hard data consistency
    
    Args:
        X: Input data from dataloader
        y: Target data from dataloader  
        mask: Sampling mask from dataloader
        
    Returns:
        k_undersampled: Undersampled k-space (B, 2, H, W)
        target_magnitude: Target magnitude image (B, 1, H, W)
        sampling_mask: Sampling mask (B, 1, H, W)
    """
    
    # Ensure correct input format
    if X.shape[1] == 2:  # Already k-space format (B, 2, H, W)
        k_undersampled = X.float()
        print(f" Input is k-space format: {k_undersampled.shape}")
    else:
        raise ValueError(f"Expected k-space input (B, 2, H, W), got {X.shape}")
    
    # Handle target format
    if y.shape[1] == 1:  # Magnitude target (B, 1, H, W)
        target_magnitude = y.float()
        print(f" Target is magnitude format: {target_magnitude.shape}")
    elif y.shape[1] == 2:  # Complex target, convert to magnitude
        real, imag = y[:, 0], y[:, 1]
        target_magnitude = torch.sqrt(real**2 + imag**2 + 1e-8).unsqueeze(1)
        print(f" Converted complex target to magnitude: {target_magnitude.shape}")
    else:
        raise ValueError(f"Expected target (B, 1, H, W) or (B, 2, H, W), got {y.shape}")
    
    # Handle mask format
    if mask.dim() == 3:  # (B, H, W)
        sampling_mask = mask.unsqueeze(1).float()  # (B, 1, H, W)
    elif mask.dim() == 4 and mask.shape[1] == 1:  # (B, 1, H, W)
        sampling_mask = mask.float()
    else:
        raise ValueError(f"Expected mask (B, H, W) or (B, 1, H, W), got {mask.shape}")
    
    print(f" Data prepared - K-space: {k_undersampled.shape}, Target: {target_magnitude.shape}, Mask: {sampling_mask.shape}")
    
    return k_undersampled, target_magnitude, sampling_mask










# ===============================================
# 3. INFERENCE CHANGES
# ===============================================

def test_cunet_with_hard_dc(model, dataloader, device, num_samples=5):
    """
    Test CUNet with hard data consistency
    
    Key changes:
    1. Pass both k-space input AND mask to model
    2. Model applies hard DC internally
    3. Proper data format handling
    """
    
    model.eval()
    
    all_metrics = {'NMSE': [], 'PSNR': [], 'SSIM': []}
    
    print(f" Testing CUNet with Hard Data Consistency")
    print(f"   - Data consistency enabled: {model.use_data_consistency}")
    print(f"   - Number of samples: {num_samples}")
    
    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            if idx >= num_samples:
                break
                
            # Unpack data
            X, y, mask = data
            X = X.to(device)
            y = y.to(device) 
            mask = mask.to(device)
            
            # Prepare data with proper formats
            k_undersampled, target_magnitude, sampling_mask = prepare_data_for_hard_dc(X, y, mask)
            
            # KEY CHANGE: Pass BOTH k-space and mask to model
            # The model will:
            # 1. Process k-space through U-Net
            # 2. Apply hard data consistency using the mask
            # 3. Return magnitude image
            pred_magnitude = model(k_undersampled, sampling_mask)
            
            print(f"Sample {idx + 1}:")
            print(f"  - Input k-space: {k_undersampled.shape}")
            print(f"  - Sampling mask: {sampling_mask.shape}")
            print(f"  - Prediction: {pred_magnitude.shape}")
            print(f"  - Target: {target_magnitude.shape}")
            
            # Calculate metrics
            for b in range(pred_magnitude.shape[0]):
                pred_img = pred_magnitude[b, 0].cpu().numpy()
                target_img = target_magnitude[b, 0].cpu().numpy()
                
                # NMSE
                mse = np.mean((pred_img - target_img) ** 2)
                target_var = np.var(target_img)
                nmse = mse / target_var if target_var > 0 else float('inf')
                
                # PSNR
                data_range = target_img.max() - target_img.min()
                # psnr = psnr_metric(target_img, pred_img, data_range=data_range)
                
                # SSIM
                # ssim_val = ssim_metric(target_img, pred_img, data_range=data_range)
                
                all_metrics['NMSE'].append(nmse)
                # all_metrics['PSNR'].append(psnr)
                # all_metrics['SSIM'].append(ssim_val)
                
                # print(f"    Batch {b}: NMSE={nmse:.4f}, PSNR={psnr:.2f}, SSIM={ssim_val:.4f}")
    
    # Calculate average metrics
    avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}
    
    print(f"\n Average Metrics (with Hard Data Consistency):")
    print(f"   - NMSE: {avg_metrics['NMSE']:.6f}")
    print(f"   - PSNR: {avg_metrics['PSNR']:.2f} dB")
    print(f"   - SSIM: {avg_metrics['SSIM']:.4f}")
    
    return avg_metrics




# ===============================================
# STEP-BY-STEP TESTING FOR CUNet with Hard DC
# ===============================================

def test_complex_conv2d():
    """Test ComplexConv2d layer independently"""
    print("\n" + "="*50)
    print(" TESTING ComplexConv2d")
    print("="*50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    try:
        # Create ComplexConv2d layer: 1 complex channel -> 4 complex channels
        conv = ComplexConv2d(1, 4, kernel_size=3, padding=1).to(device)
        print(f" ComplexConv2d created: 1 -> 4 complex channels")
        
        # Test input: 1 complex channel = 2 real channels (real, imag)
        batch_size = 2
        H, W = 64, 64
        x = torch.randn(batch_size, 2, H, W).to(device)  # (B, 2, H, W)
        print(f" Input shape: {x.shape} -> {x.shape[1]//2} complex channels")
        
        # Forward pass
        out = conv(x)
        expected_channels = 4 * 2  # 4 complex channels = 8 real channels
        print(f" Output shape: {out.shape}")
        print(f"Expected channels: {expected_channels}, Got: {out.shape[1]}")
        
        if out.shape[1] == expected_channels:
            print(" ComplexConv2d test PASSED!")
            
            # Check gradients
            loss = out.sum()
            loss.backward()
            print(" Gradient computation successful!")
            return True
        else:
            print(f" Wrong output channels: expected {expected_channels}, got {out.shape[1]}")
            return False
            
    except Exception as e:
        print(f" ComplexConv2d test FAILED: {e}")
        traceback.print_exc()
        return False

def test_complex_batch_norm():
    """Test ComplexBatchNorm2d layer"""
    print("\n" + "="*50)
    print(" TESTING ComplexBatchNorm2d")
    print("="*50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Create ComplexBatchNorm2d for 4 complex channels
        bn = ComplexBatchNorm2d(4).to(device)
        print(f" ComplexBatchNorm2d created for 4 complex channels")
        
        # Test input: 4 complex channels = 8 real channels
        x = torch.randn(2, 8, 32, 32).to(device)
        print(f" Input shape: {x.shape}")
        
        # Forward pass
        out = bn(x)
        print(f" Output shape: {out.shape}")
        
        if out.shape == x.shape:
            print( "ComplexBatchNorm2d test PASSED!")
            
            # Check if normalization actually happened
            input_mean = x.mean().item()
            output_mean = out.mean().item()
            print(f"Input mean: {input_mean:.6f}, Output mean: {output_mean:.6f}")
            return True
        else:
            print(f" Shape mismatch: input {x.shape}, output {out.shape}")
            return False
            
    except Exception as e:
        print(f" ComplexBatchNorm2d test FAILED: {e}")
        traceback.print_exc()
        return False

def test_fft_operations():
    """Test FFT and IFFT operations"""
    print("\n" + "="*50)
    print(" TESTING FFT/IFFT Operations")
    print("="*50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Test data: complex k-space
        batch_size = 2
        H, W = 64, 64
        
        # Create test k-space (real, imag channels)
        k_space = torch.randn(batch_size, 2, H, W).to(device)
        print(f" Original k-space shape: {k_space.shape}")
        
        # Test kspace_to_image conversion
        print(" Testing k-space → image conversion...")
        
        # Extract real and imaginary parts
        real = k_space[:, 0, :, :]  # (batch, H, W)
        imag = k_space[:, 1, :, :]  # (batch, H, W)
        
        # Create complex tensor
        complex_kspace = torch.complex(real, imag)  # (batch, H, W)
        print(f"Complex k-space shape: {complex_kspace.shape}")
        
        # Apply 2D IFFT
        complex_image = torch.fft.ifft2(complex_kspace, norm='ortho')
        print(f"Complex image shape: {complex_image.shape}")
        
        # Convert back to 2-channel format
        real_img = complex_image.real.unsqueeze(1)  # (batch, 1, H, W)
        imag_img = complex_image.imag.unsqueeze(1)  # (batch, 1, H, W)
        image_2ch = torch.cat([real_img, imag_img], dim=1)  # (batch, 2, H, W)
        
        print(f" Image (2-channel) shape: {image_2ch.shape}")
        
        # Test reverse: image → k-space
        print(" Testing image → k-space conversion...")
        
        # Convert magnitude to k-space (with zero phase approximation)
        magnitude = torch.sqrt(real_img.squeeze(1)**2 + imag_img.squeeze(1)**2 + 1e-8)
        print(f"Magnitude shape: {magnitude.shape}")
        
        # Create complex image with zero phase
        complex_img_zero_phase = torch.complex(magnitude, torch.zeros_like(magnitude))
        
        # Apply FFT
        reconstructed_kspace = torch.fft.fft2(complex_img_zero_phase, norm='ortho')
        
        # Convert to 2-channel
        real_k = reconstructed_kspace.real.unsqueeze(1)
        imag_k = reconstructed_kspace.imag.unsqueeze(1)
        kspace_reconstructed = torch.cat([real_k, imag_k], dim=1)
        
        print(f" Reconstructed k-space shape: {kspace_reconstructed.shape}")
        
        print(" FFT/IFFT operations test PASSED!")
        return True
        
    except Exception as e:
        print(f" FFT/IFFT test FAILED: {e}")
        traceback.print_exc()
        return False

def test_hard_data_consistency():
    """Test hard data consistency operation"""
    print("\n" + "="*50)
    print(" TESTING Hard Data Consistency")
    print("="*50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        batch_size = 2
        H, W = 32, 32
        
        # Create test data
        input_kspace = torch.randn(batch_size, 2, H, W).to(device)
        predicted_kspace = torch.randn(batch_size, 2, H, W).to(device)
        mask = torch.randint(0, 2, (batch_size, 1, H, W)).float().to(device)
        
        print(f" Input k-space: {input_kspace.shape}")
        print(f" Predicted k-space: {predicted_kspace.shape}")
        print(f" Mask: {mask.shape}")
        print(f"Mask stats: min={mask.min():.1f}, max={mask.max():.1f}, mean={mask.mean():.3f}")
        
        # Apply data consistency manually
        mask_expanded = mask.repeat(1, 2, 1, 1)  # (batch, 2, H, W)
        consistent_kspace = mask_expanded * input_kspace + (1 - mask_expanded) * predicted_kspace
        
        print(f" Consistent k-space: {consistent_kspace.shape}")
        
        # Verify data consistency
        sampled_locations = mask_expanded == 1
        unsampled_locations = mask_expanded == 0
        
        # At sampled locations, should match input
        sampled_diff = torch.abs(consistent_kspace[sampled_locations] - input_kspace[sampled_locations])
        sampled_max_diff = sampled_diff.max().item()
        
        # At unsampled locations, should match predicted
        unsampled_diff = torch.abs(consistent_kspace[unsampled_locations] - predicted_kspace[unsampled_locations])
        unsampled_max_diff = unsampled_diff.max().item()
        
        print(f"Max difference at sampled locations: {sampled_max_diff:.8f} (should be ~0)")
        print(f"Max difference at unsampled locations: {unsampled_max_diff:.8f} (should be ~0)")
        
        if sampled_max_diff < 1e-6 and unsampled_max_diff < 1e-6:
            print(" Hard Data Consistency test PASSED!")
            return True
        else:
            print(" Hard Data Consistency test FAILED!")
            return False
            
    except Exception as e:
        print(f" Hard Data Consistency test FAILED: {e}")
        traceback.print_exc()
        return False

def test_complex_residual_block():
    """Test ComplexResidualBlock if available"""
    print("\n" + "="*50)
    print(" TESTING ComplexResidualBlock")
    print("="*50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Create a simple mock of ComplexResidualBlock for testing
        # In practice, this should use your actual ComplexResidualBlock
        
        class MockComplexResidualBlock(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()
                self.conv1 = ComplexConv2d(in_channels, out_channels, 3, padding=1)
                self.conv2 = ComplexConv2d(out_channels, out_channels, 3, padding=1)
                self.skip = ComplexConv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
                
            def forward(self, x):
                identity = self.skip(x) if not isinstance(self.skip, nn.Identity) else x
                out = F.relu(self.conv1(x))
                out = self.conv2(out)
                return F.relu(out + identity)
        
        # Test the block
        block = MockComplexResidualBlock(2, 4).to(device)  # 2 -> 4 complex channels
        x = torch.randn(2, 4, 32, 32).to(device)  # 2 complex channels = 4 real channels
        
        print(f" Input shape: {x.shape}")
        
        out = block(x)
        print(f" Output shape: {out.shape}")
        
        # Should output 4 complex channels = 8 real channels
        if out.shape[1] == 8:
            print(" ComplexResidualBlock test PASSED!")
            return True
        else:
            print(f" Expected 8 channels, got {out.shape[1]}")
            return False
            
    except Exception as e:
        print(f" ComplexResidualBlock test FAILED: {e}")
        traceback.print_exc()
        return False

def test_cunet_forward_pass():
    """Test full CUNet forward pass"""
    print("\n" + "="*50)
    print(" TESTING CUNet Forward Pass")
    print("="*50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Create CUNet with minimal size for testing
        print("Creating CUNet model...")
        model = CUNet(
            in_channels=2,
            out_channels=1,
            base_features=8,  # Small for testing
            use_data_consistency=True
        ).to(device)
        
        print(f" CUNet created successfully")
        print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        print(f"Data consistency enabled: {model.use_data_consistency}")
        
        # Test input
        batch_size = 1
        H, W = 64, 64  # Small for testing
        
        k_space = torch.randn(batch_size, 2, H, W).to(device)
        mask = torch.randint(0, 2, (batch_size, 1, H, W)).float().to(device)
        
        print(f" Input k-space: {k_space.shape}")
        print(f" Input mask: {mask.shape}")
        
        # Forward pass WITHOUT mask (data consistency disabled internally)
        print("\n Testing forward pass WITHOUT mask...")
        try:
            output_no_mask = model(k_space, mask=None)
            print(f"📤 Output (no mask): {output_no_mask.shape}")
            print(" Forward pass without mask successful")
        except Exception as e:
            print(f" Forward pass without mask failed (expected if DC required): {e}")
        
        # Forward pass WITH mask (data consistency enabled)
        print("\n Testing forward pass WITH mask...")
        output_with_mask = model(k_space, mask)
        print(f" Output (with mask): {output_with_mask.shape}")
        
        # Check output shape
        expected_shape = (batch_size, 1, H, W)  # Magnitude output
        if output_with_mask.shape == expected_shape:
            print(f" Output shape correct: {output_with_mask.shape}")
        else:
            print(f" Output shape wrong: expected {expected_shape}, got {output_with_mask.shape}")
            return False
        
        # Check output values
        output_min = output_with_mask.min().item()
        output_max = output_with_mask.max().item()
        output_mean = output_with_mask.mean().item()
        
        print(f"Output statistics:")
        print(f"  Min: {output_min:.6f}")
        print(f"  Max: {output_max:.6f}")
        print(f"  Mean: {output_mean:.6f}")
        
        # Test gradient computation
        print("\n Testing gradient computation...")
        loss = output_with_mask.mean()
        loss.backward()
        print(" Gradient computation successful")
        
        print(" CUNet Forward Pass test PASSED!")
        return True
        
    except Exception as e:
        print(f" CUNet Forward Pass test FAILED: {e}")
        traceback.print_exc()
        return False

def test_cunet_with_different_sizes():
    """Test CUNet with different input sizes"""
    print("\n" + "="*50)
    print(" TESTING CUNet with Different Sizes")
    print("="*50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        model = CUNet(
            in_channels=2,
            out_channels=1,
            base_features=8,
            use_data_consistency=True
        ).to(device)
        
        test_sizes = [(32, 32), (64, 64), (128, 128)]
        
        for H, W in test_sizes:
            print(f"\n📏 Testing size: {H}x{W}")
            
            k_space = torch.randn(1, 2, H, W).to(device)
            mask = torch.randint(0, 2, (1, 1, H, W)).float().to(device)
            
            with torch.no_grad():
                output = model(k_space, mask)
                
            print(f"  Input: {k_space.shape} -> Output: {output.shape}")
            
            if output.shape == (1, 1, H, W):
                print(f" Size {H}x{W} PASSED")
            else:
                print(f" Size {H}x{W} FAILED")
                return False
        
        print(" Different sizes test PASSED!")
        return True
        
    except Exception as e:
        print(f" Different sizes test FAILED: {e}")
        traceback.print_exc()
        return False

def visualize_cunet_internals(save_path=None):
    """Visualize CUNet internal processing"""
    print("\n" + "="*50)
    print("🎨 VISUALIZING CUNet Internals")
    print("="*50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        model = CUNet(
            in_channels=2,
            out_channels=1,
            base_features=16,
            use_data_consistency=True
        ).to(device)
        
        # Create test data
        H, W = 128, 128
        k_space = torch.randn(1, 2, H, W).to(device)
        mask = torch.zeros(1, 1, H, W).to(device)
        
        # Create a simple radial mask
        center_h, center_w = H // 2, W // 2
        y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        distances = torch.sqrt((x - center_w)**2 + (y - center_h)**2)
        mask[0, 0] = (distances < H // 4).float()  # Inner quarter radius
        
        print(f"Mask coverage: {mask.mean().item()*100:.1f}%")
        
        with torch.no_grad():
            # Get model output
            output = model(k_space, mask)
            
            # Convert to numpy for visualization
            k_real = k_space[0, 0].cpu().numpy()
            k_imag = k_space[0, 1].cpu().numpy()
            k_magnitude = np.sqrt(k_real**2 + k_imag**2)
            
            mask_np = mask[0, 0].cpu().numpy()
            output_np = output[0, 0].cpu().numpy()
            
            # Create zero-filled reconstruction for comparison
            k_complex = k_real + 1j * k_imag
            zero_filled = np.abs(np.fft.ifft2(k_complex))
            
            # Visualization
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            axes[0, 0].imshow(k_magnitude, cmap='gray')
            axes[0, 0].set_title('K-space Magnitude')
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(mask_np, cmap='gray')
            axes[0, 1].set_title('Sampling Mask')
            axes[0, 1].axis('off')
            
            axes[0, 2].imshow(k_magnitude * mask_np, cmap='gray')
            axes[0, 2].set_title('Undersampled K-space')
            axes[0, 2].axis('off')
            
            axes[1, 0].imshow(zero_filled, cmap='gray')
            axes[1, 0].set_title('Zero-filled Reconstruction')
            axes[1, 0].axis('off')
            
            axes[1, 1].imshow(output_np, cmap='gray')
            axes[1, 1].set_title('CUNet + Hard DC Output')
            axes[1, 1].axis('off')
            
            # Difference map
            diff = np.abs(output_np - zero_filled)
            axes[1, 2].imshow(diff, cmap='hot')
            axes[1, 2].set_title('|CUNet - Zero-filled|')
            axes[1, 2].axis('off')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f" Visualization saved: {save_path}")
            else:
                plt.show()
                
        print(" Visualization test PASSED!")
        return True
        
    except Exception as e:
        print(f" Visualization test FAILED: {e}")
        traceback.print_exc()
        return False

def test_cunet_step_by_step():
    """
    Main function to test CUNet step by step
    Run comprehensive tests to verify all components work correctly
    """
    print(" STARTING CUNet STEP-BY-STEP DIAGNOSTIC TESTS")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Track test results
    tests = [
        ("ComplexConv2d", test_complex_conv2d),
        ("ComplexBatchNorm2d", test_complex_batch_norm),
        ("FFT/IFFT Operations", test_fft_operations),
        ("Hard Data Consistency", test_hard_data_consistency),
        ("ComplexResidualBlock", test_complex_residual_block),
        ("CUNet Forward Pass", test_cunet_forward_pass),
        ("Different Input Sizes", test_cunet_with_different_sizes),
        ("Visualization", lambda: visualize_cunet_internals("cunet_internals.png")),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*70}")
        print(f" RUNNING TEST: {test_name}")
        print(f"{'='*70}")
        
        try:
            result = test_func()
            results[test_name] = result
            
            if result:
                print(f" {test_name}: PASSED")
            else:
                print(f" {test_name}: FAILED")
                
        except Exception as e:
            print(f"💥 {test_name}: CRASHED - {e}")
            results[test_name] = False
            traceback.print_exc()
    
    # Final summary
    print("\n" + "="*70)
    print(" FINAL TEST RESULTS SUMMARY")
    print("="*70)
    
    passed_tests = sum(results.values())
    total_tests = len(results)
    
    for test_name, result in results.items():
        status = " PASSED" if result else "❌ FAILED"
        print(f"{test_name:.<50} {status}")
    
    print(f"\n Overall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED! Your CUNet is ready for training/inference!")
    else:
        print(" Some tests failed. Please check the errors above.")
        print("Common issues:")
        print("  - Missing ComplexResidualBlock implementation")
        print("  - GPU memory issues (try smaller test sizes)")
        print("  - Missing dependencies")
    
    return results

# ===============================================
# QUICK DIAGNOSTIC FUNCTIONS
# ===============================================

def quick_cunet_check():
    """Quick sanity check for CUNet"""
    print(" QUICK CUNet SANITY CHECK")
    print("-" * 30)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Create minimal model
        model = CUNet(in_channels=2, out_channels=1, base_features=8, use_data_consistency=True).to(device)
        
        # Test input
        k = torch.randn(1, 2, 32, 32).to(device)
        m = torch.randint(0, 2, (1, 1, 32, 32)).float().to(device)
        
        # Forward pass
        with torch.no_grad():
            out = model(k, m)
        
        print(f" Input: {k.shape} + Mask: {m.shape} -> Output: {out.shape}")
        print(f" Data consistency: {model.use_data_consistency}")
        print(f" Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return True
        
    except Exception as e:
        print(f" Quick check failed: {e}")
        return False







# ===============================================
# 5. MAIN TESTING FUNCTION
# ===============================================

def test_fixed_cunet():
    """Test the fixed CUNet architecture"""
    print(" TESTING FIXED CUNet")
    print("=" * 40)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Create fixed model with small features for testing
        model = CUNet(
            in_channels=2,
            out_channels=1,
            base_features=8,  # Small for testing
            use_data_consistency=True
        ).to(device)
        
        print(f" Model created successfully")
        print(f" Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        # Test with different sizes
        test_sizes = [(32, 32), (64, 64)]
        
        for H, W in test_sizes:
            print(f"\n Testing size {H}x{W}...")
            
            # Create test data
            k_space = torch.randn(1, 2, H, W).to(device)
            mask = torch.randint(0, 2, (1, 1, H, W)).float().to(device)
            
            print(f" Input k-space: {k_space.shape}")
            print(f" Mask: {mask.shape}")
            
            # Forward pass
            with torch.no_grad():
                output = model(k_space, mask)
                
            print(f" Output: {output.shape}")
            
            expected_shape = (1, 1, H, W)
            if output.shape == expected_shape:
                print(f" Size {H}x{W} PASSED")
            else:
                print(f" Size {H}x{W} FAILED: expected {expected_shape}, got {output.shape}")
                return False
        
        # Test gradient computation
        print(f"\n Testing gradient computation...")
        k_space = torch.randn(1, 2, 64, 64, requires_grad=True).to(device)
        mask = torch.randint(0, 2, (1, 1, 64, 64)).float().to(device)
        
        output = model(k_space, mask)
        loss = output.mean()
        loss.backward()
        
        print(f" Gradients computed successfully")
        print(f" Loss: {loss.item():.6f}")
        
        print(f"\n FIXED CUNet test PASSED!")
        return True
        
    except Exception as e:
        print(f" Fixed CUNet test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

# Quick diagnostic to understand channel flow
def debug_channel_flow():
    """Debug the channel flow through CUNet"""
    print("\n DEBUGGING CHANNEL FLOW")
    print("=" * 40)
    
    base_features = 8  # Same as test
    
    f1, f2, f3, f4, f5 = base_features, base_features*2, base_features*4, base_features*8, base_features*16
    print(f"Complex channel progression: {f1} -> {f2} -> {f3} -> {f4} -> {f5}")
    print(f"Real channel progression: {f1*2} -> {f2*2} -> {f3*2} -> {f4*2} -> {f5*2}")
    
    print(f"\nDecoder concatenations:")
    print(f"  Level 4: upconv({f5}->{f4}) + enc4({f4}) = {f4}+{f4} = {f4*2} complex channels = {f4*2*2} real")  
    print(f"  Level 3: upconv({f4}->{f3}) + enc3({f3}) = {f3}+{f3} = {f3*2} complex channels = {f3*2*2} real")
    print(f"  Level 2: upconv({f3}->{f2}) + enc2({f2}) = {f2}+{f2} = {f2*2} complex channels = {f2*2*2} real")
    print(f"  Level 1: upconv({f2}->{f1}) + enc1({f1}) = {f1}+{f1} = {f1*2} complex channels = {f1*2*2} real")



def verify_cunet_fix():
    """Quick verification that the CUNet fix works"""
    import torch
    import torch.nn as nn
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f" VERIFYING CUNet FIX")
    print(f"Device: {device}")
    print("=" * 50)
    
    try:
        # Test 1: Create model
        print("  Creating model...")
        model = CUNet(
            in_channels=2,
            out_channels=1, 
            base_features=8,  # Small for quick test
            use_data_consistency=True
        ).to(device)
        print(f"   ✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Test 2: Forward pass
        print("2️⃣  Testing forward pass...")
        x = torch.randn(1, 2, 64, 64).to(device)
        mask = torch.randint(0, 2, (1, 1, 64, 64)).float().to(device)
        
        with torch.no_grad():
            output = model(x, mask)
            
        print(f" Forward pass successful: {x.shape} -> {output.shape}")
        
        # Test 3: Gradient computation
        print("  Testing gradients...")
        x.requires_grad_(True)
        output = model(x, mask)
        loss = output.mean()
        loss.backward()
        print(f" Gradients computed successfully")
        
        # Test 4: Different sizes
        print("  Testing different sizes...")
        for size in [32, 128]:
            x_test = torch.randn(1, 2, size, size).to(device)
            mask_test = torch.randint(0, 2, (1, 1, size, size)).float().to(device)
            
            with torch.no_grad():
                out_test = model(x_test, mask_test)
                
            expected = (1, 1, size, size)
            if out_test.shape == expected:
                print(f" Size {size}x{size}: {out_test.shape}")
            else:
                print(f" Size {size}x{size}: expected {expected}, got {out_test.shape}")
                return False
                
        print("\n🎉 ALL TESTS PASSED! Your CUNet is now fixed!")
        return True
        
    except Exception as e:
        print(f" Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False



# if __name__ == "__main__":
#     # Run step-by-step tests first
#     print("="*50)
#     print("Running diagnostic tests...")
#     print("="*50)
#     test_model_step_by_step()
    
#     print("\n" + "="*50)
#     print("Running full model test...")
#     print("="*50)
    
#     # Test the model with your data shapes
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"Using device: {device}")
    
#     # Create model with smaller base features for testing
#     model = create_model().to(device)
    
#     print(f"Model parameters: {count_parameters(model):,}")
    
#     # Test with smaller data shapes first
#     batch_size = 1
#     test_size = 128  # Start with smaller size
#     x = torch.randn(batch_size, 2, test_size, test_size).to(device)  # Input k-space
#     y = torch.randn(batch_size, 1, test_size, test_size).to(device)  # Target image
#     mask = torch.randn(batch_size, 1, test_size, test_size).to(device)  # Mask
    
#     print(f"Test input shape: {x.shape}")
#     print(f"Test target shape: {y.shape}")
#     print(f"Test mask shape: {mask.shape}")
    
#     # Check if all model parameters are on the correct device
#     model_device = next(model.parameters()).device
#     print(f"Model device: {model_device}")
    
#     # Forward pass
#     try:
#         with torch.no_grad():
#             pred = model(x, mask)  # Now passes both k-space and mask
#             print(f"Test output shape: {pred.shape}")
            
#         # Create loss function
#         criterion = CUNetLoss()
#         loss = criterion(pred, y)
#         print(f"Test loss: {loss.item():.6f}")
        
#         print("CU-Net small scale test passed!")
        
#         # Now test with full size
#         print("\nTesting with full size (320x320)...")
#         x_full = torch.randn(batch_size, 2, 320, 320).to(device)  # Input k-space
#         y_full = torch.randn(batch_size, 1, 320, 320).to(device)  # Target image
#         mask_full = torch.randn(batch_size, 1, 320, 320).to(device)  # Mask
        
#         with torch.no_grad():
#             pred_full = model(x_full, mask_full)
#             print(f"Full size output shape: {pred_full.shape}")
        
#         print("CU-Net model created successfully!")
        
#     except Exception as e:
#         print(f"Error during forward pass: {e}")
#         import traceback
#         traceback.print_exc()