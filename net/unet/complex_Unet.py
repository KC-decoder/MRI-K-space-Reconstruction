import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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

class ComplexUpsample(nn.Module):
    """Complex Upsampling - zero-padding in frequency domain"""
    def __init__(self, scale_factor=2):
        super(ComplexUpsample, self).__init__()
        self.scale_factor = scale_factor
        
    def forward(self, x):
        batch_size, total_channels, H, W = x.shape
        channels = total_channels // 2
        
        real = x[:, :channels, :, :]
        imag = x[:, channels:, :, :]
        
        # Upsample using interpolation
        real_out = F.interpolate(real, scale_factor=self.scale_factor, mode='bilinear', align_corners=True)
        imag_out = F.interpolate(imag, scale_factor=self.scale_factor, mode='bilinear', align_corners=True)
        
        return torch.cat([real_out, imag_out], dim=1)

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

class CUNet(nn.Module):
    """Complete Complex U-Net for K-space MRI Reconstruction"""
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
        
        # Decoder
        self.upconv4 = ComplexUpsample(2)
        self.decoder4 = ComplexResidualBlock(f5 + f4, f4)
        
        self.upconv3 = ComplexUpsample(2)
        self.decoder3 = ComplexResidualBlock(f4 + f3, f3)
        
        self.upconv2 = ComplexUpsample(2)
        self.decoder2 = ComplexResidualBlock(f3 + f2, f2)
        
        self.upconv1 = ComplexUpsample(2)
        self.decoder1 = ComplexResidualBlock(f2 + f1, f1)
        
        # Output layer - converts back to 1 complex channel
        self.final_conv = ComplexConv2d(f1, in_complex_channels, 1)
        
    def forward(self, x, mask=None):
        # x shape: (batch, 2, H, W) - complex k-space
        # mask shape: (batch, 1, H, W) - optional mask (not used in forward pass for now)
        
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
        
        # Decoder path with skip connections
        up4 = self.upconv4(bottleneck)
        dec4 = self.decoder4(torch.cat([up4, enc4], dim=1))
        
        up3 = self.upconv3(dec4)
        dec3 = self.decoder3(torch.cat([up3, enc3], dim=1))
        
        up2 = self.upconv2(dec3)
        dec2 = self.decoder2(torch.cat([up2, enc2], dim=1))
        
        up1 = self.upconv1(dec2)
        dec1 = self.decoder1(torch.cat([up1, enc1], dim=1))
        
        # Final output - complex k-space
        output_kspace = self.final_conv(dec1)
        
        # Convert to image domain
        output_complex = self.kspace_to_image(output_kspace)
        
        # Convert complex image to magnitude
        output_magnitude = self.complex_to_magnitude(output_complex)
        
        # Optional data consistency
        if self.use_data_consistency and mask is not None:
            output_magnitude = self.apply_data_consistency(x, output_magnitude, mask)
        
        return output_magnitude
    
    def apply_data_consistency(self, input_kspace, pred_image, mask):
        """Apply data consistency in k-space"""
        # Convert predicted image back to k-space
        pred_kspace = self.image_to_kspace(pred_image)
        
        # Apply data consistency: keep original k-space values where mask=1
        # mask shape: (batch, 1, H, W), need to expand for complex channels (real + imaginary)
        mask_expanded = mask.repeat(1, 2, 1, 1)  # (batch, 2, H, W)
        
        # Data consistency: use original k-space where sampled, predicted where not
        consistent_kspace = mask_expanded * input_kspace + (1 - mask_expanded) * pred_kspace
        
        # Convert back to image
        consistent_image = self.kspace_to_image(consistent_kspace)
        consistent_magnitude = self.complex_to_magnitude(consistent_image)
        
        return consistent_magnitude
    
    def image_to_kspace(self, magnitude_img):
        """Convert magnitude image to k-space (approximate)"""
        # Note: This is an approximation since we've lost phase information
        # In practice, you might want to keep the complex image for better data consistency
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

# Example usage and training setup
def create_model(use_data_consistency=False, base_features=16):  # Smaller default for testing
    """Create CU-Net model"""
    model = CUNet(in_channels=2, out_channels=1, base_features=base_features, use_data_consistency=use_data_consistency)
    return model

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

# Simple test function to debug device issues
def test_complex_conv():
    """Test ComplexConv2d layer independently"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing ComplexConv2d on device: {device}")
    
    # Create a simple complex conv layer: 1 complex input channel -> 32 complex output channels
    conv = ComplexConv2d(1, 32, 3, padding=1).to(device)
    
    # Test input: 1 complex channel = 2 real channels
    x = torch.randn(1, 2, 32, 32).to(device)
    print(f"Input shape: {x.shape}, device: {x.device}")
    
    # Check conv layer devices
    print(f"conv_rr device: {next(conv.conv_rr.parameters()).device}")
    print(f"conv_ri device: {next(conv.conv_ri.parameters()).device}")
    print(f"conv_ir device: {next(conv.conv_ir.parameters()).device}")
    print(f"conv_ii device: {next(conv.conv_ii.parameters()).device}")
    
    try:
        out = conv(x)
        print(f"Output shape: {out.shape}, device: {out.device}")
        expected_channels = 2 * 32  # 32 complex channels = 64 real channels
        if out.shape[1] == expected_channels:
            print("ComplexConv2d test passed!")
            return True
        else:
            print(f"Wrong output channels: expected {expected_channels}, got {out.shape[1]}")
            return False
    except Exception as e:
        print(f"ComplexConv2d test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_step_by_step():
    """Test model step by step to isolate issues"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing model step by step on device: {device}")
    
    # Test ComplexConv2d first
    if not test_complex_conv():
        return
    
    # Create minimal model with smaller base features
    try:
        model = CUNet(in_channels=2, out_channels=1, base_features=8).to(device)  # Smaller base features
        print("Model created and moved to device successfully")
        
        # Test forward pass with smaller input size
        x = torch.randn(1, 2, 64, 64).to(device)  # 1 complex channel (2 real channels)
        print(f"Test input shape: {x.shape}, device: {x.device}")
        
        with torch.no_grad():
            out = model(x)
            print(f"Test output shape: {out.shape}, device: {out.device}")
            print("Minimal model test passed!")
            
    except Exception as e:
        print(f"Model test failed: {e}")
        import traceback
        traceback.print_exc()

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