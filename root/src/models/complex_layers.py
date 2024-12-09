import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplexBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(ComplexBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.weight_real = nn.Parameter(torch.Tensor(num_features))
            self.weight_imag = nn.Parameter(torch.Tensor(num_features))
            self.bias_real = nn.Parameter(torch.Tensor(num_features))
            self.bias_imag = nn.Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight_real', None)
            self.register_parameter('weight_imag', None)
            self.register_parameter('bias_real', None)
            self.register_parameter('bias_imag', None)

        if self.track_running_stats:
            self.register_buffer('running_mean_real', torch.zeros(num_features))
            self.register_buffer('running_mean_imag', torch.zeros(num_features))
            self.register_buffer('running_covar', torch.zeros(num_features, 2, 2))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean_real', None)
            self.register_parameter('running_mean_imag', None)
            self.register_parameter('running_covar', None)
            self.register_parameter('num_batches_tracked', None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            nn.init.ones_(self.weight_real)
            nn.init.zeros_(self.weight_imag)
            nn.init.zeros_(self.bias_real)
            nn.init.zeros_(self.bias_imag)

    def forward(self, input):
        if not torch.is_complex(input):
            raise ValueError("Input must be a complex tensor")

        input_shape = input.shape
        input_real = input.real.contiguous().view(input.size(0), self.num_features, -1)
        input_imag = input.imag.contiguous().view(input.size(0), self.num_features, -1)

        if self.training or not self.track_running_stats:
            mean_real = input_real.mean(dim=[0, 2])
            mean_imag = input_imag.mean(dim=[0, 2])
            input_centered_real = input_real - mean_real.unsqueeze(1)
            input_centered_imag = input_imag - mean_imag.unsqueeze(1)

            covar_rr = (input_centered_real * input_centered_real).mean(dim=[0, 2])
            covar_ii = (input_centered_imag * input_centered_imag).mean(dim=[0, 2])
            covar_ri = (input_centered_real * input_centered_imag).mean(dim=[0, 2])

            if self.training and self.track_running_stats:
                with torch.no_grad():
                    self.running_mean_real = self.momentum * mean_real + (1 - self.momentum) * self.running_mean_real
                    self.running_mean_imag = self.momentum * mean_imag + (1 - self.momentum) * self.running_mean_imag
                    self.running_covar[:, 0, 0] = self.momentum * covar_rr + (1 - self.momentum) * self.running_covar[:, 0, 0]
                    self.running_covar[:, 1, 1] = self.momentum * covar_ii + (1 - self.momentum) * self.running_covar[:, 1, 1]
                    self.running_covar[:, 0, 1] = self.momentum * covar_ri + (1 - self.momentum) * self.running_covar[:, 0, 1]
                    self.running_covar[:, 1, 0] = self.running_covar[:, 0, 1]
                    self.num_batches_tracked += 1
        else:
            mean_real = self.running_mean_real
            mean_imag = self.running_mean_imag
            covar_rr = self.running_covar[:, 0, 0]
            covar_ii = self.running_covar[:, 1, 1]
            covar_ri = self.running_covar[:, 0, 1]

        # Perform whitening
        det = covar_rr * covar_ii - covar_ri * covar_ri
        s = torch.sqrt(det.clamp(min=self.eps))
        t = torch.sqrt(covar_ii.clamp(min=self.eps) + covar_rr.clamp(min=self.eps) + 2 * s)
        inverse_st = 1.0 / (s * t)
        Vrr = (covar_ii + s) * inverse_st
        Vii = (covar_rr + s) * inverse_st
        Vri = -covar_ri * inverse_st

        input_centered_real = input_real - mean_real.unsqueeze(1)
        input_centered_imag = input_imag - mean_imag.unsqueeze(1)

        out_real = Vrr.unsqueeze(1) * input_centered_real + Vri.unsqueeze(1) * input_centered_imag
        out_imag = Vri.unsqueeze(1) * input_centered_real + Vii.unsqueeze(1) * input_centered_imag

        if self.affine:
            out_real = self.weight_real.unsqueeze(1) * out_real + self.bias_real.unsqueeze(1)
            out_imag = self.weight_imag.unsqueeze(1) * out_imag + self.bias_imag.unsqueeze(1)

        return torch.complex(out_real, out_imag).view(input_shape)

class ComplexConv2d(nn.Module):
    """
    Implements 2D convolution for complex-valued tensors.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(ComplexConv2d, self).__init__()
        self.conv_r = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv_i = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input):
        if not torch.is_complex(input):
            raise ValueError(f"Expected complex input, got {input.dtype}")
        # print(f"ComplexConv2d input shape: {input.shape}")
        return torch.complex(
                self.conv_r(input.real) - self.conv_i(input.imag),
                self.conv_r(input.imag) + self.conv_i(input.real)
            )

class ComplexReLU(nn.Module):
    
    """
    Implements ReLU activation for complex-valued tensors.
    Applies ReLU to both real and imaginary parts independently.
    """
    def forward(self, x):
        return torch.complex(F.relu(x.real), F.relu(x.imag))

class ComplexUpsample(nn.Module):
    """
    Implements upsampling for complex-valued tensors by zero-padding high-frequency space.
    """
    def __init__(self, scale_factor=2):
        super(ComplexUpsample, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        B, C, H, W = x.shape
        new_H, new_W = H * self.scale_factor, W * self.scale_factor
        
        # Zero-pad the high-frequency space
        padded = torch.zeros(B, C, new_H, new_W, dtype=x.dtype, device=x.device)
        padded[:, :, :H, :W] = x
        
        return padded

class ComplexDownsample(nn.Module):
    """
    Implements downsampling for complex-valued tensors in frequency space.
    """
    def __init__(self, scale_factor=2):
        super(ComplexDownsample, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        B, C, H, W = x.shape
        new_H, new_W = H // self.scale_factor, W // self.scale_factor
        
        # Truncate high-frequency components
        return x[:, :, :new_H, :new_W]

class ComplexConvTranspose2d(nn.Module):
    """
    Implements transposed convolution for complex-valued tensors.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, bias=True):
        super(ComplexConvTranspose2d, self).__init__()
        # Define conv_r and conv_i explicitly
        self.conv_r = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=bias)
        self.conv_i = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=bias)

    def forward(self, x):
        if not torch.is_complex(x):
            raise ValueError(f"Expected complex input, got {x.dtype}")

        real = self.conv_r(x.real) - self.conv_i(x.imag)
        imag = self.conv_r(x.imag) + self.conv_i(x.real)
        return torch.complex(real, imag)
    
class ComplexResidualBlock(nn.Module):
    """
    Implements a residual block for complex-valued inputs with identity mapping.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ComplexResidualBlock, self).__init__()

        # Main path layers
        self.conv1 = ComplexConv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = ComplexBatchNorm2d(out_channels)
        self.relu = ComplexReLU()

        self.conv2 = ComplexConv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = ComplexBatchNorm2d(out_channels)

        # Shortcut path to match dimensions
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                ComplexConv2d(in_channels, out_channels, kernel_size=1, padding=0),
                ComplexBatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        # Main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Add residual (shortcut)
        out += self.shortcut(x)
        out = self.relu(out)
        
        return out

class ComplexFrequencyPooling(nn.Module):
    """
    Complex Frequency Pooling for MRI k-space data.
    This module keeps the low-frequency components of k-space data and removes the high-frequency components.
    """
    def __init__(self, reduction_factor=2):
        super(ComplexFrequencyPooling, self).__init__()
        self.reduction_factor = reduction_factor

    def forward(self, x):
        # Apply 2D FFT
        x_freq = torch.fft.fft2(x)
        B, C, H, W = x_freq.shape

        # Determine cropping dimensions
        new_H = H // self.reduction_factor
        new_W = W // self.reduction_factor
        h_start = (H - new_H) // 2
        w_start = (W - new_W) // 2

        # Crop low-frequency components
        low_freq = x_freq[:, :, h_start:h_start + new_H, w_start:w_start + new_W]

        return low_freq



    

class ComplexToReal(nn.Module):
    """
    Converts complex-valued tensors to real-valued tensors by taking the magnitude.
    """
    def forward(self, x):
        return torch.abs(x)  # Magnitude of complex tensor



class ComplexAttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(ComplexAttentionBlock, self).__init__()
        
        # Ensure both inputs have the same channel size
        self.align_g = ComplexConv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True)
        self.align_x = ComplexConv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True)

        self.W_g = ComplexConv2d(F_int, F_int, kernel_size=1, stride=1, padding=0, bias=True)
        self.W_x = ComplexConv2d(F_int, F_int, kernel_size=1, stride=1, padding=0, bias=True)
        self.psi = ComplexConv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True)

        self.relu = ComplexReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, g, x):
        print(f"Before alignment: g shape: {g.shape}, x shape: {x.shape}")
        
        g = self.align_g(g)
        x = self.align_x(x)
        
        print(f"After alignment: g shape: {g.shape}, x shape: {x.shape}")

        g1 = self.W_g(g)
        x1 = self.W_x(x)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        attention = self.sigmoid(torch.abs(psi))

        return x * attention.expand_as(x)
    
    
    
    
class ComplexResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ComplexResidualBlock, self).__init__()

        self.conv1 = ComplexConv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = ComplexBatchNorm2d(out_channels)
        self.relu = ComplexReLU()

        self.conv2 = ComplexConv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = ComplexBatchNorm2d(out_channels)
        
        # Fix the shortcut path
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                ComplexConv2d(in_channels, out_channels, kernel_size=1, padding=0),
                ComplexBatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        #print(f"ComplexResidualBlock input shape: {x.shape}, type: {x.dtype}")

        residual = self.shortcut(x)
        #print(f"Residual shape after shortcut: {residual.shape}")

        out = self.relu(self.bn1(self.conv1(x)))
        #print(f"Main path after first conv: {out.shape}")

        out = self.bn2(self.conv2(out))
        #print(f"Main path after second conv: {out.shape}")

        final_output = self.relu(out + residual)
        #print(f"Final output shape: {final_output.shape}")

        return final_output

    
    
class ComplexAvgPool2d(nn.Module):
    """
    Implements average pooling for complex-valued tensors.
    """
    def __init__(self, kernel_size, stride=None, padding=0):
        super(ComplexAvgPool2d, self).__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size, stride, padding)

    def forward(self, x):
        return torch.complex(self.avg_pool(x.real), self.avg_pool(x.imag))
    
    
    
class ComplexDropout(nn.Module):
    """
    Implements dropout for complex-valued tensors.
    """
    def __init__(self, p=0.5):
        super(ComplexDropout, self).__init__()
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        mask = self.dropout(torch.ones_like(x.real))
        return x * mask
    
    
    
class ComplexAdaptiveAvgPool2d(nn.Module):
    """
    Implements adaptive average pooling for complex-valued tensors.
    """
    def __init__(self, output_size):
        super(ComplexAdaptiveAvgPool2d, self).__init__()
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d(output_size)

    def forward(self, x):
        return torch.complex(self.adaptive_avg_pool(x.real), self.adaptive_avg_pool(x.imag))
