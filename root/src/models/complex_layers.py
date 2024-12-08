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
        print(f"ComplexConv2d input shape: {input.shape}")
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
    
    
    
    
class FrequencyPooling(nn.Module):
    """
    Frequency-specific pooling for high and low frequencies.
    Implements a low-pass filter by conserving low-frequency data and truncating high-frequency data.
    """
    def __init__(self, mode='low'):
        super(FrequencyPooling, self).__init__()
        self.mode = mode

    def forward(self, x):
        B, C, H, W = x.shape
        h_mid, w_mid = H // 2, W // 2
        
        if self.mode == 'low':
            # Low-frequency: Center region
            return x[:, :, h_mid-H//4:h_mid+H//4, w_mid-W//4:w_mid+W//4]
        elif self.mode == 'high':
            # High-frequency: Corners
            high_freq = torch.zeros_like(x)
            high_freq[:, :, :h_mid//2, :w_mid//2] = x[:, :, :h_mid//2, :w_mid//2]
            high_freq[:, :, :h_mid//2, -w_mid//2:] = x[:, :, :h_mid//2, -w_mid//2:]
            high_freq[:, :, -h_mid//2:, :w_mid//2] = x[:, :, -h_mid//2:, :w_mid//2]
            high_freq[:, :, -h_mid//2:, -w_mid//2:] = x[:, :, -h_mid//2:, -w_mid//2:]
            return high_freq
        else:
            raise ValueError("Mode must be 'low' or 'high'")

class ComplexMaxPool2d(nn.Module):
    """
    Implements max pooling for complex-valued tensors in frequency space.
    """
    def __init__(self, kernel_size, stride=None, padding=0):
        super(ComplexMaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.freq_pool = FrequencyPooling(mode='low')

    def forward(self, x):
        # Convert to frequency domain
        x_freq = torch.fft.fft2(x)
        
        # Apply frequency pooling
        x_freq_pooled = self.freq_pool(x_freq)
        
        # Convert back to spatial domain
        x_spatial = torch.fft.ifft2(x_freq_pooled)
        
        # Apply traditional max pooling to magnitude
        x_mag = torch.abs(x_spatial)
        pooled_mag = F.max_pool2d(x_mag, self.kernel_size, self.stride, self.padding)
        
        # Preserve phase information
        phase = torch.angle(x_spatial)
        pooled_phase = F.interpolate(phase, size=pooled_mag.shape[-2:], mode='nearest')
        
        # Combine magnitude and phase
        return pooled_mag * torch.exp(1j * pooled_phase)
    
    
    

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

        # Main convolutional path
        self.conv1 = ComplexConv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = ComplexBatchNorm2d(out_channels)
        self.relu = ComplexReLU()

        self.conv2 = ComplexConv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = ComplexBatchNorm2d(out_channels)

        # Attention mechanism
        self.attention = ComplexAttentionBlock(out_channels, out_channels, out_channels // 2)

        # Create shortcut **only when in_channels != out_channels**
        if in_channels != out_channels:
            self.shortcut = ComplexConv2d(in_channels, out_channels, kernel_size=1, padding=0)
        else:
            self.shortcut = nn.Identity()

    def align_channels(self, tensor, target_shape):
        """
        Custom implementation for aligning channels using zero-padding or slicing.
        """
        input_channels = tensor.shape[1]
        target_channels = target_shape[1]

        if input_channels == target_channels:
            return tensor

        if input_channels > target_channels:
            # Slice the input
            return tensor[:, :target_channels, :, :]

        # Pad if input has fewer channels
        padding = torch.zeros(
            tensor.shape[0], 
            target_channels - input_channels, 
            tensor.shape[2], 
            tensor.shape[3], 
            dtype=tensor.dtype, 
            device=tensor.device
        )
        return torch.cat([tensor, padding], dim=1)

    def forward(self, x):
        print(f"ComplexResidualBlock input shape: {x.shape}, type: {x.dtype}")

        # Apply shortcut projection only when necessary
        residual = self.shortcut(x)
        print(f"Residual shape after shortcut: {residual.shape}")

        # Main path
        out = self.relu(self.bn1(self.conv1(x)))
        print(f"Main path after first conv: {out.shape}")

        out = self.bn2(self.conv2(out))
        print(f"Main path after second conv: {out.shape}")

        # Apply attention
        out = self.attention(out, out)
        print(f"After attention: {out.shape}")

        # Align Channels
        residual = self.align_channels(residual, out.shape)
        print(f"Aligned residual shape: {residual.shape}")

        # Add and activate
        final_output = self.relu(out + residual)
        print(f"Final output shape: {final_output.shape}")

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