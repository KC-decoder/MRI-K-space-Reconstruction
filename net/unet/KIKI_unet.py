import torch
import torch.nn as nn

import torch
import torch.nn as nn

# =========================
# Modern, device-safe utils
# =========================

def roll(x: torch.Tensor, shift: int, dim: int = -1) -> torch.Tensor:
    """Torch-native roll that works on any device."""
    return torch.roll(x, shifts=shift, dims=dim)

def fftshift(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Shift zero-frequency component to center along one dim."""
    # torch.fft.fftshift supports single dim
    return torch.fft.fftshift(x, dim=dim)

def fftshift2(x: torch.Tensor) -> torch.Tensor:
    """2D fftshift over the last two spatial dims (W, H) while keeping channels."""
    return torch.fft.fftshift(x, dim=(-1, -2))

# -------- internal helpers (keep your layout) --------                    # (N,H,W)

def _to_2ch(x_c: torch.Tensor) -> torch.Tensor:
    """
    (N, H, W) complex -> (N, 2, H, W) float
    """
    if not torch.is_complex(x_c):
        raise ValueError("Expected a complex tensor of shape (N,H,W)")
    x_last2 = torch.view_as_real(x_c)                       # (N,H,W,2)
    return x_last2.permute(0, 3, 1, 2).contiguous()         # (N,2,H,W)

# -------- 2D FFT/IFFT keeping your layout --------
def fft2(input_: torch.Tensor) -> torch.Tensor:
    """
    (N,2,H,W) -> (N,2,H,W) 2D FFT (default 'backward' norm, matching legacy torch.fft(..., 2))
    """
    x_c = _to_complex(input_)
    X_c = torch.fft.fft2(x_c)                               # (N,H,W) complex
    return _to_2ch(X_c)

def ifft2(input_: torch.Tensor) -> torch.Tensor:
    """
    (N,2,H,W) -> (N,2,H,W) 2D IFFT (default 'backward' norm, matching legacy)
    """
    X_c = _to_complex(input_)
    x_c = torch.fft.ifft2(X_c)                              # (N,H,W) complex
    return _to_2ch(x_c)

# -------- 1D FFT/IFFT along a spatial axis (to match your old API) --------
# In your legacy code: axis==1 ~ H (rows), axis==0 ~ W (cols)
def fft2(input_: torch.Tensor) -> torch.Tensor:
    """
    (N,C,H,W) with C in {1,2} -> (N,2,H,W).
    Upcasts to complex64 around FFT to avoid cuFFT limitations on complex-half.
    """
    real_dtype = input_.dtype  # e.g., float16 under AMP
    X_c = _to_complex(input_)                      # complex (maybe complex-half)
    X_c = X_c.to(torch.complex64)                  # upcast for cuFFT
    F_c = torch.fft.fft2(X_c)                      # complex64
    out = _to_2ch(F_c)                             # float32
    return out.to(real_dtype)                      # back to original real dtype

def ifft2(input_: torch.Tensor) -> torch.Tensor:
    real_dtype = input_.dtype
    X_c = _to_complex(input_).to(torch.complex64)
    x_c = torch.fft.ifft2(X_c)
    out = _to_2ch(x_c)
    return out.to(real_dtype)

def fft1(input_: torch.Tensor, axis: int) -> torch.Tensor:
    real_dtype = input_.dtype
    X_c = _to_complex(input_).to(torch.complex64)
    if axis == 1:   # along H
        F_c = torch.fft.fft(X_c, dim=1)
    elif axis == 0: # along W
        F_c = torch.fft.fft(X_c, dim=2)
    else:
        raise ValueError("axis must be 0 (W) or 1 (H)")
    out = _to_2ch(F_c)
    return out.to(real_dtype)

def ifft1(input_: torch.Tensor, axis: int) -> torch.Tensor:
    real_dtype = input_.dtype
    X_c = _to_complex(input_).to(torch.complex64)
    if axis == 1:
        x_c = torch.fft.ifft(X_c, dim=1)
    elif axis == 0:
        x_c = torch.fft.ifft(X_c, dim=2)
    else:
        raise ValueError("axis must be 0 (W) or 1 (H)")
    out = _to_2ch(x_c)
    return out.to(real_dtype)

# -------- Data Consistency (works for both image & k-space) --------
def _to_complex(x_ch: torch.Tensor) -> torch.Tensor:
    """
    Accepts (N,2,H,W) or (N,1,H,W).
    - (N,2,H,W): interpret as real,imag
    - (N,1,H,W): treat as purely real with imag=0
    Returns (N,H,W) complex.
    """
    if x_ch.dim() != 4:
        raise ValueError(f"Expected 4D (N,C,H,W), got {tuple(x_ch.shape)}")
    N, C, H, W = x_ch.shape
    if C == 2:
        x_last = x_ch.permute(0, 2, 3, 1).contiguous()  # (N,H,W,2)
    elif C == 1:
        zeros = torch.zeros_like(x_ch)
        x2 = torch.cat([x_ch, zeros], dim=1)            # (N,2,H,W)
        x_last = x2.permute(0, 2, 3, 1).contiguous()
    else:
        raise ValueError(f"Expected channels 1 or 2, got {C}")
    return torch.view_as_complex(x_last)                # (N,H,W)

def DataConsist(input_: torch.Tensor, k: torch.Tensor, m: torch.Tensor, is_k: bool = False) -> torch.Tensor:
    """
    input_: (N,2,H,W) or (N,1,H,W) prediction (image if is_k=False, k-space if is_k=True)
    k:      (N,2,H,W) or (N,1,H,W) measured k-space
    m:      (N,1,H,W) or (N,2,H,W) mask
    """
    # broadcast mask to 2 channels
    if m.dim() == 4 and m.size(1) == 1:
        m2 = m.repeat(1, 2, 1, 1)
    elif m.dim() == 4 and m.size(1) == 2:
        m2 = m
    else:
        raise ValueError(f"Mask must be (N,1,H,W) or (N,2,H,W), got {tuple(m.shape)}")

    # ensure k is 2-ch (pad imag=0 if needed)
    if k.size(1) == 1:
        k = torch.cat([k, torch.zeros_like(k)], dim=1)

    if is_k:
        # ensure input_ is 2-ch in k-space path
        if input_.size(1) == 1:
            input_ = torch.cat([input_, torch.zeros_like(input_)], dim=1)
        return input_ * m2 + k * (1 - m2)
    else:
        # image -> k, mix, -> image
        input_k = fft2(input_)  # accepts 1- or 2-ch
        if input_k.size(1) == 1:
            input_k = torch.cat([input_k, torch.zeros_like(input_k)], dim=1)
        mixed_k = input_k * m2 + k * (1 - m2)
        return ifft2(mixed_k)

def GenConvBlock(n_conv_layers, in_chan, out_chan, feature_maps):
    conv_block = [nn.Conv2d(in_chan, feature_maps, 3, 1, 1),
                  nn.LeakyReLU(negative_slope=0.1, inplace=True)]
    for _ in range(n_conv_layers - 2):
        conv_block += [nn.Conv2d(feature_maps, feature_maps, 3, 1, 1),
                       nn.LeakyReLU(negative_slope=0.1, inplace=True)]
    return nn.Sequential(*conv_block, nn.Conv2d(feature_maps, out_chan, 3, 1, 1))

def GenUnet():
    return 

def GenFcBlock(feat_list=[512, 1024, 1024, 512]):
    FC_blocks = []
    len_f = len(feat_list)
    for i in range(len_f - 2):
        FC_blocks += [nn.Linear(feat_list[i], feat_list[i + 1]),
                      nn.LeakyReLU(negative_slope=0.1, inplace=True)]
        
    return nn.Sequential(*FC_blocks, nn.Linear(feat_list[len_f - 2], feat_list[len_f - 1]))

# def DataConsist(input_, k, m, is_k=False):
#     if is_k:
#         return input_ * m + k * (1 - m)
#     else:
#         input_p = input_.permute(0,2,3,1); k_p = k.permute(0,2,3,1); m_p = m.permute(0,2,3,1)
#         return torch.ifft(torch.fft(input_p, 2) * m_p + k_p * (1 - m_p), 2).permute(0,3,1,2)
    




class KIKI(nn.Module):
    def __init__(self, m):
        super(KIKI, self).__init__()

        conv_blocks_K = [] 
        conv_blocks_I = []
        
        for i in range(m.iters):
            conv_blocks_K.append(GenConvBlock(m.k, m.in_ch, m.out_ch, m.fm))
            conv_blocks_I.append(GenConvBlock(m.i, m.in_ch, m.out_ch, m.fm))

        self.conv_blocks_K = nn.ModuleList(conv_blocks_K)
        self.conv_blocks_I = nn.ModuleList(conv_blocks_I)
        self.n_iter = m.iters

    def forward(self, kspace_us, mask):        
        rec = fftshift2(kspace_us)
        
        for i in range(self.n_iter):
            rec = self.conv_blocks_K[i](rec)
#            rec = DataConsist(fftshift2(rec), kspace_us, mask, is_k = True)
            rec = fftshift2(rec)
            rec = ifft2(rec)
            rec = rec + self.conv_blocks_I[i](rec)
            rec = DataConsist(rec, kspace_us, mask)
            
            if i < self.n_iter - 1:
                rec = fftshift2(fft2(rec))
        
        return rec