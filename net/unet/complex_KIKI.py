"""
KIKIRecon (complex K-UNet + complex I-block): Unrolled MRI reconstruction
for X:(B,2,320,320) k-space, M:(B,1,320,320) mask -> image (B,2,320,320).

Changes in this version
- Image block switched to **complex** (complex convs, channels-last internally).
- Tuned defaults for **low VRAM**: small widths, weight sharing across iterations.
- Optional **activation checkpointing** to further reduce memory.
- Centered FFTs only (fft2c/ifft2c) + k-space data consistency every iteration.

Usage (example)
---------------
model = KIKIRecon(
    iters=2,
    use_complex_k=True,
    k_base=8,          # very small complex K-UNet
    k_depth=3,
    i_complex=True,    # <-- complex image block
    i_layers=3,
    share_k=True,      # share weights across unrolls (saves VRAM)
    share_i=True,
    checkpoint_k=True, # activation checkpointing on K-net
    checkpoint_i=False # you can set True if still OOM
).cuda()

Y = model(X, M)  # X:(B,2,320,320), M:(B,1,320,320) -> Y:(B,2,320,320)
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# ==============================
# 2-channel <-> complex helpers
# ==============================

def channels_to_complex(x_2ch: torch.Tensor) -> torch.Tensor:
    assert x_2ch.dim() == 4 and x_2ch.size(1) == 2, f"expected (B,2,H,W), got {tuple(x_2ch.shape)}"
    return torch.complex(x_2ch[:, 0], x_2ch[:, 1])

def complex_to_channels(x_cplx: torch.Tensor) -> torch.Tensor:
    assert torch.is_complex(x_cplx), "input must be complex"
    return torch.stack([x_cplx.real, x_cplx.imag], dim=1)

# complex-last adapters (B, C, H, W, 2)

def twoch_to_complex_last(x_2ch: torch.Tensor) -> torch.Tensor:
    # (B,2,H,W) -> (B,1,H,W,2)
    x = x_2ch.permute(0,2,3,1)   # (B,H,W,2)
    return x.unsqueeze(1)        # (B,1,H,W,2)

def complex_last_to_twoch(x_cl: torch.Tensor) -> torch.Tensor:
    # (B,1,H,W,2) -> (B,2,H,W)
    return x_cl.squeeze(1).permute(0,3,1,2)

# ==============================
# Centered FFT wrappers (2ch I/O)
# ==============================
try:
    from fastmri.fftc import fft2c as _fft2c_fastmri, ifft2c as _ifft2c_fastmri  # type: ignore
    _HAS_FASTMRI = True
except Exception:
    _HAS_FASTMRI = False

def fft2c_2ch(x_2ch: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    x = channels_to_complex(x_2ch)
    if _HAS_FASTMRI:
        k = _fft2c_fastmri(x, norm=norm)
    else:
        x = torch.fft.ifftshift(x, dim=(-2,-1))
        k = torch.fft.fft2(x, dim=(-2,-1), norm=norm)
        k = torch.fft.fftshift(k, dim=(-2,-1))
    return complex_to_channels(k)

def ifft2c_2ch(k_2ch: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    k = channels_to_complex(k_2ch)
    if _HAS_FASTMRI:
        x = _ifft2c_fastmri(k, norm=norm)
    else:
        k = torch.fft.ifftshift(k, dim=(-2,-1))
        x = torch.fft.ifft2(k, dim=(-2,-1), norm=norm)
        x = torch.fft.fftshift(x, dim=(-2,-1))
    return complex_to_channels(x)

# ==============================
# Data Consistency (k-space)
# ==============================
class DataConsistency(nn.Module):
    def forward(self, k_pred_2ch: torch.Tensor, k_meas_2ch: torch.Tensor, mask_b1hw: torch.Tensor) -> torch.Tensor:
        if mask_b1hw.dim() == 3:
            mask_b1hw = mask_b1hw.unsqueeze(1)
        M = (mask_b1hw > 0).to(k_pred_2ch.dtype)
        return k_pred_2ch * (1.0 - M) + k_meas_2ch * M

# ==============================
# Complex ops (channels-last: ... , 2)
# ==============================
class ComplexConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, **kwargs):
        super().__init__()
        self.conv_re = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias, **kwargs)
        self.conv_im = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias, **kwargs)
    def forward(self, x):  # x: (B,C,H,W,2)
        real = self.conv_re(x[...,0]) - self.conv_im(x[...,1])
        imag = self.conv_re(x[...,1]) + self.conv_im(x[...,0])
        return torch.stack((real, imag), dim=-1)

class ComplexConvTranspose2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, output_padding=0, dilation=1, groups=1, bias=True, **kwargs):
        super().__init__()
        self.tconv_re = nn.ConvTranspose2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding,
                                           output_padding=output_padding, dilation=dilation, groups=groups, bias=bias, **kwargs)
        self.tconv_im = nn.ConvTranspose2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding,
                                           output_padding=output_padding, dilation=dilation, groups=groups, bias=bias, **kwargs)
    def forward(self, x):  # x: (B,C,H,W,2)
        real = self.tconv_re(x[...,0]) - self.tconv_im(x[...,1])
        imag = self.tconv_re(x[...,1]) + self.tconv_im(x[...,0])
        return torch.stack((real, imag), dim=-1)

class ComplexBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, **kwargs):
        super().__init__()
        self.bn_re = nn.BatchNorm2d(num_features=num_features, momentum=momentum, affine=affine, eps=eps, track_running_stats=track_running_stats, **kwargs)
        self.bn_im = nn.BatchNorm2d(num_features=num_features, momentum=momentum, affine=affine, eps=eps, track_running_stats=track_running_stats, **kwargs)
    def forward(self, x):
        real = self.bn_re(x[...,0])
        imag = self.bn_im(x[...,1])
        return torch.stack((real, imag), dim=-1)

class CNormAct(nn.Module):
    def __init__(self, c: int, act: str = 'lrelu'):
        super().__init__()
        self.bn = ComplexBatchNorm2d(c)
        self.act = nn.LeakyReLU(0.1, inplace=False) if act=='lrelu' else nn.GELU()
    def forward(self, x):
        x = self.bn(x)
        return torch.stack((self.act(x[...,0]), self.act(x[...,1])), dim=-1)

# ==============================
# Complex Tiny U-Net (channels-last complex)
# ==============================
class ComplexBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        self.conv = ComplexConv2d(c_in, c_out, k, stride=s, padding=p)
        self.na = CNormAct(c_out)
    def forward(self, x):
        return self.na(self.conv(x))

class ComplexDown(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.conv = ComplexConv2d(c_in, c_out, 3, stride=2, padding=1)
        self.na = CNormAct(c_out)
    def forward(self, x):
        return self.na(self.conv(x))

class ComplexUp(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.up = ComplexConvTranspose2d(c_in, c_out, 2, stride=2)
        self.post1 = ComplexBlock(c_out*2, c_out)
        self.post2 = ComplexBlock(c_out, c_out)
    def forward(self, x, skip):
        x = self.up(x)
        # pad to match skip if needed (real/imag separately)
        dh = skip.size(-3) - x.size(-3)
        dw = skip.size(-2) - x.size(-2)
        if dh or dw:
            x = torch.stack([
                F.pad(x[...,0], (0,max(dw,0), 0,max(dh,0))),
                F.pad(x[...,1], (0,max(dw,0), 0,max(dh,0)))
            ], dim=-1)
        x = torch.cat([x, skip], dim=1)
        x = self.post1(x)
        x = self.post2(x)
        return x

class ComplexTinyUNetCL(nn.Module):
    """Complex U-Net (channels-last complex) for K-space: in/out = 1 complex map."""
    def __init__(self, in_c: int = 1, base: int = 8, depth: int = 3):
        super().__init__()
        chs = [base * (2 ** i) for i in range(depth)]
        self.stem1 = ComplexBlock(in_c, base)
        self.stem2 = ComplexBlock(base, base)
        self.downs = nn.ModuleList()
        enc_c = [base]
        for c in chs:
            self.downs.append(ComplexDown(enc_c[-1], c))
            enc_c.append(c)
        mid_c = enc_c[-1]
        self.mid1 = ComplexBlock(mid_c, mid_c)
        self.mid2 = ComplexBlock(mid_c, mid_c)
        self.ups = nn.ModuleList()
        rev = list(reversed(enc_c[:-1]))
        in_up = mid_c
        for c_skip in rev:
            self.ups.append(ComplexUp(in_up, c_skip))
            in_up = c_skip
        self.head = ComplexConv2d(base, 1, 1)
    def forward(self, x):  # x: (B,1,H,W,2)
        s0 = self.stem2(self.stem1(x))
        skips = [s0]
        h = s0
        for d in self.downs:
            h = d(h)
            skips.append(h)
        h = self.mid2(self.mid1(h))
        for up in self.ups:
            h = up(h, skips.pop(-2))
        out = self.head(h)  # (B,1,H,W,2)
        return out

# Wrapper to use complex U-Net with (B,2,H,W)
class ComplexKNet2Ch(nn.Module):
    def __init__(self, base: int = 8, depth: int = 3):
        super().__init__()
        self.net = ComplexTinyUNetCL(in_c=1, base=base, depth=depth)
    def forward(self, k_2ch: torch.Tensor) -> torch.Tensor:
        x_cl = twoch_to_complex_last(k_2ch)          # (B,1,H,W,2)
        y_cl = self.net(x_cl)                        # (B,1,H,W,2)
        y_2ch = complex_last_to_twoch(y_cl)          # (B,2,H,W)
        return y_2ch

# Complex image block (residual) using complex convs
class ComplexImageBlock2Ch(nn.Module):
    def __init__(self, base: int = 8, layers: int = 3):
        super().__init__()
        mods = []
        c_in = 1
        for _ in range(layers-1):
            mods += [ComplexConv2d(c_in, base, 3, padding=1), CNormAct(base)]
            c_in = base
        mods += [ComplexConv2d(c_in, 1, 3, padding=1)]
        self.body = nn.Sequential(*mods)
    def forward(self, x_2ch):  # (B,2,H,W) -> (B,2,H,W)
        x_cl = twoch_to_complex_last(x_2ch)
        y_cl = self.body(x_cl)
        return complex_last_to_twoch(y_cl)

# ==============================
# Full model (complex K + complex I)
# ==============================
class KIKIRecon(nn.Module):
    """Unrolled recon with complex K-space U-Net and complex image residual block."""
    def __init__(
        self,
        iters: int = 2,
        use_complex_k: bool = True,
        k_base: int = 8,
        k_depth: int = 3,
        i_complex: bool = True,
        i_layers: int = 3,
        share_k: bool = True,
        share_i: bool = True,
        checkpoint_k: bool = True,
        checkpoint_i: bool = False,
    ):
        super().__init__()
        self.iters = int(iters)
        self.ckpt_k = bool(checkpoint_k)
        self.ckpt_i = bool(checkpoint_i)

        # K-space modules (complex)
        if use_complex_k:
            k_net = ComplexKNet2Ch(base=k_base, depth=k_depth)
            self.k_nets = nn.ModuleList([k_net for _ in range(self.iters)]) if share_k else \
                           nn.ModuleList([ComplexKNet2Ch(base=k_base, depth=k_depth) for _ in range(self.iters)])
        else:
            raise NotImplementedError("Set use_complex_k=True for complex K-UNet")

        # Image-space modules (complex)
        if i_complex:
            ib = ComplexImageBlock2Ch(base=max(8, k_base), layers=i_layers)
            self.i_blocks = nn.ModuleList([ib for _ in range(self.iters)]) if share_i else \
                             nn.ModuleList([ComplexImageBlock2Ch(base=max(8, k_base), layers=i_layers) for _ in range(self.iters)])
        else:
            raise NotImplementedError("This version uses complex I-block; set i_complex=True")

        self.dc = DataConsistency()

    def forward(self, kspace_us: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        assert kspace_us.dim()==4 and kspace_us.size(1)==2, f"kspace_us must be (B,2,H,W), got {tuple(kspace_us.shape)}"
        if mask.dim()==3: mask = mask.unsqueeze(1)
        mask = (mask>0).to(kspace_us.dtype)
        k = kspace_us
        for i in range(self.iters):
            # K-net (complex U-Net in k-space)
            if self.ckpt_k:
                from torch.utils.checkpoint import checkpoint
                k = checkpoint(self.k_nets[i], k, use_reentrant=False)
            else:
                k = self.k_nets[i](k)
            # DC in k-space
            k = self.dc(k, kspace_us, mask)
            # image domain + residual refine (complex)
            x = ifft2c_2ch(k)  # (B,2,H,W)
            if self.ckpt_i:
                from torch.utils.checkpoint import checkpoint
                x = x + checkpoint(self.i_blocks[i], x, use_reentrant=False)
            else:
                x = x + self.i_blocks[i](x)
            if i < self.iters - 1:
                k = fft2c_2ch(x)
        return x