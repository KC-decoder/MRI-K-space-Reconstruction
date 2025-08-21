# layer_utils_fastmri.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import matplotlib.pyplot as plt
import os
import numpy as np
try:
    from fastmri import fftc as _fftc
    _HAS_FASTMRI = hasattr(_fftc, "fft2c") and hasattr(_fftc, "ifft2c")
except Exception:
    _HAS_FASTMRI = False
    _fftc = None

from typing import Union, Optional, Tuple

# -------------------------
# Basic helpers
# -------------------------
def roll(x: torch.Tensor, shift: int, dim: int = -1) -> torch.Tensor:
    """Fast device-agnostic roll (uses torch.roll)."""
    if shift == 0:
        return x
    return torch.roll(x, shifts=shift, dims=dim)


def fftshift(x: torch.Tensor, dim: Union[int, Tuple[int, ...]] = -1) -> torch.Tensor:
    """Shift zero-frequency component to center of spectrum along given dim(s)."""
    if isinstance(dim, int):
        dim = (dim,)
    for d in dim:
        n = x.size(d)
        x = roll(x, n // 2, d)
    return x


def ifftshift(x: torch.Tensor, dim: Union[int, Tuple[int, ...]] = -1) -> torch.Tensor:
    """Inverse of fftshift."""
    if isinstance(dim, int):
        dim = (dim,)
    for d in dim:
        n = x.size(d)
        x = roll(x, -(n // 2), d)
    return x


def fftshift2(x: torch.Tensor) -> torch.Tensor:
    """2D fftshift on last two spatial dims (works for (B,C,H,W) or (B,H,W))."""
    return fftshift(fftshift(x, dim=-1), dim=-2)


def ifftshift2(x: torch.Tensor) -> torch.Tensor:
    return ifftshift(ifftshift(x, dim=-1), dim=-2)


# -------------------------
# Complex <-> 2-channel helpers
# -------------------------
def channels_to_complex(x: torch.Tensor) -> torch.Tensor:
    """
    Convert real/imag channels (B, 2, H, W) -> complex (B, H, W) dtype=complex64.
    Accepts floating input dtype (float32/64).
    """
    # print(f"shape of x: {x.shape}")
    assert x.ndim == 4 and x.size(1) == 2, "Expected (B,2,H,W)"
    # (B,2,H,W) -> (B,H,W,2)
    xr = x[:, 0, :, :].contiguous()
    xi = x[:, 1, :, :].contiguous()
    stacked = torch.stack([xr, xi], dim=-1)  # (B,H,W,2)
    return torch.view_as_complex(stacked)     # (B,H,W) complex


def complex_to_channels(z: torch.Tensor) -> torch.Tensor:
    """
    Convert complex (B,H,W) -> real/imag channels (B,2,H,W)
    """
    assert torch.is_complex(z), "Input must be complex tensor"
    real_imag = torch.view_as_real(z)        # (B,H,W,2)
    real_imag = real_imag.permute(0, 3, 1, 2).contiguous()  # (B,2,H,W)
    return real_imag


# -------------------------
# Centered FFT helpers (fastMRI convention)
# -------------------------
def fft2c(x: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    """
    Centered 2D FFT on last two dims. Accepts complex (B,H,W) or 2-channel (B,2,H,W).
    Returns complex (B,H,W).
    """
    if not torch.is_complex(x):
        x = channels_to_complex(x)  # (B,2,H,W) -> (B,H,W) complex

    if _HAS_FASTMRI:
        return _fftc.fft2c(x, norm=norm)

    # Fallback with torch.fft + manual centering
    x = torch.fft.ifftshift(x, dim=(-2, -1))
    k = torch.fft.fft2(x, dim=(-2, -1), norm=norm)
    k = torch.fft.fftshift(k, dim=(-2, -1))
    return k


def ifft2c(k: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    """
    Centered 2D IFFT on last two dims. Accepts complex (B,H,W) or 2-channel (B,2,H,W).
    Returns complex (B,H,W).
    """
    if not torch.is_complex(k):
        k = channels_to_complex(k)  # (B,2,H,W) -> (B,H,W) complex

    if _HAS_FASTMRI:
        return _fftc.ifft2c(k, norm=norm)

    # Fallback with torch.fft + manual centering
    k = torch.fft.ifftshift(k, dim=(-2, -1))
    x = torch.fft.ifft2(k, dim=(-2, -1), norm=norm)
    x = torch.fft.fftshift(x, dim=(-2, -1))
    return x


# -------------------------
# 1D FFT wrappers (optional)
# -------------------------
def fft1c(x: torch.Tensor, axis: int = -1, norm: str = 'ortho') -> torch.Tensor:
    """
    1D centered FFT along given axis.
    Accepts complex tensors. If real/imag channels provided, convert first.
    axis values supported relative to the last two dims for images, or any dim for 1D.
    """
    if not torch.is_complex(x):
        x = channels_to_complex(x)
    x = ifftshift(x, dim=axis)
    k = fastmri.fftc(x, dim=axis, norm=norm)
    k = fftshift(k, dim=axis)
    return k


def ifft1c(k: torch.Tensor, axis: int = -1, norm: str = 'ortho') -> torch.Tensor:
    if not torch.is_complex(k):
        k = channels_to_complex(k)
    k = ifftshift(k, dim=axis)
    x = fastmri.ifft2c(k, dim=axis, norm=norm)
    x = fftshift(x, dim=axis)
    return x


# -------------------------
# Data consistency (hard overwrite)
# -------------------------
# -----------------------
# Data consistency module
# -----------------------
class HardDataConsistency(nn.Module):
    """
    Simple IDC: overwrite predicted k-space at sampled locations with measured k-space.
    mask: boolean or 0/1 tensor with shape broadcastable to (B,H,W). True==sampled.
    """
    def __init__(self):
        super().__init__()

    def forward(self, k_pred: torch.Tensor, k_meas: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        k_pred: complex (B,H,W)
        k_meas: complex (B,H,W)  (measured kspace; zeros in missing points or full measured)
        mask: boolean (B,H,W) or (1,H,W) or (B,1,H,W) where True indicates measured samples
        """
        # normalize mask shape to (B,H,W) bool
        if mask.ndim == 4 and mask.size(1) == 1:
            mask_ = mask[:, 0, :, :].to(dtype=torch.bool, device=k_pred.device)
        elif mask.ndim == 3:
            mask_ = mask.to(dtype=torch.bool, device=k_pred.device)
        else:
            # try to broadcast
            mask_ = mask.to(dtype=torch.bool, device=k_pred.device)
        # ensure same shape
        if mask_.dim() == 2:
            mask_ = mask_.unsqueeze(0).expand(k_pred.shape[0], -1, -1)

        return torch.where(mask_, k_meas, k_pred)


# -------------------------
# Generator conv blocks (unchanged semantics but robust)
# -------------------------
def GenConvBlock(n_conv_layers: int, in_chan: int, out_chan: int, feature_maps: int):
    """
    Create a conv block with n_conv_layers and final 3x3 conv to out_chan.
    Semantics match your original: conv -> LeakyReLU -> (repeat conv/ReLU) -> final conv.
    """
    assert n_conv_layers >= 2, "n_conv_layers should be >= 2"
    layers = [nn.Conv2d(in_chan, feature_maps, kernel_size=3, stride=1, padding=1),
              nn.LeakyReLU(negative_slope=0.1, inplace=True)]
    for _ in range(n_conv_layers - 2):
        layers += [nn.Conv2d(feature_maps, feature_maps, kernel_size=3, stride=1, padding=1),
                   nn.LeakyReLU(negative_slope=0.1, inplace=True)]
    layers.append(nn.Conv2d(feature_maps, out_chan, kernel_size=3, stride=1, padding=1))
    return nn.Sequential(*layers)




# -------------------------
# Weight init helper
# -------------------------
def weights_init_normal(m):
    classname = m.__class__.__name__
    if 'Conv' in classname:
        nn.init.kaiming_normal_(m.weight, a=0.1, nonlinearity='leaky_relu')
        if getattr(m, 'bias', None) is not None:
            nn.init.zeros_(m.bias)
    elif 'BatchNorm' in classname or 'InstanceNorm' in classname:
        if getattr(m, 'weight', None) is not None:
            nn.init.normal_(m.weight, 1.0, 0.02)
        if getattr(m, 'bias', None) is not None:
            nn.init.zeros_(m.bias)



#---------------------------------------
# Evaluation utils
#---------------------------------------



# --------- small FFT helpers (centered) ----------
def ifft2c_2ch(k_2ch: torch.Tensor) -> torch.Tensor:
    # (B,2,H,W) -> (B,2,H,W) image (real,imag)
    k = torch.complex(k_2ch[:,0], k_2ch[:,1])
    k = torch.fft.ifftshift(k, dim=(-2,-1))
    x = torch.fft.ifft2(k, dim=(-2,-1), norm="ortho")
    x = torch.fft.fftshift(x, dim=(-2,-1))
    return torch.stack([x.real, x.imag], dim=1)

def complex_mag_2ch(x_2ch: torch.Tensor) -> torch.Tensor:
    # (B,2,H,W) -> (B,1,H,W)
    return torch.linalg.vector_norm(x_2ch, dim=1, keepdim=True)

# --------- plotting ----------
def _imshow(ax, img2d, title):
    ax.imshow(img2d, cmap="gray")
    ax.set_title(title, fontsize=10)
    ax.axis("off")

def _normalize(img):
    # per-image min-max to [0,1] for display
    imin = img.min()
    imax = img.max()
    if float(imax - imin) < 1e-12:
        return img * 0.0
    return (img - imin) / (imax - imin)

# --------- main evaluation function ----------
@torch.no_grad()
def evaluate_and_plot(
    model: torch.nn.Module,
    test_dataloader,
    device: torch.device,
    save_dir: str,
    batches: int = 1,          # how many batches to visualize
    samples_per_batch: int = 4 # how many samples per batch to plot
):
    """
    Runs evaluation and saves side-by-side plots:
      Zero-filled | Reconstruction | Ground Truth | Error map
    Assumes dataloader yields: X (k-space, B,2,H,W), y (image, B,2,H,W), mask (B,1,H,W)
    Returns: list of saved PNG paths
    """
    model = model.to(device)
    model.eval()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []

    iterator = iter(test_dataloader)
    for bidx in range(batches):
        try:
            X, y, mask = next(iterator)
        except StopIteration:
            break

        X = X.to(device).float()      # k-space (B,2,H,W)
        y = y.to(device).float()      # image  (B,2,H,W)
        mask = mask.to(device).float()# (B,1,H,W)

        # prediction: try (X,mask) first, then fallback to (X)
        try:
            y_pred = model(X, mask)   # expected KIKI signature
        except TypeError:
            y_pred = model(X)         # image-only models

        # compute zero-filled reference from k-space
        zf_img = ifft2c_2ch(X)                     # (B,2,H,W)

        # convert all to magnitude for viz
        zf_mag   = complex_mag_2ch(zf_img)         # (B,1,H,W)
        pred_mag = complex_mag_2ch(y_pred)         # (B,1,H,W)
        gt_mag   = complex_mag_2ch(y)              # (B,1,H,W)

        # error map (absolute difference), normalized for display
        err = (pred_mag - gt_mag).abs()

        B = X.size(0)
        nshow = min(samples_per_batch, B)
        cols = 4
        rows = nshow
        fig, axs = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
        if rows == 1:  # keep indexing consistent
            axs = np.expand_dims(axs, 0)

        for i in range(nshow):
            z = _normalize(zf_mag[i,0].detach().cpu().numpy())
            p = _normalize(pred_mag[i,0].detach().cpu().numpy())
            g = _normalize(gt_mag[i,0].detach().cpu().numpy())
            e = _normalize(err[i,0].detach().cpu().numpy())

            _imshow(axs[i,0], z, "Zero-filled")
            _imshow(axs[i,1], p, "Reconstruction")
            _imshow(axs[i,2], g, "Ground Truth")
            _imshow(axs[i,3], e, "Error |pred - gt|")

        plt.tight_layout()
        out_path = save_dir / f"eval_batch{bidx:03d}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(str(out_path))

    return saved_paths