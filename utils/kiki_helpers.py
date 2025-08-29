import os
import math
import time
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from debug.CUNet_debug import *


from contextlib import nullcontext


def _get_autocast(device: torch.device, enabled: bool = True):
    """
    Returns a proper autocast context for the given device.
    """
    if not enabled:
        return nullcontext()
    if device.type == "cuda":
        # modern API (silences FutureWarning)
        return torch.amp.autocast("cuda", dtype=torch.float16)
    if device.type == "cpu":
        # safe on recent PyTorch; else falls back below
        try:
            return torch.amp.autocast("cpu", dtype=torch.bfloat16)
        except Exception:
            return nullcontext()
    return nullcontext()

# -----------------------------
# Utility: make dirs & device
# -----------------------------
def _mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _align_pred_target(pred: torch.Tensor, target: torch.Tensor):
    """Updated alignment using correct magnitude calculation."""
    
    if pred.dim() != 4 or target.dim() != 4:
        raise ValueError(f"Expected 4D NCHW, got pred={tuple(pred.shape)} target={tuple(target.shape)}")
    
    if pred.size(1) == 2 and target.size(1) == 1:
        # Convert complex prediction to magnitude using correct helper
        magnitude = complex_magnitude(pred).unsqueeze(1)  # Add channel dim back
        return magnitude, target
    elif pred.size(1) == target.size(1):
        return pred, target
    else:
        raise ValueError(f"Cannot align pred channels={pred.size(1)} with target channels={target.size(1)}")
    

def build_loss_fn(loss_name: str = "l1"):
    """L1 loss is excellent for MRI magnitude comparison."""
    if loss_name.lower() == "l1":
        return nn.L1Loss()  #  Good choice for MRI
    elif loss_name.lower() in ("l2", "mse"):  
        return nn.MSELoss()  # Also okay, but L1 often better
    else:
        raise ValueError("loss_name must be 'l1' or 'l2'")
    

@torch.no_grad()
def compute_complex_metrics(pred_complex: torch.Tensor, target_magnitude: torch.Tensor):
    """
    Compute metrics for complex predictions vs magnitude targets.
    This gives you insight into what the model is actually learning.
    """
    # Convert prediction to magnitude
    pred_mag = torch.sqrt(pred_complex[:, 0:1]**2 + pred_complex[:, 1:2]**2 + 1e-12)
    
    # Standard magnitude metrics
    l1_loss = F.l1_loss(pred_mag, target_magnitude)
    mse_loss = F.mse_loss(pred_mag, target_magnitude)
    
    # Phase consistency (how stable is the learned phase?)
    phase = torch.atan2(pred_complex[:, 1:2], pred_complex[:, 0:1])
    phase_std = torch.std(phase)  # Lower is more consistent
    
    return {
        "magnitude_l1": l1_loss.item(),
        "magnitude_mse": mse_loss.item(), 
        "phase_std": phase_std.item(),
        "pred_magnitude_range": (pred_mag.min().item(), pred_mag.max().item())
    }

# -----------------------------
# Checkpoint helpers
# -----------------------------
def save_checkpoint(state: Dict[str, Any], ckpt_dir: Path, name: str):
    _mkdir(ckpt_dir)
    path = ckpt_dir / name
    torch.save(state, path)
    return path

def load_checkpoint(path: Path, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None,
                    scheduler: Optional[Any] = None, map_location: Optional[str] = None) -> Dict[str, Any]:
    chk = torch.load(path, map_location=map_location or "cpu")
    model.load_state_dict(chk["model"])
    if optimizer is not None and "optimizer" in chk and chk["optimizer"] is not None:
        optimizer.load_state_dict(chk["optimizer"])
    if scheduler is not None and "scheduler" in chk and chk["scheduler"] is not None:
        scheduler.load_state_dict(chk["scheduler"])
    return chk






# ____________________________________________
#             KIKI UTILS
# _____________________________________________



def complex_magnitude(x_2ch: torch.Tensor) -> torch.Tensor:
    """Compute magnitude from (N,2,H,W) complex format."""
    if x_2ch.dim() == 4:
        return torch.linalg.vector_norm(x_2ch, dim=1)
    elif x_2ch.dim() == 3:
        return torch.linalg.vector_norm(x_2ch, dim=0)
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got {x_2ch.shape}")

def create_zero_filled_baseline(kspace: torch.Tensor) -> torch.Tensor:
    """Create zero-filled reconstruction from undersampled k-space."""
    image_complex = ifft2(kspace)
    magnitude = complex_magnitude(image_complex)
    return magnitude.unsqueeze(1) if magnitude.dim() == 3 else magnitude

def roll(x: torch.Tensor, shift: int, dim: int = -1) -> torch.Tensor:
    """Torch-native roll that works on any device."""
    return torch.roll(x, shifts=shift, dims=dim)

def fftshift2(x: torch.Tensor) -> torch.Tensor:
    """2D fftshift over the last two spatial dims (H, W)."""
    return torch.fft.fftshift(x, dim=(-2, -1))

def ifftshift2(x: torch.Tensor) -> torch.Tensor:
    """2D ifftshift over the last two spatial dims (H, W).""" 
    return torch.fft.ifftshift(x, dim=(-2, -1))

def _to_complex(x_2ch: torch.Tensor) -> torch.Tensor:
    """Convert (N,2,H,W) format to (N,H,W) complex tensor."""
    if x_2ch.dim() not in [3, 4] or x_2ch.size(-3) != 2:
        raise ValueError(f"Expected (...,2,H,W), got {tuple(x_2ch.shape)}")
    return torch.complex(x_2ch[..., 0, :, :], x_2ch[..., 1, :, :])

def _to_2ch(x_complex: torch.Tensor) -> torch.Tensor:
    """Convert (...,H,W) complex tensor to (...,2,H,W) format."""
    if not torch.is_complex(x_complex):
        raise ValueError("Expected complex tensor")
    return torch.stack([x_complex.real, x_complex.imag], dim=-3)

def fft2(input_: torch.Tensor) -> torch.Tensor:
    """2D FFT: (N,2,H,W) -> (N,2,H,W) using ortho normalization."""
    orig_dtype = input_.dtype
    if input_.dtype == torch.float16:
        input_ = input_.float()
    
    x_complex = _to_complex(input_)
    x_complex = ifftshift2(x_complex)
    k_complex = torch.fft.fft2(x_complex, norm='ortho')
    k_complex = fftshift2(k_complex)
    result = _to_2ch(k_complex)
    return result.to(orig_dtype)

def ifft2(input_: torch.Tensor) -> torch.Tensor:
    """2D IFFT: (N,2,H,W) -> (N,2,H,W) using ortho normalization."""
    orig_dtype = input_.dtype
    if input_.dtype == torch.float16:
        input_ = input_.float()
    
    k_complex = _to_complex(input_)
    k_complex = ifftshift2(k_complex)
    x_complex = torch.fft.ifft2(k_complex, norm='ortho')
    x_complex = fftshift2(x_complex)
    result = _to_2ch(x_complex)
    return result.to(orig_dtype)


def fft1(input_: torch.Tensor, axis: int) -> torch.Tensor:
    """
    1D FFT along specified spatial axis.
    Fixed for mixed precision training
    """
    # Store original dtype for restoration
    orig_dtype = input_.dtype
    
    # Convert to float32 if needed for FFT compatibility
    if input_.dtype == torch.float16:
        input_ = input_.float()
        
    x_complex = _to_complex(input_)
    if axis == 1:   # along H dimension
        k_complex = torch.fft.fft(x_complex, dim=-2, norm='backward')
    elif axis == 0: # along W dimension  
        k_complex = torch.fft.fft(x_complex, dim=-1, norm='backward')
    else:
        raise ValueError("axis must be 0 (W) or 1 (H)")
    
    result = _to_2ch(k_complex)
    return result.to(orig_dtype)

def ifft1(input_: torch.Tensor, axis: int) -> torch.Tensor:
    """
    1D IFFT along specified spatial axis.
    Fixed for mixed precision training
    """
    # Store original dtype for restoration
    orig_dtype = input_.dtype
    
    # Convert to float32 if needed for FFT compatibility
    if input_.dtype == torch.float16:
        input_ = input_.float()
        
    X_complex = _to_complex(input_)
    if axis == 1:   # along H dimension
        x_complex = torch.fft.ifft(X_complex, dim=-2, norm='backward')  
    elif axis == 0: # along W dimension
        x_complex = torch.fft.ifft(X_complex, dim=-1, norm='backward')
    else:
        raise ValueError("axis must be 0 (W) or 1 (H)")
    
    result = _to_2ch(x_complex)
    return result.to(orig_dtype)

def DataConsist(input_: torch.Tensor, k_measured: torch.Tensor, mask: torch.Tensor, is_k: bool = False) -> torch.Tensor:
    """Data consistency with correct formula and ortho normalization."""
    orig_dtype = input_.dtype
    if input_.dtype == torch.float16:
        input_work = input_.float()
        k_work = k_measured.float() 
        mask_work = mask.float()
    else:
        input_work = input_
        k_work = k_measured.to(input_.device, input_.dtype)
        mask_work = mask.to(input_.device, input_.dtype)
    
    if mask_work.size(1) == 1 and k_work.size(1) == 2:
        mask_work = mask_work.repeat(1, 2, 1, 1)
    
    if is_k:
        result = k_work * mask_work + input_work * (1 - mask_work)
    else:
        input_k = fft2(input_work)
        consistent_k = k_work * mask_work + input_k * (1 - mask_work)
        result = ifft2(consistent_k)
    
    return result.to(orig_dtype)




def _to_gray2d(arr):
    """
    Accepts arrays in these shapes and returns (H, W) float64:
      - (H, W)
      - (1, H, W) or (H, W, 1)
      - (2, H, W) or (H, W, 2)  -> magnitude from real/imag
    """
    a = np.asarray(arr)
    if a.ndim == 2:
        return a

    if a.ndim == 3:
        # channel-first?
        if a.shape[0] in (1, 2):
            if a.shape[0] == 1:
                return a[0]
            # (2, H, W): real, imag -> magnitude
            return np.hypot(a[0], a[1])

        # channel-last?
        if a.shape[-1] in (1, 2):
            if a.shape[-1] == 1:
                return a[..., 0]
            # (H, W, 2): real, imag -> magnitude
            return np.hypot(a[..., 0], a[..., 1])

    raise ValueError(f"Don't know how to display array with shape {a.shape}")




def minmax_norm(x, dims=(2,3), eps=1e-8):
    # per-sample min/max over H,W (keeps B,C)
    x_min = x.amin(dim=dims, keepdim=True)
    x_max = x.amax(dim=dims, keepdim=True)
    scale = (x_max - x_min).clamp_min(eps)
    x_n = (x - x_min) / scale
    return x_n, (x_min, scale)

def denorm(x_n, x_min, scale):
    return x_n * scale + x_min


# ---------- small helpers ----------

def _minmax01(img2d: np.ndarray) -> np.ndarray:
    img2d = img2d.astype(np.float64)
    mn, mx = img2d.min(), img2d.max()
    return (img2d - mn) / (mx - mn + 1e-8)

def _ifft2c_2ch(k_2ch: torch.Tensor, norm="ortho") -> torch.Tensor:
    """
    k_2ch: (2,H,W) or (B,2,H,W) real+imag -> image (same rank), 2-ch real+imag
    Uses centered FFT convention.
    """
    batched = (k_2ch.dim() == 4)
    if not batched:
        k_2ch = k_2ch.unsqueeze(0)  # -> (1,2,H,W)

    k = torch.complex(k_2ch[:, 0].float(), k_2ch[:, 1].float())       # (B,H,W) complex64
    k = torch.fft.ifftshift(k, dim=(-2, -1))
    x = torch.fft.ifft2(k, dim=(-2, -1), norm=norm)
    x = torch.fft.fftshift(x, dim=(-2, -1))
    x_2ch = torch.stack([x.real, x.imag], dim=1)  # (B,2,H,W)
    return x_2ch if batched else x_2ch.squeeze(0)

def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()

