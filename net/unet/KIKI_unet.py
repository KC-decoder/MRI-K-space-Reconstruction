import math
import copy
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset

from torch.optim import Adam
from torchvision import transforms as T, utils

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA

from accelerate import Accelerator

from os import listdir
from os.path import join, isfile
import random

from scipy.io import savemat
from net.unet.unet_supermap import Unet 
from utils.KIKIUnet_utils import *
import fastmri


import torch
import torch.nn as nn

def roll(x, shift, dim=-1):
    if shift == 0:
        return x
    shift = shift % x.size(dim)
    idx = torch.arange(x.size(dim), device=x.device)
    return x.index_select(dim, torch.roll(idx, shifts=shift, dims=0))



# -------- 2ch <-> complex helpers (force fp32 before FFTs) --------
def _to_complex(x_2ch: torch.Tensor) -> torch.Tensor:
    # (B,2,H,W) or (2,H,W) -> complex (...,H,W), complex64 on CUDA
    if x_2ch.dim() == 4 and x_2ch.size(1) == 2:
        a = x_2ch[:, 0].to(torch.float32)
        b = x_2ch[:, 1].to(torch.float32)
        return torch.complex(a, b)
    elif x_2ch.dim() == 3 and x_2ch.size(0) == 2:
        a = x_2ch[0].to(torch.float32)
        b = x_2ch[1].to(torch.float32)
        return torch.complex(a, b)
    raise ValueError(f"_to_complex expects (B,2,H,W) or (2,H,W), got {tuple(x_2ch.shape)}")

def _to_2ch(x_cplx: torch.Tensor) -> torch.Tensor:
    # complex (...,H,W) -> (...,2,H,W)
    if x_cplx.dim() == 3 and x_cplx.is_complex():
        return torch.stack([x_cplx.real, x_cplx.imag], dim=1)  # (B,2,H,W)
    elif x_cplx.dim() == 2 and x_cplx.is_complex():
        return torch.stack([x_cplx.real, x_cplx.imag], dim=0)  # (2,H,W)
    raise ValueError(f"_to_2ch expects complex (B,H,W) or (H,W), got {tuple(x_cplx.shape)}")

# -------- centered FFT / IFFT in fp32 (returns 2ch real) --------
def fft2c_2ch(x_2ch: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    # (B,2,H,W) or (2,H,W) -> (same, 2ch float32)
    x = _to_complex(x_2ch)                         # complex64
    x = torch.fft.ifftshift(x, dim=(-2, -1))
    k = torch.fft.fft2(x, dim=(-2, -1), norm=norm) # <-- keyword dim=
    k = torch.fft.fftshift(k, dim=(-2, -1))
    return _to_2ch(k).to(torch.float32)

def ifft2c_2ch(k_2ch: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    # (B,2,H,W) or (2,H,W) -> (same, 2ch float32)
    k = _to_complex(k_2ch)                         # complex64
    k = torch.fft.ifftshift(k, dim=(-2, -1))
    x = torch.fft.ifft2(k, dim=(-2, -1), norm=norm)# <-- keyword dim=
    x = torch.fft.fftshift(x, dim=(-2, -1))
    return _to_2ch(x).to(torch.float32)


def ifft2(input_):
    return torch.ifft(input_.permute(0,2,3,1),2).permute(0,3,1,2)

def fft2(input_):
    return torch.fft(input_.permute(0,2,3,1),2).permute(0,3,1,2)

def ifft1(input_, axis):
    if   axis == 1:
        return torch.ifft(input_.permute(0,2,3,1),1).permute(0,3,1,2)
    elif axis == 0:
        return torch.ifft(input_.permute(0,3,2,1),1).permute(0,3,2,1)

def fft1(input_, axis):
    if   axis == 1:
        return torch.fft(input_.permute(0,2,3,1),1).permute(0,3,1,2)
    elif axis == 0:
        return torch.fft(input_.permute(0,3,2,1),1).permute(0,3,2,1)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.kaiming_normal_(m.weight.data, a=0.1, nonlinearity='leaky_relu')
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
        
def fftshift(x, dim):
    return roll(x, x.size(dim) // 2, dim)

def fftshift2(x):
    return fftshift(fftshift(x, -1), -2)

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

class DataConsistency(nn.Module):
    """
    Hard data consistency for MRI with 2-channel real (real+imag) tensors.

    Forward API matches your old function:
        forward(input_, k, m, is_k=False)

    Args:
        input_: (B,2,H,W)  -
            If is_k=True: interpreted as k-space prediction (2ch)
            If is_k=False: interpreted as image prediction (2ch)
        k:      (B,2,H,W)  undersampled measured k-space (2ch, real+imag)
        m:      (B,1,H,W)  binary mask {0,1} where 1=measured
        is_k:   bool       if True, apply DC directly in k-space; else project to
                           k-space, apply DC, and invert back to image.

    Returns:
        (B,2,H,W) same domain as `input_` (k-space if is_k=True, image if False)
    """
    def __init__(self, norm: str = "ortho"):
        super().__init__()
        self.norm = norm

    def forward(self, input_: torch.Tensor,
                      k: torch.Tensor,
                      m: torch.Tensor,
                      is_k: bool = False) -> torch.Tensor:
        # align device/dtype
        device = input_.device
        dtype  = input_.dtype

        k = k.to(device=device, dtype=dtype)
        M = (m > 0).to(device=device, dtype=dtype)  # (B,1,H,W)

        if is_k:
            # input_ already in k-space (2ch)
            # keep measured lines from k, use prediction elsewhere
            return input_ * (1 - M) + k * M
        else:
            # input_ is image (2ch). Project to k, enforce DC, invert back.
            k_pred = fft2c_2ch(input_, norm=self.norm).to(dtype)
            k_dc   = k_pred * (1 - M) + k * M
            x_dc   = ifft2c_2ch(k_dc, norm=self.norm).to(dtype)
            return x_dc
    


class KIKI(nn.Module):
    """
    KIKI (K-space -> Image -> K-space ...):
      - K-block: convs on 2ch k-space
      - DC in k-space: keep measured samples from k_meas
      - I-block: convs on 2ch image (residual)
      - Optional: back to k-space for next iteration

    Expects your helpers to be defined:
      - fft2c_2ch(x_2ch)   : (B,2,H,W)->(B,2,H,W)
      - ifft2c_2ch(k_2ch)  : (B,2,H,W)->(B,2,H,W)
      - DataConsistency()  : module with forward(input_, k, m, is_k)

    Config object `m` must have:
      m.iters (int)    : unroll iterations
      m.k (int)        : #conv layers in each K-block
      m.i (int)        : #conv layers in each I-block
      m.in_ch (int)    : input channels (use 2 for real+imag)
      m.out_ch (int)   : output channels (use 2)
      m.fm (int)       : feature maps per conv
    """
    def __init__(self, m, norm: str = "ortho", residual_image: bool = True):
        super().__init__()
        assert m.in_ch == 2 and m.out_ch == 2, \
            f"KIKI expects 2-channel (real,imag). Got in_ch={m.in_ch}, out_ch={m.out_ch}"

        self.n_iter = int(m.iters)
        self.norm = norm
        self.residual_image = residual_image

        # K-space and Image conv blocks (per iteration)
        self.conv_blocks_K = nn.ModuleList([
            GenConvBlock(m.k, m.in_ch, m.out_ch, m.fm) for _ in range(self.n_iter)
        ])
        self.conv_blocks_I = nn.ModuleList([
            GenConvBlock(m.i, m.in_ch, m.out_ch, m.fm) for _ in range(self.n_iter)
        ])

        # hard data consistency
        self.dc = DataConsistency(norm=self.norm)

    def forward(self, kspace_us: torch.Tensor, mask: torch.Tensor):
        """
        kspace_us : (B,2,H,W) undersampled k-space (real+imag)
        mask      : (B,1,H,W) binary {0,1} measured-sample map
        returns   : (B,2,H,W) reconstructed image (real+imag)
        """
        # sanitize mask shape/dtype
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        mask = (mask > 0).to(kspace_us.dtype)

        # keep CNN dtype consistent with model (handles AMP)
        param_dtype = next(self.parameters()).dtype

        # working k-space tensor
        k = kspace_us.to(dtype=param_dtype)

        for i in range(self.n_iter):
            # ----- K-block in k-space (2ch real) -----
            k = self.conv_blocks_K[i](k)

            # ----- Hard DC in k-space -----
            k = self.dc(k, kspace_us.to(k.dtype), mask, is_k=True)  # stays in k-space

            # ----- to image (2ch), I-block residual refinement -----
            x = ifft2c_2ch(k, norm=self.norm).to(param_dtype)

            i_out = self.conv_blocks_I[i](x)
            x = x + i_out if self.residual_image else i_out

            # ----- back to k-space for next iteration (unless last) -----
            if i < self.n_iter - 1:
                k = fft2c_2ch(x, norm=self.norm).to(param_dtype)

        # final output is image domain (B,2,H,W)
        return x