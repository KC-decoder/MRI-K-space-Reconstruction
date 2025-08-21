import numpy as np
import matplotlib.pyplot as plt

import os
import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

import fastmri
from fastmri.data import subsample, transforms, mri_data
from help_func import print_var_detail



class DataTransform_Diffusion:
    def __init__(
            self,
            mask_func,
            img_size=320,
            combine_coil=True,
            flag_singlecoil=False,
    ):
        """
        data transformation class applied on diffusion models

        Args:
            mask_func: mask function that output both unfolded mask and folded mask
            img_size: int, image_size of H, W
            combine_coil: bool, check if combined coil
            flag_singlecoil: bool, check if input is singlecoil
        """
        self.mask_func = mask_func
        self.img_size = img_size
        self.combine_coil = combine_coil  # whether to combine multi-coil imgs into a single channel
        self.flag_singlecoil = flag_singlecoil
        if flag_singlecoil:
            self.combine_coil = True

    def __call__(self, kspace, mask, target, data_attributes, filename, slice_num):
        if self.flag_singlecoil:
            kspace = kspace[None, ...]  # [H,W,2] to [1,H,W,2]

        # k-space, transform the data into appropriate format
        kspace = transforms.to_tensor(kspace)  # [Nc,H,W,2]

        # ====== Image reshaping ======
        # img space
        image_full = fastmri.ifft2c(kspace)  # [Nc,H,W,2]
        # center cropping
        image_full = transforms.complex_center_crop(image_full, [320, 320])  # [Nc,H,W,2]
        # resize img
        if self.img_size != 320:
            image_full = torch.einsum('nhwc->nchw', image_full)
            image_full = T.Resize(size=self.img_size)(image_full)
            image_full = torch.einsum('nchw->nhwc', image_full)

        # for now assume combined coil only
        if self.combine_coil:
            image_full = fastmri.rss(image_full).unsqueeze(0)  # [1,H,W,2]

        # img to k-space
        kspace = fastmri.fft2c(image_full)  # [Nc,H,W,2]

        # ====== Fully-sampled ===
        # img space
        image_full_abs = fastmri.complex_abs(image_full)  # [Nc,H,W]

        # ====== Under-sampled ======
        # apply mask

        # assume using same mask across image coils
        mask, mask_fold = self.mask_func()  # [1,H,W] [1,H/patch_size,W/patch_size]
        mask = torch.from_numpy(mask).float()  # [1,H,W]
        mask_fold = torch.from_numpy(mask_fold).float()  # [1,H/patch_size,W/patch_size]
        mask = mask[..., None].repeat(kspace.shape[0], 1, 1, 1)  # [Nc,H,W,2]
        masked_kspace = kspace * mask

        image_masked = fastmri.ifft2c(masked_kspace)  # [Nc,H,W,2]
        image_masked_abs = fastmri.complex_abs(image_masked)  # [Nc,H,W]
        max = torch.amax(image_masked_abs, dim=(1, 2))
        scale_coeff = 1. / max  # [Nc,]

        kspace = torch.einsum('ijkl, i -> ijkl', kspace, scale_coeff)

        return kspace, mask[0, ..., 0].unsqueeze(0), mask_fold  # [B,Nc,H,W,2]


class DataTransform_UNet:
    def __init__(
            self,
            mask_func,
            img_size=320,
            combine_coil=True,
            flag_singlecoil=False,
    ):
        self.mask_func = mask_func
        self.img_size = img_size
        self.combine_coil = combine_coil  # whether to combine multi-coil imgs into a single channel
        self.flag_singlecoil = flag_singlecoil
        if flag_singlecoil:
            self.combine_coil = True

    def __call__(self, kspace, mask, target, data_attributes, filename, slice_num):
        
        if self.flag_singlecoil:
            kspace = kspace[None, ...]  # [H,W,2] to [1,H,W,2]
        # print("Inside Transform Call 2")
        # k-space, transform the data into appropriate format
        kspace = transforms.to_tensor(kspace)  # [Nc,H,W,2]
        Nc, H, W = kspace.shape

        # # ===== Shape check before transform =====
        # if H < self.img_size or W < self.img_size:
        #     return None  # Skip this sample

        # print("Inside Transform Call 3")
        # ====== Image reshaping ======
        # img space
        # center cropping
        # print("[DEBUG] Reached transform call")
        image_full = fastmri.ifft2c(kspace)
        # print(f"[DEBUG] image_full shape before crop: {image_full.shape}")
        # center cropping
        image_full = transforms.complex_center_crop(image_full, [320, 320])
        # print("Inside Transform Call 4")
        # print(f"[DEBUG] image_full shape after crop: {image_full.shape}")
        # resize img
        if self.img_size != 320:
            image_full = torch.einsum('nhwc->nchw', image_full)
            image_full = T.Resize(size=self.img_size)(image_full)
            image_full = torch.einsum('nchw->nhwc', image_full)
        # img to k-space
        kspace = fastmri.fft2c(image_full)  # [Nc,H,W,2]

        # ====== Fully-sampled ===
        # img space
        image_full = fastmri.complex_abs(image_full)  # [Nc,H,W]

        # ====== Under-sampled ======
        # apply mask
        if isinstance(self.mask_func, subsample.MaskFunc):
            masked_kspace, mask, _ = transforms.apply_mask(kspace, self.mask_func)  # mask [1,1,W,1]
            # print("Inside Mask function in Transform")
            mask = mask.squeeze(-1).squeeze(0).repeat(kspace.shape[1], 1)  # [H,W]
        else:
            masked_kspace, mask = apply_mask(kspace, self.mask_func)  # mask [1,H,W,1]
            mask = mask.squeeze(-1).squeeze(0)  # [H,W]
        # print("Error appeared here 1")
        image_masked = fastmri.ifft2c(masked_kspace)
        image_masked = fastmri.complex_abs(image_masked)  # [Nc,H,W]
        

        # ====== RSS coil combination (knee single coil) ======
        if self.combine_coil:
            image_full = fastmri.rss(image_full, dim=0)  # [H,W]
            image_masked = fastmri.rss(image_masked, dim=0)  # [H,W]

            return image_masked.unsqueeze(0), image_full.unsqueeze(0), mask.unsqueeze(0)
        else:
            # img [B,Nc,H,W], mask [B,1,H,W]
            image_full = image_full / (image_full.max() + 1e-8) 
            image_masked = image_masked / (image_masked.max() + 1e-8)
            # print("Error appeared here 2")
            return image_masked, image_full.unsqueeze(0), mask.unsqueeze(0)

class DataTransform_UNet_Kspace:
    """
    Input:  k-space (Nc,H,W,2) or (H,W,2)
    Output:
      X: undersampled k-space
         - combine_coil=True  -> (2, H, W)      [real, imag] from one coil (coil_index)
         - combine_coil=False -> (2*Nc, H, W)   all coils stacked as channels
      Y: image target (built from fully-sampled k-space, after crop)
         - target_mode='mag':
              combine_coil=True  -> (1, H, W)   RSS magnitude
              combine_coil=False -> (1, H, W)   RSS magnitude (still a single target)
         - target_mode='complex':
              combine_coil=True  -> (2, H, W)   complex (real,imag) from coil_index
              combine_coil=False -> (2*Nc, H, W) complex for every coil, stacked
      M: mask (1, H, W), float {0,1}
    """
    def __init__(self,
                 mask_func,
                 img_size=320,
                 combine_coil=True,
                 flag_singlecoil=False,
                 coil_index=0,
                 target_mode='mag'    # 'mag' or 'complex'
                 ):
        self.mask_func      = mask_func
        self.img_size       = img_size
        self.combine_coil   = combine_coil
        self.flag_singlecoil= flag_singlecoil
        self.coil_index     = coil_index
        self.target_mode    = target_mode.lower()
        if flag_singlecoil:
            self.combine_coil = True  # single-coil dataset => treat as 1 coil

    def __call__(self, kspace, mask, target, data_attributes, filename, slice_num):
        # ---- to torch, ensure coil dim ----
        ks = transforms.to_tensor(kspace)                 # keeps last dim=2
        if ks.dim() == 3:                                 # (H,W,2) -> (1,H,W,2)
            ks = ks.unsqueeze(0)
        assert ks.dim() == 4 and ks.size(-1) == 2, f"expected (Nc,H,W,2), got {tuple(ks.shape)}"
        Nc, H, W, _ = ks.shape

        # ---- go to image, center crop, back to k-space (fully-sampled) ----
        img_fs  = fastmri.ifft2c(ks)                      # (Nc,H,W,2)
        img_fs  = transforms.complex_center_crop(img_fs, [self.img_size, self.img_size])  # (Nc,Hc,Wc,2)
        ks_fs   = fastmri.fft2c(img_fs)                   # (Nc,Hc,Wc,2)
        Nc, Hc, Wc, _ = ks_fs.shape

        # ---- apply undersampling mask ----
        if isinstance(self.mask_func, subsample.MaskFunc):
            ks_us, msk, _ = transforms.apply_mask(ks_fs, self.mask_func)   # msk: (1,1,Hc,Wc,1)
            M = msk.squeeze(-1).squeeze(0)                                 # (1,Hc,Wc)
        else:
            ks_us, msk = apply_mask(ks_fs, self.mask_func)                  # msk: (1,Hc,Wc,1)
            M = msk.squeeze(-1)                                            # (1,Hc,Wc)
        M = (M > 0).to(ks_fs.dtype).contiguous()                            # (1,Hc,Wc)

        # ---- build X (k-space undersampled) ----
        if self.combine_coil:
            c = int(max(0, min(self.coil_index, Nc - 1)))
            X = ks_us[c].permute(2, 0, 1).contiguous()                      # (2,Hc,Wc)
        else:
            X_nc2 = ks_us.permute(0, 3, 1, 2).contiguous()                  # (Nc,2,Hc,Wc)
            X = X_nc2.reshape(-1, Hc, Wc).contiguous()                      # (2*Nc,Hc,Wc)

        # ---- build Y (image target from fully-sampled ks_fs) ----
        img_fs = fastmri.ifft2c(ks_fs)                                      # (Nc,Hc,Wc,2)

        if self.target_mode == 'mag':
            # RSS magnitude (single target channel), regardless of coil handling
            mag = fastmri.complex_abs(img_fs)                               # (Nc,Hc,Wc)
            Y   = fastmri.rss(mag, dim=0).unsqueeze(0).contiguous()         # (1,Hc,Wc)
        elif self.target_mode == 'complex':
            if self.combine_coil:
                c = int(max(0, min(self.coil_index, Nc - 1)))
                Y = img_fs[c].permute(2, 0, 1).contiguous()                 # (2,Hc,Wc)
            else:
                Y_nc2 = img_fs.permute(0, 3, 1, 2).contiguous()             # (Nc,2,Hc,Wc)
                Y = Y_nc2.reshape(-1, Hc, Wc).contiguous()                  # (2*Nc,Hc,Wc)
        else:
            raise ValueError("target_mode must be 'mag' or 'complex'")

        return X, Y, M




def get_valid_sample(dataset, start=0):
    for idx in range(start, len(dataset)):
        sample = dataset[idx]
        if sample is not None:
            return idx, sample
    raise ValueError("No valid sample found.")

class DataTransform_WNet:
    def __init__(
        self,
        mask_func,
        img_size=320,
        flag_singlecoil=False,
    ):
        self.mask_func = mask_func
        self.img_size = img_size
        self.flag_singlecoil = flag_singlecoil

    def __call__(self, kspace, mask, target, data_attributes, filename, slice_num):
        if self.flag_singlecoil:
            kspace = kspace[None, ...]  # [H,W,2] to [1,H,W,2]

        # k-space, transform the data into appropriate format
        kspace = transforms.to_tensor(kspace)  # [Nc,H,W,2]

        # ====== Image reshaping ======
        # img space
        image_full = fastmri.ifft2c(kspace)  # [Nc,H,W,2]
        # center cropping
        image_full = transforms.complex_center_crop(image_full, [320, 320])  # [Nc,H,W,2]
        # resize img
        if self.img_size != 320:
            image_full = torch.einsum('nhwc->nchw', image_full)
            image_full = T.Resize(size=self.img_size)(image_full)
            image_full = torch.einsum('nchw->nhwc', image_full)
        # img to k-space
        kspace = fastmri.fft2c(image_full)  # [Nc,H,W,2]


        # ====== Fully-sampled ===
        # img space
        image_full = fastmri.complex_abs(image_full)  # [Nc,H,W]
        image_full = fastmri.rss(image_full, dim=0)  # [H,W]


        # ====== Under-sampled ======
        # apply mask
        if isinstance(self.mask_func, subsample.MaskFunc):
            masked_kspace, mask, _ = transforms.apply_mask(kspace, self.mask_func)  # mask [1,1,W,1]
            mask = mask.repeat(kspace.shape[0], kspace.shape[1], 1, kspace.shape[3])  # [Nc,H,W,2]
        else:
            masked_kspace, mask = apply_mask(kspace, self.mask_func)  # mask [1,H,W,1]
            mask = mask.repeat(kspace.shape[0], 1, 1, kspace.shape[3])  # [Nc,H,W,2]

        # kspace [B,Nc,H,W,2], mask [B,Nc,H,W,2], image [B,H,W]
        return masked_kspace, kspace, mask, image_full


class DataTransform_VarNet:
    def __init__(
        self,
        mask_func,
        img_size=320,
        flag_singlecoil=False,
    ):
        self.mask_func = mask_func
        self.img_size = img_size
        self.flag_singlecoil = flag_singlecoil

    def __call__(self, kspace, mask, target, data_attributes, filename, slice_num):
        if self.flag_singlecoil:
            kspace = kspace[None, ...]  # [H,W,2] to [1,H,W,2]

        # k-space, transform the data into appropriate format
        kspace = transforms.to_tensor(kspace)  # [Nc,H,W,2]

        # ====== Image reshaping ======
        # img space
        image_full = fastmri.ifft2c(kspace)  # [Nc,H,W,2]
        # center cropping
        image_full = transforms.complex_center_crop(image_full, [320, 320])  # [Nc,H,W,2]
        # resize img
        if self.img_size != 320:
            image_full = torch.einsum('nhwc->nchw', image_full)
            image_full = T.Resize(size=self.img_size)(image_full)
            image_full = torch.einsum('nchw->nhwc', image_full)
        # img to k-space
        kspace = fastmri.fft2c(image_full)  # [Nc,H,W,2]


        # ====== Fully-sampled ===
        # img space
        image_full = fastmri.complex_abs(image_full)  # [Nc,H,W]
        image_full = fastmri.rss(image_full, dim=0)  # [H,W]


        # ====== Under-sampled ======
        # apply mask
        if isinstance(self.mask_func, subsample.MaskFunc):
            masked_kspace, mask, _ = transforms.apply_mask(kspace, self.mask_func)  # mask [1,1,W,1]
            mask = mask.repeat(kspace.shape[0], kspace.shape[1], 1, kspace.shape[3])  # [Nc,H,W,2]
        else:
            masked_kspace, mask = apply_mask(kspace, self.mask_func)  # mask [1,H,W,1]
            mask = mask.repeat(kspace.shape[0], 1, 1, kspace.shape[3])  # [Nc,H,W,2]

        # kspace [B,Nc,H,W,2], mask [B,Nc,H,W,2], image [B,H,W]
        return masked_kspace, kspace, mask, image_full


def apply_mask(data, mask_func):
    '''
    data: [Nc,H,W,2]
    mask_func: return [Nc(1),H,W]
    '''
    # mask, _ = mask_func()
    mask_return = mask_func()
    if len(mask_return) == 1:
        mask = mask_func()
    else:
        mask, _ = mask_func()
    mask = torch.from_numpy(mask)
    # print(f"shape of mask: {mask.shape}")
    mask = mask[..., None]  # [Nc(1),H,W,1]
    masked_data = data * mask + 0.0  # the + 0.0 removes the sign of the zeros
    return masked_data, mask



class XAITransform:
    def __init__(
        self,
        img_size=320,
        combine_coil=True,
        flag_singlecoil=False,
    ):
        self.img_size = img_size
        self.combine_coil = combine_coil
        self.flag_singlecoil = flag_singlecoil
        if flag_singlecoil:
            self.combine_coil = True

    def __call__(self, kspace, mask, target, attrs, filename, slice_num):
        if self.flag_singlecoil:
            kspace = kspace[None, ...]  # [H,W,2] -> [1,H,W,2]

        # Convert to tensor
        kspace = transforms.to_tensor(kspace)  # [Nc,H,W,2]
        Nc, H, W, _ = kspace.shape

        # --- IFFT to image space ---
        image = fastmri.ifft2c(kspace)  # [Nc,H,W,2]

        # --- Center crop ---
        image = transforms.complex_center_crop(image, (self.img_size, self.img_size))  # [Nc,320,320,2]

        # --- Resize if needed ---
        if self.img_size != 320:
            image = torch.einsum("nhwc->nchw", image)  # [Nc,2,H,W]
            image = T.Resize(size=self.img_size)(image)
            image = torch.einsum("nchw->nhwc", image)  # back to [Nc,H,W,2]

        # --- FFT to return to k-space (only if needed for consistency)
        kspace = fastmri.fft2c(image)

        # --- Complex Abs ---
        image_abs = fastmri.complex_abs(image)  # [Nc,H,W]

        if self.combine_coil:
            image_rss = fastmri.rss(image_abs, dim=0)  # [H,W]
            image_rss = image_rss / (image_rss.max() + 1e-8)
            return image_rss.unsqueeze(0) # [1,H,W], [1,H,W]
        else:
            # Normalize each coil separately
            image_abs = image_abs / (image_abs.max() + 1e-8)
            return image_abs  # [Nc,H,W], [1,H,W]




