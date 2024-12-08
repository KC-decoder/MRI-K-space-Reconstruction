
# IMPORT LIBRARIES
# general imports
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import yaml
import random
import logging
import h5py
from datetime import datetime


# python math imports
import numpy as np
import pandas as pd

# plotting imports
import matplotlib.pyplot as plt
import seaborn as sns

# META fastmri imports
import fastmri
from fastmri.data import transforms as T
from fastmri.data.mri_data import fetch_dir
from fastmri.data.transforms import UnetDataTransform
from fastmri.pl_modules import FastMriDataModule
from fastmri.pl_modules import UnetModule


# pytorch imports
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import torch.nn as nn
import torch.nn.functional as F

# adapted torch imports
from torchsummary import summary

# scipy imports
from scipy.ndimage import rotate


# distributed training imports
import torch.distributed as dist
import torch.multiprocessing as mp

 
# # load training module 

from utils.model_utils import double_conv , crop_tensor 
from models.complex_layers import (
    ComplexConv2d, ComplexBatchNorm2d, ComplexReLU, FrequencyPooling,
    ComplexUpsample, ComplexToReal, ComplexAttentionBlock, ComplexResidualBlock, ComplexConvTranspose2d
)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Contracting path (Encoder)
        self.dconv_down1 = double_conv(1, 64)   # Input: 1 channel (grayscale)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Expansive path (Decoder)
        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)

        # Final 1x1 convolution to output 2 channels (real + imaginary)
        self.conv_last = nn.Conv2d(64, 2, kernel_size=1, padding=1)

    def forward(self, x):
        # Encoder (downsampling)
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        # Decoder (upsampling)
        x = self.upsample(x)
        #print(f"Upsampled x shape: {x.shape}, conv3 shape: {conv3.shape}")  # Debugging
        x = crop_tensor(x, conv3)  # Crop to match conv3
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        #print(f"Upsampled x shape: {x.shape}, conv2 shape: {conv2.shape}")  # Debugging
        x = crop_tensor(x, conv2)  # Crop to match conv2
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        #print(f"Upsampled x shape: {x.shape}, conv1 shape: {conv1.shape}")  # Debugging
        x = crop_tensor(x, conv1)  # Crop to match conv1
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        # Output 2 channels (real + imaginary)
        out = self.conv_last(x)

        return out


class UNet_Modified(nn.Module):
    def __init__(self):
        super(UNet_Modified, self).__init__()

        # Encoder (Downsampling path)
        self.enc_conv1 = self.double_conv(in_channels=2, out_channels=64)  # Input has 2 channels (real & imaginary)
        self.enc_conv2 = self.double_conv(in_channels=64, out_channels=128)
        self.enc_conv3 = self.double_conv(in_channels=128, out_channels=256)
        self.enc_conv4 = self.double_conv(in_channels=256, out_channels=512)
        self.enc_conv5 = self.double_conv(in_channels=512, out_channels=1024)

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Decoder (Upsampling path)
        self.up_trans1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.dec_conv1 = self.double_conv(in_channels=1024, out_channels=512)

        self.up_trans2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.dec_conv2 = self.double_conv(in_channels=512, out_channels=256)

        self.up_trans3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2, output_padding=1)
        self.dec_conv3 = self.double_conv(in_channels=256, out_channels=128)

        self.up_trans4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.dec_conv4 = self.double_conv(in_channels=128, out_channels=64)

        # Final output layer with 1 channel
        self.out = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1)

    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoding path
        enc1 = self.enc_conv1(x)
        enc2 = self.enc_conv2(self.pool(enc1))
        enc3 = self.enc_conv3(self.pool(enc2))
        #print(enc3.shape)
        enc4 = self.enc_conv4(self.pool(enc3))
        enc5 = self.enc_conv5(self.pool(enc4))

        # Decoding path
        dec1 = self.dec_conv1(torch.cat([enc4, self.up_trans1(enc5)], dim=1))
    
        # Crop `self.up_trans2(dec1)` to match `enc3` before concatenation
        upsampled_dec1 = self.up_trans2(dec1)
        cropped_dec1 = crop_tensor(upsampled_dec1, enc3)
        #print(cropped_dec1.shape)
        dec2 = self.dec_conv2(torch.cat([enc3, cropped_dec1], dim=1))
        

        # Use crop_tensor to match dimensions for concatenation
        dec3 = self.up_trans3(dec2)
        dec3 = crop_tensor(dec3, enc2)
        dec3 = self.dec_conv3(torch.cat([enc2, dec3], dim=1))

        dec4 = self.up_trans4(dec3)
        dec4 = crop_tensor(dec4, enc1)
        dec4 = self.dec_conv4(torch.cat([enc1, dec4], dim=1))

        # Final output layer
        output = self.out(dec4)
        output = crop_tensor(output, x) if output.shape[2:] != x.shape[2:] else output
        return output
    






class CUNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, features=32):
        super(CUNet, self).__init__()

        # Encoder Path
        self.encoder1 = ComplexResidualBlock(in_channels, features)        # 2 -> 32
        self.encoder2 = ComplexResidualBlock(features, features * 2)      # 32 -> 64
        self.encoder3 = ComplexResidualBlock(features * 2, features * 4)  # 64 -> 128

        # Bottleneck
        self.bottleneck = ComplexResidualBlock(features * 4, features * 8)  # 128 -> 256

        # Decoder Path
        self.decoder3 = ComplexResidualBlock(features * 8, features * 4)    # 256 -> 128
        self.decoder2 = ComplexResidualBlock(features * 4, features * 2)    # 128 -> 64
        self.decoder1 = ComplexResidualBlock(features * 2, features)        # 64 -> 32

        # Attention Blocks for Skip Connections
        self.attention3 = ComplexAttentionBlock(features * 4, features * 4, features * 2)
        self.attention2 = ComplexAttentionBlock(features * 2, features * 2, features)
        self.attention1 = ComplexAttentionBlock(features, features, features // 2)

        # Final Output Layer
        self.final_conv = ComplexConv2d(features, out_channels, kernel_size=1)
        self.complex_to_real = ComplexToReal()

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)            # [B, 32, H, W]
        e2 = self.encoder2(e1)           # [B, 64, H, W]
        e3 = self.encoder3(e2)           # [B, 128, H, W]

        # Bottleneck
        b = self.bottleneck(e3)          # [B, 256, H, W]

        # Decoder with Attention and Skip Connections
        d3 = self.decoder3(torch.cat([self.attention3(e3, b), b], dim=1))  # [B, 128, H, W]
        d2 = self.decoder2(torch.cat([self.attention2(e2, d3), d3], dim=1))  # [B, 64, H, W]
        d1 = self.decoder1(torch.cat([self.attention1(e1, d2), d2], dim=1))  # [B, 32, H, W]

        # Final Convolution
        output = self.final_conv(d1)
        return self.complex_to_real(output)