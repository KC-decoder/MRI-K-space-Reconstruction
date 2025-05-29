import os
import time
import h5py
import torch
import numpy as np
from utils import transforms as T
from torch import nn
from torch.nn import Conv2d, Sequential, InstanceNorm2d, ReLU, Dropout2d, Module, ModuleList, functional as F
from torch.utils.data import DataLoader
from torch.optim import RMSprop
from torch.optim.lr_scheduler import StepLR
from torchsummary import summary
from scipy.io import loadmat
from matplotlib import pyplot as plt
from utils import transforms as T


class UnetModel(Module):
    """
    PyTorch implementation of a U-Net model.
    """

    def __init__(self, in_chans, out_chans, chans, num_pool_layers, drop_prob, kernel_size):
        """
        Args:
            in_chans (int): Number of channels in the input to the U-Net model.
            out_chans (int): Number of channels in the output to the U-Net model.
            chans (int): Number of output channels of the first convolution layer.
            num_pool_layers (int): Number of down-sampling and up-sampling layers.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob
        self.kernel_size = kernel_size

        self.down_sample_layers = ModuleList([ConvBlock(in_chans, chans, drop_prob, kernel_size)])
        ch = chans
        for i in range(num_pool_layers - 1):
            self.down_sample_layers += [ConvBlock(ch, ch * 2, drop_prob, kernel_size)]
            ch *= 2
        self.conv = ConvBlock(ch, ch, drop_prob, kernel_size)

        self.up_sample_layers = ModuleList()
        for i in range(num_pool_layers - 1):
            self.up_sample_layers += [ConvBlock(ch * 2, ch // 2, drop_prob, kernel_size)]
            ch //= 2
        self.up_sample_layers += [ConvBlock(ch * 2, ch, drop_prob, kernel_size)]
        self.conv2 = Sequential(
            Conv2d(ch, ch // 2, kernel_size=1),
            Conv2d(ch // 2, out_chans, kernel_size=1),
            Conv2d(out_chans, out_chans, kernel_size=1),
        )

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        stack = []
        output = input
        # Apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.max_pool2d(output, kernel_size=2)

        output = self.conv(output)

        # Apply up-sampling layers
        for layer in self.up_sample_layers:
            output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=False)
            output = torch.cat([output, stack.pop()], dim=1)
            output = layer(output)
        return self.conv2(output)
    
    
    
class ConvBlock(Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, relu activation and dropout.
    """
    
    def __init__(self, in_chans, out_chans, drop_prob, kernel_size):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob
        self.kernel_size = kernel_size

        self.layers = Sequential(
            Conv2d(in_chans, out_chans, kernel_size=self.kernel_size),
            InstanceNorm2d(out_chans),
            ReLU(),
            Dropout2d(drop_prob),
            Conv2d(out_chans, out_chans, kernel_size=self.kernel_size),
            InstanceNorm2d(out_chans),
            ReLU(),
            Dropout2d(drop_prob)
        )

    def forward(self, input):
        """
        Args: input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        Returns: (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        return self.layers(input)

    # def __repr__(self):
    #     return ConvBlock(in_chans={self.in_chans}, out_chans={self.out_chans}, drop_prob={self.drop_prob})