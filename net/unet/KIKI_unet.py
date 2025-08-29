import torch
import torch.nn as nn

import torch
import torch.nn as nn
from utils.kiki_helpers import DataConsist , fftshift2 , ifft2, fft2
from net.unet.unet_supermap import Block, ResnetBlock, WeightStandardizedConv2d, LinearAttention, Residual, PreNorm
# =========================
# Modern, device-safe utils
# =========================



def GenUnet():
    return 

def GenFcBlock(feat_list=[512, 1024, 1024, 512]):
    FC_blocks = []
    len_f = len(feat_list)
    for i in range(len_f - 2):
        FC_blocks += [nn.Linear(feat_list[i], feat_list[i + 1]),
                      nn.LeakyReLU(negative_slope=0.1, inplace=True)]
        
    return nn.Sequential(*FC_blocks, nn.Linear(feat_list[len_f - 2], feat_list[len_f - 1]))
    




def GenConvBlock(n_conv_layers, in_chan, out_chan, feature_maps):
    conv_block = [nn.Conv2d(in_chan, feature_maps, 3, 1, 1),
                  nn.LeakyReLU(negative_slope=0.1, inplace=True)]
    for _ in range(n_conv_layers - 2):
        conv_block += [nn.Conv2d(feature_maps, feature_maps, 3, 1, 1),
                       nn.LeakyReLU(negative_slope=0.1, inplace=True)]
    return nn.Sequential(*conv_block, nn.Conv2d(feature_maps, out_chan, 3, 1, 1))

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
            rec = fftshift2(rec)
            rec = ifft2(rec)
            rec = rec + self.conv_blocks_I[i](rec)
            rec = DataConsist(rec, kspace_us, mask)
            
            if i < self.n_iter - 1:
                # NO SCALING NEEDED with ortho normalization
                rec = fftshift2(fft2(rec))
        
        return rec