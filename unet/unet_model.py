""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F
from einops import rearrange

from .unet_parts import *
from .lambda_networks import *


class UNet(nn.Module):
    def __init__(self, cfg):
        super(UNet, self).__init__()
        self.n_channels = cfg.n_channels
        self.n_classes = cfg.n_classes
        self.bilinear = cfg.bilinear

        self.inc = DoubleConv(cfg, cfg.n_channels, 64)
        self.down1 = Down(cfg, 64, 128)
        self.down2 = Down(cfg, 128, 256)
        self.down3 = Down(cfg, 256, 512)
        self.pool = nn.MaxPool2d(2)
        factor = 2 if cfg.bilinear else 1
        self.down4 = Down(cfg, 512, 1024 // factor)
        self.up1 = Up(cfg, 1024, 512 // factor, cfg.uplambdaLayer[0], cfg.bilinear)
        self.up2 = Up(cfg, 512, 256 // factor, cfg.uplambdaLayer[1], cfg.bilinear)
        self.up3 = Up(cfg, 256, 128 // factor, cfg.uplambdaLayer[2], cfg.bilinear)
        self.up4 = Up(cfg, 128, 64, cfg.uplambdaLayer[3], cfg.bilinear)
        self.outc = OutConv(64, cfg.n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
