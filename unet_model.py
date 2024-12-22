# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 13:01:49 2024

@author: Admin
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义一个卷积块，包含卷积层、批归一化和Leaky ReLU
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.01)

    def forward(self, x):
        return self.leaky_relu(self.bn(self.conv(x)))

# 下采样模块，包含两个卷积块和最大池化
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x, self.pool(x)

# 上采样模块，包含转置卷积、跳跃连接和两个卷积块
class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.upconv(x1)
        # 拼接跳跃连接
        x = torch.cat([x2, x1], dim=1)
        x = self.conv1(x)
        return self.conv2(x)

# 定义完整的 U-Net 模型
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # 下采样部分
        self.down1 = Down(1, 16)
        self.down2 = Down(16, 32)
        self.down3 = Down(32, 64)
        self.down4 = Down(64, 128)
        self.down5 = Down(128, 256)
        
        # 上采样部分
        self.up1 = Up(256, 128)
        self.up2 = Up(128, 64)
        self.up3 = Up(64, 32)
        self.up4 = Up(32, 16)

        # 最后一层输出
        self.final_conv = nn.Conv2d(16, 2, kernel_size=1)

    def forward(self, x):
        # Encoder 部分
        x1, x = self.down1(x)
        x2, x = self.down2(x)
        x3, x = self.down3(x)
        x4, x = self.down4(x)
        x5, _ = self.down5(x)
        
        # Decoder 部分
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # 输出层
        return self.final_conv(x)