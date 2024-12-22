import torch
import unet_model
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import imageio
import numpy as np
import pprint
from PIL import Image
import pandas as pd
import torch.fft as f
import torch.nn.functional as F
import cv2
from skimage import exposure
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import time
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
import math
from skimage import io, color, img_as_float
from skimage.filters import gaussian, sobel
from matplotlib.colors import LinearSegmentedColormap

wavelength = 3 * 10**8 / (548.613 * 10**9)
k = 2 * math.pi / wavelength
x1 = 0
x2 = 0.06
y1 = 0
y2 = 0.06
x = [x1, x2]  # x1 和 x2 是坐标范围的起点和终点
y = [y1, y2]  # y1 和 y2 是坐标范围的起点和终点
# 定义 M 和 N，即在 x 和 y 方向上的点的数量
M = 3136//2  # 例如，在 x 方向上的点数
N = 3136//2  # 例如，在 y 方向上的点数
# 使用 numpy.linspace 创建等间隔的值
X = torch.linspace(-(x[1]-x[0])/2, (x[1]-x[0])/2, M, device='cuda')
Y = torch.linspace(-(y[1]-y[0])/2, (y[1]-y[0])/2, N, device='cuda')
fx = torch.linspace(-M/(2*(x[1]-x[0])), M/(2*(x[1]-x[0])), M, device='cuda')
fy = torch.linspace(-N/(2*(y[1]-y[0])), N/(2*(y[1]-y[0])), N, device='cuda')
# 使用 meshgrid 创建网格
FX, FY = torch.meshgrid(fx, fy)
X, Y = torch.meshgrid(X, Y)

H_sqf = torch.nan_to_num(torch.sqrt(1 - (wavelength * FX)**2 - (wavelength * FY)**2)/torch.sqrt(1 - (wavelength * FX)**2 - (wavelength * FY)**2))
H_sqf = H_sqf.to('cuda')

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def Space(x,d):
    H = torch.exp(1j * k * d * torch.nan_to_num(torch.sqrt(1 - (wavelength * FX)**2 - (wavelength * FY)**2),0))
    x_fft = f.fft2(x)   
    x_fft = f.fftshift(x_fft)
    x_h = H * x_fft
    x_h = f.ifftshift(x_h) 
    x_ifft = f.ifft2(x_h)
    return x_ifft

def Target(x,d):
    target = Space(x,d)
    return target

def padding(x, pad_width):
    pad_left = pad_width
    pad_right = pad_width * 2 - pad_left
    out = torch.nn.functional.pad(input=x, pad=(pad_left, pad_right, pad_left, pad_right), mode='constant', value=0)
    return out


resolution_card_tmp = mpimg.imread(f'./res_card.png')[:,:]
resolution_card = np.zeros((M,M))
resolution_card[8:3128,8:3128]=resolution_card_tmp[0:-1:2,0:-1:2]/np.max(resolution_card_tmp)
resolution_card = torch.from_numpy(resolution_card).to('cuda')


model = unet_model.UNet()
model = model.to('cuda')
total_params = count_parameters(model)
print(f"total_params: {total_params}")
criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)
#scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=0.000001) # 余弦退火
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=1000)
num_epochs = 2000 #迭代次数
loss_trace = np.zeros((num_epochs))

diffraction = torch.zeros((1,1,M,M)).to('cuda')
Input = torch.zeros(1,1,M,M).to('cuda')

cmap_name = 'yellow_green'
colors = [(0, 'black'), (1, 'green')]  # 0 映射到黄色，1 映射到绿色
n_bins = [3]  # 指定颜色变化的数量
cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=512)

LR_in = (torch.abs(Target(resolution_card,0.009))*1).to('cuda') #+ torch.normal(0.5, 2e-1*torch.max(torch.abs(resolution_card))) + (1e-2*torch.max(torch.abs(resolution_card)) * (torch.sin(2 * math.pi * 0.01 * torch.arange(torch.abs(resolution_card).numel()))+1).reshape(torch.abs(resolution_card).shape).to('cuda'))
Ref1 = (torch.abs(Target(resolution_card,0.013))*2).to('cuda') #+ torch.normal(0.5, 2e-1*torch.max(torch.abs(resolution_card))) + (1e-2*torch.max(torch.abs(resolution_card)) * (torch.sin(2 * math.pi * 0.01 * torch.arange(torch.abs(resolution_card).numel()))+1).reshape(torch.abs(resolution_card).shape).to('cuda'))
Ref2 = torch.abs(Target(resolution_card,0.02)).to('cuda') #+ torch.normal(0.5, 2e-1*torch.max(torch.abs(resolution_card))) + (1e-2*torch.max(torch.abs(resolution_card)) * (torch.sin(2 * math.pi * 0.01 * torch.arange(torch.abs(resolution_card).numel()))+1).reshape(torch.abs(resolution_card).shape).to('cuda'))
diffraction[0,0,:,:] = LR_in
    
plt.imshow(exposure.equalize_hist((diffraction[0,0,:,:]).to('cpu').detach().numpy()),cmap=cmap)
plt.colorbar()
plt.show() 
plt.imshow(exposure.equalize_hist((Ref1).to('cpu').detach().numpy()),cmap=cmap)
plt.show() 
plt.imshow(exposure.equalize_hist((Ref2).to('cpu').detach().numpy()),cmap=cmap)
plt.show() 


diffraction_copy = diffraction.clone()
epoch_t = 0
epoch_t0 = 0
recovery_t = torch.complex(torch.zeros((M,M)),torch.zeros((M,M))).to('cuda')
for epoch in range(num_epochs):
    Input[0,0,:,:] = diffraction[0,0,:,:]
    recovery_real = model(Input)[0,0,:,:]
    recovery_imag = model(Input)[0,1,:,:]
    recovery = torch.complex(recovery_real,recovery_imag)   
    recovery_t = recovery
    if int(epoch/500) > epoch_t0:
        recovery_t = recovery*(1 + torch.rand(recovery.shape).to('cuda') / 1e2)
        epoch_t0 = epoch/500
    loss = 1*criterion(torch.abs(diffraction[0,0,:,:])/torch.mean(torch.abs(diffraction[0,0,:,:])), torch.abs(Target(recovery_t,0.009))/torch.mean(torch.abs(Target(recovery_t,0.009)))) + 1*criterion(torch.abs(Ref1[:,:]).float()/torch.mean(torch.abs(Ref1[:,:])).float(), torch.abs(Target(recovery_t,0.013))/torch.mean(torch.abs(Target(recovery_t,0.013))))

    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()
    loss_trace[epoch] = loss.item()
    print(f"epoch：{epoch}, lr:{optimizer.param_groups[0]['lr']}, loss:{loss_trace[epoch]:.10f}")
    scheduler.step(loss_trace[epoch])
    if int(epoch/100) > epoch_t:
        out_t = torch.abs(recovery_t[:,:]).to('cpu').detach().numpy()
        plt.imshow(exposure.equalize_hist(out_t[:,:]),cmap=cmap)
        plt.show()
        epoch_t = epoch/100
    
    
plt.imshow(exposure.equalize_hist((torch.abs(recovery_t[:,:])/torch.max(torch.abs(recovery_t[:,:]))).to('cpu').detach().numpy()),cmap=cmap)
plt.colorbar()
plt.show() 

plt.imshow((torch.arctan(torch.abs(recovery_imag[:,:]/recovery_real[:,:]))).to('cpu').detach().numpy(),cmap=cmap)
plt.colorbar()
plt.show() 


