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

cmap_name = 'green'
colors = [(0, 'black'), (1, 'green')]
n_bins = [3]  # 指定颜色变化的数量
cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=256)

cmap_name = 'yellow'
colors = [(0, 'black'), (1, 'yellow')]
n_bins = [3]  # 指定颜色变化的数量
cmap2 = LinearSegmentedColormap.from_list(cmap_name, colors, N=256)

wavelength = 3 * 10**8 / (548.613 * 10**9)
k = 2 * math.pi / wavelength
x1 = 0
x2 = 0.06
y1 = 0
y2 = 0.06
x = [x1, x2]  
y = [y1, y2]  
M = 2048 // 4 
N = 2048 // 4 
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
    H = torch.exp(1j * k * d * torch.nan_to_num(torch.sqrt(1 - (wavelength * FX)**2 - (wavelength * FY)**2)))
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

LR_in = mpimg.imread(f'./2_5mm.png')
ref1 = mpimg.imread(f'./4_5mm.png')
ref2 = mpimg.imread(f'./6_5mm.png')
ref3 = mpimg.imread(f'./8_5mm.png')
ref4 = mpimg.imread(f'./10_5mm.png')
ref5 = mpimg.imread(f'./12_5mm.png')
bg1 = mpimg.imread(f'./bg1.png')
bg2 = mpimg.imread(f'./bg2.png')

# ref1 = ref1 / np.mean(ref1[:,:]) * np.mean(LR_in[:,:])
# ref2 = ref2 / np.mean(ref2[:,:]) * np.mean(LR_in[:,:])
# ref3 = ref3 / np.mean(ref3[:,:]) * np.mean(LR_in[:,:])
# ref4 = ref4 / np.mean(ref4[:,:]) * np.mean(LR_in[:,:])
# ref5 = ref5 / np.mean(ref5[:,:]) * np.mean(LR_in[:,:])

mi=0.5
LR_in = np.abs((LR_in-1*bg2)/(bg1-bg2))**mi
ref1 = np.abs((ref1-1*bg2)/(bg1-bg2))**mi
ref2 = np.abs((ref2-1*bg2)/(bg1-bg2))**mi
ref3 = np.abs((ref3-1*bg2)/(bg1-bg2))**mi
ref4 = np.abs((ref4-1*bg2)/(bg1-bg2))**mi
ref5 = np.abs((ref5-1*bg2)/(bg1-bg2))**mi

K=5
sigma=1
LR_in = cv2.GaussianBlur(LR_in, (K, K), sigma)
ref1 = cv2.GaussianBlur(ref1, (K, K), sigma)
ref2 = cv2.GaussianBlur(ref2, (K, K), sigma)
ref3 = cv2.GaussianBlur(ref3, (K, K), sigma)
ref4 = cv2.GaussianBlur(ref4, (K, K), sigma)
ref5 = cv2.GaussianBlur(ref5, (K, K), sigma)


LR_in = torch.from_numpy(LR_in).to('cuda')
ref1 = torch.from_numpy(ref1).to('cuda')
ref2 = torch.from_numpy(ref2).to('cuda')
ref3 = torch.from_numpy(ref3).to('cuda')
ref4 = torch.from_numpy(ref4).to('cuda')
ref5 = torch.from_numpy(ref5).to('cuda')


LR_in = LR_in[0:-1:4,0:-1:4]
ref1 = ref1[0:-1:4,0:-1:4].to('cuda')
ref2 = ref2[0:-1:4,0:-1:4].to('cuda')
ref3 = ref3[0:-1:4,0:-1:4].to('cuda')
ref4 = ref4[0:-1:4,0:-1:4].to('cuda')
ref5 = ref5[0:-1:4,0:-1:4].to('cuda')


plt.imshow(((LR_in).to('cpu').detach().numpy()),cmap=cmap)
plt.show() 
plt.imshow((ref1.to('cpu').detach().numpy()),cmap=cmap)
plt.show() 
plt.imshow((ref2.to('cpu').detach().numpy()),cmap=cmap)
plt.show() 
plt.imshow(((ref3).to('cpu').detach().numpy()),cmap=cmap)
plt.show() 
plt.imshow((ref4.to('cpu').detach().numpy()),cmap=cmap)
plt.show() 
plt.imshow(((ref5).to('cpu').detach().numpy()),cmap=cmap)
plt.show() 



model = unet_model.UNet()
model = model.to('cuda')
total_params = count_parameters(model)
print(f"total_params: {total_params}")
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10000)
num_epochs = 10000 #迭代次数
loss_trace = np.zeros((num_epochs))
diffraction = torch.zeros((1,1,M,M)).to('cuda')
Input = torch.zeros(1,1,M,M).to('cuda')
diffraction[0,0,:,:] = LR_in.to('cuda')    
    
recovery_t = torch.complex(torch.zeros((M,M)),torch.zeros((M,M))).to('cuda')
diffraction_copy = diffraction.clone()
epoch_t = 0
epoch_t0 = 0
z = 0.0025
delta=0.002
s1 = 0
e1 = -1
s2 = 0
e2 = -1
sw0 = 1
sw1 = 1
sw2 = 1
sw3 = 1
sw4 = 1
sw5 = 1
for epoch in range(num_epochs):
    Input[0,0,:,:] = diffraction[0,0,:,:]
    recovery_real = model(Input)[0,0,:,:]
    recovery_imag = model(Input)[0,1,:,:]
    recovery = torch.complex(recovery_real,recovery_imag)   
    recovery_t = recovery
    if int(epoch/200) > epoch_t0:
        recovery_t = recovery*(1 + (torch.rand(recovery.shape).to('cuda') - 0.5)/ 1e2)
        epoch_t0 = epoch/200
    loss = sw0*criterion(torch.abs(diffraction[0,0,s1:e1,s2:e2])/torch.mean(torch.abs(diffraction[0,0,s1:e1,s2:e2])), (torch.abs(Target(recovery_t,z))/torch.mean(torch.abs(Target(recovery_t,z))))[s1:e1,s2:e2]) + sw1*criterion(torch.abs(ref1[s1:e1,s2:e2]).float()/torch.mean(torch.abs(ref1[s1:e1,s2:e2])).float(), (torch.abs(Target(recovery_t,z+1*delta))/torch.mean(torch.abs(Target(recovery_t,z+1*delta))))[s1:e1,s2:e2]) + sw2*criterion(torch.abs(ref2[s1:e1,s2:e2]).float()/torch.mean(torch.abs(ref2[s1:e1,s2:e2])).float(), (torch.abs(Target(recovery_t,z+2*delta)))[s1:e1,s2:e2]/torch.mean(torch.abs(Target(recovery_t,z+2*delta)))) + sw3*(criterion(torch.abs(ref3[s1:e1,s2:e2]).float()/torch.mean(torch.abs(ref3[s1:e1,s2:e2])).float(), (torch.abs(Target(recovery_t,z+3*delta)))[s1:e1,s2:e2]/torch.mean(torch.abs(Target(recovery_t,z+3*delta)))) + sw4*criterion(torch.abs(ref4[s1:e1,s2:e2]).float()/torch.mean(torch.abs(ref4[s1:e1,s2:e2])).float(), (torch.abs(Target(recovery_t,z+4*delta)))[s1:e1,s2:e2])/torch.mean(torch.abs(Target(recovery_t,z+4*delta)))) + sw5*criterion(torch.abs(ref5[s1:e1,s2:e2]).float()/torch.mean(torch.abs(ref5[s1:e1,s2:e2])).float(), (torch.abs(Target(recovery_t,z+5*delta)))[s1:e1,s2:e2]/torch.mean(torch.abs(Target(recovery_t,z+5*delta)))) 
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()
    loss_trace[epoch] = loss.item()
    print(f"epoch：{epoch}, lr:{optimizer.param_groups[0]['lr']}, loss:{loss_trace[epoch]:.10f}")
    scheduler.step(loss_trace[epoch])
    if int(epoch/100) > epoch_t:
        out_t = torch.abs(recovery_t[:,:]).to('cpu').detach().numpy()
        plt.imshow(exposure.equalize_hist(out_t[s1:e1,s2:e2]),cmap='gray')
        plt.show()
        epoch_t = epoch/100

plt.imshow(exposure.equalize_hist((torch.abs(recovery_t[:,:])/torch.max(torch.abs(recovery_t[:,:])))[s1:e1,s2:e2].to('cpu').detach().numpy()),cmap=cmap)
plt.colorbar()
plt.show()
 
plt.imshow((torch.arctan(torch.abs(recovery_imag[:,:]/recovery_real[:,:])[s1:e1,s2:e2])).to('cpu').detach().numpy(),cmap=cmap)
plt.colorbar()
plt.show() 
