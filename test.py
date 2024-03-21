# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 20:01:46 2022

@author: Zz
"""
import csv
import numpy as np
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.tri as tri
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from Loss_function import LpLoss,count_params
import operator
from functools import reduce
from functools import partial
import random
import pandas as pd
from timeit import default_timer
from scipy.interpolate import griddata
from Adam import Adam
from scipy.spatial import Delaunay, ConvexHull
import open3d as o3d
from scipy.interpolate import LinearNDInterpolator
from matplotlib.colors import Normalize
import cv2
from sklearn.metrics import mutual_info_score
from skimage.metrics import structural_similarity as ssim
from matplotlib.colors import BoundaryNorm, ListedColormap
from ChromaPalette.chroma_palette import *
from matplotlib.colors import LinearSegmentedColormap

def normalize(data):

    normalized_data = data
    for i in range(data.shape[-1]):
        channel = data[:, :, i]
        min_val = torch.min(channel)
        max_val = torch.max(channel)
        normalized_data[:, :, i] = (channel - min_val) / (max_val - min_val)
    return normalized_data

def ncc_loss(data1, data2):

    data1_normalized = normalize(data1)
    data2_normalized = normalize(data2)

    data1_flat = data1_normalized.flatten()
    data2_flat = data2_normalized.flatten()
    
    mean1 = torch.mean(data1_flat)
    mean2 = torch.mean(data2_flat)
    
    std1 = torch.std(data1_flat)
    std2 = torch.std(data2_flat)
    
    ncc = torch.sum((data1_flat - mean1) * (data2_flat - mean2)) / (len(data1_flat) * std1 * std2)
    
    return ncc

def normalized_cross_correlation(x, y):
    # 计算均值

    mean_x = torch.mean(x)
    mean_y = torch.mean(y)

    centered_x = x - mean_x
    centered_y = y - mean_y

    numerator = torch.sum(centered_x * centered_y)


    denominator_x = torch.sqrt(torch.sum(centered_x ** 2))
    denominator_y = torch.sqrt(torch.sum(centered_y ** 2))

    ncc = numerator / (denominator_x * denominator_y + 1e-8) 

    return ncc


def readfile(path):
    f=open(path)
    rows=list(csv.reader(f))
    f.close()
    return rows

def openfile(filename,dataset):
    K=readfile('./'+dataset+'/'+filename+'.csv')
    for t in range(len(K)):
        K[t]=[float(V) for V in K[t]]
    # K=np.array(K)
    return K

def save_data(filename,data):
    with open(filename, 'w') as f:
        np.savetxt(f, data, delimiter=',')
def save_pig_loss(loss_train_all,loss_test_all,imgpath):
    
    fig = plt.figure(num=1, figsize=(12, 8),dpi=100)
    plt.plot(range(len(loss_train_all)),loss_train_all,label='train loss')
    plt.plot(range(len(loss_test_all)),loss_test_all,label='test loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss value")
    plt.legend()
    plt.savefig(imgpath)

def geo_data(S):
    x = []
    y = []
    
    for i in range(S):
        for j in range(S):
            x.append(i+1)
            y.append(j+1)
    return x,y

def L2Loss(model,alpha):
    l2_loss = torch.tensor(0.0,requires_grad = True)
    for name,parma in model.named_parameters():
        if 'bias' not in name:
            l2_loss = l2_loss + (0.5*alpha * torch.sum(torch.pow(parma.abs(),2)))
            # l2_loss = l2_loss + (0.5*alpha * torch.sum(torch.pow(parma,2)))
    return l2_loss

def load_data(ntrain,ntest,S): 
    
    train_C = np.nan_to_num(np.array(openfile('train_C',train_dataset_list[0])))
    train_x = np.array(openfile('train_x_data',train_dataset_list[0]))
    train_y = np.array(openfile('train_y_data',train_dataset_list[0]))
    train_U = np.nan_to_num(np.array(openfile('train_U',train_dataset_list[0])))
    if len(train_dataset_list)>1:
        for i in range(0,len(train_dataset_list)-1):
            train_C = np.vstack((train_C,np.nan_to_num(np.array(openfile('train_C',train_dataset_list[i+1])))))
            train_x = np.vstack((train_x,np.nan_to_num(np.array(openfile('train_x_data',train_dataset_list[i+1])))))
            train_y = np.vstack((train_y,np.nan_to_num(np.array(openfile('train_y_data',train_dataset_list[i+1])))))
            train_U = np.vstack((train_U,np.nan_to_num(np.array(openfile('train_U',train_dataset_list[i+1])))))
    ntrain = train_C.shape[0]
    
    
    train_C = np.reshape(train_C,(ntrain,128,128))
  
    train_b = np.zeros((128-2,128-2))
    train_b = np.pad(train_b,pad_width = 1,mode = 'constant',constant_values = 1)
    train_b = np.reshape(train_b,(1,128,128))
    
    train_b = train_b.repeat(ntrain,axis=0)
    
    train_x = np.reshape(train_x,(ntrain,S,S))
    train_y = np.reshape(train_y,(ntrain,S,S))
    train_b = np.reshape(train_b,(ntrain,S,S))

    
    train_x2 = np.multiply(train_x,train_b)
    train_y2 = np.multiply(train_y,train_b)
    
    train_U = np.reshape(train_U,(ntrain,S,S))
    
    train_C = np.expand_dims(train_C,axis=-1)
    train_x = np.expand_dims(train_x,axis=-1)
    train_y = np.expand_dims(train_y,axis=-1)
    train_x2 = np.expand_dims(train_x2,axis=-1)
    train_y2 = np.expand_dims(train_y2,axis=-1)
    train_U = np.expand_dims(train_U,axis=-1)
    train_b = np.expand_dims(train_b,axis=-1)

    train_a = np.concatenate((train_C,train_x,train_y,train_b),axis = 3)
    train_u = np.expand_dims(train_U,axis=-1)*10

    return train_a,train_u,train_a,train_u

def load_data2(ntest,S,x_bias,y_bias,scale=False): 
    test_x = np.array(openfile('test_x_data',test_dataset_list[0]))
    test_y = np.array(openfile('test_y_data',test_dataset_list[0]))
    test_C = np.nan_to_num(np.array(openfile('test_C',test_dataset_list[0])))
    test_U = np.nan_to_num(np.array(openfile('test_U',test_dataset_list[0])))
    if scale ==True:
        test_S = np.array(openfile('test_geo_new',test_dataset_list[0]))
    
    if len(test_dataset_list)>1:
        for i in range(0,len(test_dataset_list)-1):
            test_C = np.vstack((test_C,np.nan_to_num(np.array(openfile('test_C',test_dataset_list[i+1])))))
            test_x = np.vstack((test_x,np.nan_to_num(np.array(openfile('test_x_data',test_dataset_list[i+1])))))
            test_y = np.vstack((test_y,np.nan_to_num(np.array(openfile('test_y_data',test_dataset_list[i+1])))))
            test_U = np.vstack((test_U,np.nan_to_num(np.array(openfile('test_U',test_dataset_list[i+1])))))
            if scale == True:
                test_S = np.vstack((test_S,np.nan_to_num(np.array(openfile('test_geo_new',test_dataset_list[i+1])))))
    ntest = test_C.shape[0]
    test_C = np.reshape(test_C,(ntest,S,S))
    train_b = np.zeros((S-2,S-2))
    train_b = np.pad(train_b,pad_width = 1,mode = 'constant',constant_values = 1)
    train_b = np.reshape(train_b,(1,S,S))
    
    test_b = train_b.repeat(ntest,axis=0)
    
    test_x = np.reshape(test_x,(ntest,S,S))
    test_y = np.reshape(test_y,(ntest,S,S))
    
    test_x2 = np.multiply(test_x,test_b)
    test_y2 = np.multiply(test_y,test_b)

    test_U = np.reshape(test_U,(ntest,S,S))
    
    test_C = np.expand_dims(test_C,axis=-1)
    test_x = np.expand_dims(test_x,axis=-1)
    test_y = np.expand_dims(test_y,axis=-1)
    test_x2 = np.expand_dims(test_x2,axis=-1)
    test_y2 = np.expand_dims(test_y2,axis=-1)
    test_U = np.expand_dims(test_U,axis=-1)
    test_b = np.expand_dims(test_b,axis=-1)
    
    test_a = np.concatenate((test_C,test_x,test_y,test_x2,test_y2),axis = 3)
    test_a = np.concatenate((test_C,test_x,test_y,test_b),axis = 3)
    test_u = test_U*10

    if scale == True:
        test_S = test_S[:,-1]
    else:
        test_S = np.zeros(ntest)

    return test_a,test_u,test_S

################################################################
# fourier layer
################################################################

class SpectralConv2d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d_fast, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        #(20,20,28,28)
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width):
        super(FNO2d, self).__init__()
        # model = FNO2d(12, 12, 20).cuda()
        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: simga_x ximga_y x y G
        input shape: (batchsize, x=28, y=28, c=5)
        output: the solution of the next timestep
        output shape: (batchsize, x=28, y=28, c=2)
        modes = 5
        width = 20
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.fc0 = nn.Linear(4, self.width)
        # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        self.padding =1 # pad the domain if input is non-periodic
        self.conv0 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv4 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv5 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)

        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.w4 = nn.Conv2d(self.width, self.width, 1)
        self.w5 = nn.Conv2d(self.width, self.width, 1)
        
        self.b0 = nn.Conv2d(2, self.width, 1)
        self.b1 = nn.Conv2d(2, self.width, 1)
        self.b2 = nn.Conv2d(2, self.width, 1)
        self.b3 = nn.Conv2d(2, self.width, 1)
        self.b4 = nn.Conv2d(2, self.width, 1)
        self.b5 = nn.Conv2d(2, self.width, 1)
        
        self.c0 = nn.Conv2d(3, self.width, 1)
        self.c1 = nn.Conv2d(3, self.width, 1)
        self.c2 = nn.Conv2d(3, self.width, 1)
        self.c3 = nn.Conv2d(3, self.width, 1)
        self.c4 = nn.Conv2d(3, self.width, 1)
        self.c5 = nn.Conv2d(3, self.width, 1)

        self.fc1 = nn.Linear(self.width, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, 1)

    def forward(self, x):

        grid_mesh = x[:,:,:,1:4]
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        grid_mesh = grid_mesh.permute(0, 3, 1, 2)

        grid = self.get_grid([x.shape[0], x.shape[-2], x.shape[-1]], x.device).permute(0, 3, 1, 2)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x3 = self.b0(grid)
        x4 = self.c0(grid_mesh)
        x = x1 + x2 +x3 + x4
        x = F.gelu(x)
        
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x3 = self.b1(grid)
        x4 = self.c1(grid_mesh)
        x = x1 + x2 +x3 + x4
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x3 = self.b2(grid) 
        x4 = self.c2(grid_mesh)
        x = x1 + x2 +x3 + x4
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x3 = self.b3(grid) 
        x4 = self.c3(grid_mesh)
        x = x1 + x2 +x3 + x4
        x = F.gelu(x)
        
        x1 = self.conv4(x)
        x2 = self.w4(x)
        x3 = self.b4(grid) 
        x4 = self.c4(grid_mesh)
        x = x1 + x2 +x3 + x4
        x = F.gelu(x)
        
        x1 = self.conv5(x)
        x2 = self.w5(x)
        x3 = self.b5(grid)
        x4 = self.c5(grid_mesh)
        x = x1 + x2 +x3 + x4
        x = F.gelu(x)

        
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)

        x = self.fc3(x)
        x = F.gelu(x)
        x = self.fc4(x)
        x = self.fc5(x)
        return x
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)


################################################################
# configs
################################################################



global train_dataset,train_dataset_list,ntrain
train_dataset_list = ["data_geo5_r128"]
train_dataset = ''
for i in range(len(train_dataset_list)):
    train_dataset = train_dataset+train_dataset_list[i]+'_'
train_dataset = train_dataset+'to'


global test_dataset,test_dataset_list,ntest
test_dataset_list = ["data_geo6_r128"]
test_dataset = 'to_'
for i in range(len(test_dataset_list)):
    test_dataset = test_dataset+test_dataset_list[i]+'_'
test_dataset = test_dataset+''
print(test_dataset_list)


save_result_path = './result_'+train_dataset+'/'+test_dataset

path_model = 'model_'+train_dataset
imgpath = 'loss/'+path_model
path_model = 'model/'+path_model

if not os.path.exists(save_result_path):
    if not os.path.exists('./result_'+train_dataset):
        os.mkdir('./result_'+train_dataset)
    os.mkdir(save_result_path)
    
ntrain =0
ntest = 0
# 
if_train = False


S = 128


step = 1
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

colors = color_palette(name="Melody",N=11)
colors = [colors[6],colors[3]]
n_colors = len(colors)
color_map = LinearSegmentedColormap.from_list('custom_cmap', colors, N=n_colors*20)

if if_train == False:
    
    x_bias = 0
    y_bias = 0
    myloss = LpLoss(size_average=True)
    MSEloss = torch.nn.MSELoss(reduction='mean')
    test_a,test_u,test_S = load_data2(ntest,S,x_bias,y_bias,scale=False)

    test_a = torch.FloatTensor(test_a).to(device)
    test_u = torch.FloatTensor(test_u).to(device)
    
    model = torch.load('./model/model_data_geo5_r128_to',map_location='cuda:0')
 
    
    ntest = test_a.shape[0]

    train_a,train_u,_,_ = load_data(ntrain,ntest,S=128)

    train_a = torch.FloatTensor(train_a)
    train_u = torch.FloatTensor(train_u)
    ######
    
    L2_error = np.zeros(ntest)
    L2_error_max = np.zeros(ntest)
    L2_error_min = np.zeros(ntest)
    SSIM_list = np.zeros(ntest)
    MSE_list = np.zeros(ntest)
    sim_list = np.zeros(ntest)
    scale_list = np.zeros(ntest)

    for num in range(ntest):
            sim_sub_list =[]
            for j in range(train_a.shape[0]):
                
                g_image = test_a[num,:,:,[1,2]].cpu()
                f_image= train_a[j,:,:,[1,2]].cpu()
                
                sim = ncc_loss(f_image, g_image)

                sim_sub_list.append(sim)
            sim_list[num]= np.array(sim_sub_list).mean()

            #####
            
            XY = test_a[num,:,:,1:3].squeeze().detach().cpu().numpy()
    
            XY = XY.reshape(S*S,2)
            
            a = test_a[num,:,:,0].squeeze().detach().cpu().numpy().reshape(S*S)
            truth_u = test_u[num].squeeze().detach().cpu().numpy()
     
            test_a_num = torch.unsqueeze(test_a[num,:,:,:], 0)
            predect_u = model(test_a_num)
            
            pre_u = predect_u.squeeze().detach().cpu().numpy()
    
            truth_u = truth_u.reshape(S*S)
            pre_u = pre_u.reshape(S*S)
            
            loss_y_train = myloss(test_u[num].reshape(1,-1), predect_u.reshape(1,-1),type=False)
            mse_loss = MSEloss(test_u[num].reshape(1,-1),predect_u.reshape(1,-1))

            
            L2_error[num]=loss_y_train.mean()
            MSE_list[num]=mse_loss.mean()
            
            Z = np.random.random((XY.shape[0],1))*0.5
            XYZ = np.hstack([XY,Z])
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(XYZ)
            pcd.estimate_normals()
    
            alpha = 0.80
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    
            vertices = np.asarray(mesh.vertices)
            triangles = np.asarray(mesh.triangles)
            triang = tri.Triangulation(vertices[:, 0], vertices[:, 1], triangles)

            low = -1
            up = 8
            
            levels = np.linspace(low, up, 10) 
            
            cmap = ListedColormap(['white'] + plt.cm.viridis.colors)
            fig = plt.figure(num=1, figsize=(5,12),dpi=100) 
            plt.rcParams['font.family'] = 'Arial'
            plt.rc('font', size=20)
            
            plt.subplots_adjust(left=0.17, bottom=0.10, right=0.95, top=0.95,wspace=0.05, hspace=0.3)
            plt.clf()
            

            norm = Normalize(vmin=a.min(), vmax=a.max())
            ax3 = fig.add_subplot(3,1,1)
            plt.xlabel('x',labelpad=-5)
            plt.ylabel('y')
            
            # plt.axis('off')
            sampled_temperatures = griddata((XY[:, 0], XY[:, 1]), truth_u, (vertices[:,0],vertices[:,1]), method='cubic')
            trix2 = plt.tripcolor(triang, sampled_temperatures, shading='gouraud', cmap=plt.cm.rainbow,vmin=low,vmax=up)
            cbar = plt.colorbar(trix2, ax=ax3, orientation='vertical',norm=norm,ticks=[0, 4,8])
            # plt.xticks((0,(int(num/15)/10+0.5)*10))
            # plt.yticks((0,(int(num/15)/10+0.5)*10))
            # plt.xticks((0,10))
            # plt.yticks((0,10))
            # plt.xlim(0,10)
            # plt.ylim(0,10)
            plt.axis('off')
            
            # low = pre_u.min()-1
            # up = pre_u.max()+1
            
            ax4 = fig.add_subplot(3,1,2)
            plt.xlabel('x',labelpad=-10)
            plt.ylabel('y')
            
            # plt.axis('off')
            sampled_temperatures = griddata((XY[:, 0], XY[:, 1]), pre_u, (vertices[:,0],vertices[:,1]), method='cubic')
            trix3 = plt.tripcolor(triang, sampled_temperatures, shading='gouraud', cmap=plt.cm.rainbow,vmin=low,vmax=up)
            cbar = plt.colorbar(trix3, ax=ax4, orientation='vertical',norm=norm,ticks=[0, 4, 8])
            # plt.xticks((0,(int(num/15)/10+0.5)*10))
            # plt.yticks((0,(int(num/15)/10+0.5)*10))
            # plt.xticks((0,10))
            # plt.yticks((0,10))
            # plt.xlim(0,10)
            # plt.ylim(0,10)
            plt.axis('off')
            low = 0
            up = 2
            
            ax5 = fig.add_subplot(3,1,3)
            plt.xlabel('x')
            plt.ylabel('y')
            
            sampled_temperatures = griddata((XY[:, 0], XY[:, 1]), abs(truth_u-pre_u), (vertices[:,0],vertices[:,1]), method='cubic')
            # print(abs(truth_u-pre_u)/abs(truth_u+0.0001))
            tri4 = plt.tripcolor(triang, sampled_temperatures, shading='gouraud', cmap=plt.cm.Reds,vmin=low,vmax=1)
            norm = Normalize(vmin=0, vmax=1)
            cbar = plt.colorbar(tri4, ax=ax5, orientation='vertical',norm=norm,ticks=[0, 0.5, 1])
            # plt.xticks((0,(int(num/15)/10+0.5)*10))
            # plt.yticks((0,(int(num/15)/10+0.5)*10))
            # plt.xticks((0,10))
            # plt.yticks((0,10))
            # plt.xlim(0,10)
            # plt.ylim(0,10)
            plt.axis('off')
            
            SSIM =ssim(truth_u.reshape(S,S),pre_u.reshape(S,S))
            SSIM_list[num] = SSIM
            print("test_example:",num)
            print("Sim",sim_list[num],"**SSIM",SSIM, " **L2_Related_loss",loss_y_train.detach().cpu().numpy().mean()," **MSE_loss",mse_loss.detach().cpu().numpy().mean())
            
            plt.savefig(save_result_path+"/"+test_dataset+"_"+str(num)+'.png')
            
    print("L2_error:",L2_error.mean())
    print("SSIM:",SSIM_list.mean())
    print("MSE:",MSE_list.mean())
    L2_error_list = np.append(L2_error, L2_error.mean())
    SSIM_list = np.append(SSIM_list, SSIM_list.mean())
    MSE_list = np.append(SSIM_list, MSE_list.mean())
    sim_list = np.append(sim_list, sim_list.mean())
    scale_list = np.append(scale_list, scale_list.mean())
    save_data(save_result_path+'/test_error_L2_error.csv',L2_error_list)
    save_data(save_result_path+'/test_error_SSIM.csv',SSIM_list)
    save_data(save_result_path+'/test_error_MSE.csv',MSE_list)
    save_data(save_result_path+'/test_error_Sim.csv',sim_list)
    save_data(save_result_path+'/test_error_sacle.csv',scale_list)

    
    
    
torch.cuda.empty_cache()
    
        