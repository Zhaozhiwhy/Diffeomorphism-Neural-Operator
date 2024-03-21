# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 20:01:46 2022
Darcy flow 2D
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
#import open3d as o3d
from scipy.interpolate import LinearNDInterpolator
from matplotlib.colors import Normalize

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
    
    
    test_x = np.array(openfile('test_x_data',test_dataset_list[0]))
    test_y = np.array(openfile('test_y_data',test_dataset_list[0]))
    test_C = np.nan_to_num(np.array(openfile('test_C',test_dataset_list[0])))
    test_U = np.nan_to_num(np.array(openfile('test_U',test_dataset_list[0])))
    
    
    if len(test_dataset_list)>1:
        for i in range(0,len(test_dataset_list)-1):
            test_C = np.vstack((test_C,np.nan_to_num(np.array(openfile('test_C',test_dataset_list[i+1])))))
            test_x = np.vstack((test_x,np.nan_to_num(np.array(openfile('test_x_data',test_dataset_list[i+1])))))
            test_y = np.vstack((test_y,np.nan_to_num(np.array(openfile('test_y_data',test_dataset_list[i+1])))))
            test_U = np.vstack((test_U,np.nan_to_num(np.array(openfile('test_U',test_dataset_list[i+1])))))
    ntest = test_C.shape[0]
        
    train_C = np.reshape(train_C,(ntrain,S,S))
  
    test_C = np.reshape(test_C,(ntest,S,S))
    train_b = np.zeros((S-2,S-2))
    train_b = np.pad(train_b,pad_width = 1,mode = 'constant',constant_values = 1)
    train_b = np.reshape(train_b,(1,S,S))
    
    test_b = train_b.repeat(ntest,axis=0)
    train_b = train_b.repeat(ntrain,axis=0)
    
    
    
    train_x = np.reshape(train_x,(ntrain,S,S))
    train_y = np.reshape(train_y,(ntrain,S,S))
    train_b = np.reshape(train_b,(ntrain,S,S))
    
    test_x = np.reshape(test_x,(ntest,S,S))
    test_y = np.reshape(test_y,(ntest,S,S))
    test_b = np.reshape(test_b,(ntest,S,S))
    
    train_x2 = np.multiply(train_x,train_b)
    train_y2 = np.multiply(train_y,train_b)
    test_x2 = np.multiply(test_x,test_b)
    test_y2 = np.multiply(test_y,test_b)
    
       
    train_U = np.reshape(train_U,(ntrain,S,S))
    test_U = np.reshape(test_U,(ntest,S,S))
    
    train_C = np.expand_dims(train_C,axis=-1)
    test_C = np.expand_dims(test_C,axis=-1)
    train_x = np.expand_dims(train_x,axis=-1)
    train_y = np.expand_dims(train_y,axis=-1)
    test_x = np.expand_dims(test_x,axis=-1)
    test_y = np.expand_dims(test_y,axis=-1)
    train_x2 = np.expand_dims(train_x2,axis=-1)
    train_y2 = np.expand_dims(train_y2,axis=-1)
    test_x2 = np.expand_dims(test_x2,axis=-1)
    test_y2 = np.expand_dims(test_y2,axis=-1)
    train_U = np.expand_dims(train_U,axis=-1)
    test_U = np.expand_dims(test_U,axis=-1)
    train_b = np.expand_dims(train_b,axis=-1)
    test_b = np.expand_dims(test_b,axis=-1)
    
    train_a = np.concatenate((train_C,train_x,train_y,train_b),axis = 3)

    train_u = np.expand_dims(train_U,axis=-1)*10
    
    test_a = np.concatenate((test_C,test_x,test_y,test_b),axis = 3)

    test_u = test_U*10
    return train_a,train_u,test_a,test_u

def load_data2(ntest,S,x_bias,y_bias): 

    test_x = np.array(openfile('test_x_data',test_dataset_list[0]))
    test_y = np.array(openfile('test_y_data',test_dataset_list[0]))
    test_C = np.nan_to_num(np.array(openfile('test_C',test_dataset_list[0])))
    test_U = np.nan_to_num(np.array(openfile('test_U',test_dataset_list[0])))
    
    if len(test_dataset_list)>1:
        for i in range(0,len(test_dataset_list)-1):
            test_C = np.vstack((test_C,np.nan_to_num(np.array(openfile('test_C',test_dataset_list[i+1])))))
            test_x = np.vstack((test_x,np.nan_to_num(np.array(openfile('test_x_data',test_dataset_list[i+1])))))
            test_y = np.vstack((test_y,np.nan_to_num(np.array(openfile('test_y_data',test_dataset_list[i+1])))))
            test_U = np.vstack((test_U,np.nan_to_num(np.array(openfile('test_U',test_dataset_list[i+1])))))
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
    
    test_a = np.concatenate((test_C,test_x,test_y,test_b),axis = 3)
    test_u = test_U*10
    return test_a,test_u

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
test_dataset_list = ["data_geo5_r128"]
test_dataset = 'to_'
for i in range(len(test_dataset_list)):
    test_dataset = test_dataset+test_dataset_list[i]+'_'
test_dataset = test_dataset+''



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

if_train = True
# 
modes =16
width =32

batch_size = 16
batch_size2 = batch_size
S = 128

epochs = 800
learning_rate =0.001
scheduler_step =60
scheduler_gamma = 0.98

step = 1
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if if_train == True:
    
    train_a,train_u,test_a,test_u = load_data(ntrain,ntest,S)
    
    train_a = torch.FloatTensor(train_a)
    train_u = torch.FloatTensor(train_u)
    
    test_a = torch.FloatTensor(test_a)
    test_u = torch.FloatTensor(test_u)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)
    
    ################################################################
    # training and evaluation
    ################################################################
    
    model = FNO2d(modes, modes, width).to(device)
    
    optimizer = Adam(model.parameters(), lr=learning_rate,weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    
    epo = []
    loss_train_all = []
    loss_test_all = []
    myloss = LpLoss(size_average=True)
    for ep in range(epochs):
        epo.append(ep)
        model.train()
        t1 = default_timer()
        train_l2_step = 0
        train_l2_full = 0
        for xx, yy in train_loader:

            loss = 0
            xx = xx.to(device)
            yy = yy.to(device)

            for i in range(xx.shape[0]):
                x_bias = (np.random.random()-0.5)*20
                y_bias = (np.random.random()-0.5)*20
                
                xx[i][:,:,1] = xx[i][:,:,1]+x_bias
                xx[i][:,:,2] = xx[i][:,:,2]+y_bias

            pre_y = model(xx)
            
            loss_y_train = myloss(pre_y.reshape(batch_size, -1), yy.reshape(batch_size, -1),type=False)

            l2_loss = L2Loss(model, 0.0005)
            loss = loss_y_train
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
        with torch.no_grad():
            for xx, yy in test_loader:
                test_loss = 0
                xx = xx.to(device)
                yy = yy.to(device)
                
                pre_y = model(xx)
                
                loss_y = myloss(pre_y.reshape(batch_size, -1), yy.reshape(batch_size, -1),type=False)

                test_loss = loss_y
        
        loss_train_all.append(float(loss_y_train.item()))
        loss_test_all.append(float(test_loss.item()))
        
        if ep%10==0:
            print("Lr:{}".format(optimizer.state_dict()['param_groups'][0]['lr']))
            print('epoch:',ep,' \ntrain_loss:',loss_y_train.item(),' \ntest_loss:',test_loss.item())
            torch.save(model, path_model)
        scheduler.step()
        optimizer.step()
    

    save_data('./result_'+train_dataset+'/train_loss.csv',np.array(loss_train_all))
    save_data('./result_'+train_dataset+'/test_loss.csv',np.array(loss_test_all))
    
    
    save_pig_loss(loss_train_all, loss_test_all, imgpath)
    
    num=random.randint(0,ntest-1)
    plotdata_a = test_a[num,:,:,:]
    plotdata_a = torch.unsqueeze(torch.FloatTensor(plotdata_a), 0)
    plotdata_in = plotdata_a
    plotdata_u = test_u[num,:,:,:]
    plotdata_u = torch.unsqueeze(torch.FloatTensor(plotdata_u), 0) 
    plotdata_a = plotdata_a.to(device)
    predect_u = model(plotdata_a)
    predect_u = predect_u.cpu()
    
    myloss = LpLoss(size_average=True)
    loss_t = myloss(plotdata_u.reshape(1, -1), predect_u.reshape(1, -1),type=False)
    
torch.cuda.empty_cache()
    
        