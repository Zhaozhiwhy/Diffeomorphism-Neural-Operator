# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 15:58:09 2022

@author: Zz
"""
import torch
import csv
import numpy as np
import operator
from functools import reduce

def readfile(path):
    f=open(path)
    rows=list(csv.reader(f))
    f.close()
    return rows

def openfile(filename):
    K=readfile('./data/'+filename+'.csv')
    for t in range(len(K)):
        K[t]=[float(V) for V in K[t]]
    # K=np.array(K)
    return K

#loss function with rel/abs Lp loss
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert p > 0

        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = h*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        #相对损失
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        # print(diff_norms.shape)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)
        # print(y_norms.shape)
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)
        
        return diff_norms/y_norms

    def __call__(self, x, y, type=True):
        if type==True:
            return self.abs(x, y)
        else:
            return self.rel(x, y)
        # return self.abs(x, y)
    
# print the number of parameters
def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, 
                    list(p.size()+(2,) if p.is_complex() else p.size()))
    return c

def load_data(ntrain,ntest,S): 
    #残余应力数据
    train_C = np.array(openfile('train_C'))
    test_C = np.array(openfile('test_C'))
    
    train_C = np.reshape(train_C,(ntrain,S,S))
    test_C = np.reshape(test_C,(ntest,S,S))
    train_b = np.zeros((S-2,S-2))
    train_b = np.pad(train_b,pad_width = 1,mode = 'constant',constant_values = 1)
    train_b = np.reshape(train_b,(1,S,S))
    
    test_b = train_b.repeat(ntest,axis=0)
    train_b = train_b.repeat(ntrain,axis=0)
    #网格数据
    train_x = np.array(openfile('train_x_data'))
    train_y = np.array(openfile('train_y_data'))
    test_x = np.array(openfile('test_x_data'))
    test_y = np.array(openfile('test_y_data'))
    
    train_x = np.reshape(train_x,(ntrain,S,S))
    train_x = np.reshape(train_x,(ntrain,S,S))
    train_y = np.reshape(train_y,(ntrain,S,S))
    test_x = np.reshape(test_x,(ntest,S,S))
    test_y = np.reshape(test_y,(ntest,S,S))
    
    train_x2 = np.multiply(train_x,train_b)
    train_y2 = np.multiply(train_y,train_b)
    test_x2 = np.multiply(test_x,test_b)
    test_y2 = np.multiply(test_y,test_b)
    # train_x2 = np.delete(train_x,[63],axis=1)
    # x_temp = train_x[:,0,:]
    # x_temp = np.expand_dims(x_temp,axis=-2)
    # train_x2 = np.concatenate((x_temp,train_x2),axis=1)
    # train_x2 = train_x-train_x2
    
    # train_y2 = np.delete(train_y,[63],axis=2)
    # y_temp = train_y[:,:,0]
    # y_temp = np.expand_dims(y_temp,axis=-1)
    # train_y2 = np.concatenate((y_temp,train_y2),axis=2)
    # train_y2 = train_y-train_y2
    
    # test_x2 = np.delete(test_x,[63],axis=1)
    # x_temp = test_x[:,0,:]
    # x_temp = np.expand_dims(x_temp,axis=-2)
    # test_x2 = np.concatenate((x_temp,test_x2),axis=1)
    # test_x2 = test_x-test_x2
    
    # test_y2 = np.delete(test_y,[63],axis=2)
    # y_temp = test_y[:,:,0]
    # y_temp = np.expand_dims(y_temp,axis=-1)
    # test_y2 = np.concatenate((y_temp,test_y2),axis=2)
    # test_y2 = test_y-test_y2

    #几何数据和变形数据
    train_U = np.array(openfile('train_U'))*50
    test_U = np.array(openfile('test_U'))*50
    
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
    
    train_a = np.concatenate((train_C,train_x,train_y,train_x2,train_y2),axis = 3)
    train_u = np.expand_dims(train_U,axis=-1)
    test_a = np.concatenate((test_C,test_x,test_y,test_x2,test_y2),axis = 3)
    test_u = test_U
    return train_a,train_u,test_a,test_u

def load_data2(ntest,S): 
    #残余应力数据
    test_C = np.array(openfile('test_C'))
    
    test_C = np.reshape(test_C,(ntest,S,S))
    train_b = np.zeros((S-2,S-2))
    train_b = np.pad(train_b,pad_width = 1,mode = 'constant',constant_values = 1)
    train_b = np.reshape(train_b,(1,S,S))
    
    test_b = train_b.repeat(ntest,axis=0)
    #网格数据
    test_x = np.array(openfile('test_x_data'))
    test_y = np.array(openfile('test_y_data'))
    
    test_x = np.reshape(test_x,(ntest,S,S))
    test_y = np.reshape(test_y,(ntest,S,S))
    
    test_x2 = np.multiply(test_x,test_b)
    test_y2 = np.multiply(test_y,test_b)
    # test_x2 = np.delete(test_x,[63],axis=1)
    # x_temp = test_x[:,0,:]
    # x_temp = np.expand_dims(x_temp,axis=-2)
    # test_x2 = np.concatenate((x_temp,test_x2),axis=1)
    # test_x2 = test_x-test_x2
    
    # test_y2 = np.delete(test_y,[63],axis=2)
    # y_temp = test_y[:,:,0]
    # y_temp = np.expand_dims(y_temp,axis=-1)
    # test_y2 = np.concatenate((y_temp,test_y2),axis=2)
    # test_y2 = test_y-test_y2

    #几何数据和变形数据
    test_U = np.array(openfile('test_U'))*50
    
    test_U = np.reshape(test_U,(ntest,S,S))
    
    test_C = np.expand_dims(test_C,axis=-1)
    test_x = np.expand_dims(test_x,axis=-1)
    test_y = np.expand_dims(test_y,axis=-1)
    test_x2 = np.expand_dims(test_x2,axis=-1)
    test_y2 = np.expand_dims(test_y2,axis=-1)
    test_U = np.expand_dims(test_U,axis=-1)
    test_b = np.expand_dims(test_b,axis=-1)
    
    test_a = np.concatenate((test_C,test_x,test_y,test_x2,test_y2),axis = 3)
    test_u = test_U
    return test_a,test_u

def load_data3(ntest,S): 
    #残余应力数据
    test_C = np.array(openfile('train_C'))
    
    test_C = np.reshape(test_C,(ntest,S,S))
    train_b = np.zeros((S-2,S-2))
    train_b = np.pad(train_b,pad_width = 1,mode = 'constant',constant_values = 1)
    train_b = np.reshape(train_b,(1,S,S))
    
    test_b = train_b.repeat(ntest,axis=0)
    #网格数据
    test_x = np.array(openfile('train_x_data'))
    test_y = np.array(openfile('train_y_data'))
    
    test_x = np.reshape(test_x,(ntest,S,S))
    test_y = np.reshape(test_y,(ntest,S,S))
    
    # test_x2 = np.delete(test_x,[63],axis=1)
    # x_temp = test_x[:,0,:]
    # x_temp = np.expand_dims(x_temp,axis=-2)
    # test_x2 = np.concatenate((x_temp,test_x2),axis=1)
    # test_x2 = test_x-test_x2
    
    # test_y2 = np.delete(test_y,[63],axis=2)
    # y_temp = test_y[:,:,0]
    # y_temp = np.expand_dims(y_temp,axis=-1)
    # test_y2 = np.concatenate((y_temp,test_y2),axis=2)
    # test_y2 = test_y-test_y2

    #几何数据和变形数据
    test_U = np.array(openfile('train_U'))*50
    
    test_U = np.reshape(test_U,(ntest,S,S))
    
    test_C = np.expand_dims(test_C,axis=-1)
    test_x = np.expand_dims(test_x,axis=-1)
    test_y = np.expand_dims(test_y,axis=-1)
    # test_x2 = np.expand_dims(test_x2,axis=-1)
    # test_y2 = np.expand_dims(test_y2,axis=-1)
    test_U = np.expand_dims(test_U,axis=-1)
    test_b = np.expand_dims(test_b,axis=-1)
    
    test_a = np.concatenate((test_C,test_x,test_y),axis = 3)
    test_u = test_U
    return test_a,test_u