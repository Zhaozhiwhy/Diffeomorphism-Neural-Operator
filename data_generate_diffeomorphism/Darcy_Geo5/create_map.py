# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 11:37:01 2019

@author: Zzw
"""
import numpy as np
import scipy as scp
import pylab as pyl
import matplotlib.pyplot as plt
import matplotlib
from map_utils.read_obj import read_obj
from nt_toolbox.general import *
from nt_toolbox.signal import *
from map_utils.cutting_line import cutting_line
import warnings
import matplotlib as mlp
warnings.filterwarnings('ignore')
from map_utils.compute_triang_interp import compute_triang_interp
from nt_toolbox.read_mesh import * 
from nt_toolbox.compute_boundary import *
from nt_toolbox.plot_mesh import * 
from matplotlib.pyplot import plot,savefig
import csv
import os
import multiprocessing

def calculate_angle(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return np.arctan2(y2 - y1, x2 - x1)


def find_corner_points(B):
    corner_points = []
    corner = []
    for i in range(0, len(B)):
        angle1 = calculate_angle(B[i - 1], B[i])
        if i == len(B)-1:
            angle2 = calculate_angle(B[i], B[0])
        else:
            angle2 = calculate_angle(B[i], B[i + 1])
        angle_difference = np.abs(angle2 - angle1)
        if angle_difference > np.pi / 6: 
            corner_points.append(B[i])
            corner.append(i)

    return corner_points,corner


def create_image(obj_name,sampling_size,objfilename,z,center=False):
    [X, F] = read_obj(obj_name)
    F = F-1
    F = np.unique(F,axis=1)
    if center==True:
        x_max = X[0].max()
        y_max = X[1].max()
        X[0]=X[0]-x_max/2
        X[1]=X[1]-y_max/2
        X[2]=X[2]-z

    x_min = X[0].min()
    x_max = X[0].max()
    y_min = X[1].min()
    y_max = X[1].max()
    z_max = X[2].max()
    
    point_4 = np.where((X[0]==x_min) & (X[2]==z_max))[0][0]
    point_3 = np.where((X[0]==x_max) & (X[2]==z_max))[0][0]
    
    buttomline = np.where((X[1]==y_min) & (X[2]==z_max))[0]
    
    BL = np.empty([3,np.shape(buttomline)[0]])
    for i in range(np.shape(buttomline)[0]):
        BL[:,i] = X[:,buttomline[i]]
    x_min = BL[0].min()
    x_max = BL[0].max()
    point_1 = np.where(BL[0]==x_min)[0][0]
    point_2 = np.where(BL[0]==x_max)[0][0]
    point_1 = np.where((X[0]==BL[0][point_1]) & (X[1]==BL[1][point_1]) & (X[2]==BL[2][point_1]))[0][0]
    point_2 = np.where((X[0]==BL[0][point_2]) & (X[1]==BL[1][point_2]) & (X[2]==BL[2][point_2]))[0][0]

    n = np.shape(X)[1]
    SB = compute_boundary(F,True,point_1) 

    B_points = np.transpose(X[0:2,SB])
    corners_points,corners = find_corner_points(B_points)

    
    t_1 = np.where(SB == point_1)[0][0]  
    t_2 = np.where(SB == point_2)[0][0]
    t_3 = np.where(SB == point_3)[0][0]
    t_4 = np.where(SB == point_4)[0][0]
    B = SB

    p = len(B)
    
    W = sparse.coo_matrix(np.zeros([n,n]))
    
    for i in range(3):
        i2 = (i+1)%3
        i3 = (i+2)%3
        F=F.astype('int64')
        u = X[:,F[i2,:]] - X[:,F[i,:]]
        v = X[:,F[i3,:]] - X[:,F[i,:]]
        # normalize the vectors
        u = u/np.tile(np.sqrt(np.sum(u**2,0)), (3,1))
        v = v/np.tile(np.sqrt(np.sum(v**2,0)), (3,1))
        # compute angles
        alpha = 1/np.tan(np.arccos(np.sum(u*v, 0)))
        alpha = np.maximum(alpha, 1e-2*np.ones(len(alpha))) #avoid degeneracy
        W = W + sparse.coo_matrix((alpha,(F[i2,:],F[i3,:])),(n,n))
        W = W + sparse.coo_matrix((alpha,(F[i3,:],F[i2,:])),(n,n))
        
    d = W.sum(0)
    D = sparse.diags(np.ravel(d),0)
    L = D - W
    p = len(B)

    L1 = np.copy(L.toarray()) ################################################################################
    L1[B,:] = 0
    for i in range(len(B)):
        L1[B[i], B[i]] = 1
    for i in range(L1.shape[0]):
        if L1[i][i]==0:
            L1[i][i]= 1e-6    

    from scipy.sparse import linalg
     

    sample_boundry_num = sampling_size*4
    tq_1 = (np.arange(1,sampling_size+1)-1)/(sampling_size-1)
    Z_sample = np.vstack((np.hstack((tq_1, tq_1*0 + 1, 1-tq_1, tq_1*0)),np.hstack((tq_1*0, tq_1, tq_1*0+1, 1-tq_1))))
    Z_B_sample_list= []
    for i in range(len(B)):
        Z_B_sample_list.append(Z_sample[:,int(i/len(B)*sampling_size*4)])
    Z_B_sample_list = np.array(Z_B_sample_list)
    # print(Z_B_sample_list*63+1)
    Z = np.transpose(Z_B_sample_list)
    ######################
    
    R = np.zeros([2,n])
    R[:,B] = Z
    
    Y = np.zeros([2,n])################################################################################
    Y[0,:] = linalg.spsolve(L1, R[0,:])
    Y[1,:] = linalg.spsolve(L1, R[1,:])

    E = np.linalg.norm(np.dot(L1,Y.T))/2

    q = sampling_size
    M = np.zeros([q,q,3])
    for i in range(3):
        M[:,:,i] = compute_triang_interp(F,Y,X[i,:],q)
    x_data = []
    y_data = []
    [x_data.extend(i) for i in M[:,:,0]]
    [y_data.extend(i) for i in M[:,:,1]]
    M_image = M.copy()
    for i in range(3):
        M_image[:,:,i] = (M_image[:,:,i] -M_image[:,:,i].min())/(M_image[:,:,i].max()-M_image[:,:,i].min())

    plt.clf()
    plt.scatter(x_data,y_data,s=3)
    plt.savefig("./part_img/"+objfilename+"_scatter.png")
    plt.clf()
    
    
    csvfile=open('./data/x_data.csv','a',newline='')
    writer = csv.writer(csvfile)
    writer.writerow(x_data)
    csvfile.close()  
    
    csvfile=open('./data/y_data.csv','a',newline='')
    writer = csv.writer(csvfile)
    writer.writerow(y_data)
    csvfile.close() 
    
    csvfile=open('./data/E.csv','a', encoding='utf-8',newline='')
    writer = csv.writer(csvfile)
    writer.writerow(str(E))
    csvfile.close() 
    plt.imsave("./part_img/"+objfilename+"_img.png",M_image)

   
def save_data(filename,data):
    with open(filename, 'w') as f:
        np.savetxt(f, data, delimiter=',')      
   
    
if __name__=="__main__":
    
    csvfile = open('./geo_data.csv')
    new_geo = []
    for num,row in enumerate(csvfile):
        if num>=0:
            line = row.replace("\n","")
            line = line.split(',')
            objfilename = int(float(line[0]))

            print(objfilename)
            objname = './part_obj/'+str(objfilename)+'.obj'
            create_image(objname,128,str(objfilename),0)
            
            new_line = [float(i) for i in line]

            new_geo.append(new_line)
            save_data('geo_new.csv',np.array(new_geo))
