# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 12:29:20 2019

@author: Zzw
"""
import numpy as np
def cutting_line(X,F,line="long_1"):
    

    tmp=[]
    I = []
    #I 点的序号
    new_I = []
    X = X.T
    F_initial = F.copy()
    z_min = X[:,2].min()
    z_max = X[:,2].max()
    x_min = X[:,0].min()
    x_max = X[:,0].max()
    y_min = X[:,1].min()
    y_max = X[:,1].max()
    
    new_x = X
    
    
    z_mid = (z_max+z_min)/2
    #z_mid = z_max-2
    idx_open_x=[]
    # 切口的起始点
    idx_close_x=[]
    coundry_start_point=[]
    # 切口的终止点
    if line == "short":
        
        #寻找 短边 两个点
        for i,x in enumerate(X):
            if (x==[x_max,y_min,z_mid]).all():
                idx_open_x.append(i)
            if (x==[x_max,y_max,z_mid]).all():
                idx_close_x.append(i)
            if (x==[x_min,y_min,z_mid]).all():
                coundry_start_point.append(i)
        
        idx_open_x=idx_open_x[0]
        idx_close_x=idx_close_x[0]
        
        idx_mid=X.shape[0]
        idx_mid_list = []
        #寻找切口边上的点
        #找短边为连接边
        for i,x in enumerate(X):
            if x[2]==z_mid:
                #切口不包括链接线上的点
                if x[0]==x_max and (x[1]>y_min and x[1]<y_max):
                    True
                else:
                    tmp.append(x)
                    I.append(i)
                    idx_mid_list.append(idx_mid)
                    idx_mid+=1
    elif line=="long_0":
        #找长边为连接边
        for i,x in enumerate(X):
            if (x==[x_min,y_min,z_mid]).all():
                idx_open_x.append(i)
            if (x==[x_max,y_min,z_mid]).all():
                idx_close_x.append(i)
            if (x==[x_min,y_max,z_mid]).all():
                coundry_start_point.append(i)
        
        idx_open_x=idx_open_x[0]
        idx_close_x=idx_close_x[0]
        
        idx_mid=X.shape[0]
        idx_mid_list = []
        
        for i,x in enumerate(X):
            if x[2]==z_mid:
                #切口不包括链接线上的点
                if x[1]==y_min and (x[0]>x_min and x[0]<x_max):
                    True
                else:
                    tmp.append(x)
                    I.append(i)
                    idx_mid_list.append(idx_mid)
                    idx_mid+=1   
    elif line=="long_1":
        #找长边为连接边
        
        for i,x in enumerate(X):
            if (x==[x_min,y_max,z_mid]).all():
                idx_open_x.append(i)
            if (x==[x_max,y_max,z_mid]).all():
                idx_close_x.append(i)
            if (x==[x_min,y_min,z_mid]).all():
                coundry_start_point.append(i)
        
        idx_open_x=idx_open_x[0]
        idx_close_x=idx_close_x[0]
        
        idx_mid=X.shape[0]
        idx_mid_list = []
        
        for i,x in enumerate(X):
            if x[2]==z_mid:
                #切口不包括链接线上的点
                if x[1]==y_max and (x[0]>x_min and x[0]<x_max):
                    True
                else:
                    tmp.append(x)
                    I.append(i)
                    idx_mid_list.append(idx_mid)
                    idx_mid+=1
    elif line=="long_2":
        #找长边为连接边
        for i,x in enumerate(X):
            if (x==[x_min,y_max,z_max]).all():
                idx_open_x.append(i)
            if (x==[x_max,y_max,z_max]).all():
                idx_close_x.append(i)
            if (x==[x_min,y_min,z_max]).all():
                coundry_start_point.append(i)
        
        idx_open_x=idx_open_x[0]
        idx_close_x=idx_close_x[0]
        
        idx_mid=X.shape[0]
        idx_mid_list = []
        
        for i,x in enumerate(X):
            if x[2]==z_max:
                #切口不包括链接线上的点
                if x[1]==y_max and (x[0]>x_min and x[0]<x_max):
                    True
                else:
                    tmp.append(x)
                    I.append(i)
                    idx_mid_list.append(idx_mid)
                    idx_mid+=1                           
    new_x = np.append(X,np.array(tmp),axis=0)
    
    cutting_boundry=[]
    
    start_point = idx_open_x
    stop_point = idx_close_x
    new_I = I.copy()
    new_I.remove(idx_close_x)
    new_I.remove(idx_open_x)
    cutting_boundry.append(idx_open_x)
    F = F.T
    cutting_faces=[]
    faces_number=[]
    # 按顺序 找到割线上的点 以及面 只包含线面
    for i in range(0,len(new_I)):
        for num,j in enumerate(F):
            if j[0]==start_point or j[1]==start_point or j[2]==start_point:
                if j[0] in new_I:            
                    stop_tmp = start_point
                    start_tmp = j[0]    
                    cutting_faces.append(j)  
                    faces_number.append(num)
                elif j[1] in new_I:     
                    stop_tmp = start_point
                    start_tmp = j[1]
                    cutting_faces.append(j)
                    faces_number.append(num)
                elif j[2] in new_I:
                    stop_tmp = start_point
                    start_tmp = j[2]  
                    cutting_faces.append(j)
                    faces_number.append(num)
        if i==0:
            new_I.append(idx_close_x)
        if start_point!=start_tmp:   
            cutting_boundry.append(int(start_tmp))
        stop_point = stop_tmp
        start_point = start_tmp
        if start_tmp in new_I:
            new_I.remove(start_tmp)
     
    cutting_faces = np.array(cutting_faces)        
    
    I_tmp_list = cutting_boundry.copy()

    I_tmp_list.remove(idx_open_x)
    I_tmp_list.remove(idx_close_x)
    #找到割线以下的面（线面） 然后将新增加的面上面的 的 割线上的点替换为 新生成的点
    for i in faces_number: 
        for j in F[i]:
            if j not in cutting_boundry:           
                if new_x[int(j)][2]<=z_mid:
                    for m,f in enumerate(F[i]):
                        if f in I_tmp_list:
                            F[i][m]=idx_mid_list[np.where(I==f)[0][0]]   
    #将割线以下的点面上的点换成新的点                        
    for i,f in enumerate(F):
        f = f.astype('int64')
        if f[0] in I_tmp_list:
               
            if new_x[f[1]][2] <=z_mid and new_x[f[2]][2]<=z_mid:
              
                
                F[i][0] = idx_mid_list[np.where(I==f[0])[0][0]]
        elif f[1] in I_tmp_list:
            
            if new_x[f[0]][2] <=z_mid and new_x[f[2]][2]<=z_mid:
              
                F[i][1] = idx_mid_list[np.where(I==f[1])[0][0]]
        elif f[2] in I_tmp_list:
             
            if new_x[f[1]][2] <=z_mid and new_x[f[0]][2]<=z_mid:
       
                F[i][2] = idx_mid_list[np.where(I==f[2])[0][0]]
    return new_x.T,F.T,coundry_start_point