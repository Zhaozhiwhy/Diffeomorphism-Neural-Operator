# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 19:31:12 2019

@author: Zzw
"""
import numpy as np
from nt_toolbox import perform_convolution
import matplotlib.pyplot as plt
#%   M = compute_triang_interp(face,vertex,v,n);
#%
#%   n is the size of the image
#%   vertex is a (2,n) list of points, they
#%       are assumed to lie in [1,n]^2.
#%   face is a (3,p) list of triangular faces.
#%
#%   If options.remove_nan==1, remove NaN non interpolated values using a
#%   heat diffusion.
def compute_triang_interp(face,  vertex,  v, n):
    #                    face,vertex,v,n
    

    if face.shape[1]==3 and face.shape[0]!=3:
        face = face.T
    if vertex.shape[1]==2 and vertex.shape[0]!=2:
        vertex = vertex.T  
    # print(vertex[:,[1,0,2]])
    if round(vertex.flatten().min(),6)>=0 and round(vertex.flatten().max(),6)<=1:
        vertex= vertex*(n-1)+1
        # vertex= vertex*64
    nface = face.shape[1]
    face_sampling = 0
    if len(v)==nface:
        face_sampling = 1
    M = np.zeros(n*n)
    Mnb = np.zeros(n*n)
    for i in range(0,nface):

        T = face[:,i]

        P = vertex[:,T]

        V = v[T]

        if np.all(np.round(P[0], decimals=5) ==np.round(P[0, 0], decimals=5) ):
            P[1,1] = P[1,1]+0.3
            P[1,0] = P[1,0]-0.3
            P[1,2] = P[1,2]+0
            P[0,1] = P[0,1]-0.01
            P[0,0] = P[0,0]-0.01
            P[0,2] = P[0,2]+0.1

        if np.all(np.round(P[1], decimals=5) ==np.round(P[1, 0], decimals=5) ):
            P[1,1] = P[1,1]+0.01
            P[1,0] = P[1,0]-0.01
            P[1,2] = P[1,2]-0.01

        selx = np.arange(np.floor(P[0,:]).min(),np.round(P[0,:],decimals=5).max()+1)
        sely = np.arange(np.floor(P[1,:]).min(),np.round(P[1,:],decimals=5).max()+1)

        Y,X = np.meshgrid(sely,selx)

        pos = np.array([X.T.flatten(),Y.T.flatten()])
                
        p = pos.shape[1] 
        
        #number of points
        a = np.vstack(([1,1,1],P))

        inva = np.linalg.pinv(a)
       
        b = np.vstack((np.ones([1,p]),pos))

        c = np.dot(inva,b)
        c= np.round(c,10)
        I = np.where((c[0,:]>=-10*np.spacing(1)) & (c[1,:]>=-10*np.spacing(1)) & (c[2,:]>=-10*np.spacing(1)) )
        
        if I[0].shape[0]==0:
            continue
        
        pos = pos[:,I].reshape([-1,I[0].shape[0]])

        c = c[:,I].reshape([-1,I[0].shape[0]])
        
        I = np.where((pos[0,:]<=n) & (pos[1,:]>=0) & (pos[1,:]<=n) & (pos[1,:]>=0))
        
        pos = pos[:,I].reshape([-1,I[0].shape[0]]).astype('int64')
        c = c[:,I].reshape([-1,I[0].shape[0]])
        
        
        M=M.reshape([n,n])

        pos = pos-1

        J = np.ravel_multi_index([pos[1,:].flatten(),pos[0,:].flatten()],M.shape)
        
        M=M.flatten()

        if len(J)!=0:
            if face_sampling:
                M[J] = v[i]
            
            else:
               
                M[J] = M[J] + V[0]*c[0,:] +V[1]*c[1,:] +V[2]*c[2,:]
            Mnb[J] = Mnb[J]+1
        mm = Mnb.reshape([n,n]).T
        m = M.reshape([n,n]).T
  
    Mnb = Mnb.reshape([n,n]).T
    M = M.reshape([n,n]).T
    I = np.where(Mnb>0)
    M[I] = M[I]/Mnb[I]
    

    I = np.where(Mnb==0)
    
    M[I] = np.nan
   
    
    Mask = np.isnan(M)
    M[np.where(Mask>=1)] = np.mean(M[np.where(Mask==0)])
    M0=M
    h = np.ones([3,3])/9
    niter = 20
    for i in range(20):
        M = perform_convolution.perform_convolution(M,h)

        M[np.where(Mask==0)] = M0[np.where(Mask==0)]
    
    return M
        
        