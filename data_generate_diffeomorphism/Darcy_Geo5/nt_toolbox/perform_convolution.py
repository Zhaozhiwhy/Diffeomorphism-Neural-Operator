# -*- coding: utf-8 -*-
"""
Created on Sun May  5 09:43:14 2019

@author: Zzw
"""
import numpy as np
import math
from scipy import signal
def perform_convolution(x,h,bound="sym"):
    """
        perform_convolution - compute convolution with centered filter.
        y = perform_convolution(x,h,bound);
        The filter 'h' is centred at 0 for odd
        length of the filter, and at 1/2 otherwise.
        This works either for 1D or 2D convolution.
        For 2D the matrix have to be square.
        'bound' is either 'per' (periodic extension)
        or 'sym' (symmetric extension).
        Copyright (c) 2004 Gabriel Peyre
    """
    
    if bound not in ["sym", "per"]:
        raise Exception('bound should be sym or per')

    if np.ndim(x) == 3 and np.shape(x)[2] < 4:
        #for color images
        y = x;
        for i in range(np.shape(x)[2]):
            y[:,:,i] = perform_convolution(x[:,:,i],h, bound)
        return y

    if np.ndim(x) == 3 and np.shape(x)[2] >= 4:
        raise Exception('Not yet implemented for 3D array, use smooth3 instead.')

    n = np.shape(x)
    p = np.shape(h)
    nd = np.ndim(x)

    if nd == 1:
        n = len(x)
        p = len(h)

    if bound == 'sym':

                #################################
        # symmetric boundary conditions #
        d1 = np.floor(np.asarray(p).astype(int)/2).astype('int64')  # padding before
        
        d2 = p - d1 - 1    			    # padding after

        if nd == 1:
            print(zzw)
        ################################# 1D #################################
            nx = len(x)
            xx = np.vstack((x[d1:-1:-1],x,x[nx-1:nx-d2-1:-1]))
            y = signal.convolve(xx,h)
            y = y[p:nx-p-1]

        elif nd == 2:
        ################################# 2D #################################
            #double symmetry
            nx,ny=np.shape(x)
            xx = x
            
            xx = np.vstack((xx[d1[0]-1::-1,:], xx, xx[nx-1:nx-d2[0]-1:-1,:]))
            #xx = np.vstack((xx[d1[0]:-1:-1,:], xx, xx[nx-1:nx-d2[0]-1:-1,:]))
            #xx = np.hstack((xx[:,d1[1]:-1:-1], xx, xx[:,ny-1:ny-d2[1]-1:-1]))
            xx = np.hstack((xx[:,d1[1]-1::-1], xx, xx[:,nx-1:nx-d2[0]-1:-1]))
           
            y = signal.convolve2d(xx,h,mode="full")
            
            y = y[(2*d1[0]):(2*d1[0]+n[0]), (2*d1[1]):(2*d1[1]+n[1])]
            
    else:

        ################################
        # periodic boundary conditions #

        if p > n:
            raise Exception('h filter should be shorter than x.')
        n = np.asarray(n)
        p = np.asarray(p)
        d = np.floor((p-1)/2.)
        
        if nd == 1:
            h = np.vstack((h[d:],np.vstack((np.zeros(n-p),h[:d]))))
            y = np.real(pyl.ifft(pyl.fft(x)*pyl.fft(h)))
        else:
            h = np.vstack((h[d[0]:,:],np.vstack((np.zeros([n[0]-p[0],p[1]]),h[:(d[0]),:]))))
            h = np.hstack((h[:,d[1]:],np.hstack((np.zeros([n[0],n[1]-p[1]]),h[:,:(d[1])]))))
            y = np.real(pyl.ifft2(pyl.fft2(x)*pyl.fft2(h)))
    return y