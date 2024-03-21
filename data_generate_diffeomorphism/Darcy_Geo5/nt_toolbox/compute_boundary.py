import numpy as np
from scipy import sparse
from nt_toolbox.compute_edge_face_ring import *
import warnings
def compute_boundary(face,input_point,boundary_start_point):
    """
        compute_boundary - compute the vertices on the boundary of a 3D mesh
        
          boundary=compute_boundary(face);
        
          Copyright (c) 2007 Gabriel Peyre
    """    
    
    #if np.shape(face)[0] < np.shape(face)[1]:
    #    face = np.transpose(face)
    
    # compute edges (i,j) that are adjacent to only 1 face
    A = compute_edge_face_ring(face)
    i,j,v = sparse.find(A)
    
    #retrieve python indices starting from 0 and not from 1
    v[v > 0] = v[v > 0] -1 
    
    i = i[v == -1]
    j = j[v == -1]

    # build the boundary by traversing the edges
    if input_point == True:
        j = np.hstack((j[np.where(i==boundary_start_point)[0][0]:],j[0:np.where(i==boundary_start_point)[0][0]]))
        i = np.hstack((i[np.where(i==boundary_start_point)[0][0]:],i[0:np.where(i==boundary_start_point)[0][0]]))
        
        

        boundary = [i[0]] 
        i = i[1:]
        j = j[1:]
    else:
        
        boundary = [i[0]] 
        i = i[1:]
        j = j[1:]

    while not(len(i)==0):
        b = boundary[-1]
        I = np.where(i == b)[0]
        if len(I) == 0:
            I = np.where(j == b)[0]
            
            if len(I) == 0:
                warnings.warn('Problem with boundary')
                break;
        
            boundary = boundary + [i[I][0]]

        else:
            boundary = boundary + [j[I][0]]

        i = np.delete(i,I)
        j = np.delete(j,I)
    #print(boundary)
    #cyclic boundary
    #boundary = boundary + [boundary[0]]
    
    return boundary
