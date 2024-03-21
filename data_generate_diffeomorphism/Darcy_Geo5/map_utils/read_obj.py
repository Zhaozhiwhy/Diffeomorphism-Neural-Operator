# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 14:24:30 2019

@author: Zzw
"""
"""
 read_obj - load a .obj file.

   [vertex,face,normal] = read_obj(filename);

   faces    : list of facesangle elements
   vertex  : node vertexinatates
   normal : normal vector list
"""
import numpy as np
def read_obj(filename):
    faces_s = []
    vertex_s = []
    normal_s = []
    faces = []
    vertex = []
    normal = []
    file = open(filename,"r")
    for line in file.readlines():
        if line[0] == 'v':
            line = line.replace("\n","")
            line = line.replace('v',"")
            vertex_s.append(line[1:].split(" ")) 
        elif line[0] == 'f':
            line = line.replace('f',"")
            line = line.replace("\n","")
            faces_s.append(line[1:].split(" "))
    
    for i in range(0,len(vertex_s)):
        vertex.append(list(map(lambda x:round(float(x),2),vertex_s[i])))
    for i in range(0,len(faces_s)):
        if(faces_s[i][0]!=faces_s[i][1]) and (faces_s[i][1]!=faces_s[i][2]) and (faces_s[i][0]!=faces_s[i][2]):
            faces.append(list(map(lambda x:round(float(x),2),faces_s[i])))
    vertex = np.array(vertex).T     
    faces = np.array(faces).T
    
    #vertex = (3, N) faces = [3, M]
    return vertex,faces
if __name__ == "__main__":           
    [X, F] = read_obj('./part_obj/1.obj')
    # F = F-1
    # F = np.unique(F,axis=1)
    # x_max = X[2].max()
    # x_min = X[2].min()
    # point_1 = np.where((X[0]==1) &( X[1]==28))[0][0]