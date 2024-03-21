# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 15:33:55 2023

@author: zhi-w
"""
def plt_mask(x,y):
    # cbar.set_clim(-1, 10)
    
    
    # x_range = np.linspace(0+x_bias, 10+x_bias, 100)  # 100个x坐标
    # y_range = np.linspace(0+y_bias, 10+y_bias, 100)  # 100个y坐标
    # x_grid, y_grid = np.meshgrid(x_range, y_range) 
    
    # low = truth_u.min()-1
    # up = truth_u.max()+3
    # levels = np.linspace(low, up, 10) 
    
    # ax1 = fig.add_subplot(1,4,1)
    # grid_z = griddata((XY[:, 0], XY[:, 1]), truth_u, (x_grid, y_grid), method='cubic')
    # if geometry=='L':
    #     mask = np.logical_and(x_grid < geo_information[num][1], y_grid > geo_information[num][2])
    # elif geometry=='-L':
    #     mask = np.logical_and(x_grid > geo_information[num][1], y_grid > geo_information[num][2])
    # elif geometry =='C':
    #     mask = np.logical_and(np.logical_and(x_grid > geo_information[num][1] , y_grid > 10-geo_information[num][3]),x_grid < geo_information[num][2])
    # elif geometry =='T':
    #     slope_left = -geo_information[num][5]/(geo_information[num][2]-geo_information[num][1])
    #     slope_right = geo_information[num][5]/(geo_information[num][4]-geo_information[num][3])
    #     b_left = (10+y_bias)-slope_left*(geo_information[num][1]+x_bias)
    #     b_right = (10+y_bias)-slope_right*(geo_information[num][4]+x_bias)
        
    #     mask = np.logical_and(y_grid >= (10-geo_information[num][5]+y_bias), y_grid <=10+y_bias)
    #     mask = np.logical_and(mask, x_grid >= (y_grid - b_left) / slope_left)
    #     mask = np.logical_and(mask, x_grid <= (y_grid - b_right) / slope_right)
        
    # grid_z = np.ma.masked_where(mask, grid_z)
    # plt.scatter(XY[:, 0], XY[:, 1],s = 5)
    
    # ax1 = fig.add_subplot(1,4,2)
    # grid_z = griddata((XY[:, 0], XY[:, 1]), truth_u, (x_grid, y_grid), method='cubic')
    # grid_z = np.ma.masked_where(mask, grid_z)
    # ax1.contourf(x_grid, y_grid, grid_z,levels = levels, cmap=plt.cm.jet)
    
    # ax2 = fig.add_subplot(1,4,3)
    # grid_z = griddata((XY[:, 0], XY[:, 1]), pre_u, (x_grid, y_grid), method='cubic')
    # grid_z = np.ma.masked_where(mask, grid_z)
    # ax2.contourf(x_grid, y_grid, grid_z, levels = levels,cmap=plt.cm.jet)
    
    # ax3 = fig.add_subplot(1,4,4)
    # grid_z = griddata((XY[:, 0], XY[:, 1]), truth_u-pre_u, (x_grid, y_grid), method='cubic')
    # grid_z = np.ma.masked_where(mask, grid_z)
    # contourf3 = ax3.contourf(x_grid, y_grid, grid_z, levels = levels,cmap=plt.cm.jet)
    # cbar = plt.colorbar(contourf3)
    return 0