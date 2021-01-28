"""
Created on Thu Jan 28 23:35:27 2021

@author: Mohamad Sayegh & Elias Rached

"""


import matplotlib.pyplot as plt

from a_star import AStarPlanner

import numpy as np


def is_inside(circle_x, circle_y, radius, x, y):
    
    d = (x-circle_x)**2 + (y-circle_y)**2
    
    if (d <= radius**2):
        return 1;
    else:
        return 0;
    

def create_trajectory(x0, y0, xf, yf, psx, psy, rs):
        

    
    x0 = 100*x0
    y0 = 100*y0 
    
    x0 = int(x0)
    y0 = int(y0)
    
    xf = 100*xf
    yf = 100*yf

    psx = np.array(psx)
    psx = 100*psx
    
    psy = np.array(psy)
    psy = 100*psy
    
    rs  = np.array(rs)
    rs  = 100*rs
    
    #Specify size of square grid 
    n = 100

    #Create a nxn grid of zeros   
    grid = np.zeros((n,n), dtype = int)
    
    for i in range(0, len(rs)):
        radius = rs[i]
        circle_x = psx[i]
        circle_y = psy[i]
        for x in range(0, n):
            for y in range(0, n):
                b = is_inside(circle_x, circle_y, radius, x, y)
                if b == 1:
                    grid[x,y]=1
                    
                    
    grid[0,:] = 1
    grid[:,0] = 1        

    grid[-1,:] = 1
    grid[:,-1] = 1      
                
    occupied_positions_x = np.where(grid == 1)[0];
    occupied_positions_y = np.where(grid == 1)[1];
    plt.plot(occupied_positions_x, occupied_positions_y, 'b.', markersize = 1)
    
    
    ox = occupied_positions_x.tolist()
    oy = occupied_positions_y.tolist()
    
        
    grid_size = 1  #resolution
    robot_radius = 0
    
    
    a_star = AStarPlanner(ox, oy, grid_size, robot_radius)
    rx, ry = a_star.planning(x0, y0, xf, yf)
    
    rx = np.array(rx)
    ry = np.array(ry)
    
    rx = rx/100
    ry = ry/100
    
    ox = np.array(ox)
    oy = np.array(oy)
    
    ox = ox/100
    oy = oy/100
    
    x0 = x0/100
    y0 = y0/100
    
    xf = xf/100
    yf = yf/100
    
    plt.figure()
    plt.plot(ox, oy, ".k")
    plt.plot(x0, y0, "og")
    plt.plot(xf, yf, "xb")
    plt.grid(True)
    plt.axis("equal")
    plt.plot(rx, ry, "-r")
    plt.show()
    
    return rx,ry
    
    
    
        
        
        
        
        
        