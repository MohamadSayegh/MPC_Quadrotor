"""
Created on Thu Jan 28 23:35:27 2021

@author: Mohamad Sayegh & Elias Rached

"""

import numpy as np



def is_intersecting(x1, y1, radius1, x2, y2, radius2):
    
    d = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
    if (d <= radius1 + radius2):
        return 1;
    else:
        return 0;
    
    

def update_obs(X,Y,R,Dx,Dy):

    
    delta = 5e-3
    
    
    for i in range(0,5):
        
        nb_intersections = 0
        
        x  = X[i]
        y  = Y[i]
        r  = R[i]
        dx = Dx[i]
        dy = Dy[i]
        
        x = x + dx*delta
        y = y + dy*delta
        
        bool0 = is_intersecting(X[0], Y[0], R[0], x, y, r)
        bool1 = is_intersecting(X[1], Y[1], R[1], x, y, r)
        bool2 = is_intersecting(X[2], Y[2], R[2], x, y, r)
        bool3 = is_intersecting(X[3], Y[3], R[3], x, y, r)
        bool4 = is_intersecting(X[4], Y[4], R[4], x, y, r)
    
        nb_intersections = bool0 + bool1 + bool2 + bool3 + bool4
    

        if nb_intersections >= 2:
            
            dx = (-1)*dx
            dy = (-1)*dy
            
            if i < 4:
                Dx[i+1] = -0.5
                Dy[i+1] = -0.5
            
        else:   
            if x > (1 - r) or x < r:
                dx = (-1)*dx
                
            if y > (1 - r) or y < r:
                dy = (-1)*dy
                    



        
        X[i] = x + dx*delta
        Y[i] = y + dy*delta
        Dx[i] = dx
        Dy[i] = dy
        
    return X,Y,Dx,Dy
   
  
    