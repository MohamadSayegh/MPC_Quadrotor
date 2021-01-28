"""
Created on Thu Jan 28 23:30:14 2021

@author: Mohamad Sayegh & Elias Rached

"""


from rockit import *
from rockit import FreeTime, MultipleShooting, Ocp
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi, cos, sin, tan, square
from casadi import vertcat, horzcat, sumsqr, Function, exp, vcat
from function_create_trajectory import create_trajectory
from update_obstacles_position import update_obs



#--------------- Problem parameters-------------------------------------

Nsim    = 100            # how much samples to simulate in total (affect sampling time?)
nx      = 8              # the system is composed of 8 states
nu      = 4              # the system has 4 control inputs
N       = 5              # number of control intervals = the horizon for multipls shooting
dt      = 0.05            # time interval


xf = 0.5
yf = 0.6
zf = 0.5

#-------------------- Logging variables---------------------------------


x_hist         = np.zeros((Nsim+1, N+1))
y_hist         = np.zeros((Nsim+1, N+1))
z_hist         = np.zeros((Nsim+1, N+1))
phi_hist       = np.zeros((Nsim+1, N+1))
vx_hist        = np.zeros((Nsim+1, N+1))
vy_hist        = np.zeros((Nsim+1, N+1))
vz_hist        = np.zeros((Nsim+1, N+1))
vphi_hist      = np.zeros((Nsim+1, N+1))

ux_hist         = np.zeros((Nsim+1, N+1))
uy_hist         = np.zeros((Nsim+1, N+1))
uz_hist         = np.zeros((Nsim+1, N+1))
uphi_hist       = np.zeros((Nsim+1, N+1))


#------------ initialize OCP -------------

ocp = Ocp(T = N*dt)


#------------ drone model from reference paper-------------------------

# drone model from reference paper
# model constants
k_x     = 1
k_y     = 1
k_z     = 1
k_phi   = pi/180
tau_x   = 0.8355
tau_y   = 0.7701
tau_z   = 0.5013
tau_phi = 0.5142

#Define states
x     = ocp.state()
y     = ocp.state()
z     = ocp.state()
phi   = ocp.state()
vx    = ocp.state()
vy    = ocp.state()
vz    = ocp.state()
vphi  = ocp.state()

#Defince controls
ux    = ocp.control()
uy    = ocp.control()
uz    = ocp.control()
uphi  = ocp.control()


#Specify ODEs
ocp.set_der(x   ,   vx*cos(phi) - vy*sin(phi))
ocp.set_der(y   ,   vx*sin(phi) + vy*cos(phi))
ocp.set_der(z   ,   vz)
ocp.set_der(phi ,   vphi)
ocp.set_der(vx  ,   (-vx + k_x*ux)/tau_x)
ocp.set_der(vy  ,   (-vy + k_y*uy)/tau_y)
ocp.set_der(vz  ,   (-vz + k_z*uz)/tau_z)
ocp.set_der(vphi,   (-vphi + k_phi*uphi)/tau_phi)



#------------------------------- Control constraints ----------------------

ocp.subject_to(-1 <= (ux    <= 1))
ocp.subject_to(-1 <= (uy    <= 1))
ocp.subject_to(-1 <= (uz    <= 1))
ocp.subject_to(-1 <= (uphi  <= 1))



# ------------------------ Add obstacles -----------------------------------


#round obstacles
p0 = ocp.parameter(2)
x0 = 0.2
y0 = 0.3
p0_coord = vertcat(x0,y0)
ocp.set_value(p0, p0_coord)
r0 = 0.15          

p1 = ocp.parameter(2)
x1 = 0.2
y1 = 0.8
p1_coord = vertcat(x1,y1)
ocp.set_value(p1, p1_coord)
r1 = 0.15

p2 = ocp.parameter(2)
x2 = 0.5
y2 = 0.3
p2_coord = vertcat(x2,y2)
ocp.set_value(p2, p2_coord)
r2 = 0.05

p3 = ocp.parameter(2)
x3 = 0.8
y3 = 0.2
p3_coord = vertcat(x3,y3)       
ocp.set_value(p3, p3_coord)
r3 = 0.05

p4 = ocp.parameter(2)
x4 = 0.8
y4 = 0.8
p4_coord = vertcat(x4,y4)        
ocp.set_value(p4, p4_coord)
r4 = 0.1


R = [r0,r1,r2,r3,r4]

p = vertcat(x,y,z)             # a point in 3D


X = [x0,x1,x2,x3,x4]
Y = [y0,y1,y2,y3,y4]
R = [r0,r1,r2,r3,r4]
Dx = [1 ,1 ,1 ,1 ,1 ]
Dy = [1 ,1 ,1 ,1 ,1 ]

def plotxy(xf,yf,p0_coord, p1_coord, p2_coord, R):
    
    
    r0 = R[0]
    r1 = R[1]
    r2 = R[2]
    
    #x-y plot
    fig = plt.figure(dpi = 300)
    ax = fig.add_subplot(111)

    plt.xlabel('x pos [m]')
    plt.ylabel('y pos [m]')
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.title('solution in x,y')
    ax.set_aspect('equal', adjustable='box')
    
    ts = np.linspace(0,2*pi,1000)
    plt.plot(p0_coord[0]+r0*cos(ts),p0_coord[1]+r0*sin(ts),'r-')
    plt.plot(p1_coord[0]+r1*cos(ts),p1_coord[1]+r1*sin(ts),'b-')
    plt.plot(p2_coord[0]+r2*cos(ts),p2_coord[1]+r2*sin(ts),'g-')
    plt.plot(p3_coord[0]+r3*cos(ts),p3_coord[1]+r3*sin(ts),'c-')
    plt.plot(p4_coord[0]+r4*cos(ts),p4_coord[1]+r4*sin(ts),'k-')
    plt.plot(xf,yf,'ro', markersize = 10)

  
    plt.show(block=True)


i = 0


obs_hist_0  = np.zeros((Nsim+1, 3))
obs_hist_1  = np.zeros((Nsim+1, 3))
obs_hist_2  = np.zeros((Nsim+1, 3))
obs_hist_3  = np.zeros((Nsim+1, 3))
obs_hist_4  = np.zeros((Nsim+1, 3))


while True:
    
    print("timestep", i+1, "of", Nsim)
    
    plotxy(xf,yf,p0_coord, p1_coord, p2_coord, R)

    
    #---------------- dynamic obstacles    

    X,Y,Dx,Dy =  update_obs(X,Y,R,Dx,Dy)
    
    p0_coord = vertcat(X[0],Y[0])
    ocp.set_value(p0, p0_coord)
    
    p1_coord = vertcat(X[1],Y[1])
    ocp.set_value(p1, p1_coord)

    p2_coord = vertcat(X[2],Y[2])
    ocp.set_value(p2, p2_coord)
    
    p3_coord = vertcat(X[3],Y[3])
    ocp.set_value(p3, p3_coord)
    
    p4_coord = vertcat(X[4],Y[4])
    ocp.set_value(p4, p4_coord)
    
    x0 = X[0]
    x1 = X[1]
    x2 = X[2]
    x3 = X[3]
    x4 = X[4]
    
    
    y0 = Y[0]
    y1 = Y[1]
    y2 = Y[2]
    y3 = Y[3]
    y4 = Y[4]
    
    obs_hist_0[i,0] = x0
    obs_hist_0[i,1] = y0
    obs_hist_0[i,2] = r0
    
    obs_hist_1[i,0] = x1
    obs_hist_1[i,1] = y1
    obs_hist_1[i,2] = r1
    
    obs_hist_2[i,0] = x2
    obs_hist_2[i,1] = y2
    obs_hist_2[i,2] = r2
    
    obs_hist_3[i,0] = x3
    obs_hist_3[i,1] = y3
    obs_hist_3[i,2] = r3
    
    obs_hist_4[i,0] = x4
    obs_hist_4[i,1] = y4
    obs_hist_4[i,2] = r4
    
    
    i = i + 1
    if i == Nsim :
        break




#------------------------ animation

import matplotlib.animation as animation


length = i

fig, ax  = plt.subplots(dpi = 300)

ax = plt.xlabel('x [m]')
ax = plt.ylabel('y [m]')
ax = plt.title('MPC solution')

ts = np.linspace(0,2*pi,1000)

ax = plt.axis([0,1,0,1])

P0  = plt.plot(xf, yf ,'o', markersize = 10)
O1, = plt.plot([], [] ,'g-')
O2, = plt.plot([], [] ,'k-')
O3, = plt.plot([], [] ,'c-')
O4, = plt.plot([], [] ,'-')
O5, = plt.plot([], [] ,'-')

def animate(i):
    
    O1.set_data(obs_hist_0[i,0]+obs_hist_0[i,2]*cos(ts), obs_hist_0[i,1]+obs_hist_0[i,2]*sin(ts))
    O2.set_data(obs_hist_1[i,0]+obs_hist_1[i,2]*cos(ts), obs_hist_1[i,1]+obs_hist_1[i,2]*sin(ts))
    O3.set_data(obs_hist_2[i,0]+obs_hist_2[i,2]*cos(ts), obs_hist_2[i,1]+obs_hist_2[i,2]*sin(ts))
    O4.set_data(obs_hist_3[i,0]+obs_hist_3[i,2]*cos(ts), obs_hist_3[i,1]+obs_hist_3[i,2]*sin(ts))
    O5.set_data(obs_hist_4[i,0]+obs_hist_4[i,2]*cos(ts), obs_hist_4[i,1]+obs_hist_4[i,2]*sin(ts))

    return [O1,O2,O3,O4,O5]  


myAnimation = animation.FuncAnimation(fig, animate, frames=length, interval=700, blit=True)

myAnimation.save('MPC_simulation.mp4', fps=5, extra_args=['-vcodec', 'libx264'])
    
    
    
    











