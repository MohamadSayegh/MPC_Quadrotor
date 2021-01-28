
"""
Created on Thu Jan 28 23:35:27 2021

@author: Mohamad Sayegh & Elias Rached

"""



from rockit import *
from rockit import FreeTime, MultipleShooting, Ocp
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi, cos, sin, tan, square
from casadi import vertcat, horzcat, sumsqr, Function, exp, sqrt
from update_obstacles_position import update_obs



#--------------- Problem parameters-------------------------------------

Nsim    = 150             # how much samples to simulate in total (affect sampling time?)
nx      = 8               # the system is composed of 8 states
nu      = 4               # the system has 4 control inputs
N       = 20              # number of control intervals = the horizon for multipls shooting
dt      = 0.02            # time interval

xf = 0.9
yf = 0.4
zf = 0.7

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



#------------ initialize OCP  -------------

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

#------------------------ Initial constraints----------------------------




# Initial guess
ocp.set_initial(x,         0)
ocp.set_initial(y,         0)
ocp.set_initial(z,         0)
ocp.set_initial(phi,       0)
ocp.set_initial(vx,        0)
ocp.set_initial(vy,        0)
ocp.set_initial(vz,        0)
ocp.set_initial(vphi,      0)

ocp.set_initial(ux,      0)
ocp.set_initial(uy,      0)
ocp.set_initial(uz,      0)
ocp.set_initial(uphi,    0)


#------------------------------- Control constraints ----------------------

ocp.subject_to(-1 <= (ux    <= 1))
ocp.subject_to(-1 <= (uy    <= 1))
ocp.subject_to(-1 <= (uz    <= 1))
ocp.subject_to(-1 <= (uphi  <= 1))



# ------------------------ obstacle avoidance : ellipses ---------------------------


p0 = ocp.parameter(3)
px0 = 0.2
py0 = 0.3
pz0 = 0.5
p0_coord = vertcat(px0,py0,pz0)         #center
ocp.set_value(p0, p0_coord)
r0 = ocp.parameter(3)                   #radius
r0_values = vertcat(0.1,0.1,1.0)  
ocp.set_value(r0, r0_values)

p1 = ocp.parameter(3)
px1 = 0.5
py1 = 0.3
pz1 = 0.5
p1_coord = vertcat(px1,py1,pz1)         
ocp.set_value(p1, p1_coord)
r1 = ocp.parameter(3)               
r1_values = vertcat(0.1,0.1,1.0)  
ocp.set_value(r1, r1_values)

p2 = ocp.parameter(3)
px2 = 0.5
py2 = 0.6
pz2 = 0.5
p2_coord = vertcat(px2,py2,pz2)
ocp.set_value(p2, p2_coord)
r2 = ocp.parameter(3)               
r2_values = vertcat(0.1,0.1,1.0)  
ocp.set_value(r2, r2_values)

p3 = ocp.parameter(3)
px3 = 0.5
py3 = 0.8
pz3 = 0.5
p3_coord = vertcat(px3,py3,pz3)         
ocp.set_value(p3, p3_coord)
r3 = ocp.parameter(3)                   
r3_values = vertcat(0.05,0.05,1.0)  
ocp.set_value(r3, r3_values)

p4 = ocp.parameter(3)
px4 = 0.8
py4 = 0.8
pz4 = 0.5
p4_coord = vertcat(px4,py4,pz4)         
ocp.set_value(p4, p4_coord)
r4 = ocp.parameter(3)                   
r4_values = vertcat(0.1,0.1,1.0)  
ocp.set_value(r4, r4_values)

PX =  [px0, px1, px2, px3, px4]
PY =  [py0, py1, py2, py3, py4]
R =  [0.1, 0.1, 0.1, 0.05, 0.1]
Dx = [-1 ,-1 ,1 ,1 ,1 ]
Dy = [-1 ,-1 ,1 ,1 ,1 ]


p = vertcat(x,y,z)      



#-------------------- Hard constraints 


# ocp.subject_to( sumsqr((p[0:3] - p0)/r0)  >= 1 )
# ocp.subject_to( sumsqr((p[0:3] - p1)/r1)  >= 1 )
# ocp.subject_to( sumsqr((p[0:3] - p2)/r2)  >= 1 )
# ocp.subject_to( sumsqr((p[0:3] - p3)/r3)  >= 1 )
# ocp.subject_to( sumsqr((p[0:3] - p4)/r4)  >= 1 )


#-------------- Using multiple slack variables for every constraint

# slack_0 = ocp.variable()
# slack_1 = ocp.variable()
# slack_2 = ocp.variable()
# slack_3 = ocp.variable()
# slack_4 = ocp.variable()

# ocp.subject_to(slack_0 >= 0)
# ocp.subject_to(slack_1 >= 0)
# ocp.subject_to(slack_2 >= 0)
# ocp.subject_to(slack_3 >= 0)
# ocp.subject_to(slack_4 >= 0)

# ocp.subject_to( sumsqr((p[0:3] - p0)/r0)  - 1 >= slack_0)
# ocp.subject_to( sumsqr((p[0:3] - p1)/r1)  - 1 >= slack_1)
# ocp.subject_to( sumsqr((p[0:3] - p2)/r2)  - 1 >= slack_2)
# ocp.subject_to( sumsqr((p[0:3] - p3)/r3)  - 1 >= slack_3)
# ocp.subject_to( sumsqr((p[0:3] - p4)/r4)  - 1 >= slack_4)

# ocp.add_objective(w3*(slack_0**2 + slack_1**2 + slack_2**2 + slack_3**2 + slack_4**2))

#-------------- Using one slack variable for all constraints

slack_obs = ocp.control()

ocp.subject_to(slack_obs >= 0)

ocp.subject_to( sumsqr((p[0:3] - p0)/r0)  - 1 >= slack_obs)
ocp.subject_to( sumsqr((p[0:3] - p1)/r1)  - 1 >= slack_obs)
ocp.subject_to( sumsqr((p[0:3] - p2)/r2)  - 1 >= slack_obs)
ocp.subject_to( sumsqr((p[0:3] - p3)/r3)  - 1 >= slack_obs)
ocp.subject_to( sumsqr((p[0:3] - p4)/r4)  - 1 >= slack_obs)

#-------------------------- initial point Constraint ------------------------


# Define initial parameter
X_0 = ocp.parameter(nx)
X = vertcat(x, y, z, phi, vx, vy, vz, vphi)

# initial point
ocp.subject_to(ocp.at_t0(X) == X_0 ) 



#------------------ Grid constraints: optional

# ocp.subject_to( 0 <= ( x <= 1 ) )
# ocp.subject_to( 0 <= ( y <= 1 ) )
# ocp.subject_to( 0 <= ( z <= 1 ) )

ocp.subject_to(z > 0) #above floor

#----------------- reach end point (1,1,1) ------------------------------------

pf = vertcat(xf,yf,zf) # end point

slack_tf_x = ocp.variable()
slack_tf_y = ocp.variable()
slack_tf_z = ocp.variable()

ocp.subject_to(slack_tf_x >= 0)
ocp.subject_to(slack_tf_y >= 0)
ocp.subject_to(slack_tf_z >= 0)

ocp.subject_to(-slack_tf_x <= ((ocp.at_tf(x) - pf[0]) <= slack_tf_x))
ocp.subject_to(-slack_tf_y <= ((ocp.at_tf(y) - pf[1]) <= slack_tf_y))
ocp.subject_to(-slack_tf_z <= ((ocp.at_tf(z) - pf[2]) <= slack_tf_z))



#---------------- constraints on velocity ---------------------------------

ocp.subject_to(ocp.at_tf(vx) == 0)
ocp.subject_to(ocp.at_tf(vy) == 0)
ocp.subject_to(ocp.at_tf(vz) == 0)
ocp.subject_to(ocp.at_tf(vphi) == 0)

# slack_tf_vx = ocp.variable()
# slack_tf_vy = ocp.variable()
# slack_tf_vz = ocp.variable()
# ocp.subject_to(slack_tf_vx >= 0)
# ocp.subject_to(slack_tf_vy >= 0)
# ocp.subject_to(slack_tf_vz >= 0)
# ocp.subject_to(-slack_tf_vx <= ((ocp.at_tf(vx)) <= slack_tf_vx))
# ocp.subject_to(-slack_tf_vy <= ((ocp.at_tf(vy)) <= slack_tf_vy))
# ocp.subject_to(-slack_tf_vz <= ((ocp.at_tf(vz)) <= slack_tf_vz))
# ocp.add_objective(100*(slack_tf_vx + slack_tf_vy + slack_tf_vz))



#------------------------------  Objective functions  ------------------------

w1 = 1
w2 = 1e-6
w3 = 100
w4 = 10

ocp.add_objective(w1*ocp.integral(sumsqr(p-pf)))

ocp.add_objective(w2*ocp.integral(w2*sumsqr(ux + uy + uz + uphi)))

# ocp.add_objective(w3*(slack_obs**2 + slack_obs**2 + slack_obs**2 + slack_obs**2 + slack_obs**2))

ocp.add_objective(w3*ocp.integral((slack_obs)))

ocp.add_objective(w4*(slack_tf_x + slack_tf_y + slack_tf_z))



#-------------------------  Pick a solution method -----------------------

options = {"ipopt": {"print_level": 0}}
options["expand"] = True
options["print_time"] = True
ocp.solver('ipopt', options)

# Multiple Shooting
ocp.method(MultipleShooting(N=N, M=4, intg='rk') )


#-------------------- Set initial -----------------

ocp.set_initial(x,     0.0)
ocp.set_initial(y,     0.0)
ocp.set_initial(z,     0.0)
ocp.set_initial(phi,   0.0)
ocp.set_initial(vx,    0.0)
ocp.set_initial(vy,    0.0)
ocp.set_initial(vz,    0.0)
ocp.set_initial(vphi,  0.0)

ocp.set_initial(ux,    1.0)
ocp.set_initial(uy,    1.0)
ocp.set_initial(uz,    1.0)
ocp.set_initial(uphi,  0.0)




#---------------- Solve the OCP for the first time step--------------------

# First waypoint is current position
index_closest_point = 0

current_X = vertcat(0,0,0,0,0,0,0,0)
ocp.set_value(X_0, current_X)

# Solve the optimization problem
try:
    sol = ocp.solve()
except:
    ocp.show_infeasibilities(1e-6)
    sol = ocp.non_converged_solution


# Get discretised dynamics as CasADi function to simulate the system
Sim_system_dyn = ocp._method.discrete_system(ocp)  #what is this ?


#----------------------- Log data for post-processing---------------------
  
t_sol, x_sol      = sol.sample(x,      grid='control')
t_sol, y_sol      = sol.sample(y,      grid='control')
t_sol, z_sol      = sol.sample(z,      grid='control')
t_sol, phi_sol    = sol.sample(phi,    grid='control')
t_sol, vx_sol     = sol.sample(vx,     grid='control')
t_sol, vy_sol     = sol.sample(vy,     grid='control')
t_sol, vz_sol     = sol.sample(vz,     grid='control')
t_sol, vphi_sol   = sol.sample(vphi,   grid='control')

t_sol, ux_sol     = sol.sample(ux,     grid='control')
t_sol, uy_sol     = sol.sample(uy,     grid='control')
t_sol, uz_sol     = sol.sample(uz,     grid='control')
t_sol, uphi_sol   = sol.sample(uphi,   grid='control')

t_sol, slack_sol  = sol.sample(slack_obs,     grid='control')
    

x_hist[0,:]       = x_sol
y_hist[0,:]       = y_sol
z_hist[0,:]       = z_sol
phi_hist[0,:]     = phi_sol
vx_hist[0,:]      = vx_sol
vy_hist[0,:]      = vy_sol
vz_hist[0,:]      = vz_sol
vphi_hist[0,:]    = vphi_sol

print(current_X[0])
print(current_X[1])
print(current_X[2])


#----------------------------- 3D plot function --------------------


def plot3d(p0_coord,r0_values,x_hist_1,y_hist_1,z_hist_1,x_sol,y_sol,z_sol,opt,L):
    
    fig3 = plt.figure(dpi = 300)
    ax4 = fig3.add_subplot(121, projection='3d')
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.set_xlim(0,1);
    ax4.set_ylim(0,1);
    ax4.set_zlim(0,1);
    
    # Make data
    pp0 = np.array(p0_coord)
    pp1 = np.array(p1_coord)
    pp2 = np.array(p2_coord)
    pp3 = np.array(p3_coord)
    pp4 = np.array(p4_coord)
    
    rr0 = np.array(r0_values)
    rr1 = np.array(r1_values)
    rr2 = np.array(r2_values)
    rr3 = np.array(r3_values)
    rr4 = np.array(r4_values)
    
    u = np.linspace(0,2*pi,100) 
    v = np.linspace(0,2*pi,100) 
    
    x0 = pp0[0] + rr0[0] * np.outer(np.cos(u), np.sin(v))
    y0 = pp0[1] + rr0[1] * np.outer(np.sin(u), np.sin(v))
    z0 = pp0[2] + rr0[2] * np.outer(np.ones(np.size(u)), np.cos(v))
    
    x1 = pp1[0] + rr1[0] * np.outer(np.cos(u), np.sin(v))   
    y1 = pp1[1] + rr1[1] * np.outer(np.sin(u), np.sin(v))
    z1 = pp1[2] + rr1[2] * np.outer(np.ones(np.size(u)), np.cos(v))

    
    x2 = pp2[0] + rr2[0] * np.outer(np.cos(u), np.sin(v))
    y2 = pp2[1] + rr2[1] * np.outer(np.sin(u), np.sin(v))
    z2 = pp2[2] + rr2[2] * np.outer(np.ones(np.size(u)), np.cos(v))
    
    x3 = pp3[0] + rr3[0] * np.outer(np.cos(u), np.sin(v))
    y3 = pp3[1] + rr3[1] * np.outer(np.sin(u), np.sin(v))
    z3 = pp3[2] + rr3[2] * np.outer(np.ones(np.size(u)), np.cos(v))
    
    x4 = pp4[0] + rr4[0] * np.outer(np.cos(u), np.sin(v))
    y4 = pp4[1] + rr4[1] * np.outer(np.sin(u), np.sin(v))
    z4 = pp4[2] + rr4[2] * np.outer(np.ones(np.size(u)), np.cos(v))
    
    for i in range(0,100):
        for j in range(0,100):
            if z0[i][j] < 0: z0[i,j] = 0
            elif  z0[i][j] > 1: z0[i,j] = 0
            if z1[i][j] < 0: z1[i,j] = 0
            elif  z1[i][j] > 1: z1[i,j] = 0
            if z2[i][j] < 0: z2[i,j] = 0
            elif  z2[i][j] > 1: z2[i,j] = 0
            if z3[i][j] < 0: z3[i,j] = 0
            elif  z3[i][j] > 1: z3[i,j] = 0
            if z4[i][j] < 0: z4[i,j] = 0
            elif  z4[i][j] > 1: z4[i,j] = 0
            
    # Plot the surface
    ax4.plot_wireframe(x0, y0, z0, color='y', linewidth = 0.1)
    ax4.plot_wireframe(x1, y1, z1, color='g', linewidth = 0.1)
    ax4.plot_wireframe(x2, y2, z2, color='b', linewidth = 0.1)
    ax4.plot_wireframe(x3, y3, z3, color='c', linewidth = 0.1)
    ax4.plot_wireframe(x4, y4, z4, color='k', linewidth = 0.1)
    ax4.plot3D(xf,yf,zf,'ro')
    
    ax4.xaxis.set_ticklabels([])
    ax4.yaxis.set_ticklabels([])
    ax4.zaxis.set_ticklabels([])

    xh = x_hist_1.tolist()
    yh = y_hist_1.tolist()
    zh = z_hist_1.tolist()
    
    ax4.view_init(20, -10)  #angle of view
    
    if opt == 0:    
        ax4.plot3D(xh[0:L], yh[0:L], zh[0:L], 'bo', markersize = 1.5)
    
    elif opt==1:
        ax4.plot3D(xh[L], yh[L], zh[L], 'bo', markersize = 3)
        ax4.plot3D(x_sol,y_sol,z_sol, 'r--', linewidth = 1)
    
    
    
    #-------------- second plot
    ax5 = fig3.add_subplot(122, projection='3d')
    ax5.set_xlabel('x')
    ax5.set_ylabel('y')
    ax5.set_xlim(0,1);
    ax5.set_ylim(0,1);
    ax5.set_zlim(0,1);
    
    ax5.xaxis.set_ticklabels([])
    ax5.yaxis.set_ticklabels([])
    ax5.zaxis.set_ticklabels([])
    
    # Plot the surface
    ax5.plot_wireframe(x0, y0, z0, color='y', linewidth = 0.1)
    ax5.plot_wireframe(x1, y1, z1, color='g', linewidth = 0.1)
    ax5.plot_wireframe(x2, y2, z2, color='b', linewidth = 0.1)
    ax5.plot_wireframe(x3, y3, z3, color='c', linewidth = 0.1)
    ax5.plot_wireframe(x4, y4, z4, color='k', linewidth = 0.1)
    ax5.plot3D(xf,yf,zf,'ro')
    
    ax5.view_init(90, 0)  #angle of view
        
    if opt == 0:
        ax5.plot3D(xh[0:L], yh[0:L], zh[0:L], 'bo', markersize = 1.5)
    
    elif opt==1:
        ax5.plot3D(xh[L], yh[L], zh[L], 'bo', markersize = 3)
        ax5.plot3D(x_sol,y_sol,z_sol, 'r--', linewidth = 1)
    
    
    
    plt.show(block=True)    
    
    
    
    
#----------------- Simulate the MPC solving the OCP ----------------------


clearance = 1e-4
i = 0

time             = np.zeros((Nsim+1))
x_hist_1         = np.zeros((Nsim+1))
y_hist_1         = np.zeros((Nsim+1))
z_hist_1         = np.zeros((Nsim+1))
ux_hist_1        = np.zeros((Nsim+1))
uy_hist_1        = np.zeros((Nsim+1))
uz_hist_1        = np.zeros((Nsim+1))
uphi_hist_1      = np.zeros((Nsim+1))

time                = dt + t_sol[0]
x_hist_1[i+1]       = x_sol[0]
y_hist_1[i+1]       = y_sol[0]
z_hist_1[i+1]       = z_sol[0]
ux_hist_1[i+1]      = ux_sol[0]
uy_hist_1[i+1]      = uy_sol[0]
uz_hist_1[i+1]      = uz_sol[0]
uphi_hist_1[i+1]    = uphi_sol[0]
    
t_tot = 0

obs_hist_0  = np.zeros((Nsim+1, 6))
obs_hist_1  = np.zeros((Nsim+1, 6))
obs_hist_2  = np.zeros((Nsim+1, 6))
obs_hist_3  = np.zeros((Nsim+1, 6))
obs_hist_4  = np.zeros((Nsim+1, 6))


#------------------- Start of While Loop ------------------------------------

while True:
    
    ux_hist[i, :]   = ux_sol
    uy_hist[i, :]   = uy_sol
    uz_hist[i, :]   = uz_sol
    uphi_hist[i, :] = uphi_sol
    
    # plot3d(p0_coord,r0_values,x_hist_1,y_hist_1,z_hist_1,x_sol,y_sol,z_sol,1,i)
    
    t_tot = t_tot + dt
    
    print("timestep", i+1, "of", Nsim)
    
    # Combine first control inputs
    current_U = vertcat(ux_sol[0], uy_sol[0], uz_sol[0], uphi_sol[0], slack_sol[0])
    
    # Simulate dynamics (applying the first control input) and update the current state
    current_X = Sim_system_dyn(x0=current_X, u=current_U, T = dt)["xf"]

    
    error = sumsqr(current_X[0:3] - pf)
    if error < clearance or i == Nsim:  #Nsim is max iteration
        break   #solution reached the end goal 
   
    #------------------ Set the parameter X0 to the new current_X
    ocp.set_value(X_0, current_X)
    
    #------------------ dynamic obstacle 
    
    PX,PY,Dx,Dy =  update_obs(PX,PY,R,Dx,Dy)

    px0 = PX[0]
    px1 = PX[1]
    px2 = PX[2]
    px3 = PX[3]
    px4 = PX[4]
    
    
    py0 = PY[0]
    py1 = PY[1]
    py2 = PY[2]
    py3 = PY[3]
    py4 = PY[4]
    
    p0_coord = vertcat(px0,py0,pz0)
    ocp.set_value(p0, p0_coord)
    
    p1_coord = vertcat(px1,py1,pz1)
    ocp.set_value(p1, p1_coord)
    
    p2_coord = vertcat(px2,py2,pz2)
    ocp.set_value(p2, p2_coord)
    
    p3_coord = vertcat(px3,py3,pz3)
    ocp.set_value(p3, p3_coord)
    
    p4_coord = vertcat(px4,py4,pz4)
    ocp.set_value(p4, p4_coord)

    
    #----------------- Solve the optimization problem
    try:
        
        sol = ocp.solve()
        
    except:
        
        ocp.show_infeasibilities(1e-6)
        sol = ocp.non_converged_solution
        
        #compare initial condition constraint with solution
        print(current_X)
        print(sol.value(ocp.at_t0(X)))
        break

    #-------------------------- Print ---------------------
    
    print( f' x: {current_X[0]}' )
    print( f' y: {current_X[1]}' )
    print( f' z: {current_X[2]}' )
    
    if (sumsqr((current_X[0:3] - p0_coord)/r0_values) ) >= 1: print('outside obs 1') 
    else: ('Problem! inside obs 1'); break;
    if (sumsqr((current_X[0:3] - p1_coord)/r1_values) ) >= 1: print('outside obs 2') 
    else: ('Problem! inside obs 2'); break;
    if (sumsqr((current_X[0:3] - p2_coord)/r2_values) ) >= 1: print('outside obs 3') 
    else: ('Problem! inside obs 3'); break;
    if (sumsqr((current_X[0:3] - p3_coord)/r3_values) ) >= 1: print('outside obs 4') 
    else: ('Problem! inside obs 4'); break;
    if (sumsqr((current_X[0:3] - p4_coord)/r4_values) ) >= 1: print('outside obs 5') 
    else: ('Problem! inside obs 5'); break;
    



    #---------------- Log data for post-processing ------------
    
    obs_hist_0[i,0] = px0
    obs_hist_0[i,1] = py0
    obs_hist_0[i,2] = pz0
    obs_hist_0[i,3] = r0_values[0]
    obs_hist_0[i,4] = r0_values[1]
    obs_hist_0[i,5] = r0_values[2]
    
    obs_hist_1[i,0] = px1
    obs_hist_1[i,1] = py1
    obs_hist_1[i,2] = pz1
    obs_hist_1[i,3] = r1_values[0]
    obs_hist_1[i,4] = r1_values[1]
    obs_hist_1[i,5] = r1_values[2]
    
    obs_hist_2[i,0] = px2
    obs_hist_2[i,1] = py2
    obs_hist_2[i,2] = pz2
    obs_hist_2[i,3] = r2_values[0]
    obs_hist_2[i,4] = r2_values[1]
    obs_hist_2[i,5] = r2_values[2]
    
    obs_hist_3[i,0] = px3
    obs_hist_3[i,1] = py3
    obs_hist_3[i,2] = pz3
    obs_hist_3[i,3] = r3_values[0]
    obs_hist_3[i,4] = r3_values[1]
    obs_hist_3[i,5] = r3_values[2]
    
    obs_hist_4[i,0] = px4
    obs_hist_4[i,1] = py4
    obs_hist_4[i,2] = pz4
    obs_hist_4[i,3] = r4_values[0]
    obs_hist_4[i,4] = r4_values[1]
    obs_hist_4[i,5] = r4_values[2]
    
    
    t_sol, x_sol      = sol.sample(x,        grid='control')
    t_sol, y_sol      = sol.sample(y,        grid='control')
    t_sol, z_sol      = sol.sample(z,        grid='control')
    t_sol, phi_sol    = sol.sample(phi,      grid='control')
    t_sol, vx_sol     = sol.sample(vx,       grid='control')
    t_sol, vy_sol     = sol.sample(vy,       grid='control')
    t_sol, vz_sol     = sol.sample(vz,       grid='control')
    t_sol, vphi_sol   = sol.sample(vphi,     grid='control')
    t_sol, ux_sol     = sol.sample(ux,       grid='control')
    t_sol, uy_sol     = sol.sample(uy,       grid='control')
    t_sol, uz_sol     = sol.sample(uz,       grid='control')
    t_sol, uphi_sol   = sol.sample(uphi,     grid='control')
    t_sol, slack_sol  = sol.sample(uphi,     grid='control')

    
    time                = dt + t_sol[0]
    x_hist_1[i+1]       = x_sol[0]
    y_hist_1[i+1]       = y_sol[0]
    z_hist_1[i+1]       = z_sol[0]
    
    x_hist[i+1,:]       = x_sol
    y_hist[i+1,:]       = y_sol
    z_hist[i+1,:]       = z_sol
    phi_hist[i+1,:]     = phi_sol
    vx_hist[i+1,:]      = vx_sol
    vy_hist[i+1,:]      = vy_sol
    vz_hist[i+1,:]      = vz_sol
    vphi_hist[i+1,:]    = vphi_sol
   
    
    
    # ----------------- set initial --------------------
    
    ocp.set_initial(x, x_sol)
    ocp.set_initial(y, y_sol)
    ocp.set_initial(z, z_sol)
    ocp.set_initial(phi, phi_sol)
    ocp.set_initial(vx, vx_sol)
    ocp.set_initial(vy, vy_sol)
    ocp.set_initial(vz, vz_sol)
    ocp.set_initial(vphi, vphi_sol)
    ocp.set_initial(slack_obs, slack_sol)
    
    i = i+1
    
#------------------- End of While Loop ----------------------------------




#--------------------------- Plot the results ---------------------------


#---------------------------- 2D -------------------------------------

#x-y plot
fig = plt.figure(dpi = 300)
ax1 = plt.subplot(1, 1, 1)
ax1.set_xlabel('x pos [m]')
ax1.set_ylabel('y pos [m]')
ax1.set_aspect('equal', 'box')
ax1.set_xlim(0,1.5);
ax1.set_ylim(0,1.5);

ts = np.linspace(0,2*pi,1000)


ax1.plot(p0_coord[0]+r0_values[0]*cos(ts),p0_coord[1]+r0_values[1]*sin(ts),'r-')
ax1.plot(p1_coord[0]+r1_values[0]*cos(ts),p1_coord[1]+r1_values[1]*sin(ts),'g-')
ax1.plot(p2_coord[0]+r2_values[0]*cos(ts),p2_coord[1]+r2_values[1]*sin(ts),'b-')
ax1.plot(p3_coord[0]+r3_values[0]*cos(ts),p3_coord[1]+r3_values[1]*sin(ts),'c-')
ax1.plot(p4_coord[0]+r4_values[0]*cos(ts),p4_coord[1]+r4_values[1]*sin(ts),'k-')
ax1.plot(xf,yf,'ro')

for k in range(len(x_hist_1)):
    ax1.plot(x_hist_1[k], y_hist_1[k], 'b-')
    ax1.plot(x_hist_1[k], y_hist_1[k], 'g.')

plt.show(block=True)




#x-z plot
fig = plt.figure(dpi = 300)
ax1 = plt.subplot(1, 1, 1)
ax1.set_xlabel('x pos [m]')
ax1.set_ylabel('z pos [m]')
ax1.set_aspect('equal', 'box')
ax1.set_xlim(0,1.5);
ax1.set_ylim(0,1.5);

ts = np.linspace(0,2*pi,1000)

ax1.plot(p0_coord[0]+r0_values[0]*cos(ts),p0_coord[2]+r0_values[2]*sin(ts),'r-')
ax1.plot(p1_coord[0]+r1_values[0]*cos(ts),p1_coord[2]+r1_values[2]*sin(ts),'g-')
ax1.plot(p2_coord[0]+r2_values[0]*cos(ts),p2_coord[2]+r2_values[2]*sin(ts),'b-')
ax1.plot(p3_coord[0]+r3_values[0]*cos(ts),p3_coord[2]+r3_values[2]*sin(ts),'c-')
ax1.plot(p4_coord[0]+r4_values[0]*cos(ts),p4_coord[2]+r4_values[2]*sin(ts),'k-')
ax1.plot(xf,zf,'ro')

for k in range(len(x_hist_1)):
    ax1.plot(x_hist_1[k], z_hist_1[k], 'b-')
    ax1.plot(x_hist_1[k], z_hist_1[k], 'g.')

plt.show(block=True)


# ------------------ 3d plot----------------------

plot3d(p0_coord,r0_values,x_hist_1,y_hist_1,z_hist_1,x_sol,y_sol,z_sol,0,i)

    
# ------------------- Results

print(f'Total execution time is: {t_tot}')


timestep = np.linspace(0,t_tot, len(ux_hist[0:i,0]))

fig2 = plt.figure(dpi = 300, figsize=(4,2))
plt.plot(timestep, ux_hist[0:i,0], "-b", label="ux")
plt.plot(timestep, uy_hist[0:i,0], "-r", label="uy")
plt.plot(timestep, uz_hist[0:i,0], "-g", label="uz")
plt.plot(timestep, uphi_hist[0:i,0], "-k", label="uphi")
plt.title("Control Inputs")
plt.ylim(-1.02, 1.02)
plt.xlabel("Time (s)")
plt.ylabel("Control Inputs (m/s^2)")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show(block=True)

fig3 = plt.figure(dpi = 300, figsize=(4,2))
plt.plot(timestep, vx_hist[0:i,0], "-b", label="vx", linewidth = 1)
plt.plot(timestep, vy_hist[0:i,0], "-r", label="vy", linewidth = 1)
plt.plot(timestep, vz_hist[0:i,0], "-g", label="vz", linewidth = 1)
plt.plot(timestep, vphi_hist[0:i,0], "-k", label="vphi", linewidth = 1)
plt.title("Velocity")
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show(block=True)

fig4 = plt.figure(dpi = 300, figsize=(4,2))
plt.plot(timestep, x_hist[0:i,0], "b.", label="x")
plt.plot(timestep, y_hist[0:i,0], "r.", label="y")
plt.plot(timestep, z_hist[0:i,0], "g.", label="z")
plt.plot(timestep, phi_hist[0:i,0], "k.", label="phi")
plt.plot(timestep, xf*np.ones(i),'b--', linewidth = 1, label='x goal')
plt.plot(timestep, zf*np.ones(i),'r--', linewidth = 1, label='y and z goal')
plt.title("Position")
plt.xlabel("Time (s)")
plt.ylabel("Positon (m)")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show(block=True)



