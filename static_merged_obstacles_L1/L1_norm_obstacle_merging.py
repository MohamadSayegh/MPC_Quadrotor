"""
Created on Thu Jan 28 23:35:27 2021

@author: Mohamad Sayegh & Elias Rached

"""

from rockit import *
from rockit import FreeTime, MultipleShooting, Ocp
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from math import sqrt
import numpy as np
from numpy import pi, cos, sin, tan, square
from casadi import vertcat, horzcat, sumsqr, Function, exp




#--------------- Problem parameters-------------------------------------

Nsim    = 50             # how much samples to simulate in total (affect sampling time?)
nx      = 8              # the system is composed of 8 states
nu      = 4              # the system has 4 control inputs
N       = 10             # number of control intervals = the horizon for multipls shooting
dt      = 0.1            # time interval


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

#------------------------------- Control constraints ----------------------

ocp.subject_to(-1 <= (ux    <= 1))
ocp.subject_to(-1 <= (uy    <= 1))
ocp.subject_to(-1 <= (uz    <= 1))
ocp.subject_to(-1 <= (uphi  <= 1))


# ------------------------ Add obstacles -----------------------------------

#elliptical obstacles
p0 = ocp.parameter(2)
p0_coord = vertcat(0.0,0.3)         #center
ocp.set_value(p0, p0_coord)
# r0 = 0.2                            #radius
r0 = vertcat(0.2,0.1)

p1 = ocp.parameter(2)
p1_coord = vertcat(0.1,0.8)
ocp.set_value(p1, p1_coord)
# r1 = 0.2
r1 = vertcat(0.2,0.3)

p2 = ocp.parameter(2)
p2_coord = vertcat(0.35,0.3)
ocp.set_value(p2, p2_coord)
# r2 = 0.2
r2 = vertcat(0.2,0.15)

p3 = ocp.parameter(2)
p3_coord = vertcat(0.8,0.2)         #center
ocp.set_value(p3, p3_coord)
# r3 = 0.1
r3 = vertcat(0.1,0.2)

p4 = ocp.parameter(2)
p4_coord = vertcat(0.8,0.8)         #center
ocp.set_value(p4, p4_coord)
# r4 = 0.2
r4 = vertcat(0.2,0.2)

p = vertcat(x,y,z)             # a point in 3D

nb_obstacles = 5
obstacles = np.empty([5,4])
obstacles[0,:] = [p0_coord[0],p0_coord[1],r0[0],r0[1]]
obstacles[1,:] = [p1_coord[0],p1_coord[1],r1[0],r1[1]]
obstacles[2,:] = [p2_coord[0],p2_coord[1],r2[0],r2[1]]
obstacles[3,:] = [p3_coord[0],p3_coord[1],r3[0],r3[1]]
obstacles[4,:] = [p4_coord[0],p4_coord[1],r4[0],r4[1]]


plot_merged = False
is_merged = False

for i in range(0,nb_obstacles):
    for j in range(i+1,nb_obstacles):
        if (((obstacles[i,0] - obstacles[j,0])**2) < ((obstacles[i,2] + obstacles[j,2])**2)) and (obstacles[i,1] == obstacles[j,1]):
            a = (obstacles[i,2] + obstacles[j,2] + abs(obstacles[i,0] - obstacles[j,0]))/2
            cx = (obstacles[i,0] + obstacles[j,0])/2
            cy = obstacles[i,1]
            b = sqrt(a*max(obstacles[i,3], obstacles[j,3]))
            ocp.subject_to(((p[0] - cx)**2/(a**2))+((p[1] - cy)**2/(b**2)) >= 1)
            print(f'Merged obstacles {i+1} and {j+1} along x')
            is_merged = True
            plot_merged = True
            p_merged = vertcat(cx,cy)
            r_merged = vertcat(a,b)
            break

        elif (((obstacles[i,1] - obstacles[j,1])**2) < ((obstacles[i,3] + obstacles[j,3])**2)) and (obstacles[i,0] == obstacles[j,0]):
            b = (obstacles[i,3] + obstacles[j,3] + abs(obstacles[i,1] - obstacles[j,1]))/2
            cx = obstacles[i,0]
            cy = (obstacles[i,1] + obstacles[j,1])/2
            a = sqrt(a*max(obstacles[i,2], obstacles[j,2]))
            ocp.subject_to(((p[0] - cx )**2/(a**2))+((p[1] - cy)**2/(b**2)) >= 1)  
            print(f'Merged obstacles {i+1} and {j+1} along y')
            is_merged = True
            plot_merged = True
            p_merged = vertcat(cx,cy)
            r_merged = vertcat(a,b)
            break
                    
        else:
            continue
    
    if not is_merged:
        ocp.subject_to(((p[0] - obstacles[i,0])**2/obstacles[i,2]**2) + ((p[1] - obstacles[i,1])**2/obstacles[i,3]**2) >= 1)
        print(f'Obstacle {i+1} is not merged')
    is_merged = False

#-------------------------- Constraints -----------------------------------

# Define initial parameter
X_0 = ocp.parameter(nx)
X = vertcat(x, y, z, phi, vx, vy, vz, vphi)

#initial point
ocp.subject_to(ocp.at_t0(X) == X_0 ) 

# ocp.subject_to( 0  <=  (x    <= 1))
# ocp.subject_to( 0  <=  (y    <= 1))
# ocp.subject_to( 0  <=  (z    <= 1))


#----------------- reach end point (1,1,1) ------------------------------------


pf = vertcat(xf,yf,zf) # end point

# ocp.subject_to(ocp.at_tf(x) == xf)
# ocp.subject_to(ocp.at_tf(y) == yf)
# ocp.subject_to(ocp.at_tf(z) == zf)


slack_tf_x = ocp.variable()
slack_tf_y = ocp.variable()
slack_tf_z = ocp.variable()

ocp.subject_to(slack_tf_x >= 0)
ocp.subject_to(slack_tf_y >= 0)
ocp.subject_to(slack_tf_z >= 0)

ocp.subject_to(-slack_tf_x <= ((ocp.at_tf(x) - pf[0]) <= slack_tf_x))
ocp.subject_to(-slack_tf_y <= ((ocp.at_tf(y) - pf[1]) <= slack_tf_y))
ocp.subject_to(-slack_tf_z <= ((ocp.at_tf(z) - pf[2]) <= slack_tf_z))

ocp.add_objective(10*(slack_tf_x + slack_tf_y + slack_tf_z))


#---------------- constraints on velocity ---------------------------------

v_final = vertcat(0,0,0,0)

ocp.subject_to(ocp.at_tf(vx) == 0)
ocp.subject_to(ocp.at_tf(vy) == 0)
ocp.subject_to(ocp.at_tf(vz) == 0)
ocp.subject_to(ocp.at_tf(vphi) == 0)


#------------------------------  Objective Function ------------------------

# ocp.add_objective(50*ocp.integral(sumsqr(p-pf)))

l1_gain = 100

slack_x_l1 = ocp.control()
slack_y_l1 = ocp.control()
slack_z_l1 = ocp.control()

ocp.subject_to(-slack_x_l1 <= ((p[0] - pf[0])<= slack_x_l1))
ocp.subject_to(-slack_y_l1 <= ((p[1] - pf[1])<= slack_y_l1))
ocp.subject_to(-slack_y_l1 <= ((p[2] - pf[2])<= slack_y_l1))

ocp.subject_to(0 >= -slack_x_l1)
ocp.subject_to(0 >= -slack_y_l1)
ocp.subject_to(0 >= -slack_z_l1)

ocp.add_objective(l1_gain*ocp.integral(slack_x_l1 + slack_y_l1 + slack_z_l1))

ocp.add_objective((9e-4)*ocp.integral(sumsqr(ux + uy + uz + uphi)))
ocp.add_objective((5e-1)*ocp.integral(sumsqr(vx + vy + vz + vphi)))

# ocp.add_objective(ocp.T)


#-------------------------  Pick a solution method: ipopt --------------------

options = {"ipopt": {"print_level": 0}}
# options = {'ipopt': {"max_iter": 1000, 'hessian_approximation':'limited-memory', 'limited_memory_max_history' : 5, 'tol':1e-3}}
options["expand"] = True
options["print_time"] = True
ocp.solver('ipopt', options)



#-------------------------- try other solvers here -------------------


# # Multiple Shooting
# # ocp.method(MultipleShooting(N=N, M=1, intg='rk', grid=FreeGrid(min=0.05, max=10)))
ocp.method(MultipleShooting(N=N, M=2, intg='rk') )


#-------------------- Set initial-----------------

ux_init = np.ones(N)
uy_init = np.ones(N)
uz_init = np.zeros(N)
uphi_init = np.zeros(N)

vx_init = np.empty(N)
vx_init[0] = 0
vy_init = np.empty(N)
vy_init[0] = 0
vz_init = np.empty(N)
vz_init[0] = 0
vphi_init = np.empty(N)
vphi_init[0] = 0

x_init = np.empty(N)
x_init[0] = 0
y_init = np.empty(N)
y_init[0] = 0
z_init = np.empty(N)
z_init[0] = 0
phi_init = np.empty(N)
phi_init[0] = 0

for i in range(1,N):
    vx_init[i] = vx_init[i-1] + ux_init[i-1]*dt
    vy_init[i] = vy_init[i-1] + uy_init[i-1]*dt
    vz_init[i] = vz_init[i-1] + uz_init[i-1]*dt
    vphi_init[i] = vphi_init[i-1] + uphi_init[i-1]*dt

    phi_init[i] = phi_init[i-1] + vphi_init[i-1]*dt
    z_init[i] = z_init[i-1] + vz_init[i-1]*dt
    x_init[i] = x_init[i-1] + ((vx_init[i-1]*cos(phi_init[i-1])) - (vy_init[i-1]*sin(phi_init[i-1])))*dt
    y_init[i] = y_init[i-1] + ((vx_init[i-1]*sin(phi_init[i-1])) + (vy_init[i-1]*cos(phi_init[i-1])))*dt


ocp.set_initial(x, x_init)
ocp.set_initial(y, y_init)
ocp.set_initial(z, z_init)
ocp.set_initial(phi, phi_init)
ocp.set_initial(vx, vx_init)
ocp.set_initial(vy, vy_init)
ocp.set_initial(vz, vz_init)
ocp.set_initial(vphi, vphi_init)

ocp.set_initial(ux, ux_init)
ocp.set_initial(uy, uy_init)
ocp.set_initial(uz, uz_init)
ocp.set_initial(uphi, uphi_init)


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
Sim_system_dyn = ocp._method.discrete_system(ocp)  



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

t_sol, slack_x_l1_sol, = sol.sample(slack_x_l1, grid='control')
t_sol, slack_y_l1_sol, = sol.sample(slack_y_l1, grid='control')
t_sol, slack_z_l1_sol, = sol.sample(slack_z_l1, grid='control')


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




#------------------ plot function------------------- 


def plotxy(p0_coord, p1_coord, p2_coord, p3_coord, p4_coord, x_hist_1, y_hist_1, opt, x_sol, y_sol):
    
    #x-y plot
    fig = plt.figure(dpi = 300)
    ax = fig.add_subplot(111)

    plt.xlabel('x pos [m]')
    plt.ylabel('y pos [m]')
    plt.xlim(0,1.1)
    plt.ylim(0,1.1)
    plt.title('solution in x,y')
    ax.set_aspect('equal', adjustable='box')

    ts = np.linspace(0,2*pi,1000)
    plt.plot(p0_coord[0]+r0[0]*cos(ts),p0_coord[1]+r0[1]*sin(ts),'r-')
    plt.plot(p1_coord[0]+r1[0]*cos(ts),p1_coord[1]+r1[1]*sin(ts),'b-')
    plt.plot(p2_coord[0]+r2[0]*cos(ts),p2_coord[1]+r2[1]*sin(ts),'g-')
    plt.plot(p3_coord[0]+r3[0]*cos(ts),p3_coord[1]+r3[1]*sin(ts),'c-')
    plt.plot(p4_coord[0]+r4[0]*cos(ts),p4_coord[1]+r4[1]*sin(ts),'k-')
    
    if plot_merged == True:
        plt.plot(p_merged[0]+r_merged[0]*cos(ts),p_merged[1]+r_merged[1]*sin(ts),'k--')
    

    plt.plot(xf,yf,'ro', markersize = 10)
    
    if opt == 1:
        plt.plot(x_sol, y_sol, 'go' )
        plt.plot(x_hist[:,0], y_hist[:,0], 'bo', markersize = 3)
        
    else:
        plt.plot(x_hist[:,0], y_hist[:,0], 'bo', markersize = 3)
  
    plt.show(block=True)
    




#----------------- Simulate the MPC solving the OCP ----------------------


clearance = 1e-3
clearance_v = 1e-5    #should become lower if possible
i = 0

time             = np.zeros((Nsim+1))
x_hist_1         = np.zeros((Nsim+1))
y_hist_1         = np.zeros((Nsim+1))
z_hist_1         = np.zeros((Nsim+1))
ux_hist_1        = np.zeros((Nsim+1))
uy_hist_1        = np.zeros((Nsim+1))
uz_hist_1        = np.zeros((Nsim+1))
uphi_hist_1      = np.zeros((Nsim+1))

t_tot = 0


while True:
    
    print("timestep", i+1, "of", Nsim)
    
    # plotxy(p0_coord, p1_coord, p2_coord, p3_coord, p4_coord, x_hist_1, y_hist_1, 1, x_sol, y_sol)

    ux_hist[i, :] = ux_sol
    uy_hist[i, :] = uy_sol
    uz_hist[i, :] = uz_sol
    uphi_hist[i, :] = uphi_sol
   
    # Combine first control inputs
    current_U = vertcat(ux_sol[0], uy_sol[0], uz_sol[0], uphi_sol[0], slack_x_l1_sol[0], slack_y_l1_sol[0], slack_z_l1_sol[0])
    
    # Simulate dynamics (applying the first control input) and update the current state
    current_X = Sim_system_dyn(x0=current_X, u=current_U, T = dt)["xf"]
    
    #if freetime
    # current_X = Sim_system_dyn(x0=current_X, u=current_U, T=t_sol[1]-t_sol[0])["xf"]
    t_tot = t_tot + dt

    print( f' x: {current_X[0]}' )
    print( f' y: {current_X[1]}' )
    print( f' z: {current_X[2]}' )
    
    
    print('obstacle 0:', sumsqr(current_X[0:2] - p0_coord)   -  r0**2  )
    print('obstacle 1:', sumsqr(current_X[0:2] - p1_coord)   -  r1**2  )
    print('obstacle 2:', sumsqr(current_X[0:2] - p2_coord)   -  r2**2  )  
    print('obstacle 3:', sumsqr(current_X[0:2] - p3_coord)   -  r3**2  )
    print('obstacle 4:', sumsqr(current_X[0:2] - p4_coord)   -  r4**2  )
    
    error = sumsqr(current_X[0:3] - pf)
    error_v = sumsqr(current_X[4:8] - v_final)

    if (error < clearance and error_v < clearance_v) or i == Nsim:  #Nsim is max iteration
        # plotxy(p0_coord, p1_coord, p2_coord, p3_coord, p4_coord, x_hist_1, y_hist_1, 0, x_sol, y_sol)
        print('Desired goal reached!')
        break   #solution reached the end goal
   
    # Set the parameter X0 to the new current_X
    ocp.set_value(X_0, current_X)
        
       
        
    # Solve the optimization problem
    try:
        sol = ocp.solve()
    except:
        ocp.show_infeasibilities(1e-6)
        sol = ocp.non_converged_solution
        break

    # Log data for post-processing  
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

    t_sol, slack_x_l1_sol, = sol.sample(slack_x_l1, grid='control')
    t_sol, slack_y_l1_sol, = sol.sample(slack_y_l1, grid='control')
    t_sol, slack_z_l1_sol, = sol.sample(slack_z_l1, grid='control')
    
    x_hist[i+1,:]       = x_sol
    y_hist[i+1,:]       = y_sol
    z_hist[i+1,:]       = z_sol
    phi_hist[i+1,:]     = phi_sol
    vx_hist[i+1,:]      = vx_sol
    vy_hist[i+1,:]      = vy_sol
    vz_hist[i+1,:]      = vz_sol
    vphi_hist[i+1,:]    = vphi_sol

    
    #--------------------- Initial guess
    # can be added  = less nb iterations and sol time, but higher execution time

    ocp.set_initial(x, x_sol)
    ocp.set_initial(y, y_sol)
    ocp.set_initial(z, z_sol)
    ocp.set_initial(phi, phi_sol)
    ocp.set_initial(vx, vx_sol)
    ocp.set_initial(vy, vy_sol)
    ocp.set_initial(vz, vz_sol)
    ocp.set_initial(vphi, vphi_sol)
    
    i = i+1






# ------------------- Results

print(f'Total execution time is: {t_tot}')

plotxy(p0_coord, p1_coord, p2_coord, p3_coord, p4_coord, x_hist[0:i,0], y_hist[0:i,0], 0, x_sol, y_sol)


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
# plt.legend(loc="upper right")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# plt.show(block=True)

fig3 = plt.figure(dpi = 300, figsize=(4,2))
plt.plot(timestep, vx_hist[0:i,0], "-b", label="vx")
plt.plot(timestep, vy_hist[0:i,0], "-r", label="vy")
plt.plot(timestep, vz_hist[0:i,0], "-g", label="vz")
plt.plot(timestep, vphi_hist[0:i,0], "-k", label="vphi")
plt.title("Velocity")
# plt.ylim(-1.02, 1.02)
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
# plt.legend(loc="upper right")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# plt.show(block=True)

fig4 = plt.figure(dpi = 300, figsize=(4,2))
plt.plot(timestep, x_hist[0:i,0], "-b", label="x")
plt.plot(timestep, y_hist[0:i,0], "-r", label="y")
plt.plot(timestep, z_hist[0:i,0], "-g", label="z")
plt.plot(timestep, phi_hist[0:i,0], "-k", label="phi")
plt.title("Position")
# plt.ylim(-1.02, 1.02)
plt.xlabel("Time (s)")
plt.ylabel("Positon (m)")
# plt.legend(loc="upper right")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show(block=True)








    
    
    
    
    

