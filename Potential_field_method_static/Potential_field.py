"""
Created on Thu Jan 28 23:35:27 2021

@author: Mohamad Sayegh & Elias Rached

"""

from rockit import *
from rockit import FreeTime, MultipleShooting, Ocp
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi, cos, sin, tan, square
from casadi import vertcat, horzcat, sumsqr, Function, exp, vcat, sum1
from function_create_trajectory import create_trajectory
from scipy import interpolate




#--------------- Problem parameters-------------------------------------

Nsim    = 60             # how much samples to simulate in total (affect sampling time?)
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


#------------ initialize OCP -------------

ocp = Ocp(T = N*dt)


#------------ drone model from reference paper--------------

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
x0 = 0.0
y0 = 0.3
p0_coord = vertcat(x0,y0)
ocp.set_value(p0, p0_coord)
r0 = 0.2                            

p1 = ocp.parameter(2)
x1 = 0.1
y1 = 0.8
p1_coord = vertcat(x1,y1)
ocp.set_value(p1, p1_coord)
r1 = 0.2

p2 = ocp.parameter(2)
x2 = 0.4
y2 = 0.3
p2_coord = vertcat(x2,y2)
ocp.set_value(p2, p2_coord)
r2 = 0.25

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


p = vertcat(x,y,z)             # a point in 3D



#-------------------------- Constraints -----------------------------------

# Define initial parameter
X_0 = ocp.parameter(nx)
X = vertcat(x, y, z, phi, vx, vy, vz, vphi)

#initial point
ocp.subject_to(ocp.at_t0(X) == X_0 ) 

ocp.subject_to( 0  <=  (x    <= 1))
ocp.subject_to( 0  <=  (y    <= 1))
ocp.subject_to( 0  <=  (z    <= 1))


#----------------- reach end point (1,1,1) ------------------------------------

pf = ocp.parameter(3)
p_final = vertcat(xf,yf,zf) # end point
ocp.set_value(pf, p_final)


slack_tf_x = ocp.variable()
slack_tf_y = ocp.variable()
slack_tf_z = ocp.variable()

ocp.subject_to(slack_tf_x >= 0)
ocp.subject_to(slack_tf_y >= 0)
ocp.subject_to(slack_tf_z >= 0)

ocp.subject_to((ocp.at_tf(x) - pf[0]) <= slack_tf_x)
ocp.subject_to((ocp.at_tf(y) - pf[1]) <= slack_tf_y)
ocp.subject_to((ocp.at_tf(z) - pf[2]) <= slack_tf_z)


#---------------- constraints on velocity ---------------------------------

v_final = vertcat(0,0,0,0)

ocp.subject_to(ocp.at_tf(vx) == 0)
ocp.subject_to(ocp.at_tf(vy) == 0)
ocp.subject_to(ocp.at_tf(vz) == 0)
ocp.subject_to(ocp.at_tf(vphi) == 0)



#---------------------- Potential field function 

X = [x0,x1,x2,x3,x4]
Y = [y0,y1,y2,y3,y4]
R = [r0,r1,r2,r3,r4]


ox  =  ocp.parameter(len(X), 1)
oy  =  ocp.parameter(len(Y), 1)
ro  =  ocp.parameter(len(R), 1)

so =  - 0.01   #affects the spread of function

e = 1

we = ocp.parameter(1)
ocp.set_value(we,  e)

g = we*sum1(exp(- ((x - ox )**2/((ro+so)**2)) - ((y - oy)**2/((ro+so)**2))))

costf = Function('costf',[ox,oy,ro,x,y,we],[g])

ocp.set_value(ox,  X)
ocp.set_value(oy,  Y)
ocp.set_value(ro , R)


potential_field = np.zeros((100,100))

    
for i in range(0,100):
    for j in range(0,100):     
        
        potential_field[i,j] = costf( X , Y , R, 0.01*i , 0.01*j, 1)
      
plt.figure(dpi = 300)        
plt.imshow(potential_field.T, cmap='hot', interpolation = 'none', origin='lower')
ts = np.linspace(0,2*pi,1000)

plt.plot(100*p0_coord[0]+100*r0*cos(ts),100*p0_coord[1]+100*r0*sin(ts),'r-')
plt.plot(100*p1_coord[0]+100*r1*cos(ts),100*p1_coord[1]+100*r1*sin(ts),'b-')
plt.plot(100*p2_coord[0]+100*r2*cos(ts),100*p2_coord[1]+100*r2*sin(ts),'g-')
plt.plot(100*p3_coord[0]+100*r3*cos(ts),100*p3_coord[1]+100*r3*sin(ts),'c-')
plt.plot(100*p4_coord[0]+100*r4*cos(ts),100*p4_coord[1]+100*r4*sin(ts),'k-')
plt.plot(100*xf,100*yf,'bo', markersize = 10)
plt.xlim([0,100])
plt.ylim([0,100])
plt.xlabel('x [cm]')
plt.ylabel('y [cm]')
plt.title('Potential Field')    

print(potential_field[int(100*xf)  , int(100*yf) ]) #value at xf,yf

#---------------------- objective functions -----------------------------


# weights
w0 = 15
w1 = 20
w2 = 1e-6
w3 = 100


ocp.add_objective(w0*ocp.integral(g))

ocp.add_objective(w1*ocp.integral(sumsqr(p-pf)))

ocp.add_objective(w2*ocp.integral(sumsqr(ux + uy + uz + uphi)))

ocp.add_objective(w3*(slack_tf_x**2 + slack_tf_y**2 + slack_tf_z**2))    




# to evaluate objective function 
obj =   w0*sum1( 10*exp(- ((x - ox )**2/((ro+so)**2)) - ((y - oy)**2/((ro+so)**2))))+ \
        w1*ocp.integral(sumsqr(p-pf)) + \
        w2*ocp.integral(sumsqr(ux + uy + uz + uphi)) + \
        w3*(slack_tf_x + slack_tf_y + slack_tf_z)



#-------------------------  Pick a solution method: ipopt --------------------

options = {"ipopt": {"print_level": 5}}
# options = {'ipopt': {"max_iter": 1000, 'hessian_approximation':'limited-memory', 'limited_memory_max_history' : 5, 'tol':1e-3}}
options["expand"] = True
options["print_time"] = True
ocp.solver('ipopt', options)



#-------------------------- try other solvers here -------------------


# Multiple Shooting
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

t_sol, sx_sol      = sol.sample(slack_tf_x,        grid='control')
t_sol, sy_sol      = sol.sample(slack_tf_y,        grid='control')
t_sol, sz_sol      = sol.sample(slack_tf_z,        grid='control')


x_hist[0,:]       = x_sol
y_hist[0,:]       = y_sol
z_hist[0,:]       = z_sol
phi_hist[0,:]     = phi_sol
vx_hist[0,:]      = vx_sol
vy_hist[0,:]      = vy_sol
vz_hist[0,:]      = vz_sol
vphi_hist[0,:]    = vphi_sol



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
    plt.plot(p0_coord[0]+r0*cos(ts),p0_coord[1]+r0*sin(ts),'r-')
    plt.plot(p1_coord[0]+r1*cos(ts),p1_coord[1]+r1*sin(ts),'b-')
    plt.plot(p2_coord[0]+r2*cos(ts),p2_coord[1]+r2*sin(ts),'g-')
    plt.plot(p3_coord[0]+r3*cos(ts),p3_coord[1]+r3*sin(ts),'c-')
    plt.plot(p4_coord[0]+r4*cos(ts),p4_coord[1]+r4*sin(ts),'k-')
    plt.plot(xf,yf,'ro', markersize = 10)
    
    if opt == 1:
        plt.plot(x_sol, y_sol, 'go' )
        plt.plot(x_hist[:,0], y_hist[:,0], 'bo', markersize = 3)
        
    else:
        plt.plot(x_hist[:,0], y_hist[:,0], 'bo', markersize = 3)
  
    plt.show(block=True)
    




#----------------- Simulate the MPC solving the OCP ----------------------

#clearances for stopping or using trajectory points
clearance_obj        = 1e-5
clearance_v          = 1e-4   
clearance            = 1e-2
local_min_clearance  = 1e-1


obs_hist_0  = np.zeros((Nsim+1, 3))
obs_hist_1  = np.zeros((Nsim+1, 3))
obs_hist_2  = np.zeros((Nsim+1, 3))
obs_hist_3  = np.zeros((Nsim+1, 3))
obs_hist_4  = np.zeros((Nsim+1, 3))


intermediate_points = []
intermediate_points_required = False
new_path_not_needed = False
intermediate_points_index = 0
is_stuck = False


i = 0

t_tot = 0


while True:
    
    
    print("timestep", i+1, "of", Nsim)
    
    #----- plot rin real time (optional) --------------------
    # plotxy(p0_coord, p1_coord, p2_coord, p3_coord, p4_coord, x_hist[0:i,0], y_hist[0:i,0], 1, x_sol, y_sol)

    ux_hist[i, :] = ux_sol
    uy_hist[i, :] = uy_sol
    uz_hist[i, :] = uz_sol
    uphi_hist[i, :] = uphi_sol
   
    # Combine first control inputs
    current_U = vertcat(ux_sol[0], uy_sol[0], uz_sol[0], uphi_sol[0])
    
    # Simulate dynamics (applying the first control input) and update the current state
    current_X = Sim_system_dyn(x0=current_X, u=current_U, T = dt)["xf"]
    
    t_tot = t_tot + dt


    print( f' x: {current_X[0]}' )
    print( f' y: {current_X[1]}' )
    print( f' z: {current_X[2]}' )
    

    if ( sumsqr(current_X[0:2] - p0_coord)   -  r0**2  ) >= 0: print('outside obs 1') 
    else: print('Problem! inside obs 1')
    if ( sumsqr(current_X[0:2] - p1_coord)   -  r1**2  ) >= 0: print('outside obs 2') 
    else: print('Problem! inside obs 2')
    if ( sumsqr(current_X[0:2] - p2_coord)   -  r2**2  ) >= 0: print('outside obs 3') 
    else: print('Problem! inside obs 3')
    if ( sumsqr(current_X[0:2] - p3_coord)   -  r3**2  ) >= 0: print('outside obs 4') 
    else: print('Problem! inside obs 4')
    if ( sumsqr(current_X[0:2] - p4_coord)   -  r4**2  ) >= 0: print('outside obs 5') 
    else: print('Problem! inside obs 5')




    error_v = sumsqr(current_X[4:8] - v_final)
 
    if intermediate_points_required:
        error = sumsqr(current_X[0:3] - intermediate_points[intermediate_points_index-1])
    else:
        error   = sumsqr(current_X[0:3] - p_final)

    if is_stuck or i == Nsim:
        break

    if intermediate_points_index == len(intermediate_points):  #going to end goal
        clearance = 1e-3
    else:
        clearance = 1e-2
        
    if abs( y_sol[0] - yf ) < 5e-1:  #turn off potential field function
        ocp.set_value(we,  1e-5)
    
    if error < clearance:
        
        if intermediate_points_index == len(intermediate_points):
            print('Location reached, now reducing veolcity to zero')
            
            if error_v < clearance_v:
                print('Desired goal reached!')
                break
            
        else:
            
            print('Intermediate point reached! Diverting to next point.')
            
            intermediate_points_index = intermediate_points_index + 1
            ocp.set_value(pf, vcat(intermediate_points[intermediate_points_index-1]))
            
   
    #----------------- Set the parameter X0 to the new current_X
    ocp.set_value(X_0, current_X)

    #---------------------- Solve the optimization problem
    try:
        sol = ocp.solve()
    except:
        ocp.show_infeasibilities(1e-6)
        sol = ocp.non_converged_solution
        break


    #---------------------- Log data for post-processing  
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
    
    t_sol, sx_sol      = sol.sample(slack_tf_x,        grid='control')
    t_sol, sy_sol      = sol.sample(slack_tf_y,        grid='control')
    t_sol, sz_sol      = sol.sample(slack_tf_z,        grid='control')

    
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
plt.show(block=True)

fig3 = plt.figure(dpi = 300, figsize=(4,2))
plt.plot(timestep, vx_hist[0:i,0], "-b", label="vx")
plt.plot(timestep, vy_hist[0:i,0], "-r", label="vy")
plt.plot(timestep, vz_hist[0:i,0], "-g", label="vz")
plt.plot(timestep, vphi_hist[0:i,0], "-k", label="vphi")
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
plt.plot(timestep, yf*np.ones(i),'r--', linewidth = 0.5, label='y goal')
plt.plot(timestep, zf*np.ones(i),'g--', linewidth = 0.5, label='x and z goal')
plt.title("Position")
plt.xlabel("Time (s)")
plt.ylabel("Positon (m)")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show(block=True)


# print(x_hist[0:i,0])
# print(y_hist[0:i,0])
# print(z_hist[0:i,0])
# print(phi_hist[0:i,0])


# print(vx_hist[0:i,0])
# print(vy_hist[0:i,0])
# print(vz_hist[0:i,0])
# print(vphi_hist[0:i,0])


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
P,  = plt.plot([], [] ,'bo', markersize = 5)
Px, = plt.plot([], [] ,'r--', markersize = 2)

def animate(i):
    
    O1.set_data(obs_hist_0[i,0]+obs_hist_0[i,2]*cos(ts), obs_hist_0[i,1]+obs_hist_0[i,2]*sin(ts))
    O2.set_data(obs_hist_1[i,0]+obs_hist_1[i,2]*cos(ts), obs_hist_1[i,1]+obs_hist_1[i,2]*sin(ts))
    O3.set_data(obs_hist_2[i,0]+obs_hist_2[i,2]*cos(ts), obs_hist_2[i,1]+obs_hist_2[i,2]*sin(ts))
    O4.set_data(obs_hist_3[i,0]+obs_hist_3[i,2]*cos(ts), obs_hist_3[i,1]+obs_hist_3[i,2]*sin(ts))
    O5.set_data(obs_hist_4[i,0]+obs_hist_4[i,2]*cos(ts), obs_hist_4[i,1]+obs_hist_4[i,2]*sin(ts))
    P.set_data(x_hist[0:i,0],y_hist[0:i,0])
    Px.set_data(x_hist[i,:],y_hist[i,:])
    
    return [O1,O2,O3,O4,O5,P,Px]  


myAnimation = animation.FuncAnimation(fig, animate, frames=length, interval=700, blit=True)

myAnimation.save('MPC_simulation.mp4', fps=5, extra_args=['-vcodec', 'libx264'])
    
    
    
    
    

