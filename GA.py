import numpy as np
import matplotlib.pyplot as plt

#Variables


#Number of iterations:
iterations = 10

#Time Steps:
dt = 0.004

#initial variables:
theta = np.zeros(iterations)
theta_dot = np.zeros(iterations)

phi = np.zeros(iterations)
phi_dot = np.zeros(iterations)

x = np.zeros(iterations)
x_dot = np.zeros(iterations)

#Variables to be solved for
theta_ddot = np.zeros(iterations)
phi_ddot = np.zeros(iterations)
x_ddot = np.zeros(iterations)
T = np.zeros(iterations)

#Position Arrays:
Ox = np.zeros(iterations)
Oy = np.zeros(iterations)

Ax = np.zeros(iterations)
Ay = np.zeros(iterations)

Bx = np.zeros(iterations)
By = np.zeros(iterations)

Px = np.zeros(iterations)
Py = np.zeros(iterations)



#Lengths
La = 2.82    #m
Lb = 0.245  #m
Lg = 0.96   #m
Lp = 2.5    #m

#Inertial Terms
Ib = 2.909  #Kgm^2
Ic = 10     #Kgm^2
mb = 4.2    #kg
mc = 86     #kg
###########################
#Set this to an Extremely high value to revert it to the previous solution.
mf = 10     #kg


###########################


mp = 0.5    #kg
I_tot = (1/12)*mb*(La+Lb)**2 + (mc*Lb**2) + mb*Lg**2

#Initial Conditions:
#NB!! the number before d2r is in degrees!! i.e. it is 45 degrees * d2r to give a value in radians.
theta[0] = np.deg2rad(-45)       #rad
phi[0] = np.deg2rad(-180)       #rad

theta_dot[0] = 0    #rad/s
phi_dot[0] = 0      #rad/s

#Initial Positions:
#This x position may redundant for now, but it allows us to choose initial position.
x[0] = 0    

Ox[0] = x[0]
Oy[0] = -La*np.sin(theta[0])

Ax[0] = Ox[0] + La*np.cos(theta[0])
Ay[0] = Oy[0] + La*np.sin(theta[0])

Bx[0] = Ox[0] - Lb*np.cos(theta[0])
By[0] = Oy[0] - Lb*np.sin(theta[0])

Px[0] = Ox[0] + La*np.cos(theta[0]) + Lp*np.cos(phi[0])
Py[0] = Oy[0] + La*np.sin(theta[0]) + Lp*np.sin(phi[0])

#gravity terms
alpha = 3000
y0 = -0.0003
y =  Py[0]
g = 9.81    #m/s^2
gp = g * (1 - np.exp(-alpha*(y-y0)))
gmod = gp/g  #m/s^2

A = np.array([
    [],
    [],
    [],
    [],
])
