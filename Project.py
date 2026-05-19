#import math
import numpy as np
import matplotlib.pyplot as plt

#Variables


#Number of iterations:
iterations = 400

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
mf = 10000     #kg


###########################


mp = 0.5    #kg
I_tot = (1/12)*mb*(La+Lb)**2 + (mc*Lb**2) + mb*Lg**2

print(I_tot)
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


#Time Steps:
dt = 0.004

#Setting up Matrices:
#The System follows the Form Ax = b.
def ddot_solver(theta,theta_dot,phi,phi_dot,Py):

    gp = g * (1 - np.exp(-alpha*(Py - y0)))

# --- 2. Matrix A (4x4) ---
# Use np.sin() and np.cos() for calculations
    A = np.array([
    [I_tot, 0, (mc*Lb - mb*Lg)*np.sin(theta), La*np.sin(theta - phi)],
    [(mb*Lg - mc*Lb)*np.sin(theta), 0, (mc + mb + mf), -np.cos(phi)],
    [-La*np.sin(theta - phi), 0, np.cos(phi), 1/mp],
    [La*np.cos(theta - phi), Lp, -np.sin(phi), 0]
    ])

    # --- 3. Vector b (4x1) ---
    # Use np.sin() and np.cos() for calculations
    b = np.array([
    mc*g*Lb*np.sin(theta) - mb*g*Lg*np.cos(theta),
    (mb*Lg - mc*Lb)*(theta_dot**2)*np.cos(theta),
    La*np.cos(theta - phi)*(theta_dot**2) + Lp*phi_dot**2 - gp*np.sin(phi),
    La*np.sin(theta-phi)*(theta_dot**2) - gp*np.cos(phi)
    ])

    #Solving for the x matrix which is x = [theta_ddot,phi_ddot,x_ddot,Tension]
    theta_ddot, phi_ddot,x_ddot,T = np.linalg.solve(A,b)

    return theta_ddot, phi_ddot, x_ddot, T



#Euler integrals:

#Get Accelerations
theta_ddot[0], phi_ddot[0],x_ddot[0],T[0] = ddot_solver(theta[0],theta_dot[0],phi[0],phi_dot[0], Py[0])

#Finding Velocites:
theta_dot[1] = theta_dot[0] + theta_ddot[0]*dt
phi_dot[1] = phi_dot[0] + phi_ddot[0]*dt
x_dot[1] = x_dot[0] + x_ddot[0]*dt


#Finding Positions:
theta[1] = theta[0] + theta_dot[1]*dt
phi[1] = phi[0] + phi_dot[1]*dt
x[1] = x[0] + x_dot[1]*dt


#Calculate new positions
Ox[1] = x[1]
#Oy[1] = -La*np.sin(theta[1])
Oy[1] = Oy[0]

Ax[1] = Ox[1] + La*np.cos(theta[1])
Ay[1] = Oy[0] + La*np.sin(theta[1]) 

By[1] = Oy[0] - Lb*np.sin(theta[1]) 
Bx[1] = Ox[1] - Lb*np.cos(theta[1]) 

Py[1] = Oy[0] + La*np.sin(theta[1]) + Lp*np.sin(phi[1])
Px[1] = Ox[1] + La*np.cos(theta[1]) + Lp*np.cos(phi[1]) 

#Update Gravity term:
y =  Py[1]
gp = g * (1 - np.exp(-alpha*(y-y0)))
gmod = gp/g  #m/s^2

#2 Step Adams-Bashfort

#Get Accelerations
theta_ddot[1], phi_ddot[1],x_ddot[1],T[1] = ddot_solver(theta[1],theta_dot[1],phi[1],phi_dot[1], Py[1])

#Finding Velocites:
theta_dot[2] = theta_dot[1] + 0.5*dt*(3*theta_ddot[1] - theta_ddot[0])
phi_dot[2] = phi_dot[1] + 0.5*dt*(3*phi_ddot[1] - phi_ddot[0])
x_dot[2] = x_dot[1] + 0.5*dt*(3*x_ddot[1] - x_ddot[0])

#Finding Positions:
theta[2] = theta[1] + 0.5*dt*(3*theta_dot[1] - theta_dot[0])
phi[2] = phi[1] + 0.5*dt*(3*phi_dot[1] - phi_dot[0])
x[2] = x[1] + 0.5*dt*(3*x_dot[1] - x_dot[0])

#Calculate new positions
Ox[2] = x[2]
Oy[2] = Oy[1]

Ax[2] = Ox[2] + La*np.cos(theta[2])
Ay[2] = Oy[0] + La*np.sin(theta[2]) 

By[2] = Oy[0] - Lb*np.sin(theta[2]) 
Bx[2] = Ox[2] - Lb*np.cos(theta[2]) 

Py[2] = Oy[0] + La*np.sin(theta[2]) + Lp*np.sin(phi[2])
Px[2] = Ox[2] + La*np.cos(theta[2]) + Lp*np.cos(phi[2]) 

#Update Gravity term:
y =  Py[2]
gp = g * (1 - np.exp(-alpha*(y-y0)))
gmod = gp/g  #m/s^2


#3 Step Adams Bashfort
#Get Accelerations
theta_ddot[2], phi_ddot[2],x_ddot[2],T[2] = ddot_solver(theta[2],theta_dot[2],phi[2],phi_dot[2], Py[2])

#Finding Velocites:
theta_dot[3] = theta_dot[2] + (1/12)*dt*(23*theta_ddot[2] - 16*theta_ddot[1] + 5*theta_ddot[0])
phi_dot[3] = phi_dot[2] + (1/12)*dt*(23*phi_ddot[2] - 16*phi_ddot[1] + 5*phi_ddot[0])
x_dot[3] = x_dot[2] + (1/12)*dt*(23*x_ddot[2] - 16*x_ddot[1] + 5*x_ddot[0])

#Finding Positions:
theta[3] = theta[2] + (1/12)*dt*(23*theta_dot[2] - 16*theta_dot[1] + 5*theta_dot[0])
phi[3] = phi[2] + (1/12)*dt*(23*phi_dot[2] - 16*phi_dot[1] + 5*phi_dot[0])
x[3] = x[2] + (1/12)*dt*(23*x_dot[2] - 16*x_dot[1] + 5*x_dot[0])

#Calculate new positions
Ox[3] = x[3]
Oy[3] = Oy[2]

Ax[3] = Ox[3] + La*np.cos(theta[3])
Ay[3] = Oy[0] + La*np.sin(theta[3]) 

By[3] = Oy[0] - Lb*np.sin(theta[3]) 
Bx[3] = Ox[3] - Lb*np.cos(theta[3]) 

Py[3] = Oy[0] + La*np.sin(theta[3]) + Lp*np.sin(phi[3])
Px[3] = Ox[3] + La*np.cos(theta[3]) + Lp*np.cos(phi[3]) 

#Update Gravity term:
y =  Py[3]
gp = g * (1 - np.exp(-alpha*(y-y0)))
gmod = gp/g  #m/s^2



for i in range(3,iterations-1):
    theta_ddot[i], phi_ddot[i],x_ddot[i],T[i] = ddot_solver(theta[i],theta_dot[i],phi[i],phi_dot[i], Py[i])

    # # Print solved values
    # print(
    #     f"Step {i}: "
    #     f"theta_ddot = {theta_ddot[i]:.6f}, "
    #     f"phi_ddot = {phi_ddot[i]:.6f}, "
    #     f"x_ddot = {x_ddot[i]:.6f}, "
    #     f"T = {T[i]:.6f}"
    # )

    #Finding Velocites:
    theta_dot[i+1] = theta_dot[i] + (1/12)*dt*(23*theta_ddot[i] - 16*theta_ddot[i-1] + 5*theta_ddot[i-2])
    phi_dot[i+1] = phi_dot[i] + (1/12)*dt*(23*phi_ddot[i] - 16*phi_ddot[i-1] + 5*phi_ddot[i-2])
    x_dot[i+1] = x_dot[i] + (1/12)*dt*(23*x_ddot[i] - 16*x_ddot[i-1] + 5*x_ddot[i-2])

    #Finding Positions:
    theta[i+1] = theta[i] + (1/12)*dt*(23*theta_dot[i] - 16*theta_dot[i-1] + 5*theta_dot[i-2])
    phi[i+1] = phi[i] + (1/12)*dt*(23*phi_dot[i] - 16*phi_dot[i-1] + 5*phi_dot[i-2])
    x[i+1] = x[i] + (1/12)*dt*(23*x_dot[i] - 16*x_dot[i-1] + 5*x_dot[i-2])

    #Calculate new positions
    Ox[i+1] = x[i+1]
    Oy[i+1] = Oy[0]

    Ax[i+1] = Ox[i+1] + La*np.cos(theta[i+1])
    Ay[i+1] = Oy[0] + La*np.sin(theta[i+1]) 

    By[i+1] = Oy[0] - Lb*np.sin(theta[i+1]) 
    Bx[i+1] = Ox[i+1] - Lb*np.cos(theta[i+1]) 

    Py[i+1] = Oy[0] + La*np.sin(theta[i+1]) + Lp*np.sin(phi[i+1])
    Px[i+1] = Ox[i+1] + La*np.cos(theta[i+1]) + Lp*np.cos(phi[i+1]) 

    #Update Gravity term:
    y =  Py[i+1]
    gp = g * (1 - np.exp(-alpha*(y-y0)))
    gmod = gp/g
    
    print(
        f"{i+1:6d} "
        f"{Ox[i+1]:12.5f} {Oy[i+1]:12.5f} "
        f"{Ax[i+1]:12.5f} {Ay[i+1]:12.5f} "
        f"{Bx[i+1]:12.5f} {By[i+1]:12.5f} "
        f"{Px[i+1]:12.5f} {Py[i+1]:12.5f}"
        f"{gmod:12.5f} {gp:12.5f}"
        f"{x[i+1]:12.5f}" 
    )


#Plotting Projectile
plt.figure(figsize=(8,6))

# Plot Px vs Py
plt.plot(Px, Py, 'b-', linewidth=2)

# Optional: show points
plt.plot(Px, Py, 'ro', markersize=3)

plt.xlabel('Px [m]')
plt.ylabel('Py [m]')
plt.title('P Trajectory')
plt.grid(True)

plt.axis('equal')   # Keeps x and y scaling equal
plt.show()

# Energy Norm