import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
#Variables

#Simulation Duration
duration = 1    #s
#Time Steps:
dt = 0.004      #s

#Number of iterations:
iterations = int(duration/dt)

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
mp = 0.5    #kg
###########################
#Set this to an Extremely high value to revert it to the previous solution.
mf = 10    #kg
###########################
I_tot = (1/12)*mb*(La+Lb)**2 + (mc*Lb**2) + mb*Lg**2

###########################
#Set this to an Extremely high value to revert it to the previous solution.
mf = 10     #kg

#Initial Conditions:
#NB!! the number before d2r is in degrees!! i.e. it is 45 degrees * d2r to give a value in radians.
theta[0] = np.radians(-45)       #rad
phi[0] = np.radians(-180)       #rad

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

Px[0] = Ax[0] + Lp*np.cos(phi[0])
Py[0] = Ay[0] + Lp*np.sin(phi[0])

#gravity terms
alpha = 3000
y0 = -0.0003
y =  Py[0]
g = 9.81    #m/s^2
gp = g * (1 - np.exp(-alpha*(y-y0)))
gmod = gp/g  #m/s^2

def solve(theta,theta_dot,phi,phi_dot,Py):
    alpha = 3000
    y0 = -0.0003
    g = 9.81    #m/s^2
    gp = g * (1 - np.exp(-alpha*(Py-y0)))
    
 # Use np.sin() and np.cos() for calculations
    A = np.array([
    [I_tot,                             0,      (mc*Lb - mb*Lg)*np.sin(theta),  La*np.sin(theta - phi)],
    [(mc*Lb-mb*Lg)*np.sin(theta),       0,      (mc + mb + mf),                 -np.cos(phi)],
    [-La*np.sin(theta - phi),           0,      np.cos(phi),                    1/mp],
    [La*np.cos(theta - phi),            Lp,     -np.sin(phi),                   0]
    ])

    # --- 3. Vector b (4x1) ---
    # Use np.sin() and np.cos() for calculations
    b = np.array([
    (mc*Lb - mb*Lg)*g*np.cos(theta),
    -(mc*Lb-mb*Lg)*(theta_dot**2)*np.cos(theta),
    La*np.cos(theta - phi)*(theta_dot**2) + Lp*phi_dot**2 - gp*np.sin(phi),
    La*np.sin(theta-phi)*(theta_dot**2) - gp*np.cos(phi)
    ])

    #Solving for the x matrix which is x = [theta_ddot,phi_ddot,x_ddot,Tension]
    theta_ddot, phi_ddot,x_ddot,T = np.linalg.solve(A,b)

    return theta_ddot, phi_ddot,x_ddot,T
# def derivatives(theta,theta_dot,phi,phi_dot,Py):
    
#     d = theta - phi
#     alpha = 3000
#     y0 = -0.0003
#     g = 9.81    #m/s^2
#     gp = g * (1 - np.exp(-alpha*(Py-y0)))
#     theta_ddot = (
#         mp*La*gp*np.sin(phi)*np.sin(d)
#         + (mc*Lb - mb*Lg)*g*np.cos(theta)
#         - (La*theta_dot**2*np.cos(d) + Lp*phi_dot**2)
#         * mp*La*np.sin(d)
#     ) / (mp*La**2*np.sin(d)**2 + IO)

#     phi_ddot = (
#         La*theta_dot**2*np.sin(d)
#         - La*theta_ddot*np.cos(d)
#         - gp*np.cos(phi)
#     ) / Lp

#     return np.array([theta_ddot,phi_ddot])
#Euler integrals:

#Get Accelerations
theta_ddot[0], phi_ddot[0],x_ddot[0],T[0] = solve(theta[0],theta_dot[0],phi[0],phi_dot[0],Py[0])
#theta_ddot[0], phi_ddot[0] = derivatives(theta[0],theta_dot[0],phi[0],phi_dot[0],Py[0])
#Finding Velocites:
theta_dot[1] = theta_dot[0] + theta_ddot[0]*dt
phi_dot[1] = phi_dot[0] + phi_ddot[0]*dt
x_dot[1] = x_dot[0] + x_ddot[0]*dt


#Finding Positions:
theta[1] = theta[0] + theta_dot[0]*dt
phi[1] = phi[0] + phi_dot[0]*dt
x[1] = x[0] + x_dot[1]*dt

#Calculate new positions
Ox[1] = x[1]
#Oy[1] = -La*np.sin(theta[1])
Oy[1] = Oy[0]

Ax[1] = Ox[1] + La*np.cos(theta[1])
Ay[1] = Oy[0] + La*np.sin(theta[1]) 

By[1] = Oy[0] - Lb*np.sin(theta[1]) 
Bx[1] = Ox[1] - Lb*np.cos(theta[1]) 

Py[1] = Ay[1] + Lp*np.sin(phi[1])
Px[1] = Ax[1] + Lp*np.cos(phi[1]) 


#2 Step Adams-Bashfort

#Get Accelerations
theta_ddot[1], phi_ddot[1],x_ddot[1],T[1] = solve(theta[1],theta_dot[1],phi[1],phi_dot[1],Py[1])
# theta_ddot[1], phi_ddot[1] = derivatives(theta[1],theta_dot[1],phi[1],phi_dot[1],Py[1])
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

Py[2] = Ay[2] + Lp*np.sin(phi[2])
Px[2] = Ax[2] + Lp*np.cos(phi[2]) 

#Update Gravity term:
y =  Py[2]
gp = g * (1 - np.exp(-alpha*(y-y0)))
gmod = gp/g  #m/s^2

#3 Step Adams Bashfort
#Get Accelerations
theta_ddot[2], phi_ddot[2],x_ddot[2],T[2] = solve(theta[2],theta_dot[2],phi[2],phi_dot[2],Py[2])
# theta_ddot[2], phi_ddot[2], = derivatives(theta[2],theta_dot[2],phi[2],phi_dot[2],Py[2])
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

Py[3] = Ay[3] + Lp*np.sin(phi[3])
Px[3] = Ax[3] + Lp*np.cos(phi[3]) 

#Update Gravity term:
y =  Py[3]
gp = g * (1 - np.exp(-alpha*(y-y0)))
gmod = gp/g  #m/s^2

# duration = 0    #s
# global i
# i = 3
# while(duration < 2):    
for i in range(3,iterations-1):
    theta_ddot[i], phi_ddot[i],x_ddot[i],T[i] = solve(theta[i],theta_dot[i],phi[i],phi_dot[i],Py[i])
    # theta_ddot[i], phi_ddot[i], = derivatives(theta[i],theta_dot[i],phi[i],phi_dot[i],Py[i])
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

    Py[i+1] = Ay[i+1] + Lp*np.sin(phi[i+1])
    Px[i+1] = Ax[i+1] + Lp*np.cos(phi[i+1]) 

    #Update Gravity term:
    y =  Py[i+1]
    gp = g * (1 - np.exp(-alpha*(y-y0)))
    gmod = gp/g

    # duration += dt
    # i += 1

print("="*100)
print(i)
print("="*100)
# ==========================================
# INTERACTIVE PLOT
# ==========================================

fig, ax = plt.subplots(figsize=(8,6))
plt.subplots_adjust(bottom=0.25)

# Initial plot
[line] = ax.plot(Px, Py, 'b-', linewidth=3)
ax.plot(Px, Py, 'ro', markersize=1)

ax.set_xlabel('Px [m]')
ax.set_ylabel('Py [m]')
ax.set_title('P Trajectory')
ax.grid(True)
ax.axis('equal')

# ------------------------------------------
# Slider axis
# ------------------------------------------
ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])

time_slider = Slider(
    ax=ax_slider,
    label='Frame',
    valmin=0,
    valmax=iterations-1,
    valinit=iterations-1,
    valstep=1
)

# ------------------------------------------
# Update function
# ------------------------------------------
def update(val):

    idx = int(time_slider.val)

    line.set_xdata(Px[:idx+1])
    line.set_ydata(Py[:idx+1])

    fig.canvas.draw_idle()

# Connect slider
time_slider.on_changed(update)

plt.show()

# ==========================================
# SECOND PLOT: FIRST 40 ITERATIONS
# ==========================================

plt.figure(figsize=(8,6))

# Plot only first 40 iterations
plt.plot(Px[:200], Py[:200], 'g-', linewidth=2)
plt.plot(Px[:200], Py[:200], 'ko', markersize=4)

plt.xlabel('Px [m]')
plt.ylabel('Py [m]')
plt.title('P Trajectory (First 150 Iterations)')
plt.grid(True)

plt.ylim(-0.001, 0.002)
plt.xlim(-0.6, 0.5)

plt.show()



# import pandas as pd
# # Create a dictionary excluding x variables
# data = {
#     "theta": theta,
#     "theta_dot": theta_dot,
#     "phi": phi,
#     "phi_dot": phi_dot,
#     "theta_ddot": theta_ddot,
#     "phi_ddot": phi_ddot,
#     "T": T,
#     "gmod": gmod,
#     "Px": Px,
#     "Py": Py,
#     "phi deg": np.rad2deg(phi) 
# }

# # Create a DataFrame
# df = pd.DataFrame(data)

# # Display the table
# print(df)

import pandas as pd

# Show full dataframe in Jupyter
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# Create a dictionary excluding x variables
data = {
    "theta": theta,
    "theta_dot": theta_dot,
    "phi": phi,
    "phi_dot": phi_dot,
    "theta_ddot": theta_ddot,
    "phi_ddot": phi_ddot,
    "T": T,
    "gmod": gmod,
    "Px": Px,
    "Py": Py,
    "phi deg": np.rad2deg(phi) 
}

# Create DataFrame
df = pd.DataFrame(data)

# Display full table
print(df)

Etot = Lb*(np.sin(theta[0] + 1))*mc*g

#Energy Calcs:

#velocity Arrays
vp_x = np.zeros(iterations)
vp_y = np.zeros(iterations)
vp= np.zeros(iterations)
#Energy Arrays
K_BC = np.zeros(iterations)
K_P = np.zeros(iterations)
K_frame = np.zeros(iterations)
Uc = np.zeros(iterations)
Ub = np.zeros(iterations)
Up = np.zeros(iterations)
U_BC = np.zeros(iterations)

#Energy Norm Arrays:
KP_Norm = np.zeros(iterations)
KBC_Norm = np.zeros(iterations)
Kframe_Norm = np.zeros(iterations)
UBC_Norm = np.zeros(iterations)
Up_Norm = np.zeros(iterations)
Etot_Norm = np.zeros(iterations)

#Time array
time = np.zeros(iterations)

for i in range(iterations-1):
    gp = g * (1 - np.exp(-alpha*(Py[i]-y0)))
    #Velocity for P
    vp_x[i] = -La*theta_dot[i]*np.sin(theta[i]) - Lp*phi_dot[i]*np.sin(phi[i])
    vp_y[i] = La*theta_dot[i]*np.cos(theta[i]) + Lp*phi_dot[i]*np.cos(phi[i])
    vp[i] = np.sqrt(vp_x[i]**2 + vp_y[i]**2)

    #Kinetic Energies
    K_BC[i] = 0.5*I_tot*theta_dot[i]**2
    K_P[i] = 0.5*mp*vp[i]**2
    K_frame[i] = 0.5*(mc+mb+mf)*x_dot[i]**2

    #Potential Energies
    Uc[i] = mc*g*Lb*(1-np.sin(theta[i]))
    Ub[i] = mb*g*Lg*(np.sin(theta[i]) + np.sin(theta[0]))
    Up[i] = mp*gp*Py[i]

    U_BC[i] = Uc[i] + Ub[i]

    KP_Norm[i] = K_P[i] / Etot
    KBC_Norm[i] = K_BC[i] / Etot
    Kframe_Norm[i] = K_frame[i] / Etot
    UBC_Norm[i] = U_BC[i] / Etot
    Up_Norm[i] = Up[i] / Etot
    Etot_Norm[i] = KP_Norm[i] + KBC_Norm[i] + Kframe_Norm[i] + UBC_Norm[i]

    if i == 1:
        time[i] = 0
    else:
        time[i] = time[i-1] + dt    




plt.figure(figsize=(8,6))

# Plot Energies
plt.plot(time, KP_Norm, 'g-', linewidth=2)
plt.plot(time, KBC_Norm, 'b-', linewidth=2)
plt.plot(time, Kframe_Norm, 'c-', linewidth=2)
plt.plot(time, UBC_Norm, 'm-', linewidth=2)
plt.plot(time, Etot_Norm, 'r', linewidth=2)

plt.xlabel('Time [s]')
plt.ylabel('Energy fraction')
plt.title('Energy Norms')
plt.grid(True)

plt.show()