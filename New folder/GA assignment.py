import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ============================================================
# GIVEN VARIABLES
# ============================================================

# Lengths in metres (m)
L_b = 1.0
L_B = 1.0   # Counterweight connection distance from pivot/point O to B
L_a = 2.82
L_p = 2.5
L_g = 0.96
L_c = 1.573      # Distance from B to counterweight point mass

# Masses in kg
m_p = 0.5
m_c = 86
m_b = 4.2

# Moments of inertia in kg.m^2

I_b = 2.909

# ============================================================
# IMPORTANT I_o NOTE
# ============================================================
# Since the counterweight is now modelled separately in the 5x5 matrix,
# I_o is defined here WITHOUT m_c*L_b**2 to avoid double-counting.
#
# If your board equation defines I_o including the counterweight contribution,
# change this line back to:
#
# I_o = I_b + m_b*L_g**2 + I_c + m_c*L_b**2
#
I_o = I_b + m_b * L_g**2

# Gravitational acceleration
g = 9.81

# Penalty/contact parameters from your original code
alpha_penalty = 3000
y_penalty = -3e-4

# If you want the projectile gravity modifier active, keep this True.
# If you want to use the exact board equations with plain g, set this False.
use_penalty_gravity = True

# ============================================================
# INITIAL CONDITIONS
# ============================================================

dt = 0.004

theta = np.radians(-45)
theta_dot = 0.0

phi = np.radians(-180)
phi_dot = 0.0

# Reasonable assumption for beta initial angle, as requested
beta = np.radians(90)
beta_dot = 0.0

# Original vertical offset from your code
Oy = -L_a * np.sin(theta)

# ============================================================
# STORAGE LISTS
# ============================================================

accel_history = []
vel_history = []

time_list = []
theta_list = []
phi_list = []
beta_list = []

T_p_history = []
T_c_history = []

yp_history = []
gmod_history = []

# ============================================================
# TIME INTEGRATION LOOP
# ============================================================

for t in np.arange(0, 1.004, dt):

    # --------------------------------------------------------
    # Store current state
    # --------------------------------------------------------
    time_list.append(t)
    theta_list.append(theta)
    phi_list.append(phi)
    beta_list.append(beta)

    # Projectile vertical position from your original code
    yp = Oy + L_a * np.sin(theta) + L_p * np.sin(phi)

    # Gravity/contact modifier from your original code
    g_mod = 1 - np.exp(-alpha_penalty * (yp - y_penalty))

    if use_penalty_gravity:
        g_eff = g * g_mod
    else:
        g_eff = g

    yp_history.append(yp)
    gmod_history.append(g_mod)

    # --------------------------------------------------------
    # 5x5 MATRIX SYSTEM
    #
    # Unknown vector:
    #
    # [theta_ddot, phi_ddot, beta_ddot, T_p, T_c]
    #
    # --------------------------------------------------------

    M = np.array([
        [
            L_a * np.sin(theta - phi),
            0,
            0,
            0,
            -1/m_p
        ],

        [
            L_a * np.cos(theta - phi),
            L_p,
            0,
            0,
            0
        ],

        [
            I_o,
            0,
            0,
            -L_b * np.sin(theta - beta),
            L_a*np.sin(theta - phi)
        ],

        [
            -L_b * np.sin(theta - beta),
            0,
            0,
            -1/m_c,
            0
        ],
        [
            L_b * np.cos(theta - beta),0
            ,
            -L_c,
            0,
            0
        ]
    ], dtype=float)

    N = np.array([(
        
            -L_a *np.cos(theta-phi)* theta_dot**2
            + g*g_mod* np.sin(phi)
            - L_p * phi_dot**2 
        ),

        (
            L_a * theta_dot**2 * np.sin(theta - phi)
            - g*g_mod * np.cos(phi)
        ),
        (

        -m_b * g * L_g * np.cos(theta)
        ),
        (
        -L_c*beta_dot**2
            +g * np.sin(beta)
            + L_b * theta_dot**2*np.cos(theta - beta)
        ),

        (
            +theta_dot**2 * L_b * np.sin(theta - beta)
            + g * np.cos(beta)
        )
    ], dtype=float)

    # --------------------------------------------------------
    # Solve system
    # --------------------------------------------------------

    try:
        solution = np.linalg.solve(M, N)
    except np.linalg.LinAlgError:
        print(f"Matrix became singular at t = {t:.4f} s")
        break

    theta_ddot = solution[0]
    phi_ddot = solution[1]
    beta_ddot = solution[2]
    T_p = solution[3]
    T_c = solution[4]

    # Store only real accelerations for integration
    accel_history.append(np.array([theta_ddot, phi_ddot, beta_ddot]))

    # Store velocities before updating them
    current_vels = np.array([theta_dot, phi_dot, beta_dot])
    vel_history.append(current_vels)

    # Store tensions separately
    T_p_history.append(T_p)
    T_c_history.append(T_c)

    # --------------------------------------------------------
    # Optional printout
    # --------------------------------------------------------

    print("=" * 60)
    print(f"t = {t:.3f} s")
    print(f"yp = {yp:.6f} m")
    print(f"g_mod = {g_mod:.3f}")
    print(f"theta = {theta:.3f} rad")
    print(f"theta_dot = {theta_dot:.3f} rad/s")
    print(f"theta_ddot = {theta_ddot:.3f} rad/s^2")
    print(f"phi = {phi:.3f} rad")
    print(f"phi_dot = {phi_dot:.3f} rad/s")
    print(f"phi_ddot = {phi_ddot:.3f} rad/s^2")
    print(f"beta = {beta:.3f} rad")
    print(f"beta_dot = {beta_dot:.3f} rad/s")
    print(f"beta_ddot = {beta_ddot:.3f} rad/s^2")
    print(f"T_p = {T_p:.3f} N")
    print(f"T_c = {T_c:.3f} N")

    # --------------------------------------------------------
    # Adams-Bashforth integration
    # --------------------------------------------------------

    if len(accel_history) >= 3:

        # 3-point Adams-Bashforth update for angular velocities
        accel_update = (dt / 12) * (
            23 * accel_history[-1]
            - 16 * accel_history[-2]
            + 5 * accel_history[-3]
        )

        theta_dot += accel_update[0]
        phi_dot += accel_update[1]
        beta_dot += accel_update[2]

        # 3-point Adams-Bashforth update for angular positions
        vel_update = (dt / 12) * (
            23 * vel_history[-1]
            - 16 * vel_history[-2]
            + 5 * vel_history[-3]
        )

        theta += vel_update[0]
        phi += vel_update[1]
        beta += vel_update[2]

    elif len(accel_history) == 2:

        # 2-point Adams-Bashforth update for angular velocities
        accel_update = (dt / 2) * (
            3 * accel_history[-1]
            - accel_history[-2]
        )

        theta_dot += accel_update[0]
        phi_dot += accel_update[1]
        beta_dot += accel_update[2]

        # 2-point Adams-Bashforth update for angular positions
        vel_update = (dt / 2) * (
            3 * vel_history[-1]
            - vel_history[-2]
        )

        theta += vel_update[0]
        phi += vel_update[1]
        beta += vel_update[2]

    elif len(accel_history) == 1:

        # Euler start-up step
        theta_dot += theta_ddot * dt
        phi_dot += phi_ddot * dt
        beta_dot += beta_ddot * dt

        theta += theta_dot * dt
        phi += phi_dot * dt
        beta += beta_dot * dt


# ============================================================
# CONVERT LISTS TO ARRAYS
# ============================================================

time = np.array(time_list)

theta_arr = np.array(theta_list)
phi_arr = np.array(phi_list)
beta_arr = np.array(beta_list)

vel_arr = np.array(vel_history)
accel_arr = np.array(accel_history)

theta_dot_arr = vel_arr[:, 0]
phi_dot_arr = vel_arr[:, 1]
beta_dot_arr = vel_arr[:, 2]

theta_ddot_arr = accel_arr[:, 0]
phi_ddot_arr = accel_arr[:, 1]
beta_ddot_arr = accel_arr[:, 2]

T_p_arr = np.array(T_p_history)
T_c_arr = np.array(T_c_history)

yp_arr = np.array(yp_history)
gmod_arr = np.array(gmod_history)

# ============================================================
# PLOTS
# ============================================================

# Angular positions
plt.figure(figsize=(10, 6))
plt.plot(time, np.degrees(theta_arr), label=r'$\theta$ (deg)')
plt.plot(time, np.degrees(phi_arr), label=r'$\phi$ (deg)')
plt.plot(time, np.degrees(beta_arr), label=r'$\beta$ (deg)')
plt.xlabel('Time (s)')
plt.ylabel('Angle (degrees)')
plt.title('Angular Positions vs Time')
plt.legend()
plt.grid(True)

# Angular velocities
plt.figure(figsize=(10, 6))
plt.plot(time, np.degrees(theta_dot_arr), label=r'$\dot{\theta}$ (deg/s)')
plt.plot(time, np.degrees(phi_dot_arr), label=r'$\dot{\phi}$ (deg/s)')
plt.plot(time, np.degrees(beta_dot_arr), label=r'$\dot{\beta}$ (deg/s)')
plt.xlabel('Time (s)')
plt.ylabel('Angular Velocity (deg/s)')
plt.title('Angular Velocities vs Time')
plt.legend()
plt.grid(True)

# Angular accelerations
plt.figure(figsize=(10, 6))
plt.plot(time, np.degrees(theta_ddot_arr), label=r'$\ddot{\theta}$ (deg/s$^2$)')
plt.plot(time, np.degrees(phi_ddot_arr), label=r'$\ddot{\phi}$ (deg/s$^2$)')
plt.plot(time, np.degrees(beta_ddot_arr), label=r'$\ddot{\beta}$ (deg/s$^2$)')
plt.xlabel('Time (s)')
plt.ylabel('Angular Acceleration (deg/s$^2$)')
plt.title('Angular Accelerations vs Time')
plt.legend()
plt.grid(True)

# Tensions
plt.figure(figsize=(10, 6))
plt.plot(time, T_p_arr, label=r'$T_p$')
plt.plot(time, T_c_arr, label=r'$T_c$')
plt.xlabel('Time (s)')
plt.ylabel('Tension (N)')
plt.title('Rope Tensions vs Time')
plt.legend()
plt.grid(True)

# Projectile vertical position and gravity modifier
plt.figure(figsize=(10, 6))
plt.plot(time, yp_arr, label=r'$y_p$')
plt.xlabel('Time (s)')
plt.ylabel('Projectile vertical position (m)')
plt.title('Projectile Vertical Position vs Time')
plt.legend()
plt.grid(True)

plt.figure(figsize=(10, 6))
plt.plot(time, gmod_arr, label=r'$g_{mod}$')
plt.xlabel('Time (s)')
plt.ylabel('Gravity modifier')
plt.title('Gravity Modifier vs Time')
plt.legend()
plt.grid(True)

plt.show()


# ============================================================
# ANIMATION
# ============================================================

theta_anim = theta_arr
phi_anim = phi_arr
beta_anim = beta_arr
time_anim = time

# Correct fixed origin position
x_origin = 0.0
y_origin = 2.5

# Projectile/linkage animation points
xA = x_origin + L_a * np.cos(theta_anim)
yA = y_origin + L_a * np.sin(theta_anim)

xP = xA + L_p * np.cos(phi_anim)
yP = yA + L_p * np.sin(phi_anim)

# Counterweight animation assumption:
# B is L_B from the origin along theta
# C is L_c from B at absolute angle beta
xB = x_origin - L_B * np.cos(theta_anim)
yB = y_origin - L_B * np.sin(theta_anim)

xC = xB + L_c * np.cos(beta_anim) 
yC = yB - L_c * np.sin(beta_anim)

# Figure setup
fig, ax = plt.subplots(figsize=(7, 7))

max_length = L_a + L_p + L_c + 0.5
ax.set_xlim(-max_length, max_length)
ax.set_ylim(-max_length, max_length)
ax.set_aspect('equal')
ax.grid(True)

# Lines and markers
projectile_line, = ax.plot([], [], 'o-', lw=3, label='Projectile linkage')
counterweight_line, = ax.plot([], [], 's-', lw=3, label='Counterweight')
origin_marker, = ax.plot([], [], 'ko', markersize=6)

time_text = ax.text(0.05, 0.92, '', transform=ax.transAxes)

ax.set_title('2 DOF System with Counterweight DOF')
ax.set_xlabel('x-position (m)')
ax.set_ylabel('y-position (m)')
ax.legend(loc='upper right')


def init():
    projectile_line.set_data([], [])
    counterweight_line.set_data([], [])
    origin_marker.set_data([x_origin], [y_origin])
    time_text.set_text('')
    return projectile_line, counterweight_line, origin_marker, time_text


def animate(i):
    # Projectile linkage: Origin -> A -> projectile P
    projectile_xs = [x_origin, xA[i], xP[i]]
    projectile_ys = [y_origin, yA[i], yP[i]]

    # Counterweight: Origin -> B -> counterweight C
    counterweight_xs = [x_origin, xB[i], xC[i]]
    counterweight_ys = [y_origin, yB[i], yC[i]]

    projectile_line.set_data(projectile_xs, projectile_ys)
    counterweight_line.set_data(counterweight_xs, counterweight_ys)

    time_text.set_text(f"Time = {time_anim[i]:.3f} s")

    return projectile_line, counterweight_line, origin_marker, time_text


ani = animation.FuncAnimation(
    fig,
    animate,
    frames=len(time_anim),
    init_func=init,
    interval=dt * 1000,
    blit=True
)

plt.show()

# ============================================================
# SAVE ANIMATION
# ============================================================

try:
    ani.save("simulation_with_counterweight.mp4", fps=1 / dt)
    print("Animation saved as simulation_with_counterweight.mp4")
except Exception as e:
    print("Animation could not be saved as MP4.")
    print("Reason:", e)
    print("The animation should still display in the matplotlib window")