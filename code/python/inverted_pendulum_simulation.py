import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button
from scipy.integrate import solve_ivp
from control import lqr

# === System Parameters ===
M = 0.5     # cart mass
m = 0.2     # pendulum mass
l = 0.3     # pendulum length to COM
g = 9.81

# Linearized system
A = np.array([
    [0, 1, 0, 0],
    [0, 0, (m * g) / M, 0],
    [0, 0, 0, 1],
    [0, 0, ((M + m) * g) / (M * l), 0]
])
B = np.array([[0], [1 / M], [0], [1 / (M * l)]])

# LQR Controller
Q = np.diag([10, 1, 100, 1])
R = np.array([[0.001]])
K, _, _ = lqr(A, B, Q, R)
K = np.array(K).flatten()

# Closed-loop dynamics
def closed_loop_dynamics(t, x):
    u = -K @ x
    dxdt = (A - B @ K.reshape(1, -1)) @ x
    return dxdt

# Initial simulation
x0 = [0.0, 0.0, 0.2, 0.0]
t_span = (0, 10)
t_eval = np.linspace(*t_span, 500)
sol = solve_ivp(closed_loop_dynamics, t_span, x0, t_eval=t_eval)

t = t_eval
x_data = sol.y[0].copy()
theta_data = sol.y[2].copy()
current_state = sol.y[:, -1].copy()

# === Plot and Animation Setup ===
fig = plt.figure(figsize=(12, 6))
gs = fig.add_gridspec(2, 2, height_ratios=[5, 1], width_ratios=[2, 1])

# Animation axis
ax = fig.add_subplot(gs[0, 0])
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-0.5, 1.2)
ax.set_aspect('equal')
ax.set_title("Inverted Pendulum Simulation")
ax.grid()

# Control/cart elements
cart_width = 0.3
cart_height = 0.2
line, = ax.plot([], [], 'r-', lw=3)
cart_patch = plt.Rectangle((0, 0), cart_width, cart_height, fc='b')
pend_mass, = ax.plot([], [], 'ko', markersize=8)
ax.add_patch(cart_patch)
trail_line, = ax.plot([], [], 'c--', lw=1, alpha=0.7)
force_arrow = ax.arrow(0, 0, 0, 0, color='green', width=0.01)

# Trail data
trail = []
trail_length = 50

# Live plot axis
ax_plot = fig.add_subplot(gs[0, 1])
ax_plot.set_xlim(0, 10)
ax_plot.set_ylim(-np.pi/2, np.pi/2)
ax_plot.set_title("Pendulum Angle θ(t)")
ax_plot.set_xlabel("Time (s)")
ax_plot.set_ylabel("θ (rad)")
theta_line, = ax_plot.plot([], [], 'm-', lw=2)
time_vals = []
value_vals = []

# Plot toggle state
plot_mode = {'theta': True}

def init():
    line.set_data([], [])
    pend_mass.set_data([], [])
    cart_patch.set_xy((-100, -100))
    trail_line.set_data([], [])
    theta_line.set_data([], [])
    return line, pend_mass, cart_patch, trail_line, theta_line

def animate(i):
    global force_arrow, trail, time_vals, value_vals

    xc = x_data[i]
    th = theta_data[i]

    pend_x = xc + l * np.sin(th)
    pend_y = cart_height + l * np.cos(th)

    trail.append((pend_x, pend_y))
    if len(trail) > trail_length:
        trail.pop(0)
    trail_x, trail_y = zip(*trail)
    trail_line.set_data(trail_x, trail_y)

    line.set_data([xc, pend_x], [cart_height, pend_y])
    pend_mass.set_data([pend_x], [pend_y])
    cart_patch.set_xy((xc - cart_width / 2, 0))

    # Update control force arrow
    x_vec = np.array([x_data[i], 0, theta_data[i], 0])
    u = -K @ x_vec
    u_clipped = np.clip(u, -10, 10)
    arrow_len = u_clipped * 0.05
    force_arrow.remove()
    force_arrow = ax.arrow(
        xc, cart_height + 0.05,
        arrow_len, 0,
        color='green', width=0.01, head_width=0.07,
        length_includes_head=True
    )

    # Update live plot
    current_time = t[i]
    time_vals.append(current_time)
    val = th if plot_mode['theta'] else u
    value_vals.append(val)

    theta_line.set_data(time_vals, value_vals)
    ax_plot.set_xlim(0, max(10, current_time + 0.1))
    ax_plot.set_ylim(-2, 2)
    ax_plot.set_ylabel("θ (rad)" if plot_mode['theta'] else "u (N)")
    ax_plot.set_title("Pendulum Angle θ(t)" if plot_mode['theta'] else "Control Force u(t)")

    return line, pend_mass, cart_patch, trail_line, theta_line, force_arrow

# Create animation
ani = animation.FuncAnimation(
    fig, animate, init_func=init,
    frames=len(t), interval=1000 * (t[1] - t[0]),
    blit=True
)

# === Buttons Setup ===
# Button axes
ax_left = fig.add_subplot(gs[1, 0])
ax_left.axis('off')
ax_right = fig.add_subplot(gs[1, 1])
ax_right.axis('off')

# Define button areas manually
b1 = plt.axes([0.1, 0.05, 0.15, 0.05])
b2 = plt.axes([0.3, 0.05, 0.15, 0.05])
b3 = plt.axes([0.7, 0.05, 0.15, 0.05])

btn_left = Button(b1, 'Impulse Left')
btn_right = Button(b2, 'Impulse Right')
btn_toggle = Button(b3, 'Toggle Plot')

# Impulse handler
def apply_impulse(direction):
    global current_state, x_data, theta_data, ani, time_vals, value_vals, trail

    impulse = 0.5
    current_state[1] += impulse * direction

    sol2 = solve_ivp(closed_loop_dynamics, (0, 10), current_state, t_eval=np.linspace(0, 10, 500))
    x_data[:] = sol2.y[0]
    theta_data[:] = sol2.y[2]
    current_state[:] = sol2.y[:, -1]

    # Reset trail and plot
    trail.clear()
    time_vals.clear()
    value_vals.clear()
    ani.frame_seq = ani.new_frame_seq()
    ani.event_source.start()

btn_left.on_clicked(lambda event: apply_impulse(-1))
btn_right.on_clicked(lambda event: apply_impulse(1))

# Toggle button handler
def toggle_plot(event):
    plot_mode['theta'] = not plot_mode['theta']
    time_vals.clear()
    value_vals.clear()

btn_toggle.on_clicked(toggle_plot)

plt.show()
