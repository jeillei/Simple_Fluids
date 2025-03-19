import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

width, height = 200, 100
viscosity = 0.1
omega = 1.0 / (3.0 * viscosity + 0.5)
density = 1.0

e = np.array([[0, 0],
              [1, 0], [0, 1], [-1, 0], [0, -1],
              [1, 1], [-1, 1], [-1, -1], [1, -1]])
w = np.array([4/9,
              1/9, 1/9, 1/9, 1/9,
              1/36, 1/36, 1/36, 1/36])

f = np.ones((9, height, width)) * density / 9.0
f_eq = np.zeros((9, height, width))
rho = np.ones((height, width)) * density
ux = np.zeros((height, width))
uy = np.zeros((height, width))

center_x, center_y = width // 2, height // 2
radius = height // 9
obstacle = np.zeros((height, width), bool)
for y in range(height):
    for x in range(width):
        if (x - center_x)**2 + (y - center_y)**2 < radius**2:
            obstacle[y, x] = True

def equil():
    global f_eq
    for i in range(9):
        cu = 3.0 * (e[i, 0] * ux + e[i, 1] * uy)
        f_eq[i] = rho * w[i] * (1.0 + cu + 0.5 * cu**2 - 1.5 * (ux**2 + uy**2))

def stream():
    global f
    for i in range(9):
        f[i] = np.roll(f[i], shift=(e[i, 1], e[i, 0]), axis=(0, 1))

def collide():
    global f, rho, ux, uy
    rho = np.sum(f, axis=0)
    rho[rho < 0.1] = 0.1
    ux = np.sum(f * e[:, 0, np.newaxis, np.newaxis], axis=0) / rho
    uy = np.sum(f * e[:, 1, np.newaxis, np.newaxis], axis=0) / rho
    velocity_mag = np.sqrt(ux**2 + uy**2)
    max_velocity = 0.2
    mask = velocity_mag > max_velocity
    if np.any(mask):
        scale = max_velocity / velocity_mag[mask]
        ux[mask] *= scale
        uy[mask] *= scale
    ux[obstacle] = 0
    uy[obstacle] = 0
    equil()
    f = f * (1 - omega) + f_eq * omega
    opposite = [0, 3, 4, 1, 2, 7, 8, 5, 6]
    for i in range(1, 9):
        f[i, obstacle] = f[opposite[i], obstacle]

fig, ax = plt.subplots(figsize=(10, 5))
fixed_max = 0.5
image = ax.imshow(np.sqrt(ux**2 + uy**2), cmap='viridis', vmin=0, vmax=fixed_max)
plt.colorbar(image, label='Velocity magnitude')

def animate(frame):
    global f, rho, ux, uy, f_eq
    inlet_region = slice(0, 5)
    inlet_velocity = 0.5 + 0.1 * np.sin(frame * 0.2)
    ux[:, inlet_region] = inlet_velocity
    uy[:, inlet_region] = 0.0
    rho[:, inlet_region] = 1.0
    equil()
    f[:, :, inlet_region] = f_eq[:, :, inlet_region]
    for _ in range(5):
        stream()
        collide()
    damping_factor = 0.5
    ux *= damping_factor
    uy *= damping_factor
    velocity_magnitude = np.sqrt(ux**2 + uy**2)
    image.set_array(velocity_magnitude)
    return [image]

anim = FuncAnimation(fig, animate, frames=300, interval=50, blit=True)
plt.tight_layout()
plt.show()
