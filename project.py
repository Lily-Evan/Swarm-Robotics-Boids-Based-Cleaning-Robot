# swarm_cleaners_boids.py
# ----------------------
# Swarm Robotics: Boids-based Cleaning Robots (pure Python + matplotlib)
# Features:
# - 2D continuous space with discretized "dust" cells to clean
# - 10–30 agents (boids) following: Separation, Alignment, Cohesion
# - Obstacle avoidance + world boundaries
# - Collaborative cleaning (cell becomes clean when any agent passes through)
# - Live animation + coverage(%) over time plot
# - Optional MP4 export (FFMpegWriter) at end

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# =============== Parameters ===============
SEED = 7
np.random.seed(SEED)

# World/Grid
WORLD_W, WORLD_H = 60, 60          # continuous space extents (also grid size for dust)
CELL_SIZE = 1.0                     # dust cell size (1 unit)
DUST_DENSITY = 0.25                 # probability each cell initially has dust (0..1)
OBSTACLE_DENSITY = 0.06             # probability each cell is obstacle (0..1)
MAX_STEPS = 4000

# Agents
N_AGENTS = 18
INIT_SPEED = 0.8
MAX_SPEED = 2.0
MAX_FORCE = 0.05

# Boids radii
NEIGHBOR_RADIUS = 6.0
SEPARATION_RADIUS = 2.0
OBSTACLE_RADIUS = 3.0

# Weights
W_ALIGN = 0.8
W_COHESION = 0.6
W_SEPARATION = 1.4
W_OBS_AVOID = 2.0
W_DUST_BIAS = 0.5      # gentle bias toward nearest dust cluster (heuristic)

# Animation / plotting
INTERVAL_MS = 20
TRAIL_LEN = 20         # how many past positions to draw per agent (nice visuals)

# =============== World Setup ===============
# Grid codes: -1 obstacle, 0 dirty, 1 clean
grid = np.zeros((WORLD_W, WORLD_H), dtype=int)

# Obstacles: avoid blocking entire areas—create noise then erode edges
obs_mask = np.random.rand(WORLD_W, WORLD_H) < OBSTACLE_DENSITY
# Ensure borders are free (to reduce trapped regions)
obs_mask[[0, -1], :] = False
obs_mask[:, [0, -1]] = False
grid[obs_mask] = -1

# Dust: place on non-obstacle cells
dust_mask = (np.random.rand(WORLD_W, WORLD_H) < DUST_DENSITY) & (grid != -1)
grid[dust_mask] = 0  # explicit; already zero by default

# Coverage helpers
def coverage_percent():
    free = np.count_nonzero(grid != -1)
    cleaned = np.count_nonzero(grid == 1)
    return 100.0 * cleaned / free if free > 0 else 100.0

# =============== Agent (Boid) Setup ===============
pos = np.stack([
    np.random.uniform(5, WORLD_W-5, size=N_AGENTS),
    np.random.uniform(5, WORLD_H-5, size=N_AGENTS)
], axis=1)
vel = np.stack([
    np.random.uniform(-1, 1, size=N_AGENTS),
    np.random.uniform(-1, 1, size=N_AGENTS)
], axis=1)
# Normalize to initial speed
def norm_rows(v, target_mag):
    m = np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    return v / m * target_mag
vel = norm_rows(vel, INIT_SPEED)

# Trails for pretty viz
trail = [ [pos[i].copy()] for i in range(N_AGENTS) ]

# Precompute obstacle coordinates for quick neighbor checks
obs_coords = np.argwhere(grid == -1)  # array of [x,y]

# =============== Utility ===============
def limit(vec, max_mag):
    mag = np.linalg.norm(vec)
    if mag > max_mag:
        return vec * (max_mag / (mag + 1e-9))
    return vec

def wrap_or_bounce(p, v):
    # Soft bounce on boundaries (keeps them inside)
    x, y = p
    if x < 0: x = 0; v[0] = abs(v[0])
    if x > WORLD_W - 1: x = WORLD_W - 1; v[0] = -abs(v[0])
    if y < 0: y = 0; v[1] = abs(v[1])
    if y > WORLD_H - 1: y = WORLD_H - 1; v[1] = -abs(v[1])
    return np.array([x, y]), v

def nearest_dust_direction(p, sample_radius=10):
    # Look around p in a square to bias motion toward dusty cells
    x0, y0 = np.clip(np.round(p).astype(int), 0, [WORLD_W-1, WORLD_H-1])
    x_min = max(0, x0 - sample_radius); x_max = min(WORLD_W-1, x0 + sample_radius)
    y_min = max(0, y0 - sample_radius); y_max = min(WORLD_H-1, y0 + sample_radius)
    region = grid[x_min:x_max+1, y_min:y_max+1]
    dust_coords = np.argwhere(region == 0)
    if dust_coords.size == 0:
        return np.zeros(2)
    # Choose the densest dust cell by proximity-weighted sampling
    # Weight inversely by distance
    world_coords = dust_coords + np.array([x_min, y_min])
    diffs = world_coords - p
    dists = np.linalg.norm(diffs, axis=1) + 1e-6
    weights = 1.0 / dists
    idx = np.random.choice(len(world_coords), p=weights/weights.sum())
    dir_vec = diffs[idx]
    mag = np.linalg.norm(dir_vec)
    return dir_vec / (mag + 1e-9)

# =============== Boids Rules ===============
def boids_step():
    global pos, vel

    # Neighbor indices by distance (O(N^2), fine for N<=100)
    diffs = pos[:, None, :] - pos[None, :, :]
    dists = np.linalg.norm(diffs, axis=2) + np.eye(N_AGENTS) * 1e9

    # Initialize accelerations
    acc = np.zeros_like(vel)

    for i in range(N_AGENTS):
        # --- Neighborhood sets ---
        neigh_mask = dists[i] < NEIGHBOR_RADIUS
        sep_mask = dists[i] < SEPARATION_RADIUS

        # Alignment: steer toward average heading of neighbors
        align = np.zeros(2)
        if np.any(neigh_mask):
            avg_vel = vel[neigh_mask].mean(axis=0)
            align = avg_vel - vel[i]

        # Cohesion: steer toward center of mass of neighbors
        cohesion = np.zeros(2)
        if np.any(neigh_mask):
            center = pos[neigh_mask].mean(axis=0)
            cohesion = center - pos[i]

        # Separation: steer away from very close neighbors
        separation = np.zeros(2)
        if np.any(sep_mask):
            # vector from neighbors to me → sum normalized away vectors
            close_vecs = pos[i] - pos[sep_mask]
            inv = close_vecs / (np.linalg.norm(close_vecs, axis=1, keepdims=True) + 1e-9)
            separation = inv.sum(axis=0)

        # Obstacle avoidance: repel from nearby obstacles
        obs_avoid = np.zeros(2)
        if obs_coords.size > 0:
            # sample small subset for speed
            if len(obs_coords) > 800:
                sample_idx = np.random.choice(len(obs_coords), size=800, replace=False)
                sample = obs_coords[sample_idx]
            else:
                sample = obs_coords
            dif = pos[i] - sample
            dd = np.linalg.norm(dif, axis=1)
            mask = dd < OBSTACLE_RADIUS
            if np.any(mask):
                rep = dif[mask] / (dd[mask].reshape(-1,1) + 1e-9)
                # closer obstacles push more strongly
                w = (OBSTACLE_RADIUS - dd[mask]).reshape(-1,1)
                obs_avoid = (rep * w).sum(axis=0)

        # Dust bias (heuristic target seeking)
        dust_bias = nearest_dust_direction(pos[i])

        # Combine
        steer = (
            W_ALIGN * align +
            W_COHESION * cohesion +
            W_SEPARATION * separation +
            W_OBS_AVOID * obs_avoid +
            W_DUST_BIAS * dust_bias
        )

        # Limit steering
        steer = limit(steer, MAX_FORCE)
        acc[i] = steer

    # Integrate
    vel = vel + acc
    # Limit speed
    speeds = np.linalg.norm(vel, axis=1, keepdims=True) + 1e-9
    vel = np.where(speeds > MAX_SPEED, vel / speeds * MAX_SPEED, vel)

    new_pos = pos + vel
    # Boundary handling
    for i in range(N_AGENTS):
        new_pos[i], vel[i] = wrap_or_bounce(new_pos[i], vel[i])
    pos[:] = new_pos

    # Clean dust at agent positions
    for i in range(N_AGENTS):
        gx, gy = np.clip(np.round(pos[i]).astype(int), 0, [WORLD_W-1, WORLD_H-1])
        if grid[gx, gy] != -1:
            grid[gx, gy] = 1  # cleaned
        # trail
        trail[i].append(pos[i].copy())
        if len(trail[i]) > TRAIL_LEN:
            trail[i].pop(0)

# =============== Animation ===============
fig, (ax_world, ax_cov) = plt.subplots(1, 2, figsize=(12, 6))

# Base image (transposed so x→horizontal, y→vertical with origin lower-left)
img = ax_world.imshow(
    np.where(grid == -1, 0.1, np.where(grid == 1, 0.7, 1.0)).T,
    origin='lower', vmin=0, vmax=1
)
scat = ax_world.scatter(pos[:,0], pos[:,1], s=40, c='tab:red', edgecolor='k', linewidths=0.5, label='Agents')
trail_lines = [ax_world.plot([], [], linewidth=1, alpha=0.5)[0] for _ in range(N_AGENTS)]
ax_world.set_xlim(0, WORLD_W-1); ax_world.set_ylim(0, WORLD_H-1)
ax_world.set_title("Swarm Cleaners (Boids) — Environment")
ax_world.legend(loc='upper right')

cov_line, = ax_cov.plot([], [], lw=2)
ax_cov.set_xlim(0, MAX_STEPS)
ax_cov.set_ylim(0, 100)
ax_cov.set_xlabel("Steps")
ax_cov.set_ylabel("Coverage (%)")
ax_cov.set_title("Coverage over Time")
cov_history = []

paused = False

def on_click(event):
    # Left click to toggle dust; right click to toggle obstacle (for live editing)
    if event.inaxes != ax_world: return
    gx = int(round(event.xdata))
    gy = int(round(event.ydata))
    if gx < 0 or gx >= WORLD_W or gy < 0 or gy >= WORLD_H:
        return
    global obs_coords
    if event.button == 1:
        # toggle dust/clean (skip obstacles)
        if grid[gx, gy] != -1:
            grid[gx, gy] = 0 if grid[gx, gy] == 1 else 1
    elif event.button == 3:
        # toggle obstacle (cannot place on current agent location)
        if not np.any(np.all(np.round(pos).astype(int) == np.array([gx, gy]), axis=1)):
            if grid[gx, gy] == -1:
                grid[gx, gy] = 0
            else:
                grid[gx, gy] = -1
            obs_coords = np.argwhere(grid == -1)

fig.canvas.mpl_connect('button_press_event', on_click)

def update(frame):
    if paused:
        return img, scat, cov_line, *trail_lines

    boids_step()

    # Update world image colors: obstacle=0.1 (dark), dirty=1.0 (white), clean=0.7 (gray)
    world_colors = np.where(grid == -1, 0.1, np.where(grid == 1, 0.7, 1.0))
    img.set_data(world_colors.T)

    # Agents
    scat.set_offsets(pos)

    # Trails
    for i, ln in enumerate(trail_lines):
        if len(trail[i]) >= 2:
            t = np.array(trail[i])
            ln.set_data(t[:,0], t[:,1])
        else:
            ln.set_data([], [])

    # Coverage
    cov = coverage_percent()
    cov_history.append(cov)
    cov_line.set_data(np.arange(len(cov_history)), cov_history)
    ax_cov.set_xlim(0, max(MAX_STEPS, len(cov_history)))
    ax_world.set_title(f"Swarm Cleaners (Boids) — Coverage: {cov:.2f}%")

    # Stop condition: all reachable cells cleaned
    if cov >= 99.9:
        ani.event_source.stop()

    return img, scat, cov_line, *trail_lines

ani = animation.FuncAnimation(fig, update, frames=MAX_STEPS, interval=INTERVAL_MS, blit=False)

plt.tight_layout()
plt.show()

# =============== Optional: Save to MP4 ===============
# from matplotlib.animation import FFMpegWriter
# writer = FFMpegWriter(fps=int(1000/INTERVAL_MS), bitrate=3000)
# ani.save("swarm_cleaners_boids.mp4", writer=writer)
