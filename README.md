# Swarm-Robotics-Boids-Based-Cleaning-Robot


This project simulates a group of autonomous cleaning robots using **Boids-inspired swarm behavior**.  
Instead of a single robot vacuum, multiple agents collaborate to clean a room filled with dust and obstacles.  

Agents follow three simple rules:
- **Separation**: avoid collisions with neighbors  
- **Alignment**: match the average heading of nearby agents  
- **Cohesion**: move toward the center of mass of neighbors  

Extensions:
- Obstacle avoidance  
- Dust cleaning (cells become "clean" when visited)  
- Coverage tracking (% of cleaned environment)  
- Interactive environment editing with mouse clicks  

---

## 🔹 Features
- 2D environment (60x60 cells)
- Obstacles placed randomly (black cells)
- Dust distribution (white = dirty, gray = cleaned)
- 18 cleaning robots following swarm rules
- Visualization:
  - Left panel → Environment with robots
  - Right panel → Coverage (%) over time
- Optional: Save animation as MP4 video
- Interactive: Left-click to add/remove dust, Right-click to add/remove obstacles

---

## 🔧 Requirements
- Python 3.8+
- Libraries:
  ```bash
  pip install numpy matplotlib
▶️ Run
Execute:

bash

python swarm_cleaners_boids.py
You will see:

Environment:

Red circles = robots

Trails = recent robot paths

Black = obstacles

White = dirty cells

Gray = cleaned cells

Coverage plot: Percentage of cleaned environment increases as robots explore.

💾 Save as Video (Optional)
To save the animation as MP4:

Install ffmpeg:

Linux: sudo apt install ffmpeg

Windows/Mac: Download here and add it to PATH

Uncomment these lines at the end of the script:

python

from matplotlib.animation import FFMpegWriter
writer = FFMpegWriter(fps=int(1000/INTERVAL_MS), bitrate=3000)
ani.save("swarm_cleaners_boids.mp4", writer=writer)
📚 References
Reynolds, C. W. (1987). Flocks, herds and schools: A distributed behavioral model. SIGGRAPH.

Boids Algorithm

Swarm Robotics Overview

Matplotlib Animation Docs
