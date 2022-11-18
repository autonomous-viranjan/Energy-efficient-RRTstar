# Energy Efficient RRT* Planner
RRT* based path planner for energy efficient motion planning in off-road environment

Following features are added to the RRT* algorithm:
- Cost: Minimizes Kinetic Energy, Potential Energy, rolling, drag and braking losses
- Elevation Check function: eliminates paths with unsafe approach angles

Output shows the asymptotically optimal path on a 2D colormap of an off-road terrain 
