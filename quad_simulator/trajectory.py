import numpy as np

class FigureEightTrajectory:
    def __init__(self, altitude=1.0, scale=1.0, speed=0.5):
        """
        Initializes a figure-eight trajectory.
        
        Args:
            altitude (float): The constant Z height for the trajectory.
            scale (float): The size of the figure-eight.
            speed (float): How fast to move along the path.
        """
        self.altitude = altitude
        self.scale = scale
        self.speed = speed

    def get_target(self, t):
        """
        Get the target position (x, y, z) and yaw for a given time t.
        """
        # Parametric equation for a figure-eight (Lissajous curve)
        x = self.scale * np.sin(self.speed * t)
        y = self.scale * np.sin(self.speed * t) * np.cos(self.speed * t)
        z = self.altitude
        
        # Calculate desired yaw (tangent to the path)
        # dx/dt
        dx_dt = self.scale * self.speed * np.cos(self.speed * t)
        # dy/dt
        dy_dt = self.scale * self.speed * (np.cos(self.speed * t)**2 - np.sin(self.speed * t)**2)
        
        # Yaw is the angle of the velocity vector
        yaw = np.arctan2(dy_dt, dx_dt)
        
        return np.array([x, y, z]), yaw
