import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation

class Visualizer:
    """
    Creates a 3D Matplotlib animation of the quadcopter.
    """
    def __init__(self, history, arm_length=0.2):
        self.history = history
        self.arm_length = arm_length
        self.states = history['state']
        self.target_pos_history = history['target_pos'] 
        self.target_pos = self.target_pos_history[0]    

        # *** THIS IS THE FIX: This line was missing ***
        self.body_frame_arms = np.array([
            [arm_length, 0, 0],
            [-arm_length, 0, 0],
            [0, arm_length, 0],
            [0, -arm_length, 0]
        ])
        # *** END OF FIX ***

    def _get_world_frame_arms(self, state):
        pos = state[0:3]
        angles = state[6:9]
        phi, theta, psi = angles
        
        R = Rotation.from_euler('zyx', [psi, theta, phi]).as_matrix()
        
        world_frame_arms = (R @ self.body_frame_arms.T).T + pos
        return world_frame_arms

    def _animate(self, i):
        # Downsample for faster animation
        frame_idx = i * 10
        if frame_idx >= len(self.states):
            return self.lines
            
        state = self.states[frame_idx]
        pos = state[0:3]
        
        # Get arm positions in world frame
        arms = self._get_world_frame_arms(state)
        
        # Update lines for arms
        self.lines[0].set_data_3d([arms[0, 0], arms[1, 0]], [arms[0, 1], arms[1, 1]], [arms[0, 2], arms[1, 2]])
        self.lines[1].set_data_3d([arms[2, 0], arms[3, 0]], [arms[2, 1], arms[3, 1]], [arms[2, 2], arms[3, 2]])
        
        # Update trajectory
        self.trajectory.set_data_3d(self.states[:frame_idx, 0], self.states[:frame_idx, 1], self.states[:frame_idx, 2])
        
        return self.lines

    def run(self):
        fig = plt.figure(figsize=(10, 8))
        self.ax = fig.add_subplot(111, projection='3d')
        
        # Find simulation bounds
        all_pos = self.states[:, 0:3]
        all_x = np.concatenate([all_pos[:, 0], self.target_pos_history[:, 0]]) 
        all_y = np.concatenate([all_pos[:, 1], self.target_pos_history[:, 1]]) 
        all_z = np.concatenate([all_pos[:, 2], self.target_pos_history[:, 2]]) 
        
        min_x, max_x = np.min(all_x) - self.arm_length, np.max(all_x) + self.arm_length
        min_y, max_y = np.min(all_y) - self.arm_length, np.max(all_y) + self.arm_length
        min_z, max_z = np.min(all_z) - self.arm_length, np.max(all_z) + self.arm_length
        
        # Make axes equal
        max_range = np.array([max_x - min_x, max_y - min_y, max_z - min_z]).max() / 2.0
        mid_x = (max_x + min_x) / 2.0
        mid_y = (max_y + min_y) / 2.0
        mid_z = (max_z + min_z) / 2.0
        
        self.ax.set_xlim(mid_x - max_range, mid_x + max_range)
        self.ax.set_ylim(mid_y - max_range, mid_y + max_range)
        self.ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_zlabel("Z (m)")
        self.ax.set_title("6-DOF Quadcopter Simulation")
        
        # Plot *entire* target trajectory
        self.ax.plot(self.target_pos_history[:, 0], self.target_pos_history[:, 1], self.target_pos_history[:, 2], 'r:', label='Target Path') 
        
        # Plot *final* target
        self.ax.plot([self.target_pos_history[-1, 0]], [self.target_pos_history[-1, 1]], [self.target_pos_history[-1, 2]], 'rx', markersize=10, label='Final Target') 
        
        # Plot ground
        self.ax.plot_surface(
            np.array([min_x, max_x]),
            np.array([min_y, max_y]),
            np.array([[0, 0], [0, 0]]),
            alpha=0.2, color='g'
        )

        # Initialize quadcopter lines
        line1, = self.ax.plot([], [], [], 'b-', lw=3)
        line2, = self.ax.plot([], [], [], 'b-', lw=3)
        self.lines = [line1, line2]
        
        # Initialize trajectory line
        self.trajectory, = self.ax.plot([], [], [], 'g:', lw=1, label='Trajectory')
        
        self.ax.legend()
        
        num_frames = len(self.states) // 10
        
        _ = FuncAnimation(
            fig, self._animate,
            frames=num_frames,
            interval=self.history['time'][1] * 10 * 1000, # dt * 10 (downsample) * 1000 (ms)
            blit=False
        )
        
        plt.show()
