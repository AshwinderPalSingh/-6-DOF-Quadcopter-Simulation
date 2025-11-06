import numpy as np

class Simulation:
    """
    Runs the main simulation loop.
    """
    def __init__(self, quad, controller, sim_time=10.0, dt=0.01):
        self.quad = quad
        self.controller = controller
        self.sim_time = sim_time
        self.dt = dt
        self.num_steps = int(sim_time / dt)
        
        # History buffers
        self.history = {
            'time': np.zeros(self.num_steps),
            'state': np.zeros((self.num_steps, 12)),
            'control': np.zeros((self.num_steps, 4)),
            'target_pos': np.zeros((self.num_steps, 3)),
            'target_yaw': np.zeros(self.num_steps)
        }

    def run(self, trajectory): # <-- THIS IS THE FIX
        print(f"Running simulation for {self.sim_time} seconds...")
        
        # Set the initial target (for t=0)
        target_pos, target_yaw = trajectory.get_target(0.0) 
        self.controller.set_target(target_pos, target_yaw)
        
        for i in range(self.num_steps):
            current_time = i * self.dt 
            
            # Get moving target from trajectory
            target_pos, target_yaw = trajectory.get_target(current_time) 
            
            # Set the controller's target for this step
            self.controller.set_target(target_pos, target_yaw) 
            
            # Get current state
            current_state = self.quad.get_state()
            
            # Calculate control inputs
            control_inputs = self.controller.update(current_state, self.dt)
            
            # Apply control and update physics
            self.quad.update(self.dt, control_inputs)
            
            # Log history
            self.history['time'][i] = current_time 
            self.history['state'][i, :] = current_state
            self.history['control'][i, :] = control_inputs
            self.history['target_pos'][i, :] = target_pos
            self.history['target_yaw'][i] = target_yaw
            
            # Simple simulation progress
            if i % (self.num_steps // 10) == 0:
                print(f"Simulation {int(i / self.num_steps * 100)}% complete...")
        
        print("Simulation complete.")
        return self.history
