import numpy as np
from quad_simulator.quadcopter import Quadcopter
from quad_simulator.controller import Controller
from quad_simulator.simulation import Simulation
from quad_simulator.visualizer import Visualizer
from quad_simulator.trajectory import FigureEightTrajectory # <-- ADDED
import plot_results

def main():
    # --- Parameters ---
    quad_params = {
        'm': 0.5,     # Mass (kg)
        'L': 0.2,     # Arm length (m)
        'Ixx': 0.002, # Inertia (kg*m^2)
        'Iyy': 0.002,
        'Izz': 0.004,
        'g': 9.81
    }
    
    # --- Simulation Setup ---
    sim_time = 15.0  # seconds
    dt = 0.01      # time step
    
    # --- SOTA Feature: Trajectory Planning ---
    trajectory = FigureEightTrajectory(altitude=1.0, scale=1.0, speed=0.5) # <-- ADDED
    
    # --- Initialization ---
    quad = Quadcopter(m=quad_params['m'], L=quad_params['L'], 
                     Ixx=quad_params['Ixx'], Iyy=quad_params['Iyy'], Izz=quad_params['Izz'])
    controller = Controller(quad_params)
    sim = Simulation(quad, controller, sim_time=sim_time, dt=dt)
    
    # --- Run Simulation ---
    history = sim.run(trajectory) # <-- CHANGED
    
    # --- Plot Results ---
    print("Plotting results...")
    plot_results.plot_all(history)
    print(f"Saved plots to sim_position_plots.png and sim_attitude_plots.png")
    
    # --- Run 3D Visualization ---
    print("Starting 3D visualization... (Close window to exit)")
    vis = Visualizer(history, arm_length=quad_params['L'])
    vis.run()

if __name__ == "__main__":
    main()
