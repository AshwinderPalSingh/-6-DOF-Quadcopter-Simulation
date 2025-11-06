import matplotlib.pyplot as plt
import numpy as np

def plot_all(history):
    plot_positions(history)
    plot_attitudes(history)

def plot_positions(history):
    time = history['time']
    states = history['state']
    targets = history['target_pos']
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle('Position vs. Time', fontsize=16)
    
    labels = ['X (m)', 'Y (m)', 'Z (m)']
    
    for i in range(3):
        axes[i].plot(time, states[:, i], 'b-', label='Actual')
        axes[i].plot(time, targets[:, i], 'r--', label='Target')
        axes[i].set_ylabel(labels[i])
        axes[i].legend()
        axes[i].grid(True)
        
    axes[-1].set_xlabel('Time (s)')
    plt.savefig('sim_position_plots.png')
    plt.close(fig)

def plot_attitudes(history):
    time = history['time']
    states = history['state']
    target_yaw = history['target_yaw']
    
    # Note: Roll/Pitch targets are dynamic (from PID), so we just plot states
    angles_deg = np.rad2deg(states[:, 6:9])
    target_yaw_deg = np.rad2deg(target_yaw)
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle('Attitude vs. Time', fontsize=16)
    
    labels = ['Roll (deg)', 'Pitch (deg)', 'Yaw (deg)']
    
    for i in range(3):
        axes[i].plot(time, angles_deg[:, i], 'b-', label='Actual')
        if i == 2: # Plot yaw target
            axes[i].plot(time, target_yaw_deg, 'r--', label='Target')
            axes[i].legend()
            
        axes[i].set_ylabel(labels[i])
        axes[i].grid(True)
        
    axes[-1].set_xlabel('Time (s)')
    plt.savefig('sim_attitude_plots.png')
    plt.close(fig)

if __name__ == "__main__":
    print("This script is meant to be imported, but you can run main.py to generate plots.")
    # As a fallback, try to load a history file if run standalone (not used by main.py)
    try:
        import os
        if os.path.exists('sim_history.npz'):
            history = np.load('sim_history.npz')
            plot_all(history)
    except Exception as e:
        print(f"Could not load history. Run main.py first. Error: {e}")
