import matplotlib.pyplot as plt
import numpy as np

def plot_prediction_vs_gt(ground_truth, predictions, target_type, save_path=None):
    """
    Plots the Scatter plot of Predicted vs Real values.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    if target_type == 'radius':
        ax.set_title('Real Radius vs Predicted Radius')
        ax.set_ylabel('Predicted Radius [mm]')
        ax.set_xlabel('Real Radius [mm]')
        limit = 6
        ax.axis([0, limit, 0, limit])
        
    elif target_type == 'angle':
        ax.set_title('Real Angle vs Predicted Angle')
        ax.set_ylabel('Predicted Angle [deg]')
        ax.set_xlabel('Real Angle [deg]')
        limit = 360
        ax.axis([0, limit, 0, limit])
        
    # Plot scatter
    ax.scatter(ground_truth, predictions, color='black', alpha=0.6, label='Predictions')
    
    # Plot ideal line (y=x)
    x = np.linspace(0, limit, 100)
    ax.plot(x, x, 'r--', linewidth=2, label='Ideal')
    
    ax.grid(True, which='both', linestyle='--', alpha=0.7)
    ax.legend()
    ax.set_aspect('equal')
    
    if save_path:
        plt.savefig(save_path)
        print(f"Graph saved to {save_path}")
    
    plt.show()