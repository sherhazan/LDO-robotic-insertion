import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from scipy.stats import norm

def plot_regression_with_margins(ground_truth, predictions, title, unit, save_path, filter_radius=None, rad_values=None):
    """
    Plots Ground Truth vs Prediction with statistical margins (1-sigma, 2-sigma).
    """
    # Filter logic (e.g., remove small radii for angle calculation)
    if filter_radius is not None and rad_values is not None:
        mask = np.array(rad_values) >= filter_radius
        gt = np.array(ground_truth)[mask]
        pred = np.array(predictions)[mask]
    else:
        gt = np.array(ground_truth)
        pred = np.array(predictions)

    errors = pred - gt
    mu, std = norm.fit(errors)
    
    # Bounds
    lb1, ub1 = mu - 1*std, mu + 1*std
    lb2, ub2 = mu - 2*std, mu + 2*std
    
    # Colors based on error magnitude
    colors = np.where((errors >= lb1) & (errors <= ub1), 'k', 'gray')
    colors = np.where((errors >= lb2) & (errors <= ub2), colors, 'lightgray')
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title(title, fontsize=20)
    ax.set_ylabel(f'Predict {unit}', fontsize=15)
    ax.set_xlabel(f'Real {unit}', fontsize=15)
    
    # Limits
    limit = 360 if 'deg' in unit else 6
    ax.axis([0, limit, 0, limit])
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Plot
    ax.scatter(gt, pred, color=colors, alpha=0.6, label='Predict')
    x_line = np.linspace(0, limit)
    ax.plot(x_line, x_line, 'r', linewidth=2, label='Ground Truth')
    
    # Margins
    ax.plot(x_line, x_line + lb1, '--', color='gray', label=f'1$\sigma$ ({std:.2f})')
    ax.plot(x_line, x_line + ub1, '--', color='gray')
    ax.plot(x_line, x_line + lb2, '--', color='lightgray', label=f'2$\sigma$ ({2*std:.2f})')
    ax.plot(x_line, x_line + ub2, '--', color='lightgray')
    
    # Metrics
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    
    textstr = f'RMSE: {rmse:.2f}\nMAE: {mae:.2f}'
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=15, verticalalignment='top', bbox=props)
    
    ax.legend(fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    plt.close()

def plot_in_out_accuracy(df, threshold, save_path):
    """
    Bar chart showing classification accuracy for Insertion (In/Out).
    """
    # Logic
    # GT > Thresh & Pred > Thresh -> Out-Out (Correct Rejection)
    # GT < Thresh & Pred < Thresh -> In-In (Correct Insertion)
    # GT < Thresh & Pred > Thresh -> In-Out (False Rejection - Safe fail)
    # GT > Thresh & Pred < Thresh -> Out-In (False Insertion - Dangerous!)
    
    gt = df['gt_rad']
    pred = df['pred_rad']
    
    out_out = ((gt > threshold) & (pred > threshold)).sum()
    in_in   = ((gt <= threshold) & (pred <= threshold)).sum()
    in_out  = ((gt <= threshold) & (pred > threshold)).sum()
    out_in  = ((gt > threshold) & (pred <= threshold)).sum()
    
    total_out = out_out + out_in
    total_in = in_in + in_out
    
    counts = [out_out, out_in, in_in, in_out]
    labels = [
        f'{out_out} ({out_out/total_out*100:.1f}%)',
        f'{out_in} ({out_in/total_out*100:.1f}%)',
        f'{in_in} ({in_in/total_in*100:.1f}%)',
        f'{in_out} ({in_out/total_in*100:.1f}%)'
    ]
    
    colors = [[0,0,0.5], [0.5,0.5,1], [0.5,0,0], [1,0.5,0.5]]
    legend_labels = ['GT=Out | Pred=Out', 'GT=Out | Pred=In', 'GT=In | Pred=In', 'GT=In | Pred=Out']
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title(f'Insertion Classification (Thresh={threshold}mm)', fontsize=20)
    ax.set_ylabel('Count', fontsize=15)
    
    x_pos = np.arange(len(counts))
    plt.bar(x_pos, counts, color=colors)
    plt.xticks(x_pos, labels, fontsize=12)
    
    patches = [Patch(color=c, label=l) for c, l in zip(colors, legend_labels)]
    ax.legend(handles=patches, fontsize=12)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    plt.close()

def plot_spatial_error(gt_rads, gt_angles_deg, error_values, title, max_error_scale, save_path):
    """
    Plots error heatmap on X-Y plane.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title(title, fontsize=20)
    ax.set_ylabel('Y [mm]', fontsize=15)
    ax.set_xlabel('X [mm]', fontsize=15)
    ax.axis([-6, 6, -6, 6])
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--')
    
    # Convert to Cartesian (Fix: Convert degrees to radians!)
    gt_angles_rad = np.deg2rad(gt_angles_deg)
    x = gt_rads * np.sin(gt_angles_rad)
    y = gt_rads * np.cos(gt_angles_rad)
    
    # Custom Colormap (Blue to Red)
    colors1 = [(1, 1, 1), (0, 0, 1)] # White to Blue
    colors2 = [(1, 0, 0), (0.5, 0, 0)] # Red to Dark Red
    cmap1 = LinearSegmentedColormap.from_list('blue', colors1)
    cmap2 = LinearSegmentedColormap.from_list('red', colors2)
    # Combine
    combined_cmap = ListedColormap([cmap1(i) for i in range(cmap1.N)] + [cmap2(i) for i in range(cmap2.N)])

    sc = ax.scatter(x, y, c=error_values, cmap=combined_cmap, vmin=0, vmax=max_error_scale)
    cbar = plt.colorbar(sc)
    cbar.set_label('Absolute Error', fontsize=12)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    plt.close()