import os
import pandas as pd
import numpy as np
from src.visualization.results_plotting import (
    plot_regression_with_margins,
    plot_in_out_accuracy,
    plot_spatial_error
)

# --- CONFIGURATION ---
CONFIG = {
    'case': 'rgb_rad',  # Name of the experiment folder in results
    'ver': 0,           # Version number used in results folder
    'results_root': './results'
}

def main():
    # Construct paths
    exp_name = f"{CONFIG['case'].split('_')[0]}_v{CONFIG['ver']}"
    exp_dir = os.path.join(CONFIG['results_root'], exp_name)
    csv_path = os.path.join(exp_dir, 'graphs_data.csv')
    
    if not os.path.exists(csv_path):
        print(f"Error: Data file not found at {csv_path}")
        print("Did you run 'generate_results.py' first?")
        return

    print(f"--- Generating Graphs for {exp_name} ---")
    df = pd.read_csv(csv_path)
    
    # 1. Angle Regression (Filter out small radii < 2mm where angle is unstable)
    save_p = os.path.join(exp_dir, 'angle_regression.png')
    plot_regression_with_margins(
        ground_truth=df['gt_angle'],
        predictions=df['pred_angle'],
        title='Real vs Predicted Angle',
        unit='[deg]',
        save_path=save_p,
        filter_radius=2.0,
        rad_values=df['gt_rad']
    )

    # 2. Radius Regression
    save_p = os.path.join(exp_dir, 'radius_regression.png')
    plot_regression_with_margins(
        ground_truth=df['gt_rad'],
        predictions=df['pred_rad'],
        title='Real vs Predicted Radius',
        unit='[mm]',
        save_path=save_p
    )

    # 3. In/Out Classification
    save_p = os.path.join(exp_dir, 'in_out_classification.png')
    plot_in_out_accuracy(df, threshold=2.0, save_path=save_p)

    # 4. Spatial Error (Angle)
    # Calculate smallest difference considering circularity (0 == 360)
    angle_diff = np.abs(df['pred_angle'] - df['gt_angle'])
    angle_err = np.where(angle_diff < 180, angle_diff, 360 - angle_diff)
    
    save_p = os.path.join(exp_dir, 'spatial_error_angle.png')
    plot_spatial_error(
        gt_rads=df['gt_rad'],
        gt_angles_deg=df['gt_angle'],
        error_values=angle_err,
        title='Angle Error Map',
        max_error_scale=20, # Cap colors at 20 degrees error
        save_path=save_p
    )

    # 5. Spatial Error (Radius)
    rad_err = np.abs(df['pred_rad'] - df['gt_rad'])
    save_p = os.path.join(exp_dir, 'spatial_error_radius.png')
    plot_spatial_error(
        gt_rads=df['gt_rad'],
        gt_angles_deg=df['gt_angle'],
        error_values=rad_err,
        title='Radius Error Map',
        max_error_scale=2.0, # Cap colors at 2mm error
        save_path=save_p
    )

    print("All graphs generated successfully!")

if __name__ == "__main__":
    main()