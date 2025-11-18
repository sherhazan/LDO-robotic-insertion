import pandas as pd
import numpy as np
import os

def add_classification_labels(csv_path, angle_bin_size=22.5, rad_bin_size=1.0):
    """
    Adds discrete class labels to the CSV based on continuous values.
    Logic follows the paper's discretization strategy (16 classes for angle).
    """
    print(f"Generating classification labels for {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # 1. Angle Classification (0-15 for 360 degrees)
    # Formula: class = int(angle // 22.5)
    df['label_angle'] = (df['Angle [deg]'] // angle_bin_size).astype(int)
    
    # Clip to ensure no index out of bounds (e.g. 360.0 -> class 16 -> should be 0)
    df['label_angle'] = df['label_angle'] % 16 

    # 2. Radius Classification
    # Formula: class = int(radius // 1.0)
    df['label_radius'] = (df['Radial distance [mm]'] // rad_bin_size).astype(int)
    
    # Save back to CSV
    df.to_csv(csv_path, index=False)
    print(f"Updated CSV with labels: {csv_path}")
    print(f"Unique Angle Classes: {sorted(df['label_angle'].unique())}")
    print(f"Unique Radius Classes: {sorted(df['label_radius'].unique())}")

if __name__ == "__main__":
    # For testing manually
    add_classification_labels('./dataset/final_split/train/train_data.csv')