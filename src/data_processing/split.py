import pandas as pd
import numpy as np
import shutil
import os

def split_train_test(source_csv_path, source_root_folder, output_root_folder, test_size=1500, seed=42):
    """
    Splits the dataset into Train and Test sets.
    
    Args:
        source_csv_path: Path to the source CSV (cropped_data.csv).
        source_root_folder: Path containing 'rgb' and 'depth' folders.
        output_root_folder: Where to create 'train' and 'test' folders.
        test_size: Number of samples for the test set.
        seed: Random seed for reproducibility (CRITICAL for thesis comparison).
    """
    print(f"Splitting data with random seed {seed}...")
    np.random.seed(seed) # Fix the randomness for reproducible results
    
    # Load Data
    df = pd.read_csv(source_csv_path)
    
    # Check if we have enough data
    if len(df) <= test_size:
        raise ValueError(f"Dataset size ({len(df)}) is smaller than requested test size ({test_size})!")

    # Create random indices
    shuffled_indices = np.random.permutation(len(df))
    test_indices = shuffled_indices[:test_size]
    train_indices = shuffled_indices[test_size:]
    
    # Split DataFrames
    train_df = df.iloc[train_indices].copy()
    test_df = df.iloc[test_indices].copy()
    
    # Define Paths
    splits = {
        'train': train_df,
        'test': test_df
    }
    
    for split_name, split_df in splits.items():
        split_folder = os.path.join(output_root_folder, split_name)
        target_rgb = os.path.join(split_folder, 'rgb')
        target_depth = os.path.join(split_folder, 'depth')
        
        os.makedirs(target_rgb, exist_ok=True)
        os.makedirs(target_depth, exist_ok=True)
        
        # Save split CSV
        csv_name = f"{split_name}_data.csv"
        csv_path = os.path.join(split_folder, csv_name)
        split_df.to_csv(csv_path, index=False)
        print(f"Saved {split_name} CSV with {len(split_df)} samples to {csv_path}")

        # Copy Files
        print(f"Copying images for {split_name} set...")
        src_rgb_folder = os.path.join(source_root_folder, 'rgb')
        src_depth_folder = os.path.join(source_root_folder, 'depth')
        
        for _, row in split_df.iterrows():
            img_name = row['Picture name']
            
            # Source files
            src_rgb = os.path.join(src_rgb_folder, img_name)
            src_depth = os.path.join(src_depth_folder, img_name)
            
            # Check if source file exists before copying
            if not os.path.exists(src_rgb):
                print(f"Warning: Missing file {src_rgb}")
                continue

            # Copy
            shutil.copy(src_rgb, os.path.join(target_rgb, img_name))
            shutil.copy(src_depth, os.path.join(target_depth, img_name))

    print("Data split completed successfully.")