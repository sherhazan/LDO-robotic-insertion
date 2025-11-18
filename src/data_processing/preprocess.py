import os
import shutil
import pandas as pd
import numpy as np
import cv2

from src.utils import center_blue_object
from src.augmentations import random_crop_resize
from src.data_processing.split import split_train_test
from src.data_processing.labeling import add_classification_labels

def step1_organize_and_process_raw(raw_root, output_root, csv_output_path):
    """
    Step 1: Flattens the directory structure, renames files, and creates a unified master CSV.
    Converts units: Radians -> Degrees, Meters -> Millimeters.
    """
    print("\n--- Step 1: Organizing Raw Data & Processing CSV ---")
    
    rgb_out = os.path.join(output_root, 'rgb')
    depth_out = os.path.join(output_root, 'depth')
    os.makedirs(rgb_out, exist_ok=True)
    os.makedirs(depth_out, exist_ok=True)

    dataframes = []
    
    for folder in os.listdir(raw_root):
        folder_path = os.path.join(raw_root, folder)
        # Skip non-directories or the 'data' folder itself
        if not os.path.isdir(folder_path) or folder == 'data':
            continue

        # Assume folder ends with set ID (e.g., 'set1')
        set_id = folder[-1:] 
        
        # Source Paths
        src_rgb_dir = os.path.join(folder_path, 'RGB_black_background_dataset')
        src_depth_dir = os.path.join(folder_path, 'Depth_black_background_dataset')
        src_csv = os.path.join(folder_path, 'black_background_dataset.csv')
        
        if not os.path.exists(src_rgb_dir): continue

        # 1. Process CSV
        if os.path.exists(src_csv):
            df = pd.read_csv(src_csv)
            # Rename picture column to match new naming convention
            df['Picture name'] = df['Picture name'].apply(lambda x: f"set{set_id}_img{x.split('_')[-1]}")
            dataframes.append(df)

        # 2. Copy & Rename Images
        files = sorted([f for f in os.listdir(src_rgb_dir) if f.endswith('.png')], 
                       key=lambda x: int(x.split('_')[-1].split('.')[0]))

        for filename in files:
            idx = filename.split('_')[-1].split('.')[0]
            new_name = f"set{set_id}_img{idx}.png"
            
            shutil.copy(os.path.join(src_rgb_dir, filename), os.path.join(rgb_out, new_name))
            shutil.copy(os.path.join(src_depth_dir, filename), os.path.join(depth_out, new_name))

    # Merge Dataframes & Convert Units
    if dataframes:
        combined_df = pd.concat(dataframes, ignore_index=True)
        
        # Convert Radians to Degrees
        if 'Angel [rad]' in combined_df.columns:
            combined_df['Angle [deg]'] = combined_df['Angel [rad]'] * (180 / np.pi)
        
        # Convert Meters to Millimeters
        if 'Radial distance [m]' in combined_df.columns:
            combined_df['Radial distance [mm]'] = combined_df['Radial distance [m]'] * 1000

        # Select final columns
        final_cols = ['Angle [deg]', 'Radial distance [mm]', 'Picture name']
        combined_df = combined_df[[c for c in final_cols if c in combined_df.columns]]
        
        combined_df.to_csv(csv_output_path, index=False)
        print(f"Saved unified CSV to {csv_output_path}")
    else:
        print("Warning: No CSV files found in raw data!")

def step2_center_images(data_folder):
    """
    Step 2: Applies 'center_blue_object' to all images in the processed folder.
    Overwrites the images in place.
    """
    print("\n--- Step 2: Centering Images (ROI) ---")
    rgb_dir = os.path.join(data_folder, 'rgb')
    depth_dir = os.path.join(data_folder, 'depth')
    
    count = 0
    files = os.listdir(rgb_dir)
    for img_name in files:
        rgb_path = os.path.join(rgb_dir, img_name)
        depth_path = os.path.join(depth_dir, img_name)
        
        if not os.path.exists(depth_path): continue
        
        # Read (Important: Load Depth as Unchanged)
        img_rgb = cv2.imread(rgb_path)
        img_depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

        if img_rgb is None or img_depth is None: continue

        # Center
        centered_rgb, centered_depth = center_blue_object(img_rgb, img_depth)

        # Save (Overwrite)
        cv2.imwrite(rgb_path, centered_rgb)
        cv2.imwrite(depth_path, centered_depth)
        count += 1
        
    print(f"Centered {count} images.")

def step3_augment_data(input_folder, output_folder, input_csv, output_csv):
    """
    Step 3: Generates random crops (Augmentation).
    Reads from 'input_folder', writes crops to 'output_folder' and creates a new CSV.
    """
    print("\n--- Step 3: Augmentation (Random Cropping) ---")
    
    os.makedirs(os.path.join(output_folder, 'rgb'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'depth'), exist_ok=True)

    # Load Metadata for fast lookup
    df = pd.read_csv(input_csv)
    # Create dict: 'set1_img5' -> {'Angle...': 10, 'Radial...': 5}
    # Removing .png extension from key if present in CSV Picture name
    meta_data = df.set_index('Picture name')[['Angle [deg]', 'Radial distance [mm]']].to_dict('index')

    new_csv_rows = []
    src_rgb_dir = os.path.join(input_folder, 'rgb')
    src_depth_dir = os.path.join(input_folder, 'depth')

    files = [f for f in os.listdir(src_rgb_dir) if f.endswith('.png')]

    for file_name in files:
        # Load images
        img_rgb = cv2.imread(os.path.join(src_rgb_dir, file_name))
        img_depth = cv2.imread(os.path.join(src_depth_dir, file_name), cv2.IMREAD_UNCHANGED)

        if img_rgb is None or img_depth is None: continue

        # Generate Crops
        crops = random_crop_resize(img_rgb, img_depth, num_crops=3, output_size=(240, 240))
        
        # Get Labels
        name_no_ext = os.path.splitext(file_name)[0]
        if name_no_ext not in meta_data:
            continue
            
        angle = meta_data[name_no_ext]['Angle [deg]']
        radius = meta_data[name_no_ext]['Radial distance [mm]']

        # Save Crops & Update CSV Data
        for i, (crop_rgb, crop_depth) in enumerate(crops):
            new_name = f"{name_no_ext}_crop{i}.png"
            
            cv2.imwrite(os.path.join(output_folder, 'rgb', new_name), crop_rgb)
            cv2.imwrite(os.path.join(output_folder, 'depth', new_name), crop_depth)
            
            new_csv_rows.append([new_name, angle, radius])

    # Save Augmented CSV
    aug_df = pd.DataFrame(new_csv_rows, columns=['Picture name', 'Angle [deg]', 'Radial distance [mm]'])
    aug_df.to_csv(output_csv, index=False)
    print(f"Generated {len(aug_df)} augmented samples.")

# --- MAIN EXECUTION FLOW ---
if __name__ == "__main__":
    # --- Configuration Paths ---
    # Change these paths according to your local setup if needed
    RAW_DATA_DIR = './dataset/original_data/'         # Original folders (set1, set2...)
    INTERMEDIATE_DIR = './dataset/processed_temp/'    # Organized & Centered data
    CROPPED_DIR = './dataset/cropped_data/'           # Augmented data
    FINAL_SPLIT_DIR = './dataset/final_split/'        # Train/Test split folders
    
    # CSV Paths
    INTERMEDIATE_CSV = os.path.join(INTERMEDIATE_DIR, 'data.csv')
    CROPPED_CSV = os.path.join(CROPPED_DIR, 'cropped_data.csv')

    # --- Execute Pipeline ---
    
    # 1. Organize & Rename
    step1_organize_and_process_raw(RAW_DATA_DIR, INTERMEDIATE_DIR, INTERMEDIATE_CSV)
    
    # 2. Center (In-place on intermediate dir)
    step2_center_images(INTERMEDIATE_DIR)
    
    # 3. Augment (Crop)
    step3_augment_data(INTERMEDIATE_DIR, CROPPED_DIR, INTERMEDIATE_CSV, CROPPED_CSV)
    
    # 4. Split Train/Test (External Module)
    print("\n--- Step 4: Train/Test Split ---")
    split_train_test(
        source_csv_path=CROPPED_CSV,
        source_root_folder=CROPPED_DIR,
        output_root_folder=FINAL_SPLIT_DIR,
        test_size=1500,
        seed=42
    )

    # 5. Generate Class Labels (External Module)
    print("\n--- Step 5: Generating Class Labels (Discretization) ---")
    train_csv_path = os.path.join(FINAL_SPLIT_DIR, 'train', 'train_data.csv')
    test_csv_path = os.path.join(FINAL_SPLIT_DIR, 'test', 'test_data.csv')
    
    add_classification_labels(train_csv_path, angle_bin_size=22.5, rad_bin_size=1.0)
    add_classification_labels(test_csv_path, angle_bin_size=22.5, rad_bin_size=1.0)

    print("\n✅ Data Pipeline Completed Successfully!")
    print(f"Ready for training. Data located at: {FINAL_SPLIT_DIR}")