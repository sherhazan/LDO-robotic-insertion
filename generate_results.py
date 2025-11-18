import os
import torch
import pandas as pd
import numpy as np
import cv2
from torchvision import transforms

# Import our modules
from src.model import build_model
from src.sensor import estimate_continuous_value, get_probabilities

# --- CONFIGURATION ---
CONFIG = {
    'case': 'Prgbd',       # Can be 'Lrgbd', 'Prgbd', 'rgb', 'depth', 'Ergbd'
    'ver': 1,              # Version for the output folder name
    'weights_ver': 3,      # Version of the trained weights file
    
    'test_dir': './dataset/test',
    'csv_path': './dataset/test/test_data.csv',
    'weights_dir': './Weights',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

def load_trained_model(model_type, target_type, version, device):
    """Loads a specific model (e.g., rgb_angle_v3.pth)"""
    num_classes = 16 if target_type == 'angle' else 6
    model = build_model(model_type, num_classes)
    
    # Construct weight path based on naming convention: "{type}_{target}_v{ver}.pth"
    # Example: Lrgbd_angle_v1.pth OR rgb_rad_v3.pth
    weight_name = f"{model_type}_{target_type}_v{version}.pth"
    weight_path = os.path.join(CONFIG['weights_dir'], weight_name)
    
    try:
        model.load_state_dict(torch.load(weight_path, map_location=device))
        model.to(device)
        model.eval()
        print(f"Loaded: {weight_name}")
        return model
    except FileNotFoundError:
        print(f"CRITICAL ERROR: Weight file not found: {weight_path}")
        exit()

def prepare_tensors(rgb_path, depth_path):
    """Reads images and converts to tensors with batch dim"""
    # Read images
    rgb_cv = cv2.imread(rgb_path)
    rgb_cv = cv2.cvtColor(rgb_cv, cv2.COLOR_BGR2RGB)
    depth_cv = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    
    # Transforms
    t_rgb = transforms.ToTensor()(rgb_cv).unsqueeze(0) # (1, 3, H, W)
    
    # Depth Normalization & Grayscale
    depth_norm = depth_cv.astype("float32") / 65535.0
    t_depth = torch.from_numpy(depth_norm).unsqueeze(0).unsqueeze(0) # (1, 1, H, W)
    
    return t_rgb, t_depth

def main():
    # 1. Setup Output
    output_dir = f"./results/{CONFIG['case']}_v{CONFIG['ver']}"
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, 'graphs_data.csv')
    
    device = CONFIG['device']
    is_parallel = CONFIG['case'].startswith('P') # Check if Parallel Fusion
    
    # 2. Load Models
    models = {}
    
    if is_parallel:
        print("--- Running PARALLEL Fusion (4 Models) ---")
        # Parallel uses standard RGB and Depth models separately
        # Load 4 models: RGB-Angle, Depth-Angle, RGB-Rad, Depth-Rad
        models['rgb_angle'] = load_trained_model('rgb', 'angle', CONFIG['weights_ver'], device)
        models['depth_angle'] = load_trained_model('depth', 'angle', CONFIG['weights_ver'], device)
        models['rgb_rad'] = load_trained_model('rgb', 'rad', CONFIG['weights_ver'], device)
        models['depth_rad'] = load_trained_model('depth', 'rad', CONFIG['weights_ver'], device)
    else:
        print(f"--- Running STANDARD Inference ({CONFIG['case']}) ---")
        # Load 2 models: Case-Angle, Case-Rad
        # E.g., Lrgbd_angle, Lrgbd_rad
        model_type = CONFIG['case'] # e.g., 'Lrgbd'
        models['angle'] = load_trained_model(model_type, 'angle', CONFIG['weights_ver'], device)
        models['rad'] = load_trained_model(model_type, 'rad', CONFIG['weights_ver'], device)

    # 3. Process Data
    df = pd.read_csv(CONFIG['csv_path'])
    results = []
    
    print(f"Processing {len(df)} samples...")
    
    for idx, row in df.iterrows():
        img_name = row['Picture name']
        rgb_p = os.path.join(CONFIG['test_dir'], 'rgb', img_name)
        depth_p = os.path.join(CONFIG['test_dir'], 'depth', img_name)
        
        if not os.path.exists(rgb_p): continue
        
        # Prepare Inputs
        t_rgb, t_depth = prepare_tensors(rgb_p, depth_p)
        
        # --- INFERENCE LOGIC ---
        
        if is_parallel:
            # Get probabilities from each stream
            p_rgb_ang = get_probabilities(models['rgb_angle'], t_rgb, t_depth, device, 'rgb')
            p_depth_ang = get_probabilities(models['depth_angle'], t_rgb, t_depth, device, 'depth')
            
            p_rgb_rad = get_probabilities(models['rgb_rad'], t_rgb, t_depth, device, 'rgb')
            p_depth_rad = get_probabilities(models['depth_rad'], t_rgb, t_depth, device, 'depth')
            
            # Average the distributions (Fusion happens here!)
            prob_angle = (p_rgb_ang + p_depth_ang) / 2.0
            prob_rad = (p_rgb_rad + p_depth_rad) / 2.0
            
        else:
            # Standard case
            model_type = CONFIG['case']
            prob_angle = get_probabilities(models['angle'], t_rgb, t_depth, device, model_type)
            prob_rad = get_probabilities(models['rad'], t_rgb, t_depth, device, model_type)

        # --- INTERPOLATION ---
        pred_angle = estimate_continuous_value(prob_angle, 'angle', threshold=0.01)
        pred_rad = estimate_continuous_value(prob_rad, 'radius', threshold=0.05)
        
        # Post-processing (Angle wrap handling for error calc - visually)
        gt_angle = row['Angle [deg]']
        if np.abs(pred_angle - gt_angle) > 180:
            pred_angle_corrected = 360 - pred_angle # Just for logic, usually we store raw
        else:
            pred_angle_corrected = pred_angle

        # Save result
        results.append({
            'Picture name': img_name,
            'gt_rad': row['Radial distance [mm]'],
            'pred_rad': pred_rad,
            'gt_angle': gt_angle,
            'pred_angle': pred_angle # Saving raw prediction
        })
        
        if idx % 200 == 0:
            print(f"Processed {idx}...")

    # 4. Save CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    print(f"Done! Results saved to: {output_csv}")

if __name__ == "__main__":
    main()