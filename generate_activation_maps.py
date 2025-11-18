import torch
import os
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

# Import our modules
from src.model import build_model
from src.visualization.activation import ActivationMapGenerator, save_activation_grid
import pandas as pd

# --- Configuration ---
CONFIG = {
    'case': 'rgb_rad',      # Options: 'rgb_rad', 'depth_angle', 'Lrgbd_angle'...
    'ver': 3,
    'data_root': './dataset/final_split/test', # Using the organized test set
    'weights_dir': './results/models',
    'output_dir': './results/figures/activations',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

def main():
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    # Parse Case
    parts = CONFIG['case'].split('_')
    model_type = parts[0]   # rgb, depth, Lrgbd
    target_type = parts[1]  # angle, rad
    
    num_classes = 16 if target_type == 'angle' else 6
    
    print(f"--- Generating Activations for {CONFIG['case']} ---")

    # 1. Load Model
    model = build_model(model_type, num_classes)
    weight_path = os.path.join(CONFIG['weights_dir'], f"{CONFIG['case']}_v{CONFIG['ver']}_best.pth")
    
    try:
        model.load_state_dict(torch.load(weight_path, map_location=CONFIG['device']))
        print("Weights loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Weights not found at {weight_path}")
        return

    # 2. Initialize Generator
    cam_gen = ActivationMapGenerator(model, model_type, CONFIG['device'])
    
    # 3. Load Test Data (From CSV to get correct file names)
    csv_path = os.path.join(CONFIG['data_root'], 'test_data.csv')
    df = pd.read_csv(csv_path)
    
    # Select 9 random samples or specific ones
    # df = df.sample(9, random_state=42) 
    samples = df.iloc[:9] # Just taking first 9 for demo

    results = []
    
    for _, row in samples.iterrows():
        img_name = row['Picture name']
        rgb_path = os.path.join(CONFIG['data_root'], 'rgb', img_name)
        depth_path = os.path.join(CONFIG['data_root'], 'depth', img_name)
        
        if not os.path.exists(rgb_path): continue

        # --- Preprocessing ---
        # Load RGB
        rgb_cv = cv2.imread(rgb_path)
        rgb_cv = cv2.cvtColor(rgb_cv, cv2.COLOR_BGR2RGB)
        t_rgb = transforms.ToTensor()(rgb_cv) # (3, H, W)
        
        # Load Depth
        depth_cv = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        # Normalize depth similarly to training
        depth_norm = depth_cv.astype("float32") / 65535.0
        t_depth = torch.from_numpy(depth_norm).unsqueeze(0) # (1, H, W)
        
        # --- Input Preparation ---
        input_tensor = None
        overlay_bg = None
        
        if model_type == 'rgb':
            input_tensor = t_rgb.unsqueeze(0) # Batch dim
            overlay_bg = Image.fromarray(rgb_cv)
            
        elif model_type == 'depth':
            input_tensor = t_depth.unsqueeze(0) # Batch dim
            # For depth, we use the normalized depth as background, converted to RGB
            depth_vis = (depth_norm * 255).astype(np.uint8)
            overlay_bg = Image.fromarray(depth_vis).convert("RGB")
            
        elif model_type == 'Lrgbd':
            # Lrgbd expects separate inputs. 
            # Note: ActivationMapGenerator handles the tuple logic inside .generate()
            input_tensor = [t_rgb.unsqueeze(0).to(CONFIG['device']), 
                            t_depth.unsqueeze(0).to(CONFIG['device'])]
            overlay_bg = Image.fromarray(rgb_cv)

        # --- Generate ---
        if input_tensor is not None:
            heatmap = cam_gen.generate(input_tensor, input_img_for_overlay=overlay_bg)
            results.append(heatmap)

    # 4. Save Grid
    if results:
        save_name = f"{CONFIG['case']}_activation_map.png"
        save_path = os.path.join(CONFIG['output_dir'], save_name)
        save_activation_grid(results, save_path)

if __name__ == "__main__":
    main()