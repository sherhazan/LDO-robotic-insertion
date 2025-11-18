import torch
import os
import pandas as pd
import numpy as np
import cv2
from torchvision import transforms

# Import our modules
from src.model import build_model
from src.sensor import estimate_continuous_value, predict_single_image
from src.visualization.plots import plot_prediction_vs_gt

# --- Configuration ---
CONFIG = {
    'case': 'Lrgbd_angle',  # Ensure this matches your trained model
    'ver': 3,              # Version number used in training
    'data_root': './dataset/final_split/test',
    'weights_dir': './results/models',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

def prepare_input(rgb_path, depth_path, model_type):
    """Loads and preprocesses images for inference"""
    # Load
    rgb = cv2.imread(rgb_path)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    
    # Transforms
    to_tensor = transforms.ToTensor()
    rgb_t = to_tensor(rgb)
    
    # Depth processing
    depth = depth.astype(np.float32) / 65535.0
    depth_t = torch.from_numpy(depth).unsqueeze(0) # (1, H, W)
    
    return rgb_t, depth_t

def main():
    # Parse config
    parts = CONFIG['case'].split('_')
    model_type = parts[0]
    target_type = parts[1]
    
    # Paths
    csv_path = os.path.join(CONFIG['data_root'], 'test_data.csv')
    rgb_dir = os.path.join(CONFIG['data_root'], 'rgb')
    depth_dir = os.path.join(CONFIG['data_root'], 'depth')
    
    weight_path = os.path.join(CONFIG['weights_dir'], f"{CONFIG['case']}_v{CONFIG['ver']}_best.pth")
    
    print(f"--- Evaluating {CONFIG['case']} ---")
    print(f"Loading weights from: {weight_path}")
    
    # 1. Load Model
    # Determine num_classes
    num_classes = 16 if target_type == 'angle' else 6 # Must match training!
    
    model = build_model(model_type, num_classes)
    try:
        model.load_state_dict(torch.load(weight_path, map_location=CONFIG['device']))
    except FileNotFoundError:
        print(f"Error: Model weights not found at {weight_path}")
        return

    model.to(CONFIG['device'])
    model.eval()
    
    # 2. Load Data
    df = pd.read_csv(csv_path)
    ground_truths = []
    predictions = []
    
    print("Running inference...")
    for idx, row in df.iterrows():
        img_name = row['Picture name']
        
        # Get Ground Truth
        if target_type == 'angle':
            gt = row['Angle [deg]']
        else:
            gt = row['Radial distance [mm]']
            
        # Prepare Input
        rgb_p = os.path.join(rgb_dir, img_name)
        depth_p = os.path.join(depth_dir, img_name)
        
        if not os.path.exists(rgb_p): continue
        
        rgb_t, depth_t = prepare_input(rgb_p, depth_p, model_type)
        
        # Handle Model Inputs based on type
        if model_type == 'Lrgbd':
             # Pass separately
            rgb_in = rgb_t.unsqueeze(0).to(CONFIG['device'])
            depth_in = depth_t.unsqueeze(0).to(CONFIG['device'])
            
            with torch.no_grad():
                out = model(rgb_in, depth_in)
                prob = torch.nn.functional.softmax(out, dim=1)[0].cpu().numpy()
                
        else:
            # Concatenate or single input logic here (omitted for brevity, similar to train.py)
            # For now assume Lrgbd as per your snippet
            pass

        # Estimate Continuous Value
        pred = estimate_continuous_value(prob, target_type)
        
        # Angle Wrap-around fix for error calculation (optional visual fix)
        if target_type == 'angle':
            if np.abs(pred - gt) > 180:
                # If pred is 359 and gt is 1, error is huge unless we wrap
                # But for the plot, we usually leave it or wrap the visualization
                pass 

        ground_truths.append(gt)
        predictions.append(pred)
        
        if idx % 100 == 0:
            print(f"Processed {idx}/{len(df)}")

    # 3. Plot
    print("Plotting results...")
    plot_prediction_vs_gt(ground_truths, predictions, target_type, save_path='./results/figures/evaluation.png')

if __name__ == "__main__":
    main()