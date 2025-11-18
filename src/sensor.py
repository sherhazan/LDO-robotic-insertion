import numpy as np
import torch

def estimate_continuous_value(probability_vector, target_type, threshold=0.01):
    """
    Converts a discrete probability vector into a continuous value using interpolation.
    Includes noise suppression (thresholding).
    """
    # 1. Thresholding (Suppress noise)
    # If prob < threshold, set to 0
    prob = np.where(probability_vector < threshold, 0, probability_vector)
    
    # 2. Re-normalize
    if np.sum(prob) == 0:
        return 0.0 # Safety fallback
    prob = prob / np.sum(prob)
    
    num_classes = len(prob)

    if target_type == 'radius':
        # Linear interpolation: Dot product of probabilities and steps
        # Assuming domain 0 to 6 mm
        domain_steps = np.linspace(0, 6, num_classes) 
        estimated_value = np.dot(domain_steps, prob)
        
    elif target_type == 'angle':
        # Circular Mean Interpolation
        step_rad = 2 * np.pi / num_classes
        # Calculate middle of each slice
        mid_slices_rad = [step_rad/2 + step_rad * i for i in range(num_classes)]
        
        sin_vals = np.sin(mid_slices_rad)
        cos_vals = np.cos(mid_slices_rad)
        
        dot_sin = np.dot(sin_vals, prob)
        dot_cos = np.dot(cos_vals, prob)
        
        # Arctan2 handles the quadrants correctly
        angle_rad = np.arctan2(dot_sin, dot_cos)
        estimated_value = np.rad2deg(angle_rad)
        
        # Normalize to [0, 360)
        if estimated_value < 0:
            estimated_value += 360.0
            
    else:
        raise ValueError(f"Unknown target_type: {target_type}")
        
    return estimated_value

def get_probabilities(model, rgb, depth, device, model_type):
    """Helper to run inference and get softmax probabilities"""
    model.eval()
    with torch.no_grad():
        # Prepare inputs based on model type
        rgb = rgb.to(device)
        depth = depth.to(device)
        
        if model_type == 'Lrgbd':
            # Late fusion expects two inputs
            out = model(rgb, depth)
        elif model_type == 'rgb':
            out = model(rgb)
        elif model_type == 'depth':
            out = model(depth)
        elif model_type == 'Ergbd':
             # Early fusion expects concat
            inp = torch.cat([rgb, depth], dim=1)
            out = model(inp)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Softmax to get probabilities
        prob = torch.nn.functional.softmax(out, dim=1)
        return prob[0].cpu().numpy()