import torch
from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

class ActivationMapGenerator:
    def __init__(self, model, model_type, device):
        self.model = model
        self.model_type = model_type
        self.device = device
        self.model.eval()
        self.model.to(device)
        
        # --- Smart Hooking Logic ---
        # If it's a standard ResNet (RGB or Depth case), we don't need to specify a layer usually,
        # but specifying 'layer4' ensures we get the last conv layer safely.
        
        target_layer = None
        
        if model_type == 'Lrgbd':
            # For Late Fusion, we visualize the RGB stream's last layer
            target_layer = model.conv_rgb.layer4
        elif model_type == 'Ergbd':
            # Early Fusion is just a ResNet with 4 input channels
            target_layer = model.layer4
        else: 
            # Standard RGB or Depth ResNet
            target_layer = model.layer4
            
        # Initialize CAM
        self.cam_extractor = SmoothGradCAMpp(model, target_layer=target_layer)

    def generate(self, input_tensor, input_img_for_overlay=None):
        """
        Generates the heatmap overlay.
        """
        # Ensure tensor is on device
        input_tensor = input_tensor.to(self.device)
        
        # Forward Pass logic (handles single vs dual input)
        if self.model_type == 'Lrgbd':
            # Assuming input_tensor is a tuple/list [rgb, depth]
            # We need to reconstruct the forward pass to get the logits
            out = self.model(input_tensor[0], input_tensor[1])
        else:
            out = self.model(input_tensor)
            
        # Get the class index with highest probability
        class_idx = out.squeeze(0).argmax().item()
        
        # Generate CAM (Pass class_idx and the model output)
        activation_map = self.cam_extractor(class_idx, out)
        
        # --- Visualization Logic ---
        # If no background image provided, try to reconstruct from tensor
        if input_img_for_overlay is None:
            if self.model_type == 'Lrgbd':
                t = input_tensor[0].squeeze(0) # Take RGB part
            else:
                t = input_tensor.squeeze(0)
            
            # If Depth (1 channel), convert to RGB for prettier plotting
            if t.shape[0] == 1:
                pil_img = to_pil_image(t)
                pil_img = pil_img.convert("RGB")
            else:
                pil_img = to_pil_image(t)
        else:
            pil_img = input_img_for_overlay.convert("RGB")

        # Create Overlay
        # alpha=0.5 makes the heatmap semi-transparent
        result = overlay_mask(pil_img, to_pil_image(activation_map[0], mode='F'), alpha=0.5)
        
        return result

def save_activation_grid(images, save_path, grid_size=(3, 3)):
    """Saves a grid of activation maps"""
    rows, cols = grid_size
    fig, ax = plt.subplots(rows, cols, figsize=(15, 15))
    
    for i in range(rows * cols):
        r, c = divmod(i, cols)
        if i < len(images):
            ax[r, c].imshow(images[i])
        ax[r, c].axis('off')
            
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved activation map grid to {save_path}")
    plt.close()