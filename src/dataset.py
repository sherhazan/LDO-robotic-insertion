import os
import torch
import pandas as pd
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

class RGBDDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None, target_type='angle'):
        """
        Args:
            root_dir (string): Directory with 'rgb' and 'depth' subfolders.
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
            target_type (string): 'angle' or 'radius' - determines which label to return.
        """
        self.root_dir = root_dir
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.target_type = target_type
        
        # Validations
        if target_type == 'angle':
            self.label_col = 'label_angle'
            self.cont_col = 'Angle [deg]'
        elif target_type == 'radius':
            self.label_col = 'label_radius'
            self.cont_col = 'Radial distance [mm]'
        else:
            raise ValueError("target_type must be 'angle' or 'radius'")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.data.iloc[idx]['Picture name']
        
        # Paths
        rgb_path = os.path.join(self.root_dir, 'rgb', img_name)
        depth_path = os.path.join(self.root_dir, 'depth', img_name)

        # Load Images
        # RGB: Convert BGR to RGB
        image_rgb = cv2.imread(rgb_path)
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)
        
        # Depth: Load as 16-bit unchanged
        image_depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        
        # Normalize Depth (Simple normalization for now, can be improved based on L515 specs)
        # Converting to float and normalizing to 0-1 range generally
        image_depth = image_depth.astype(np.float32) / 65535.0 
        # Add channel dimension to depth (H, W) -> (H, W, 1)
        image_depth = np.expand_dims(image_depth, axis=2)

        # Get Labels
        class_label = int(self.data.iloc[idx][self.label_col])
        continuous_val = float(self.data.iloc[idx][self.cont_col])

        # Apply Transforms (e.g. ToTensor)
        if self.transform:
            # Transforms usually expect PIL images or specific shapes. 
            # For custom RGBD, we usually implement custom transforms or just ToTensor
            to_tensor = transforms.ToTensor()
            image_rgb = to_tensor(image_rgb)
            image_depth = to_tensor(image_depth)

        return {
            'rgb': image_rgb,       # Shape: (3, H, W)
            'depth': image_depth,   # Shape: (1, H, W)
            'label': class_label,   # LongTensor for CrossEntropyLoss
            'continuous': continuous_val, # For interpolation later
            'name': img_name
        }