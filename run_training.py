import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import os

# Import our modules
from src.dataset import RGBDDataset
from src.model import build_model
from src.train import train_model

# --- Configuration ---
CONFIG = {
    'case': 'depth_angle',   # Options: 'rgb_angle', 'depth_angle', 'Lrgbd_angle', 'Ergbd_radius'...
    'ver': 3,
    'batch_size': 64,
    'lr': 1e-4,
    'epochs': 25,
    'data_root': './dataset/final_split', # Created by preprocess.py
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

def main():
    # Parse case name (e.g., 'Lrgbd_angle' -> model='Lrgbd', target='angle')
    parts = CONFIG['case'].split('_')
    model_type = parts[0]   # 'rgb', 'depth', 'Lrgbd', 'Ergbd'
    target_type = parts[1]  # 'angle', 'radius'
    
    print(f"--- Experiment: {model_type} prediction of {target_type} ---")

    # 1. Transforms
    # Note: We apply ColorJitter only to RGB usually. Depth is handled differently in dataset.py
    train_transform = transforms.Compose([
        transforms.ColorJitter(brightness=.5, hue=.1, saturation=[0.5, 1.5], contrast=[0.5, 1.5]),
        # Resize/Crop is already handled in preprocess or can be added here
    ])

    # 2. Datasets (Using the CSV-based loader we built)
    train_dir = os.path.join(CONFIG['data_root'], 'train')
    test_dir = os.path.join(CONFIG['data_root'], 'test')
    
    train_ds = RGBDDataset(
        root_dir=train_dir, 
        csv_file=os.path.join(train_dir, 'train_data.csv'),
        transform=train_transform,
        target_type=target_type
    )
    
    val_ds = RGBDDataset(
        root_dir=test_dir, 
        csv_file=os.path.join(test_dir, 'test_data.csv'),
        transform=None, # No jitter for validation
        target_type=target_type
    )

    # 3. DataLoaders
    train_dl = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=4)
    val_dl = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=4)
    
    # 4. Build Model
    # Determine number of classes automatically from the data
    # E.g. 0-15 angle classes -> num_classes = 16
    if target_type == 'angle':
        # Assuming 16 classes for angle based on the paper
        num_classes = 16 
    else:
        # Assuming radius max range (e.g. 6mm)
        num_classes = 7 # Adjust logic if needed
        
    model = build_model(model_type, num_classes)
    model.to(CONFIG['device'])

    # 5. Run Training
    experiment_name = f"{CONFIG['case']}_v{CONFIG['ver']}"
    
    train_model(
        model=model,
        train_dl=train_dl,
        val_dl=val_dl,
        epochs=CONFIG['epochs'],
        lr=CONFIG['lr'],
        device=CONFIG['device'],
        model_type=model_type,
        save_dir='./results/models',
        experiment_name=experiment_name
    )

if __name__ == "__main__":
    main()