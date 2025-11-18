import torch
import torch.nn as nn
import torchvision.models as models

class LateFusionNet(nn.Module):
    def __init__(self, num_classes):
        super(LateFusionNet, self).__init__()
        # --- RGB Stream ---
        self.conv_rgb = models.resnet18(weights=None)
        self.conv_rgb.fc = nn.Identity() # Remove original FC
        
        # --- Depth Stream ---
        self.conv_depth = models.resnet18(weights=None)
        # Modify first layer to accept 1 channel instead of 3
        self.conv_depth.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv_depth.fc = nn.Identity() # Remove original FC
        
        # --- Fusion Layers ---
        # ResNet18 output before FC is 512. So 512*2 = 1024
        self.fc1 = nn.Linear(512 * 2, 128)  
        self.dropout = nn.Dropout(0.3) # Added dropout for regularization
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, rgb, depth):
        # RGB Stream
        x_rgb = self.conv_rgb(rgb)
        
        # Depth Stream
        x_depth = self.conv_depth(depth)
        
        # Concatenate
        combined = torch.cat((x_rgb, x_depth), dim=1)
        
        # Fusion
        x = self.fc1(combined)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def build_model(model_type, num_classes):
    """
    Factory function to create the correct model based on configuration.
    model_type: 'rgb', 'depth', 'Lrgbd' (Late Fusion), 'Ergbd' (Early Fusion)
    """
    print(f"Building model: {model_type} with {num_classes} classes.")
    
    if model_type == 'Lrgbd':
        return LateFusionNet(num_classes)
        
    elif model_type == 'rgb':
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(512, num_classes)
        return model
        
    elif model_type == 'depth':
        model = models.resnet18(weights=None)
        # Modify for 1 channel input
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(512, num_classes)
        return model
        
    elif model_type == 'Ergbd': # Early Fusion
        model = models.resnet18(weights=None)
        # Modify for 4 channels input (3 RGB + 1 Depth)
        model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(512, num_classes)
        return model
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")