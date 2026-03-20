"""
Phase 5: Dual-View CNN Architecture
Breast Cancer Detection - Dual-View CNN with ResNet50

This module defines the neural network architecture for breast cancer detection.
The model uses two ResNet50 branches for CC and MLO views with feature fusion.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path

# ==================== DATASET CLASS ====================

class DualViewMammographyDataset(Dataset):
    """Dataset class for dual-view mammography images"""
    
    def __init__(self, image_files, labels, data_dir):
        """
        Args:
            image_files: List of image filenames
            labels: List of labels (0 or 1)
            data_dir: Path to directory containing preprocessed images
        """
        self.image_files = image_files
        self.labels = labels
        self.data_dir = Path(data_dir)
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load the preprocessed dual-view image
        img_path = self.data_dir / self.image_files[idx]
        dual_view = np.load(str(img_path))  # Shape: (2, 512, 512)
        
        # Convert to torch tensor
        image = torch.from_numpy(dual_view).float()
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return image, label


# ==================== DUAL-VIEW CNN MODEL ====================

class DualViewCNN(nn.Module):
    """
    Dual-View CNN for Breast Cancer Detection
    
    Architecture:
    - Two ResNet50 branches (CC and MLO views)
    - Feature extraction from both branches
    - Feature fusion via concatenation
    - Classification head with dropout
    """
    
    def __init__(self, num_classes=2, dropout_rate=0.5, freeze_early_layers=True):
        """
        Initialize the dual-view CNN model
        
        Args:
            num_classes: Number of output classes (2 for binary: benign/malignant)
            dropout_rate: Dropout rate for regularization
            freeze_early_layers: Whether to freeze early ResNet50 layers
        """
        super(DualViewCNN, self).__init__()
        
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Load pretrained ResNet50 models
        resnet50_cc = models.resnet50(pretrained=True)
        resnet50_mlo = models.resnet50(pretrained=True)
        
        # Modify first layer to accept grayscale (1 channel) -> (3 channels)
        # Method: Repeat the single channel 3 times
        original_conv1 = resnet50_cc.conv1
        self.conv1_cc = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        # Initialize with pretrained weights (average across channels)
        with torch.no_grad():
            self.conv1_cc.weight.copy_(original_conv1.weight.mean(dim=1, keepdim=True))
        
        original_conv1_mlo = resnet50_mlo.conv1
        self.conv1_mlo = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        with torch.no_grad():
            self.conv1_mlo.weight.copy_(original_conv1_mlo.weight.mean(dim=1, keepdim=True))
        
        # Replace conv1 layers
        resnet50_cc.conv1 = self.conv1_cc
        resnet50_mlo.conv1 = self.conv1_mlo
        
        # Remove classification head (avgpool and fc)
        self.cc_branch = nn.Sequential(*list(resnet50_cc.children())[:-2])  # Remove avgpool and fc
        self.mlo_branch = nn.Sequential(*list(resnet50_mlo.children())[:-2])
        
        # Add adaptive average pooling to ensure consistent output size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Feature dimension from ResNet50: 2048
        self.feature_dim = 2048
        self.fused_dim = self.feature_dim * 2  # CC + MLO concatenation
        
        # Freeze early layers if specified
        if freeze_early_layers:
            self._freeze_early_layers()
        
        # Classification head
        self.classification_head = nn.Sequential(
            nn.Linear(self.fused_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            
            nn.Linear(256, num_classes)
        )
    
    def _freeze_early_layers(self):
        """
        Freeze early layers of ResNet50 branches
        Only unfreeze the last residual block (layer4)
        """
        def freeze_module(module):
            for param in module.parameters():
                param.requires_grad = False
        
        # Freeze everything except layer4 in both branches
        freeze_module(self.cc_branch[0])  # conv1
        freeze_module(self.cc_branch[1])  # bn1
        freeze_module(self.cc_branch[2])  # layer1
        freeze_module(self.cc_branch[3])  # layer2
        freeze_module(self.cc_branch[4])  # layer3
        
        freeze_module(self.mlo_branch[0])
        freeze_module(self.mlo_branch[1])
        freeze_module(self.mlo_branch[2])
        freeze_module(self.mlo_branch[3])
        freeze_module(self.mlo_branch[4])
        
        # layer4 (index 5) remains unfrozen for fine-tuning
    
    def unfreeze_early_layers(self, num_layers=50):
        """
        Unfreeze the last num_layers for fine-tuning
        
        Args:
            num_layers: Number of layers to unfreeze (from the end)
        """
        # Count total parameters
        total_params = sum(1 for _ in self.cc_branch.parameters()) + \
                      sum(1 for _ in self.mlo_branch.parameters())
        
        # Calculate which layers to unfreeze
        layers_to_unfreeze = max(0, total_params - num_layers)
        
        # Unfreeze the specified number of layers
        param_count = 0
        for param in list(self.cc_branch.parameters()) + list(self.mlo_branch.parameters()):
            if param_count >= layers_to_unfreeze:
                param.requires_grad = True
            param_count += 1
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, 2, 512, 512)
               Where x[:, 0, :, :] is CC view and x[:, 1, :, :] is MLO view
        
        Returns:
            logits: Classification logits of shape (batch_size, num_classes)
        """
        # Extract CC and MLO views
        cc_view = x[:, 0:1, :, :]  # (batch_size, 1, 512, 512)
        mlo_view = x[:, 1:2, :, :]  # (batch_size, 1, 512, 512)
        
        # Process CC view through CC branch
        cc_features = self.cc_branch(cc_view)  # (batch_size, 2048, H', W')
        cc_features = self.adaptive_pool(cc_features)  # (batch_size, 2048, 1, 1)
        cc_features = cc_features.view(cc_features.size(0), -1)  # (batch_size, 2048)
        
        # Process MLO view through MLO branch
        mlo_features = self.mlo_branch(mlo_view)  # (batch_size, 2048, H', W')
        mlo_features = self.adaptive_pool(mlo_features)  # (batch_size, 2048, 1, 1)
        mlo_features = mlo_features.view(mlo_features.size(0), -1)  # (batch_size, 2048)
        
        # Fuse features via concatenation
        fused_features = torch.cat([cc_features, mlo_features], dim=1)  # (batch_size, 4096)
        
        # Classification head
        logits = self.classification_head(fused_features)  # (batch_size, num_classes)
        
        return logits


# ==================== MODEL INITIALIZATION ====================

def create_dual_view_model(device='cuda', num_classes=2, dropout_rate=0.5):
    """
    Create and initialize the dual-view CNN model
    
    Args:
        device: Device to place model on ('cuda' or 'cpu')
        num_classes: Number of output classes
        dropout_rate: Dropout rate
    
    Returns:
        model: Initialized DualViewCNN model
    """
    model = DualViewCNN(
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        freeze_early_layers=True
    )
    model = model.to(device)
    
    # Print model summary
    print("\n" + "="*80)
    print("DUAL-VIEW CNN MODEL ARCHITECTURE")
    print("="*80)
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print("="*80 + "\n")
    
    return model


# ==================== UTILITY FUNCTIONS ====================

def count_parameters(model):
    """Count total and trainable parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_model_summary_stats(model):
    """Get detailed parameter statistics"""
    total, trainable = count_parameters(model)
    frozen = total - trainable
    
    return {
        'total_parameters': total,
        'trainable_parameters': trainable,
        'frozen_parameters': frozen,
        'trainable_percentage': (trainable / total * 100) if total > 0 else 0
    }
