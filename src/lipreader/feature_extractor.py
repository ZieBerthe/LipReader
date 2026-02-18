"""
Feature extraction module for converting lip regions to features.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional

from .config import Config


class FeatureExtractor:
    """Extracts features from preprocessed lip regions."""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the feature extractor.
        
        Args:
            config: Configuration object. If None, uses default Config.
        """
        self.config = config or Config()
        self.device = torch.device(self.config.DEVICE)
        
        # Simple CNN-based feature extractor
        self.model = self._build_feature_extractor()
        self.model.to(self.device)
        self.model.eval()
    
    def _build_feature_extractor(self) -> nn.Module:
        """
        Build a simple CNN for feature extraction.
        
        Returns:
            PyTorch module for feature extraction.
        """
        class SimpleCNN(nn.Module):
            def __init__(self, feature_dim=512):
                super().__init__()
                self.features = nn.Sequential(
                    # Conv block 1
                    nn.Conv2d(1, 32, kernel_size=3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                    
                    # Conv block 2
                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                    
                    # Conv block 3
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                    
                    # Conv block 4
                    nn.Conv2d(128, 256, kernel_size=3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                )
                
                # Calculate feature map size (128x128 -> 8x8 after 4 pooling layers)
                self.flatten = nn.Flatten()
                self.fc = nn.Linear(256 * 8 * 8, feature_dim)
            
            def forward(self, x):
                x = self.features(x)
                x = self.flatten(x)
                x = self.fc(x)
                return x
        
        return SimpleCNN(feature_dim=self.config.FEATURE_DIM)
    
    def extract_features(self, lip_regions: np.ndarray) -> np.ndarray:
        """
        Extract features from lip regions.
        
        Args:
            lip_regions: Numpy array of shape (T, H, W) containing lip regions.
            
        Returns:
            Feature array of shape (T, feature_dim).
        """
        # Normalize to [0, 1]
        lip_regions = lip_regions.astype(np.float32) / 255.0
        
        # Convert to torch tensor: (T, 1, H, W)
        lip_tensor = torch.from_numpy(lip_regions).unsqueeze(1)
        lip_tensor = lip_tensor.to(self.device)
        
        # Extract features frame by frame
        features = []
        with torch.no_grad():
            for i in range(lip_tensor.size(0)):
                frame = lip_tensor[i:i+1]
                feat = self.model(frame)
                features.append(feat.cpu().numpy())
        
        features = np.concatenate(features, axis=0)
        return features
    
    def extract_features_batch(self, lip_regions: np.ndarray, batch_size: int = 16) -> np.ndarray:
        """
        Extract features from lip regions using batching for efficiency.
        
        Args:
            lip_regions: Numpy array of shape (T, H, W) containing lip regions.
            batch_size: Batch size for processing.
            
        Returns:
            Feature array of shape (T, feature_dim).
        """
        # Normalize to [0, 1]
        lip_regions = lip_regions.astype(np.float32) / 255.0
        
        # Convert to torch tensor: (T, 1, H, W)
        lip_tensor = torch.from_numpy(lip_regions).unsqueeze(1)
        lip_tensor = lip_tensor.to(self.device)
        
        # Extract features in batches
        features = []
        with torch.no_grad():
            for i in range(0, lip_tensor.size(0), batch_size):
                batch = lip_tensor[i:i+batch_size]
                feat = self.model(batch)
                features.append(feat.cpu().numpy())
        
        features = np.concatenate(features, axis=0)
        return features
