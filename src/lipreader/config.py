"""
Configuration settings for the lip reader pipeline.
"""

import os
from typing import Dict, Any


class Config:
    """Configuration class for lip reading pipeline."""
    
    # Video processing settings
    TARGET_FPS = 25
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    
    # Lip region settings
    LIP_REGION_SIZE = (128, 128)
    LIP_REGION_PADDING = 20
    
    # Feature extraction settings
    FEATURE_DIM = 512
    SEQUENCE_LENGTH = 75  # Maximum number of frames to process
    
    # Model settings
    MODEL_PATH = os.path.join("models", "lipreading_model.pth")
    DEVICE = "cpu"  # "cuda" if GPU available
    
    # Face detection settings
    FACE_DETECTION_CONFIDENCE = 0.5
    
    # Output settings
    OUTPUT_DIR = "output"
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create config from dictionary."""
        config = cls()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            key: value
            for key, value in vars(self).items()
            if not key.startswith("_") and not callable(value)
        }
