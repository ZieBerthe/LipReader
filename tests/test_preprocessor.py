"""
Tests for the video preprocessor module.
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lipreader.preprocessor import VideoPreprocessor
from lipreader.config import Config


class TestVideoPreprocessor:
    """Test suite for VideoPreprocessor."""
    
    def test_initialization(self):
        """Test preprocessor initialization."""
        preprocessor = VideoPreprocessor()
        assert preprocessor is not None
        assert preprocessor.config is not None
        assert preprocessor.face_mesh is not None
    
    def test_initialization_with_config(self):
        """Test preprocessor initialization with custom config."""
        config = Config()
        config.TARGET_FPS = 30
        preprocessor = VideoPreprocessor(config)
        assert preprocessor.config.TARGET_FPS == 30
    
    def test_detect_lip_region_with_fake_frame(self):
        """Test lip region detection with a simple frame."""
        preprocessor = VideoPreprocessor()
        
        # Create a blank frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Detection should return None or a bbox (depends on MediaPipe)
        bbox = preprocessor.detect_lip_region(frame)
        # Just check it doesn't crash
        assert bbox is None or isinstance(bbox, tuple)
    
    def test_extract_lip_regions_empty_list(self):
        """Test extracting lip regions from empty frame list."""
        preprocessor = VideoPreprocessor()
        frames = []
        lip_regions = preprocessor.extract_lip_regions(frames)
        assert len(lip_regions) == 0
    
    def test_extract_lip_regions_single_frame(self):
        """Test extracting lip regions from single frame."""
        preprocessor = VideoPreprocessor()
        
        # Create a simple frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frames = [frame]
        
        lip_regions = preprocessor.extract_lip_regions(frames)
        assert len(lip_regions) == 1
        assert lip_regions[0].shape == preprocessor.config.LIP_REGION_SIZE


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
