"""
Tests for the feature extractor module.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lipreader.feature_extractor import FeatureExtractor
from lipreader.config import Config


class TestFeatureExtractor:
    """Test suite for FeatureExtractor."""
    
    def test_initialization(self):
        """Test feature extractor initialization."""
        extractor = FeatureExtractor()
        assert extractor is not None
        assert extractor.config is not None
        assert extractor.model is not None
    
    def test_extract_features_single_frame(self):
        """Test feature extraction from single frame."""
        config = Config()
        extractor = FeatureExtractor(config)
        
        # Create fake lip region
        lip_regions = np.random.randint(0, 256, (1, 128, 128), dtype=np.uint8)
        
        features = extractor.extract_features(lip_regions)
        
        assert features.shape == (1, config.FEATURE_DIM)
        assert features.dtype == np.float32 or features.dtype == np.float64
    
    def test_extract_features_multiple_frames(self):
        """Test feature extraction from multiple frames."""
        config = Config()
        extractor = FeatureExtractor(config)
        
        # Create fake lip regions
        num_frames = 10
        lip_regions = np.random.randint(0, 256, (num_frames, 128, 128), dtype=np.uint8)
        
        features = extractor.extract_features(lip_regions)
        
        assert features.shape == (num_frames, config.FEATURE_DIM)
    
    def test_extract_features_batch(self):
        """Test batch feature extraction."""
        config = Config()
        extractor = FeatureExtractor(config)
        
        # Create fake lip regions
        num_frames = 20
        lip_regions = np.random.randint(0, 256, (num_frames, 128, 128), dtype=np.uint8)
        
        features = extractor.extract_features_batch(lip_regions, batch_size=8)
        
        assert features.shape == (num_frames, config.FEATURE_DIM)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
