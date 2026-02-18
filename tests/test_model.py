"""
Tests for the lip reading model.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lipreader.model import LipReadingModel
from lipreader.config import Config


class TestLipReadingModel:
    """Test suite for LipReadingModel."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = LipReadingModel()
        assert model is not None
        assert model.config is not None
        assert model.model is not None
        assert model.vocabulary is not None
        assert len(model.vocabulary) > 0
    
    def test_vocabulary_contains_special_tokens(self):
        """Test that vocabulary contains special tokens."""
        model = LipReadingModel()
        
        assert "<pad>" in model.vocabulary
        assert "<sos>" in model.vocabulary
        assert "<eos>" in model.vocabulary
        assert "<unk>" in model.vocabulary
    
    def test_predict_with_random_features(self):
        """Test prediction with random features."""
        config = Config()
        model = LipReadingModel(config)
        
        # Create random features
        num_frames = 20
        features = np.random.randn(num_frames, config.FEATURE_DIM).astype(np.float32)
        
        predictions = model.predict(features, top_k=5)
        
        assert len(predictions) == 5
        assert all(isinstance(p, tuple) and len(p) == 2 for p in predictions)
        assert all(isinstance(p[0], str) and isinstance(p[1], float) for p in predictions)
    
    def test_predict_sequence(self):
        """Test sequence prediction."""
        config = Config()
        model = LipReadingModel(config)
        
        # Create random features
        num_frames = 20
        features = np.random.randn(num_frames, config.FEATURE_DIM).astype(np.float32)
        
        predicted_text = model.predict_sequence(features)
        
        assert isinstance(predicted_text, str)
        assert len(predicted_text) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
