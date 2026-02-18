"""
Tests for the main pipeline.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lipreader.pipeline import LipReaderPipeline
from lipreader.config import Config


class TestLipReaderPipeline:
    """Test suite for LipReaderPipeline."""
    
    def test_initialization(self):
        """Test pipeline initialization."""
        pipeline = LipReaderPipeline()
        assert pipeline is not None
        assert pipeline.config is not None
        assert pipeline.preprocessor is not None
        assert pipeline.feature_extractor is not None
        assert pipeline.model is not None
    
    def test_get_pipeline_info(self):
        """Test getting pipeline information."""
        pipeline = LipReaderPipeline()
        info = pipeline.get_pipeline_info()
        
        assert "version" in info
        assert "config" in info
        assert "components" in info
        assert "vocabulary_size" in info
        assert info["vocabulary_size"] > 0
    
    def test_pipeline_with_custom_config(self):
        """Test pipeline with custom configuration."""
        config = Config()
        config.TARGET_FPS = 30
        pipeline = LipReaderPipeline(config)
        
        assert pipeline.config.TARGET_FPS == 30


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
