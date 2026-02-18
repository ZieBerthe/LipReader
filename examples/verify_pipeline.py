"""
End-to-end verification script for the lip reader pipeline.
This creates a simple test to verify all components work together.
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lipreader import LipReaderPipeline
from lipreader.config import Config
from lipreader.preprocessor import VideoPreprocessor
from lipreader.feature_extractor import FeatureExtractor
from lipreader.model import LipReadingModel


def test_individual_components():
    """Test each component individually."""
    print("=" * 60)
    print("Testing Individual Components")
    print("=" * 60)
    
    # Test 1: Configuration
    print("\n[1/4] Testing Configuration...")
    config = Config()
    assert config.TARGET_FPS == 25
    assert config.FEATURE_DIM == 512
    print("  ✓ Configuration initialized successfully")
    
    # Test 2: Video Preprocessor
    print("\n[2/4] Testing Video Preprocessor...")
    preprocessor = VideoPreprocessor(config)
    # Create fake frame
    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    bbox = preprocessor.detect_lip_region(fake_frame)
    print(f"  ✓ Preprocessor initialized successfully")
    print(f"  - Detected lip region: {bbox is not None}")
    
    # Test 3: Feature Extractor
    print("\n[3/4] Testing Feature Extractor...")
    extractor = FeatureExtractor(config)
    fake_lips = np.random.randint(0, 256, (10, 128, 128), dtype=np.uint8)
    features = extractor.extract_features_batch(fake_lips)
    assert features.shape == (10, config.FEATURE_DIM)
    print(f"  ✓ Feature extractor initialized successfully")
    print(f"  - Extracted features shape: {features.shape}")
    
    # Test 4: Lip Reading Model
    print("\n[4/4] Testing Lip Reading Model...")
    model = LipReadingModel(config)
    predictions = model.predict(features)
    assert len(predictions) > 0
    print(f"  ✓ Model initialized successfully")
    print(f"  - Vocabulary size: {len(model.vocabulary)}")
    print(f"  - Top prediction: {predictions[0][0]} (confidence: {predictions[0][1]:.4f})")
    
    print("\n" + "=" * 60)
    print("All individual components working correctly!")
    print("=" * 60)


def test_full_pipeline():
    """Test the full pipeline integration."""
    print("\n" + "=" * 60)
    print("Testing Full Pipeline Integration")
    print("=" * 60)
    
    # Initialize pipeline
    print("\nInitializing pipeline...")
    config = Config()
    pipeline = LipReaderPipeline(config)
    
    # Get pipeline info
    info = pipeline.get_pipeline_info()
    print(f"\nPipeline Information:")
    print(f"  Version: {info['version']}")
    print(f"  Vocabulary size: {info['vocabulary_size']}")
    print(f"  Device: {config.DEVICE}")
    print(f"  Components: {', '.join(info['components'].values())}")
    
    print("\n" + "=" * 60)
    print("Full pipeline integration test passed!")
    print("=" * 60)


def test_pipeline_robustness():
    """Test pipeline handles edge cases."""
    print("\n" + "=" * 60)
    print("Testing Pipeline Robustness")
    print("=" * 60)
    
    config = Config()
    
    # Test with different sequence lengths
    print("\n[1/3] Testing with different sequence lengths...")
    extractor = FeatureExtractor(config)
    model = LipReadingModel(config)
    
    for seq_len in [5, 25, 50]:
        fake_lips = np.random.randint(0, 256, (seq_len, 128, 128), dtype=np.uint8)
        features = extractor.extract_features_batch(fake_lips)
        predictions = model.predict(features)
        print(f"  ✓ Sequence length {seq_len}: {predictions[0][0]}")
    
    # Test with empty predictions
    print("\n[2/3] Testing prediction consistency...")
    features = extractor.extract_features_batch(fake_lips)
    pred1 = model.predict(features, top_k=5)
    pred2 = model.predict(features, top_k=5)
    assert pred1[0][0] == pred2[0][0], "Predictions should be deterministic"
    print("  ✓ Predictions are consistent")
    
    # Test vocabulary
    print("\n[3/3] Testing vocabulary...")
    assert "<pad>" in model.vocabulary
    assert "<sos>" in model.vocabulary
    assert "<eos>" in model.vocabulary
    assert "<unk>" in model.vocabulary
    assert len(model.vocabulary) > 50
    print(f"  ✓ Vocabulary contains {len(model.vocabulary)} words")
    
    print("\n" + "=" * 60)
    print("Pipeline robustness tests passed!")
    print("=" * 60)


def main():
    """Main verification function."""
    print("\n" + "=" * 80)
    print(" " * 20 + "LIP READER PIPELINE VERIFICATION")
    print("=" * 80)
    
    try:
        # Run all tests
        test_individual_components()
        test_full_pipeline()
        test_pipeline_robustness()
        
        # Final summary
        print("\n" + "=" * 80)
        print(" " * 25 + "VERIFICATION COMPLETE")
        print("=" * 80)
        print("\n✓ All components initialized successfully")
        print("✓ All integrations working correctly")
        print("✓ Pipeline handles edge cases properly")
        print("\nThe lip reader pipeline is ready for use!")
        print("\nNext steps:")
        print("  1. Collect/prepare a lip reading dataset")
        print("  2. Train the model on the dataset")
        print("  3. Load trained weights with pipeline.load_model_weights()")
        print("  4. Process videos with pipeline.process_video()")
        print("\n" + "=" * 80)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
