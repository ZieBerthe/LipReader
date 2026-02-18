"""
Main pipeline for lip reading from video.
"""

import numpy as np
from typing import Optional, Dict, Any, List
from pathlib import Path
import time

from .config import Config
from .preprocessor import VideoPreprocessor
from .feature_extractor import FeatureExtractor
from .model import LipReadingModel


class LipReaderPipeline:
    """End-to-end pipeline for lip reading."""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the lip reader pipeline.
        
        Args:
            config: Configuration object. If None, uses default Config.
        """
        self.config = config or Config()
        
        # Initialize components
        print("Initializing lip reader pipeline...")
        self.preprocessor = VideoPreprocessor(self.config)
        self.feature_extractor = FeatureExtractor(self.config)
        self.model = LipReadingModel(self.config)
        
        print("Pipeline initialized successfully!")
    
    def process_video(
        self,
        video_path: str,
        return_intermediate: bool = False
    ) -> Dict[str, Any]:
        """
        Process a video file and predict the spoken words.
        
        Args:
            video_path: Path to the input video file.
            return_intermediate: If True, returns intermediate results.
            
        Returns:
            Dictionary containing predictions and optional intermediate results.
        """
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        results = {
            "video_path": video_path,
            "status": "success"
        }
        
        try:
            # Step 1: Preprocess video
            print(f"Processing video: {video_path}")
            start_time = time.time()
            
            print("  [1/3] Extracting lip regions...")
            lip_regions = self.preprocessor.preprocess_video(video_path)
            preprocess_time = time.time() - start_time
            print(f"  Extracted {len(lip_regions)} frames in {preprocess_time:.2f}s")
            
            if return_intermediate:
                results["lip_regions"] = lip_regions
            
            # Step 2: Extract features
            print("  [2/3] Extracting features...")
            start_time = time.time()
            features = self.feature_extractor.extract_features_batch(lip_regions)
            feature_time = time.time() - start_time
            print(f"  Extracted features of shape {features.shape} in {feature_time:.2f}s")
            
            if return_intermediate:
                results["features"] = features
            
            # Step 3: Predict words
            print("  [3/3] Predicting words...")
            start_time = time.time()
            predictions = self.model.predict(features, top_k=5)
            predicted_text = self.model.predict_sequence(features)
            prediction_time = time.time() - start_time
            print(f"  Generated predictions in {prediction_time:.2f}s")
            
            results["predictions"] = predictions
            results["predicted_text"] = predicted_text
            results["num_frames"] = len(lip_regions)
            
            # Print results
            print("\nPrediction Results:")
            print(f"  Top prediction: {predicted_text}")
            print(f"  Top 5 predictions:")
            for i, (word, prob) in enumerate(predictions, 1):
                print(f"    {i}. {word} (confidence: {prob:.4f})")
            
        except Exception as e:
            results["status"] = "error"
            results["error"] = str(e)
            print(f"Error during processing: {e}")
            raise
        
        return results
    
    def process_video_batch(
        self,
        video_paths: List[str],
        return_intermediate: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Process multiple videos.
        
        Args:
            video_paths: List of paths to video files.
            return_intermediate: If True, returns intermediate results.
            
        Returns:
            List of result dictionaries, one per video.
        """
        results = []
        
        print(f"Processing {len(video_paths)} videos...")
        for i, video_path in enumerate(video_paths, 1):
            print(f"\n--- Processing video {i}/{len(video_paths)} ---")
            try:
                result = self.process_video(video_path, return_intermediate)
                results.append(result)
            except Exception as e:
                print(f"Failed to process {video_path}: {e}")
                results.append({
                    "video_path": video_path,
                    "status": "error",
                    "error": str(e)
                })
        
        return results
    
    def load_model_weights(self, model_path: Optional[str] = None):
        """
        Load pre-trained model weights.
        
        Args:
            model_path: Path to model weights file.
        """
        self.model.load_weights(model_path)
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """
        Get information about the pipeline configuration.
        
        Returns:
            Dictionary with pipeline information.
        """
        return {
            "version": "0.1.0",
            "config": self.config.to_dict(),
            "components": {
                "preprocessor": "VideoPreprocessor",
                "feature_extractor": "FeatureExtractor",
                "model": "LipReadingModel"
            },
            "vocabulary_size": len(self.model.vocabulary)
        }
