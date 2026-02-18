"""
Example script demonstrating the lip reader pipeline.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lipreader import LipReaderPipeline, Config


def main():
    """Main function to demonstrate the pipeline."""
    
    print("=" * 60)
    print("Lip Reader Pipeline Demo")
    print("=" * 60)
    
    # Create pipeline with default configuration
    config = Config()
    pipeline = LipReaderPipeline(config)
    
    # Print pipeline information
    print("\nPipeline Information:")
    info = pipeline.get_pipeline_info()
    print(f"  Version: {info['version']}")
    print(f"  Vocabulary size: {info['vocabulary_size']}")
    print(f"  Device: {config.DEVICE}")
    print(f"  Target FPS: {config.TARGET_FPS}")
    print(f"  Lip region size: {config.LIP_REGION_SIZE}")
    
    # Check for video file
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        
        if not Path(video_path).exists():
            print(f"\nError: Video file not found: {video_path}")
            print("\nUsage: python demo.py <path_to_video>")
            sys.exit(1)
        
        print(f"\nProcessing video: {video_path}")
        print("-" * 60)
        
        # Process the video
        results = pipeline.process_video(video_path)
        
        print("\n" + "=" * 60)
        print("Processing Complete!")
        print("=" * 60)
        
        if results["status"] == "success":
            print(f"\nPredicted text: {results['predicted_text']}")
            print(f"Number of frames processed: {results['num_frames']}")
            print("\nTop predictions:")
            for i, (word, prob) in enumerate(results['predictions'], 1):
                print(f"  {i}. {word:15s} (confidence: {prob:.4f})")
        else:
            print(f"\nError: {results.get('error', 'Unknown error')}")
    
    else:
        print("\nNo video file provided.")
        print("Usage: python demo.py <path_to_video>")
        print("\nExample:")
        print("  python demo.py data/sample_video.mp4")
        print("\nNote: This is a demonstration pipeline.")
        print("For production use, you'll need to:")
        print("  1. Collect and prepare a lip reading dataset")
        print("  2. Train the model on the dataset")
        print("  3. Load the trained weights using pipeline.load_model_weights()")


if __name__ == "__main__":
    main()
