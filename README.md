# LipReader

A comprehensive lip reading pipeline for reconstructing words/speech from video of a person talking with no sound.

## Overview

LipReader is a Python-based system that uses computer vision and deep learning to read lips from video footage and predict the spoken words. The pipeline processes video frames, extracts lip regions, generates visual features, and uses a sequence-to-sequence model to predict text.

## Features

- **Video Preprocessing**: Automatic face and lip region detection using MediaPipe
- **Feature Extraction**: CNN-based visual feature extraction from lip movements
- **Sequence Modeling**: LSTM-based model with attention for temporal sequence processing
- **End-to-End Pipeline**: Simple API for processing videos from input to text output
- **Modular Design**: Easy to extend and customize each component
- **Batch Processing**: Support for processing multiple videos efficiently

## Architecture

The pipeline consists of four main components:

1. **VideoPreprocessor**: Extracts and normalizes lip regions from video frames
2. **FeatureExtractor**: Converts lip images to feature vectors using CNNs
3. **LipReadingModel**: Predicts words/phrases using LSTM with attention
4. **LipReaderPipeline**: Orchestrates the entire process

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/ZieBerthe/LipReader.git
cd LipReader
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package in development mode:
```bash
pip install -e .
```

## Quick Start

### Basic Usage

```python
from lipreader import LipReaderPipeline

# Initialize the pipeline
pipeline = LipReaderPipeline()

# Process a video
results = pipeline.process_video("path/to/video.mp4")

# Get predictions
print(f"Predicted text: {results['predicted_text']}")
print(f"Top predictions: {results['predictions']}")
```

### Using the Demo Script

```bash
# Run the demo with a video file
python examples/demo.py path/to/your/video.mp4
```

### Custom Configuration

```python
from lipreader import LipReaderPipeline, Config

# Create custom configuration
config = Config()
config.TARGET_FPS = 30
config.FEATURE_DIM = 1024

# Initialize pipeline with custom config
pipeline = LipReaderPipeline(config)
```

## Project Structure

```
LipReader/
├── src/lipreader/          # Main package
│   ├── __init__.py         # Package initialization
│   ├── config.py           # Configuration settings
│   ├── preprocessor.py     # Video preprocessing
│   ├── feature_extractor.py # Feature extraction
│   ├── model.py            # Lip reading model
│   └── pipeline.py         # Main pipeline
├── tests/                  # Unit tests
│   ├── test_preprocessor.py
│   ├── test_feature_extractor.py
│   ├── test_model.py
│   └── test_pipeline.py
├── examples/               # Example scripts
│   └── demo.py            # Demo script
├── data/                   # Data directory (videos, datasets)
├── models/                 # Trained model weights
├── requirements.txt        # Python dependencies
├── setup.py               # Package setup
└── README.md              # This file
```

## Testing

Run the test suite:

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=lipreader --cov-report=html
```

## Training Your Own Model

This repository provides the pipeline architecture. To train your own model:

1. **Collect a dataset**: Use public datasets like:
   - LRW (Lip Reading in the Wild)
   - LRS2/LRS3 (Lip Reading Sentences)
   - GRID Corpus

2. **Prepare the data**: Extract lip regions and create feature vectors

3. **Train the model**: Implement a training script using the provided architecture

4. **Load trained weights**:
```python
pipeline = LipReaderPipeline()
pipeline.load_model_weights("path/to/trained_model.pth")
```

## Configuration Options

Key configuration parameters in `config.py`:

- `TARGET_FPS`: Frame rate for video processing (default: 25)
- `LIP_REGION_SIZE`: Size of extracted lip regions (default: 128x128)
- `FEATURE_DIM`: Dimension of feature vectors (default: 512)
- `SEQUENCE_LENGTH`: Maximum frames to process (default: 75)
- `DEVICE`: Computing device "cpu" or "cuda" (default: "cpu")

## Performance Notes

- The current model uses randomly initialized weights for demonstration
- For production use, train the model on a proper lip reading dataset
- GPU acceleration is recommended for real-time processing
- Typical processing time: ~1-2 seconds per second of video (on CPU)

## Known Limitations

- Requires clear frontal view of face for best results
- Performance depends on video quality and lighting
- Current vocabulary is limited to common words
- Requires trained weights for accurate predictions

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Acknowledgments

- MediaPipe for face and landmark detection
- PyTorch for deep learning framework
- OpenCV for video processing

## Citation

If you use this code in your research, please cite:

```bibtex
@software{lipreader2026,
  title={LipReader: A Lip Reading Pipeline},
  author={LipReader Team},
  year={2026},
  url={https://github.com/ZieBerthe/LipReader}
}
```

## Contact

For questions or issues, please open an issue on GitHub.
