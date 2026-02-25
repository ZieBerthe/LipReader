import json
import os
import numpy as np
import torch
from typing import Dict, Any
import glob
from tqdm import tqdm

# Global audio parameters (fixed for all samples)
AUDIO_CONFIG = {
    'sr': 8000,
    'hop_length': 80,
    'win_length': 200,
    'n_fft': 256,
    'mag_mean': None,  # Computed from dataset
    'mag_std': None,   # Computed from dataset
}

def compute_global_audio_stats(video_dir: str, config_save_path: str = 'audio_config.json', max_samples: int = None) -> Dict[str, Any]:
    """Compute global magnitude mean and std from training videos.
    
    Args:
        video_dir: Directory containing training videos
        config_save_path: Where to save the config
        max_samples: Max number of samples to use (None = all)
        
    Returns:
        Updated config dict with mean/std
    """
    import librosa
    
    video_paths = sorted(glob.glob(os.path.join(video_dir, '*.mpg')))
    if max_samples:
        video_paths = video_paths[:max_samples]
    
    print(f"Computing global statistics from {len(video_paths)} videos...")
    
    all_magnitudes = []
    
    for video_path in tqdm(video_paths):
        try:
            y, sr_loaded = librosa.load(video_path, sr=AUDIO_CONFIG['sr'])
            
            stft_complex = librosa.stft(
                y=y,
                n_fft=AUDIO_CONFIG['n_fft'],
                hop_length=AUDIO_CONFIG['hop_length'],
                win_length=AUDIO_CONFIG['win_length'],
                window='hann',
                center=True,
                pad_mode='reflect'
            )
            
            magnitude = np.abs(stft_complex)
            log_magnitude = np.log1p(magnitude)
            
            all_magnitudes.append(log_magnitude.flatten())
        except Exception as e:
            print(f"Warning: Failed to process {video_path}: {e}")
            continue
    
    # Concatenate all magnitudes
    all_magnitudes = np.concatenate(all_magnitudes)
    
    # Compute global stats
    global_mean = float(all_magnitudes.mean())
    global_std = float(all_magnitudes.std())
    
    # Update config
    config = AUDIO_CONFIG.copy()
    config['mag_mean'] = global_mean
    config['mag_std'] = global_std
    
    # Save to file
    with open(config_save_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nGlobal statistics computed:")
    print(f"  Mean: {global_mean:.4f}")
    print(f"  Std:  {global_std:.4f}")
    print(f"Saved to: {config_save_path}")
    
    return config

def load_audio_config(config_path: str = 'audio_config.json') -> Dict[str, Any]:
    """Load audio config from file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Audio config not found at {config_path}. "
            f"Run compute_global_audio_stats() first to create it."
        )
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config

def get_audio_config() -> Dict[str, Any]:
    """Get audio config, loading from file if it exists, otherwise return defaults."""
    try:
        return load_audio_config()
    except FileNotFoundError:
        print("Warning: Using default config (mean/std not computed yet)")
        return AUDIO_CONFIG.copy()

if __name__ == "__main__":
    # Example: Compute stats from training data
    video_dir = 'data/s1'
    config = compute_global_audio_stats(video_dir, max_samples=100)  # Use first 100 videos
    print("\nConfig:", config)
