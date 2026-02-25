import os
import cv2
import numpy as np
import torch 
from typing import List, Tuple, Dict, Any
from matplotlib import pyplot as plt
import imageio
import gdown
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import librosa
import soundfile as sf
from audio_config import get_audio_config
torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
# MediaPipe for face mesh and landmark detection
# Model path for MediaPipe
model_path = "models/face_landmarker.task"  # path to downloaded model
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
VisionRunningMode = vision.RunningMode

#for the vocabulary, and loss calculation purpose
# Reserve index 0 for padding/blank token
vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!1234567890 "]
char_to_num = {char: idx + 1 for idx, char in enumerate(vocab)}  # Start from index 1
num_to_char = {idx: char for char, idx in char_to_num.items()}
num_to_char[0] = ''  # Index 0 is padding/blank (empty string)
def encode_chars(chars):
    """Convert list of characters to indices"""
    return [char_to_num[c] for c in chars]

def decode_indices(indices):
    """Convert list of indices back to characters"""
    return ''.join([num_to_char[i] for i in indices])

def _crop_mouth(gray_frame: np.ndarray, landmarks: list, padding_ratio: float) -> np.ndarray:
    """Crop mouth region from MediaPipe FaceLandmarker (478-point model).
    
    Args:
        gray_frame: Grayscale frame
        landmarks: List of facial landmarks
        padding_ratio: Relative padding as a ratio of mouth size (e.g., 0.5 = 50% padding)
    """
    h, w = gray_frame.shape[:2]
    
    # MediaPipe FaceLandmarker mouth landmark indices
    # Outer lips boundary
    mouth_indices = [
        # Upper outer lip
        61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
        # Lower outer lip  
        291, 375, 321, 405, 314, 17, 84, 181, 91, 146,
        # Inner lips for better coverage
        78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
        308, 415, 310, 311, 312, 13, 82, 81, 80, 191
    ]
    
    xs = [int(landmarks[i][0] * w) for i in mouth_indices]
    ys = [int(landmarks[i][1] * h) for i in mouth_indices]
    
    # Calculate mouth bounding box size
    mouth_width = max(xs) - min(xs)
    mouth_height = max(ys) - min(ys)
    
    # Calculate padding based on mouth size
    padding_x = int(mouth_width * padding_ratio)
    padding_y = int(mouth_height * padding_ratio)
    
    x_min = max(min(xs) - padding_x, 0)
    y_min = max(min(ys) - padding_y, 0)
    x_max = min(max(xs) + padding_x, w - 1)
    y_max = min(max(ys) + padding_y, h - 1)
    
    return gray_frame[y_min:y_max + 1, x_min:x_max + 1]
def load_video(video_path: str, padding_ratio: float = 0.8, target_size: Tuple[int, int] = (200, 100)) -> List[np.ndarray]:
    """Load video and extract mouth regions using MediaPipe FaceLandmarker.
    
    Args:
        video_path: Path to video file
        padding_ratio: Relative padding as ratio of mouth size (0.5 = 50% padding, 1.0 = 100%)
        target_size: Target size (width, height) for resizing mouth crops
        
    Returns:
        List of cropped and resized mouth frames
    """
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_faces=1
    )
    landmarker = FaceLandmarker.create_from_options(options)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")

    # Use a fixed increment for timestamp to guarantee monotonicity (e.g., 40ms per frame = 25 FPS)
    TIMESTAMP_INCREMENT_MS = 40
    frames = []
    frame_idx = 0
    last_valid_mouth = None
    timestamp_ms = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Use strictly increasing timestamp (fixed increment per frame)
        results = landmarker.detect_for_video(
            mp_image,
            timestamp_ms=timestamp_ms
        )

        if results.face_landmarks:
            landmarks = results.face_landmarks[0]
            landmark_tuples = [(lm.x, lm.y, lm.z) for lm in landmarks]

            mouth = _crop_mouth(gray, landmark_tuples, padding_ratio)

            # Validate and resize the cropped mouth
            if mouth.size > 0 and mouth.shape[0] > 0 and mouth.shape[1] > 0:
                # Use INTER_AREA for better quality when downscaling
                mouth_resized = cv2.resize(mouth, target_size, interpolation=cv2.INTER_AREA)
                frames.append(mouth_resized)
                last_valid_mouth = mouth_resized
            elif last_valid_mouth is not None:
                # Use last valid frame if crop failed
                frames.append(last_valid_mouth)
        else:
            # Use last valid mouth frame if face not detected
            if last_valid_mouth is not None:
                frames.append(last_valid_mouth)
            else:
                print(f"Warning: No face detected in frame {frame_idx}")

        frame_idx += 1
        timestamp_ms += TIMESTAMP_INCREMENT_MS

    cap.release()

    if not frames:
        raise ValueError(f"No valid mouth frames extracted from {video_path}")

    print(f"Total frames extracted: {len(frames)}")

    frames_tensor = torch.tensor(np.array(frames), dtype=torch.float32)
    mean = frames_tensor.mean()
    std = frames_tensor.std()
    return (frames_tensor - mean) / std if std > 0 else frames_tensor


#for the vocabulary, and allignement of the data with the text
def load_alignment(alignment_path: str) -> List[int]:
    """Load alignment file and return encoded character indices."""
    with open(alignment_path, 'r') as f:
        lines = f.readlines()
    
    tokens = ''
    for line in lines:
        parts = line.split()
        if len(parts) >= 3 and parts[2] != 'sil':
            tokens += ' ' + parts[2]
    
    # Convert string to list of characters, then encode
    return encode_chars(list(tokens))

def load_audio(
    video_path: str,
    audio_config: Dict[str, Any] | None = None,
) -> torch.Tensor:
    """Load audio from video file and extract STFT magnitude.
    
    Uses global audio config for all parameters (sr, hop_length, etc.) and normalization.
    
    Args:
        video_path: Path to video file
        audio_config: Audio config dict (if None, loads from audio_config.json)

    Returns:
        Normalized STFT magnitude tensor
    """
    if audio_config is None:
        audio_config = get_audio_config()
    
    sr = audio_config['sr']
    hop_length = audio_config['hop_length']
    win_length = audio_config['win_length']
    n_fft = audio_config['n_fft']
    global_mean = audio_config.get('mag_mean', 0.0)
    global_std = audio_config.get('mag_std', 1.0)

    try:
        # Load audio from video
        y, sr_loaded = librosa.load(video_path, sr=sr)
    except Exception as exc:
        raise ValueError(f"Could not load audio from {video_path}. Ensure ffmpeg is installed.")

    if y.size == 0:
        raise ValueError(f"No audio samples found in {video_path}.")

    # Compute STFT (returns complex-valued spectrogram)
    stft_complex = librosa.stft(
        y=y,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window='hann',
        center=True,
        pad_mode='reflect'
    )
    
    # Extract magnitude only (phase not needed - Griffin-Lim will estimate it)
    magnitude = np.abs(stft_complex).T  # (time, freq)
    
    # Apply log compression to magnitude for better numerical stability
    log_magnitude = np.log1p(magnitude)  # log(1 + x) is more stable than log(x)
    
    # Normalize magnitude using global stats
    mag_tensor = torch.tensor(log_magnitude, dtype=torch.float32)
    mag_normalized = (mag_tensor - global_mean) / global_std if global_std > 0 else mag_tensor

    return mag_normalized

def save_audio_from_video(video_path: str, output_wav_path: str, audio_config: Dict[str, Any] | None = None) -> None:
    """Extract audio from a video and save as WAV.

    This preserves the original waveform.
    """
    if audio_config is None:
        audio_config = get_audio_config()
    
    sr = audio_config['sr']
    y, _ = librosa.load(video_path, sr=sr)
    sf.write(output_wav_path, y, sr)

def reconstruct_audio_from_magnitude_only(
    log_magnitude_normalized: np.ndarray | torch.Tensor,
    magnitude_mean: float,
    magnitude_std: float,
    sr: int,
    hop_length: int,
    win_length: int,
    n_fft: int,
    n_iter: int = 60,
) -> np.ndarray:
    """Reconstruct audio from STFT magnitude only using Griffin-Lim (no phase needed).
    
    Use this when your model predicts magnitude only.
    Better than mel-based Griffin-Lim, but still has some artifacts.
    """
    if isinstance(log_magnitude_normalized, torch.Tensor):
        log_magnitude_normalized = log_magnitude_normalized.cpu().numpy()

    # Denormalize magnitude
    log_magnitude = (log_magnitude_normalized * magnitude_std) + magnitude_mean
    magnitude = np.expm1(log_magnitude)  # inverse of log1p
    
    # Griffin-Lim on STFT magnitude
    # (time, freq) -> (freq, time) for librosa
    y = librosa.griffinlim(
        magnitude.T,
        n_iter=n_iter,
        hop_length=hop_length,
        win_length=win_length,
        window='hann',
        center=True,
        length=None
    )
    return y

def reconstruct_audio_from_stft(
    log_magnitude_normalized: np.ndarray | torch.Tensor,
    phase: np.ndarray | torch.Tensor,
    magnitude_mean: float,
    magnitude_std: float,
    sr: int,
    hop_length: int,
    win_length: int,
    n_fft: int,
) -> np.ndarray:
    """Reconstruct audio from normalized STFT magnitude and phase using inverse STFT.
    
    This provides perfect reconstruction (no Griffin-Lim noise) since we preserve phase.
    Use this for testing/validation with ground truth data.
    """
    if isinstance(log_magnitude_normalized, torch.Tensor):
        log_magnitude_normalized = log_magnitude_normalized.cpu().numpy()
    if isinstance(phase, torch.Tensor):
        phase = phase.cpu().numpy()

    # Denormalize magnitude
    log_magnitude = (log_magnitude_normalized * magnitude_std) + magnitude_mean
    magnitude = np.expm1(log_magnitude)  # inverse of log1p
    
    # Reconstruct complex STFT from magnitude and phase
    # (time, freq) -> (freq, time) for librosa
    stft_complex = magnitude.T * np.exp(1j * phase.T)
    
    # Inverse STFT for perfect reconstruction
    y = librosa.istft(
        stft_complex,
        hop_length=hop_length,
        win_length=win_length,
        window='hann',
        center=True,
    )
    return y

# Legacy function kept for backward compatibility
def reconstruct_audio_from_log_mel(
    log_mel_normalized: np.ndarray | torch.Tensor,
    log_mel_mean: float,
    log_mel_std: float,
    sr: int,
    hop_length: int,
    win_length: int,
    n_fft: int,
    n_iter: int = 192,
) -> np.ndarray:
    """Reconstruct audio from normalized log-mel features using Griffin-Lim.
    
    WARNING: This method produces noisy results. Use reconstruct_audio_from_stft instead.
    """
    if isinstance(log_mel_normalized, torch.Tensor):
        log_mel_normalized = log_mel_normalized.cpu().numpy()

    log_mel = (log_mel_normalized * log_mel_std) + log_mel_mean
    mel = np.exp(log_mel).T
    y = librosa.feature.inverse.mel_to_audio(
        mel,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_iter=n_iter,
        power=2.0,
    )
    return y

def align_audio_to_video(
    audio_mel: torch.Tensor,
    num_video_frames: int,
) -> torch.Tensor:
    """Align audio mel frames to video frames via interpolation.
    
    Args:
        audio_mel: Audio features of shape (audio_frames, n_mels)
        num_video_frames: Number of video frames to align to
        
    Returns:
        Aligned audio features of shape (num_video_frames, n_mels)
    """
    # (T, M) -> (1, M, T) for interpolate
    mel = audio_mel.T.unsqueeze(0)
    aligned = torch.nn.functional.interpolate(
        mel, size=num_video_frames, mode='linear', align_corners=False
    )
    return aligned.squeeze(0).T  # back to (num_video_frames, n_mels)
# def load_alignment(alignment_path: str) -> List[Tuple[float, float, str]]:
#     """Load alignment file and return list of (start_time, end_time, word) tuples."""
#     with open(alignment_path, 'r') as f:
#         lines = f.readlines()
#     tokens = []
#     for line in lines:
#         parts = line.split()
#         if len(parts) >= 3 and parts[2] != 'sil':
#             tokens = [*tokens,' ', parts[2]]
#     return encode_chars(torch.reshaped(torch.strings.unicode_split(tokens,input_encoding='UTF-8'),(-1,)).tolist())

def load_data(
    video_path: str,
    alignment_path: str,
    audio_config: Dict[str, Any] | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """Load video, audio and alignment data.
    
    Uses global audio config for consistent processing.
    
    Args:
        video_path: Path to video file
        alignment_path: Path to alignment file
        audio_config: Audio config dict (if None, loads from audio_config.json)
        
    Returns:
        Tuple of (mouth_frames, audio_stft_magnitude, char_indices)
    """
    if audio_config is None:
        audio_config = get_audio_config()
    
    mouth_frames = load_video(video_path)
    audio_stft_mag = load_audio(video_path, audio_config)
    char_indices = load_alignment(alignment_path)
    
    return (
        mouth_frames,
        audio_stft_mag,
        char_indices,
    )

if __name__ == "__main__":
    print("Vocabulary size:", len(vocab))
    
    # Load global audio config
    audio_config = get_audio_config()
    print(f"\nAudio config: {audio_config}")
    
    preprocessed = load_data(
        video_path='data/s1/bbaf2n.mpg', 
        alignment_path='data/alignments/s1/bbaf2n.align',
        audio_config=audio_config
    )
    
    (
        mouth_frames,
        audio_stft_mag,
        char_indices,
    ) = preprocessed
    
    print(f'\nMouth frames shape: {mouth_frames.shape}')
    print(f'Audio STFT magnitude shape: {audio_stft_mag.shape}')
    print(f'Character indices length: {len(char_indices)}')
    print(f'Character sequence: {decode_indices(char_indices)}')
    
    # Save original audio for comparison
    save_audio_from_video('data/s1/bbaf2n.mpg', 'original_audio.wav', audio_config)
    print("\nSaved original audio to original_audio.wav")
    
    # Reconstruct with magnitude only (Griffin-Lim)
    y_reconstructed_magnitude_only = reconstruct_audio_from_magnitude_only(
        audio_stft_mag,
        audio_config['mag_mean'],
        audio_config['mag_std'],
        sr=audio_config['sr'],
        hop_length=audio_config['hop_length'],
        win_length=audio_config['win_length'],
        n_fft=audio_config['n_fft'],
        n_iter=60,
    )
    sf.write('reconstructed_magnitude_only.wav', y_reconstructed_magnitude_only, audio_config['sr'])
    print("Saved reconstructed_magnitude_only.wav (uses Griffin-Lim)")
    print("\nCompare original vs reconstructed to verify quality!")