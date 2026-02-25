import os
import cv2
import numpy as np
import torch 
from typing import List, Tuple
from matplotlib import pyplot as plt
import imageio
import gdown
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import librosa
import soundfile as sf
torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
# MediaPipe for face mesh and landmark detection
model_path = "models/face_landmarker.task"  # path to downloaded model

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1
)

landmarker = FaceLandmarker.create_from_options(options)
#for the vocabulary, and loss calculation purpose
vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!1234567890 "]
char_to_num = {char: idx for idx, char in enumerate((vocab))}
num_to_char = {idx: char for char, idx in char_to_num.items()}
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
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")
    
    # Get actual video FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 25.0  # Default fallback
    frame_time_ms = int(1000 / fps)
    
    frames = []
    frame_idx = 0
    last_valid_mouth = None

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

        # timestamp must increase for VIDEO mode
        results = landmarker.detect_for_video(
            mp_image,
            timestamp_ms=frame_idx * frame_time_ms
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

    cap.release()
    
    if not frames:
        raise ValueError(f"No valid mouth frames extracted from {video_path}")
    
    print(f"Video FPS: {fps:.2f}, Total frames extracted: {len(frames)}")

    # Normalize with torch to keep everything on the same side of the fence.
    frames_tensor = torch.tensor(frames, dtype=torch.float32)
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
    sr: int = 16000,
    n_mels: int = 128,
    hop_length: int | None = None,
    win_length: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, int, int, float, float, int, int]:
    """Load audio from video file and extract log-mel features.
    
    Args:
        video_path: Path to video file
        sr: Sample rate for audio processing
        n_mels: Number of mel bins
        hop_length: Hop length in samples (auto-aligned to video FPS if None)
        win_length: Window length in samples (defaults to 2 * hop_length if None)

    Returns:
        Tuple of (log_mel_features, waveform, sample_rate, hop_length,
        log_mel_mean, log_mel_std, n_fft, win_length)
    """
    # Use fixed hop/win for better audio quality
    # Alignment to video frames can be done via interpolation if needed
    if hop_length is None:
        hop_length = 160  # 10ms at 16kHz - good for quality
    if win_length is None:
        win_length = 400  # 25ms at 16kHz

    try:
        # Load audio from video
        y, sr_loaded = librosa.load(video_path, sr=sr)
    except Exception as exc:
        raise ValueError(f"Could not load audio from {video_path}. Ensure ffmpeg is installed.")

    if y.size == 0:
        raise ValueError(f"No audio samples found in {video_path}.")

    # Extract log-mel features (better target for reconstruction than MFCCs)
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr_loaded,
        n_mels=n_mels,
        hop_length=hop_length,
        win_length=win_length,
        power=2.0,
    )
    log_mel = np.log(np.maximum(mel, 1e-10)).T

    # Normalize
    mel_tensor = torch.tensor(log_mel, dtype=torch.float32)
    mean = mel_tensor.mean().item()
    std = mel_tensor.std().item()
    mel_normalized = (mel_tensor - mean) / std if std > 0 else mel_tensor

    n_fft = max(1024, win_length)

    return mel_normalized, y, sr_loaded, hop_length, mean, std, n_fft, win_length

def save_audio_from_video(video_path: str, output_wav_path: str, sr: int = 16000) -> None:
    """Extract audio from a video and save as WAV.

    This preserves the original waveform (resampled to sr) instead of reconstructing from MFCCs.
    """
    _, y, sr_loaded, _, _, _, _, _ = load_audio(video_path, sr=sr)
    sf.write(output_wav_path, y, sr_loaded)

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
    """Reconstruct audio from normalized log-mel features using Griffin-Lim."""
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int, float, float, int, int, List[int]]:
    """Load video, audio and alignment data.
    
    Args:
        video_path: Path to video file
        alignment_path: Path to alignment file
        
    Returns:
        Tuple of (mouth_frames, audio_log_mel, audio_waveform, audio_sr, hop_length,
        log_mel_mean, log_mel_std, n_fft, win_length, char_indices)
    """
    mouth_frames = load_video(video_path)
    audio_log_mel, audio_waveform, audio_sr, hop_length, log_mel_mean, log_mel_std, n_fft, win_length = load_audio(video_path)
    char_indices = load_alignment(alignment_path)
    return (
        mouth_frames,
        audio_log_mel,
        audio_waveform,
        audio_sr,
        hop_length,
        log_mel_mean,
        log_mel_std,
        n_fft,
        win_length,
        char_indices,
    )
if __name__ == "__main__":
    print("Vocabulary size:", len(vocab))
    preprocessed = load_data(
        video_path='data/s1/bbaf3s.mpg', 
        alignment_path='data/alignments/s1/bbaf3s.align'
    )
    
    (
        mouth_frames,
        audio_log_mel,
        audio_waveform,
        audio_sr,
        hop_length,
        log_mel_mean,
        log_mel_std,
        n_fft,
        win_length,
        char_indices,
    ) = preprocessed
    print(f'Mouth frames shape: {mouth_frames.shape}')
    print(f'Audio log-mel shape: {audio_log_mel.shape}')
    print(f'Character indices length: {len(char_indices)}')
    print(f'Character sequence: {decode_indices(char_indices)}')
    # Save extracted audio for verification
    # save_audio_from_video('data/s1/bbaf3s.mpg', 'extracted_audio.wav', sr=audio_sr)
    # check loaded audio
    print(f'Extracted audio waveform shape: {audio_waveform.shape}, Sample rate: {audio_sr}')
    # reconstruct audio from log-mel features to verify correctness
    y_reconstructed = reconstruct_audio_from_log_mel(
        audio_log_mel,
        log_mel_mean,
        log_mel_std,
        sr=audio_sr,
        hop_length=hop_length,
        win_length=win_length,
        n_fft=n_fft,
        n_iter=128,
    )
    sf.write('reconstructed_audiobbaf3s.wav', y_reconstructed, audio_sr)
    print("Saved reconstructed audio to reconstructed_audio.wav")