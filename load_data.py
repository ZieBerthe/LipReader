import os
import sys
import warnings
import logging

# ── Suppress ALL mediapipe / TFLite / absl noise ──────────────────────────
# These must be set BEFORE importing mediapipe or TFLite.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'          # TFLite: only FATAL
os.environ['GLOG_minloglevel'] = '3'               # glog: only FATAL
os.environ['GLOG_logtostderr'] = '0'                # glog: don't log to stderr
os.environ['GLOG_stderrthreshold'] = '3'            # glog: stderr threshold FATAL
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'           # silence oneDNN info
os.environ['ABSL_MIN_LOG_LEVEL'] = '3'              # absl C++ logging

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
logging.getLogger('mediapipe').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import cv2
import numpy as np
import torch
from typing import List, Tuple, Dict, Any
from matplotlib import pyplot as plt
import imageio
import gdown

# Redirect stderr briefly while importing mediapipe to catch C++ warnings
_stderr_fd = sys.stderr.fileno()
_old_stderr = os.dup(_stderr_fd)
with open(os.devnull, 'w') as _devnull:
    os.dup2(_devnull.fileno(), _stderr_fd)
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
os.dup2(_old_stderr, _stderr_fd)
os.close(_old_stderr)

torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# MediaPipe for face mesh and landmark detection
model_path = "models/face_landmarker.task"
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
VisionRunningMode = vision.RunningMode

# GRID corpus alignment constants
ALIGN_SAMPLE_RATE = 25000  # Alignment file timestamp rate (samples/sec)
VIDEO_FPS = 25

# Vocabulary and character encoding
# Reserve index 0 for padding/blank token (CTC blank)
vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!1234567890 "]
char_to_num = {char: idx + 1 for idx, char in enumerate(vocab)}  # Start from 1
num_to_char = {idx: char for char, idx in char_to_num.items()}
num_to_char[0] = ''  # Index 0 is padding/blank


def encode_chars(chars):
    """Convert list of characters to indices."""
    return [char_to_num[c] for c in chars]


def decode_indices(indices):
    """Convert list of indices back to characters."""
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

    mouth_width = max(xs) - min(xs)
    mouth_height = max(ys) - min(ys)
    padding_x = int(mouth_width * padding_ratio)
    padding_y = int(mouth_height * padding_ratio)

    x_min = max(min(xs) - padding_x, 0)
    y_min = max(min(ys) - padding_y, 0)
    x_max = min(max(xs) + padding_x, w - 1)
    y_max = min(max(ys) + padding_y, h - 1)

    return gray_frame[y_min:y_max + 1, x_min:x_max + 1]


def load_video(video_path: str, padding_ratio: float = 0.8, target_size: Tuple[int, int] = (200, 100)) -> torch.Tensor:
    """Load video and extract mouth regions using MediaPipe FaceLandmarker.

    Args:
        video_path: Path to video file
        padding_ratio: Relative padding as ratio of mouth size
        target_size: Target size (width, height) for resizing mouth crops

    Returns:
        Normalised tensor of mouth frames (T, H, W)
    """
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_faces=1
    )
    # Suppress C++ warnings emitted when MediaPipe creates the landmarker
    _fd = sys.stderr.fileno()
    _old = os.dup(_fd)
    with open(os.devnull, 'w') as _dn:
        os.dup2(_dn.fileno(), _fd)
        landmarker = FaceLandmarker.create_from_options(options)
    os.dup2(_old, _fd)
    os.close(_old)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")

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

        results = landmarker.detect_for_video(mp_image, timestamp_ms=timestamp_ms)

        if results.face_landmarks:
            landmarks = results.face_landmarks[0]
            landmark_tuples = [(lm.x, lm.y, lm.z) for lm in landmarks]
            mouth = _crop_mouth(gray, landmark_tuples, padding_ratio)

            if mouth.size > 0 and mouth.shape[0] > 0 and mouth.shape[1] > 0:
                mouth_resized = cv2.resize(mouth, target_size, interpolation=cv2.INTER_AREA)
                frames.append(mouth_resized)
                last_valid_mouth = mouth_resized
            elif last_valid_mouth is not None:
                frames.append(last_valid_mouth)
        else:
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


def load_alignment(
    alignment_path: str,
    num_frames: int = 75,
) -> Tuple[List[int], torch.Tensor, List[Tuple[str, float, float]]]:
    """Load alignment file and return text + per-frame word timing labels.

    The alignment file format (GRID corpus):
        start_sample end_sample word
        e.g. "23750 29500 bin"

    Timestamps are in units of 1/25000 s.  At 25 fps the conversion is
        frame_idx = timestamp * VIDEO_FPS / ALIGN_SAMPLE_RATE  (= timestamp / 1000)

    Args:
        alignment_path: Path to .align file
        num_frames: Number of video frames

    Returns:
        char_indices:      Encoded character indices for CTC loss
        frame_word_labels: (num_frames,) tensor — word position index per frame
                           0 = silence, 1 = first word, 2 = second word, …
        word_timings:      List of (word, start_sec, end_sec) tuples
    """
    with open(alignment_path, 'r') as f:
        lines = f.readlines()

    # Parse alignment segments
    segments = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 3:
            start = int(parts[0])
            end = int(parts[1])
            word = parts[2]
            segments.append((start, end, word))

    # ---- Character sequence (for CTC) ----
    tokens = ''
    for start, end, word in segments:
        if word != 'sil':
            tokens += ' ' + word
    char_indices = encode_chars(list(tokens))

    # ---- Per-frame word labels & word timing list ----
    word_timings: List[Tuple[str, float, float]] = []
    frame_word_labels = torch.zeros(num_frames, dtype=torch.long)
    word_idx = 0  # 0 = silence

    for start, end, word in segments:
        start_sec = start / ALIGN_SAMPLE_RATE
        end_sec = end / ALIGN_SAMPLE_RATE

        if word != 'sil':
            word_idx += 1
            word_timings.append((word, start_sec, end_sec))

        # Map to frame indices
        start_frame = int(start * VIDEO_FPS / ALIGN_SAMPLE_RATE)
        end_frame = int(end * VIDEO_FPS / ALIGN_SAMPLE_RATE)
        start_frame = max(0, min(start_frame, num_frames))
        end_frame = max(0, min(end_frame, num_frames))

        label = word_idx if word != 'sil' else 0
        frame_word_labels[start_frame:end_frame] = label

    return char_indices, frame_word_labels, word_timings


def load_data(
    video_path: str,
    alignment_path: str,
) -> Tuple[torch.Tensor, List[int], torch.Tensor, List[Tuple[str, float, float]]]:
    """Load video and alignment data for lip reading.

    Args:
        video_path:     Path to video file
        alignment_path: Path to alignment file

    Returns:
        mouth_frames:      (T, H, W) tensor of mouth crops
        char_indices:      Encoded character indices for CTC
        frame_word_labels: (T,) tensor of per-frame word indices (0 = silence)
        word_timings:      List of (word, start_sec, end_sec) tuples
    """
    mouth_frames = load_video(video_path)
    num_frames = mouth_frames.shape[0]
    char_indices, frame_word_labels, word_timings = load_alignment(
        alignment_path, num_frames=num_frames
    )

    return mouth_frames, char_indices, frame_word_labels, word_timings


if __name__ == "__main__":
    print("Vocabulary size:", len(vocab))

    mouth_frames, char_indices, frame_word_labels, word_timings = load_data(
        video_path='data/s1/bbaf2n.mpg',
        alignment_path='data/alignments/s1/bbaf2n.align',
    )

    print(f'\nMouth frames shape: {mouth_frames.shape}')
    print(f'Character indices length: {len(char_indices)}')
    print(f'Character sequence: {decode_indices(char_indices)}')
    print(f'\nFrame word labels shape: {frame_word_labels.shape}')
    print(f'Frame word labels: {frame_word_labels}')
    print(f'\nWord timings:')
    for word, start, end in word_timings:
        print(f'  {word}: {start:.3f}s - {end:.3f}s (duration: {end-start:.3f}s)')
