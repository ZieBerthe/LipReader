"""Mouth-region extraction from GRID videos with MediaPipe FaceLandmarker."""
import os
import warnings
import logging

# ── Suppress mediapipe / TFLite / absl noise (must happen before the import) ──
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'
os.environ['GLOG_logtostderr'] = '0'
os.environ['GLOG_stderrthreshold'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['ABSL_MIN_LOG_LEVEL'] = '3'

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
logging.getLogger('mediapipe').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from typing import Tuple

import cv2
import numpy as np
import torch

from lipreader.paths import FACE_LANDMARKER_MODEL
from lipreader.utils import suppress_stderr

with suppress_stderr():
    import mediapipe as mp
    from mediapipe.tasks.python import vision

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
VisionRunningMode = vision.RunningMode

MOUTH_SIZE = (200, 100)  # (width, height) of the resized mouth crops


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


def load_video(video_path: str, padding_ratio: float = 0.8, target_size: Tuple[int, int] = MOUTH_SIZE) -> torch.Tensor:
    """Load video and extract mouth regions using MediaPipe FaceLandmarker.

    Args:
        video_path: Path to video file
        padding_ratio: Relative padding as ratio of mouth size
        target_size: Target size (width, height) for resizing mouth crops

    Returns:
        Normalised tensor of mouth frames (T, H, W)
    """
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=FACE_LANDMARKER_MODEL),
        running_mode=VisionRunningMode.VIDEO,
        num_faces=1
    )
    with suppress_stderr():  # C++ warnings emitted when MediaPipe creates the landmarker
        landmarker = FaceLandmarker.create_from_options(options)
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
    landmarker.close()

    if not frames:
        raise ValueError(f"No valid mouth frames extracted from {video_path}")

    print(f"Total frames extracted: {len(frames)}")

    frames_tensor = torch.tensor(np.array(frames), dtype=torch.float32)
    mean = frames_tensor.mean()
    std = frames_tensor.std()
    return (frames_tensor - mean) / std if std > 0 else frames_tensor
