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
    return frames
if __name__ == "__main__":
    video_path = "data/s1/bbaf2n.mpg"  # Update with your video path
    mouth_frames = load_video(video_path)
    print(f"Loaded {len(mouth_frames)} mouth frames from the video.")
    # Display the first mouth frame as an example
    for i in range(0,len(mouth_frames),5):
        if mouth_frames:
            plt.imshow(mouth_frames[i], cmap='gray')
            plt.title("Example Mouth Frame")
            plt.axis('off')
            plt.show()