"""
Video preprocessing module for extracting lip regions from video frames.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path

from .config import Config


class VideoPreprocessor:
    """Preprocesses video to extract lip regions."""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the video preprocessor.
        
        Args:
            config: Configuration object. If None, uses default Config.
        """
        self.config = config or Config()
        
        # Try to initialize MediaPipe Face Detector
        # MediaPipe 0.10+ uses the tasks API
        try:
            import mediapipe as mp
            from mediapipe import solutions
            from mediapipe.framework.formats import landmark_pb2
            
            # Try legacy solutions API (pre-0.10)
            self.face_mesh = solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=self.config.FACE_DETECTION_CONFIDENCE,
                min_tracking_confidence=self.config.FACE_DETECTION_CONFIDENCE
            )
            self.use_legacy_api = True
        except (ImportError, AttributeError):
            # MediaPipe 0.10+ doesn't have solutions.face_mesh
            # Use simpler face detection approach with cv2
            self.face_detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            self.use_legacy_api = False
            print("Warning: Using OpenCV face detection. For better results, use MediaPipe < 0.10 or implement face landmark detection.")
        
        # Lip landmark indices (MediaPipe Face Mesh)
        # Outer lip: 61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 185
        # Inner lip: 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 191
        self.lip_indices = [
            61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 185,  # Outer
            78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 191     # Inner
        ]
    
    def extract_frames(self, video_path: str) -> List[np.ndarray]:
        """
        Extract frames from video file.
        
        Args:
            video_path: Path to video file.
            
        Returns:
            List of video frames as numpy arrays.
        """
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        target_fps = self.config.TARGET_FPS
        frame_interval = max(1, int(fps / target_fps))
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                frames.append(frame)
            
            frame_count += 1
        
        cap.release()
        return frames
    
    def detect_lip_region(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect lip region in a frame.
        
        Args:
            frame: Input frame as numpy array.
            
        Returns:
            Tuple of (x, y, w, h) for lip region bounding box, or None if not detected.
        """
        if self.use_legacy_api:
            # Use MediaPipe Face Mesh
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            if not results.multi_face_landmarks:
                return None
            
            # Get first face
            face_landmarks = results.multi_face_landmarks[0]
            h, w = frame.shape[:2]
            
            # Get lip landmarks using instance variable
            lip_points = []
            for idx in self.lip_indices:
                landmark = face_landmarks.landmark[idx]
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                lip_points.append((x, y))
            
            # Calculate bounding box
            lip_points = np.array(lip_points)
            x_min, y_min = lip_points.min(axis=0)
            x_max, y_max = lip_points.max(axis=0)
        else:
            # Use OpenCV face detection and estimate lip region
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) == 0:
                return None
            
            # Get first face
            (fx, fy, fw, fh) = faces[0]
            h, w = frame.shape[:2]
            
            # Estimate lip region (lower third of face)
            x_min = fx + int(fw * 0.2)
            x_max = fx + int(fw * 0.8)
            y_min = fy + int(fh * 0.6)
            y_max = fy + int(fh * 0.9)
        
        # Add padding
        padding = self.config.LIP_REGION_PADDING
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)
        
        return (x_min, y_min, x_max - x_min, y_max - y_min)
    
    def extract_lip_regions(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Extract lip regions from all frames.
        
        Args:
            frames: List of video frames.
            
        Returns:
            List of cropped and resized lip regions.
        """
        lip_regions = []
        last_valid_bbox = None
        
        for frame in frames:
            bbox = self.detect_lip_region(frame)
            
            # Use last valid bbox if current detection fails
            if bbox is None and last_valid_bbox is not None:
                bbox = last_valid_bbox
            elif bbox is not None:
                last_valid_bbox = bbox
            
            if bbox is not None:
                x, y, w, h = bbox
                lip_region = frame[y:y+h, x:x+w]
                
                # Resize to target size
                lip_region = cv2.resize(
                    lip_region,
                    self.config.LIP_REGION_SIZE,
                    interpolation=cv2.INTER_LINEAR
                )
                
                # Convert to grayscale
                lip_region = cv2.cvtColor(lip_region, cv2.COLOR_BGR2GRAY)
                
                lip_regions.append(lip_region)
            else:
                # Create blank frame if detection fails
                blank = np.zeros(self.config.LIP_REGION_SIZE, dtype=np.uint8)
                lip_regions.append(blank)
        
        return lip_regions
    
    def preprocess_video(self, video_path: str) -> np.ndarray:
        """
        Complete preprocessing pipeline for a video.
        
        Args:
            video_path: Path to video file.
            
        Returns:
            Preprocessed lip regions as numpy array of shape (T, H, W).
        """
        frames = self.extract_frames(video_path)
        lip_regions = self.extract_lip_regions(frames)
        
        # Limit sequence length
        if len(lip_regions) > self.config.SEQUENCE_LENGTH:
            lip_regions = lip_regions[:self.config.SEQUENCE_LENGTH]
        
        return np.array(lip_regions)
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'face_mesh') and self.use_legacy_api:
            self.face_mesh.close()
