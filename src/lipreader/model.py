"""
Lip reading model for converting visual features to text.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List
from pathlib import Path

from .config import Config


class LipReadingModel:
    """Model for lip reading inference."""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the lip reading model.
        
        Args:
            config: Configuration object. If None, uses default Config.
        """
        self.config = config or Config()
        self.device = torch.device(self.config.DEVICE)
        
        # Simple vocabulary (can be extended) - must be created before model
        self.vocabulary = self._build_vocabulary()
        self.idx_to_word = {idx: word for word, idx in self.vocabulary.items()}
        
        # Build the model
        self.model = self._build_model()
        self.model.to(self.device)
        self.model.eval()
    
    def _build_vocabulary(self) -> dict:
        """
        Build a simple vocabulary for lip reading.
        
        Returns:
            Dictionary mapping words to indices.
        """
        # Basic vocabulary with common words
        words = [
            "<pad>", "<sos>", "<eos>", "<unk>",
            "hello", "hi", "thank", "you", "please", "yes", "no",
            "good", "morning", "afternoon", "evening", "night",
            "how", "are", "what", "where", "when", "who", "why",
            "I", "you", "he", "she", "we", "they",
            "is", "am", "are", "was", "were",
            "do", "does", "did", "can", "could", "will", "would",
            "the", "a", "an", "this", "that", "these", "those",
            "and", "or", "but", "if", "because",
            "see", "hear", "speak", "talk", "say", "tell",
            "go", "come", "walk", "run", "stop",
            "eat", "drink", "sleep", "work", "play",
            "happy", "sad", "angry", "tired", "hungry",
            "big", "small", "hot", "cold", "fast", "slow",
        ]
        return {word: idx for idx, word in enumerate(words)}
    
    def _build_model(self) -> nn.Module:
        """
        Build the lip reading model architecture.
        
        Returns:
            PyTorch model for sequence-to-sequence prediction.
        """
        class LipReadingNet(nn.Module):
            def __init__(self, feature_dim, vocab_size, hidden_dim=256):
                super().__init__()
                
                # Temporal encoder (LSTM)
                self.encoder = nn.LSTM(
                    feature_dim,
                    hidden_dim,
                    num_layers=2,
                    batch_first=True,
                    bidirectional=True,
                    dropout=0.3
                )
                
                # Attention mechanism
                self.attention = nn.MultiheadAttention(
                    hidden_dim * 2,
                    num_heads=4,
                    batch_first=True
                )
                
                # Decoder
                self.decoder = nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(hidden_dim, vocab_size)
                )
            
            def forward(self, x):
                # x: (batch, seq_len, feature_dim)
                encoder_out, _ = self.encoder(x)
                
                # Apply attention
                attn_out, _ = self.attention(encoder_out, encoder_out, encoder_out)
                
                # Global average pooling over time
                pooled = torch.mean(attn_out, dim=1)
                
                # Decode to vocabulary
                output = self.decoder(pooled)
                return output
        
        vocab_size = len(self.vocabulary)
        return LipReadingNet(
            self.config.FEATURE_DIM,
            vocab_size,
            hidden_dim=256
        )
    
    def load_weights(self, model_path: Optional[str] = None):
        """
        Load pre-trained model weights.
        
        Args:
            model_path: Path to model weights. If None, uses config path.
        """
        if model_path is None:
            model_path = self.config.MODEL_PATH
        
        if Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model weights from {model_path}")
        else:
            print(f"Warning: Model weights not found at {model_path}")
            print("Using randomly initialized model (for demonstration only)")
    
    def predict(self, features: np.ndarray, top_k: int = 5) -> List[tuple]:
        """
        Predict words from features.
        
        Args:
            features: Feature array of shape (T, feature_dim).
            top_k: Number of top predictions to return.
            
        Returns:
            List of (word, probability) tuples.
        """
        # Convert to tensor and add batch dimension
        features_tensor = torch.from_numpy(features).float().unsqueeze(0)
        features_tensor = features_tensor.to(self.device)
        
        # Forward pass
        with torch.no_grad():
            logits = self.model(features_tensor)
            probs = torch.softmax(logits, dim=-1)
        
        # Get top-k predictions
        probs = probs.squeeze(0).cpu().numpy()
        top_indices = np.argsort(probs)[-top_k:][::-1]
        
        predictions = []
        for idx in top_indices:
            word = self.idx_to_word.get(idx, "<unk>")
            prob = probs[idx]
            predictions.append((word, float(prob)))
        
        return predictions
    
    def predict_sequence(self, features: np.ndarray) -> str:
        """
        Predict a sequence of words from features (simplified version).
        
        Args:
            features: Feature array of shape (T, feature_dim).
            
        Returns:
            Predicted text string.
        """
        predictions = self.predict(features, top_k=1)
        if predictions:
            return predictions[0][0]
        return "<unk>"
