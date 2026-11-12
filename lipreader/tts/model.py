"""Video-style TTS: text + mouth-video style embedding -> mel spectrogram.

Architecture (non-autoregressive):
    TextEncoder   : char embedding → Conv1D → BiLSTM → (B, T_text, 256)
    VideoStyleEnc : per-frame 2-D CNN → mean pool → FC → style vector (128)
    MelDecoder    : interpolate text features to mel length, concat style,
                    Conv1D stack → mel prediction (B, T_mel, n_mels)
    PostNet       : Conv1D residual refinement of mel
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from lipreader.data.vocab import NUM_CLASSES

# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------
VOCAB_SIZE   = NUM_CLASSES             # +1 for padding/blank at index 0
MAX_TEXT_LEN = 50                       # characters (GRID sentences are short)
MAX_VIDEO_FRAMES = 75
MAX_MEL_FRAMES   = 350                  # ~4 s at 22 050 Hz / hop 256

# Model hyper-parameters
ENC_DIM      = 256
STYLE_DIM    = 128
DEC_HIDDEN   = 512
POSTNET_CH   = 256
POSTNET_LAYERS = 3


# ---------------------------------------------------------------------------
#  Model components
# ---------------------------------------------------------------------------

class TextEncoder(nn.Module):
    """Character embedding → 3 × Conv1D + BN + ReLU → BiLSTM."""

    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=ENC_DIM, out_dim=ENC_DIM):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        convs = []
        for _ in range(3):
            convs += [
                nn.Conv1d(embed_dim, embed_dim, kernel_size=5, padding=2),
                nn.BatchNorm1d(embed_dim),
                nn.ReLU(),
                nn.Dropout(0.5),
            ]
        self.convs = nn.Sequential(*convs)

        self.lstm = nn.LSTM(
            embed_dim, out_dim // 2,
            num_layers=1, batch_first=True, bidirectional=True,
        )

    def forward(self, text_ids, text_lengths):
        """
        Args:
            text_ids:     (B, T_text)  long
            text_lengths: (B,)
        Returns:
            enc_out: (B, T_text, out_dim)
        """
        x = self.embedding(text_ids)                      # (B, T, E)
        x = self.convs(x.transpose(1, 2)).transpose(1, 2) # (B, T, E)

        # Pack → LSTM → unpack
        packed = nn.utils.rnn.pack_padded_sequence(
            x, text_lengths.cpu().clamp(min=1), batch_first=True, enforce_sorted=False,
        )
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        return out                                         # (B, T_text, out_dim)


class VideoStyleEncoder(nn.Module):
    """Per-frame 2-D CNN → BiLSTM → attention pool → style vector.

    Instead of simple mean-pooling (which loses all temporal dynamics),
    we run a BiLSTM over the per-frame CNN features and use a small
    attention layer to produce a single global style vector that captures
    speaking rhythm, pace and emphasis patterns from the video.
    """

    def __init__(self, style_dim=STYLE_DIM):
        super().__init__()
        cnn_out = 128

        # Spatial feature extraction (per frame)
        self.cnn = nn.Sequential(
            nn.Conv2d(1,  32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, cnn_out, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),                # → (B*T, 128, 1, 1)
        )

        # Temporal modelling: captures rhythm / speaking rate
        self.lstm = nn.LSTM(
            input_size=cnn_out,
            hidden_size=style_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # Attention pooling over time (learns *which* frames matter most)
        self.attn = nn.Sequential(
            nn.Linear(style_dim, style_dim),
            nn.Tanh(),
            nn.Linear(style_dim, 1),            # score per frame
        )

        self.out_proj = nn.Sequential(
            nn.Linear(style_dim, style_dim),
            nn.ReLU(),
        )

    def forward(self, frames, frame_lengths):
        """
        Args:
            frames:        (B, T_vid, H, W)
            frame_lengths: (B,)
        Returns:
            style: (B, style_dim)
        """
        B, T, H, W = frames.shape

        # ---- Per-frame spatial features ----
        x = frames.reshape(B * T, 1, H, W)
        x = self.cnn(x).squeeze(-1).squeeze(-1)          # (B*T, 128)
        x = x.view(B, T, -1)                             # (B, T, 128)

        # ---- Temporal modelling (BiLSTM) ----
        packed = nn.utils.rnn.pack_padded_sequence(
            x, frame_lengths.cpu().clamp(min=1),
            batch_first=True, enforce_sorted=False,
        )
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(   # (B, T, style_dim)
            lstm_out, batch_first=True, total_length=T,
        )

        # ---- Attention pooling (learns which frames are important) ----
        mask = (
            torch.arange(T, device=frames.device)
            .unsqueeze(0).expand(B, -1)
            < frame_lengths.unsqueeze(1)
        )                                                 # (B, T) bool

        scores = self.attn(lstm_out).squeeze(-1)          # (B, T)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # (B, T, 1)

        pooled = (lstm_out * weights).sum(dim=1)          # (B, style_dim)

        return self.out_proj(pooled)                      # (B, style_dim)


class MelDecoder(nn.Module):
    """Conv1D stack: (enc_dim + style_dim) → n_mels."""

    def __init__(self, in_dim=ENC_DIM + STYLE_DIM, n_mels=80, hidden=DEC_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, hidden, 5, padding=2), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Conv1d(hidden, hidden, 5, padding=2), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Conv1d(hidden, hidden, 5, padding=2), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Conv1d(hidden, n_mels, 1),
        )

    def forward(self, x):
        """x: (B, T_mel, in_dim) → (B, T_mel, n_mels)"""
        return self.net(x.transpose(1, 2)).transpose(1, 2)


class PostNet(nn.Module):
    """Residual Conv1D refinement of the mel output."""

    def __init__(self, n_mels=80, hidden=POSTNET_CH, n_layers=POSTNET_LAYERS):
        super().__init__()
        layers = []
        in_ch = n_mels
        for i in range(n_layers - 1):
            layers += [
                nn.Conv1d(in_ch, hidden, 5, padding=2),
                nn.BatchNorm1d(hidden),
                nn.Tanh(),
                nn.Dropout(0.1),
            ]
            in_ch = hidden
        layers.append(nn.Conv1d(hidden, n_mels, 5, padding=2))
        self.net = nn.Sequential(*layers)

    def forward(self, mel):
        """mel: (B, T, n_mels) → residual (B, T, n_mels)"""
        return self.net(mel.transpose(1, 2)).transpose(1, 2)


# ---------------------------------------------------------------------------
#  Full model
# ---------------------------------------------------------------------------

class VideoStyleTTS(nn.Module):
    """Text-to-mel with video style conditioning.

    Forward:
        1. Encode text → (B, T_text, enc_dim)
        2. Encode video → (B, style_dim)
        3. Interpolate text features up to mel length
        4. Concat style vector at every mel frame
        5. Decode → mel prediction
        6. PostNet → mel refinement (residual)
    """

    def __init__(self, n_mels=80):
        super().__init__()
        self.text_encoder    = TextEncoder()
        self.style_encoder   = VideoStyleEncoder()
        self.mel_decoder     = MelDecoder(in_dim=ENC_DIM + STYLE_DIM, n_mels=n_mels)
        self.post_net        = PostNet(n_mels=n_mels)

    def forward(self, text_ids, text_lengths, frames, frame_lengths, mel_lengths):
        """
        All lengths are 1-D (B,) int tensors.
        Returns mel_pred, mel_postnet — both (B, T_mel_max, n_mels).
        """
        # 1. Text encoding
        enc_out = self.text_encoder(text_ids, text_lengths)  # (B, T_text, enc_dim)

        # 2. Video style
        style = self.style_encoder(frames, frame_lengths)    # (B, style_dim)

        # 3. Up-sample text features to mel length (linear interpolation)
        T_mel_max = mel_lengths.max().item()
        enc_up = F.interpolate(
            enc_out.transpose(1, 2),                         # (B, enc_dim, T_text)
            size=T_mel_max, mode="linear", align_corners=False,
        ).transpose(1, 2)                                    # (B, T_mel_max, enc_dim)

        # 4. Broadcast style to every mel frame & concat
        style_exp = style.unsqueeze(1).expand(-1, T_mel_max, -1)  # (B, T_mel, style_dim)
        dec_in = torch.cat([enc_up, style_exp], dim=-1)           # (B, T_mel, enc_dim+style_dim)

        # 5. Mel prediction
        mel_pred = self.mel_decoder(dec_in)                  # (B, T_mel, n_mels)

        # 6. PostNet refinement (residual)
        mel_postnet = mel_pred + self.post_net(mel_pred)

        return mel_pred, mel_postnet

    @torch.no_grad()
    def infer(self, text_ids, text_lengths, frames, frame_lengths, mel_length: int):
        """Inference helper – generates mel of a given length."""
        mel_lengths = torch.tensor([mel_length], device=text_ids.device)
        _, mel_post = self.forward(
            text_ids, text_lengths, frames, frame_lengths, mel_lengths,
        )
        return mel_post                                      # (1, mel_length, n_mels)
