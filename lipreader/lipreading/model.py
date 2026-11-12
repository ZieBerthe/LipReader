"""Lip-reading network: 3D-CNN front-end + 2-layer BiLSTM + linear CTC head."""
import torch
import torch.nn as nn

from lipreader.data.dataset import MAX_WORDS


def adaptive_avg_pool_2x2(x):
    """Same result as nn.AdaptiveAvgPool3d((None, 2, 2)) on (B, C, T, H, W).

    Written with slicing because MPS supports neither AdaptiveAvgPool3d nor
    adaptive pooling with non-divisible sizes (here W=25 → 2).
    Bins follow PyTorch's rule: [floor(i*n/2), ceil((i+1)*n/2)).
    """
    H, W = x.shape[-2:]
    hb = [(0, (H + 1) // 2), (H // 2, H)]
    wb = [(0, (W + 1) // 2), (W // 2, W)]
    rows = [torch.stack([x[..., h0:h1, w0:w1].mean(dim=(-2, -1)) for w0, w1 in wb], dim=-1)
            for h0, h1 in hb]
    return torch.stack(rows, dim=-2)


class LipReadingModel(nn.Module):
    """Lip reading model that predicts text (CTC only).

    Outputs:
        text_logits: (B, T, num_chars) — for CTC decoding
    """

    def __init__(self, num_chars, max_word_slots=MAX_WORDS):
        super(LipReadingModel, self).__init__()
        # CNN sized for ~21k samples (31 speakers)
        self.cnn = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),

            nn.Conv3d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm3d(96),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),  # (B, 96, T, 12, 25)
        )
        # Collapse spatial dims per frame: (12, 25) → (2, 2) — see adaptive_avg_pool_2x2
        self.spatial_pool = adaptive_avg_pool_2x2
        self.feature_size = 96 * 2 * 2  # = 384

        # 2-layer BiLSTM
        self.bilstm = nn.LSTM(
            input_size=self.feature_size, hidden_size=128,
            num_layers=2, batch_first=True, bidirectional=True,
            dropout=0.3,
        )

        # CTC text prediction head
        self.text_head = nn.Linear(256, num_chars)

        self.dropout = nn.Dropout(0.3)

    OUT_DIM = 256  # size of the per-frame features returned by encode()

    def encode(self, x):
        """Mouth crops (B, T, H, W) → per-frame visual features (B, T, 256)."""
        B, T, H, W = x.size()
        x = x.unsqueeze(1)  # (B, 1, T, H, W)

        features = self.cnn(x)  # (B, 96, T, 12, 25)
        T_out = features.size(2)
        features = self.spatial_pool(features)                          # (B, 96, T, 2, 2)
        features = features.permute(0, 2, 1, 3, 4).reshape(B, T_out, -1)  # (B, T, 384)

        lstm_out, _ = self.bilstm(features)   # (B, T, 256)
        return self.dropout(lstm_out)

    def forward(self, x):
        return self.text_head(self.encode(x))      # (B, T, num_chars)
