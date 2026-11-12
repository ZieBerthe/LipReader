"""Lip-to-speech with conditional flow matching.

    mouth crops (B, 75, H, W) ──LipReadingModel.encode──▶ (B, 75, 256) ──interpolate──▶ cond (B, T_mel, 256)
    noise x0 ~ N(0, I) ──Euler ODE with FlowDecoder(x_t, t, cond)──▶ normalised mel x1 (B, T_mel, 80)

Training uses the rectified-flow / OT-CFM objective (sigma_min = 0):
    x_t = (1 - t) x0 + t x1,   target velocity = x1 - x0,   loss = MSE(v_theta(x_t, t, cond), x1 - x0)
Classifier-free guidance: the condition is zeroed with probability `cfg_dropout` during training,
and at sampling v = v_uncond + w (v_cond - v_uncond).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from lipreader.data.vocab import NUM_CLASSES
from lipreader.lipreading.model import LipReadingModel


def sinusoidal_embedding(x: torch.Tensor, dim: int) -> torch.Tensor:
    """(N,) positions or timesteps → (N, dim) sin/cos features."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=x.device, dtype=torch.float32) / half)
    args = x.float()[:, None] * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class FlowDecoder(nn.Module):
    """Predicts the flow velocity for every mel frame with a small Transformer."""

    def __init__(self, n_mels=80, cond_dim=256, d_model=256, n_layers=6, n_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.in_proj = nn.Linear(n_mels + cond_dim, d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.SiLU(), nn.Linear(4 * d_model, d_model),
        )
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=4 * d_model, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, n_mels)
        nn.init.zeros_(self.out.weight)  # start by predicting zero velocity
        nn.init.zeros_(self.out.bias)

    def forward(self, x_t, t, cond):
        """x_t (B, T, n_mels), t (B,) in [0, 1], cond (B, T, cond_dim) → velocity (B, T, n_mels)."""
        B, T, _ = x_t.shape
        h = self.in_proj(torch.cat([x_t, cond], dim=-1))
        h = h + sinusoidal_embedding(torch.arange(T, device=h.device), self.d_model)[None]
        h = h + self.time_mlp(sinusoidal_embedding(t * 1000.0, self.d_model))[:, None]
        return self.out(self.norm(self.transformer(h)))


class LipToSpeech(nn.Module):
    def __init__(self, t_mel=256, n_mels=80, cfg_dropout=0.1, **decoder_kwargs):
        super().__init__()
        self.t_mel = t_mel
        self.n_mels = n_mels
        self.cfg_dropout = cfg_dropout
        # The lip reader's CNN + BiLSTM is the visual encoder; its text head stays unused for now.
        self.encoder = LipReadingModel(num_chars=NUM_CLASSES)
        self.decoder = FlowDecoder(n_mels=n_mels, cond_dim=LipReadingModel.OUT_DIM, **decoder_kwargs)
        # Per-bin log-mel normalisation (set from data/mels/stats.npz, saved with the checkpoint)
        self.register_buffer("mel_mean", torch.zeros(n_mels))
        self.register_buffer("mel_std", torch.ones(n_mels))

    def set_mel_stats(self, mean, std):
        self.mel_mean.copy_(torch.as_tensor(mean))
        self.mel_std.copy_(torch.as_tensor(std))

    def init_encoder(self, lipreader_ckpt: str):
        """Start the visual encoder from a trained lip reader (checkpoints/lipreading_best.pth)."""
        state = torch.load(lipreader_ckpt, map_location="cpu")
        self.encoder.load_state_dict(state, strict=False)

    def condition(self, frames):
        """(B, 75, H, W) mouth crops → (B, t_mel, 256) visual features at the mel frame rate."""
        feats = self.encoder.encode(frames)                       # (B, 75, 256)
        feats = F.interpolate(feats.transpose(1, 2), size=self.t_mel, mode="linear", align_corners=False)
        return feats.transpose(1, 2)

    def loss(self, frames, mel, generator: torch.Generator = None):
        """Flow-matching loss. `mel` is the raw log-mel (B, t_mel, n_mels).
        Pass a CPU `generator` to make the noise deterministic (validation)."""
        B = mel.shape[0]
        x1 = (mel - self.mel_mean) / self.mel_std
        cond = self.condition(frames)
        if self.training and self.cfg_dropout > 0:
            keep = (torch.rand(B, device=cond.device) >= self.cfg_dropout).float()
            cond = cond * keep[:, None, None]

        x0 = torch.randn(x1.shape, generator=generator).to(x1.device)
        t = torch.rand(B, generator=generator).to(x1.device)
        x_t = (1 - t[:, None, None]) * x0 + t[:, None, None] * x1
        v = self.decoder(x_t, t, cond)
        return F.mse_loss(v, x1 - x0)

    @torch.no_grad()
    def sample(self, frames, steps: int = 32, cfg_scale: float = 2.0):
        """Generate raw log-mels (B, t_mel, n_mels) from mouth crops with an Euler ODE solver."""
        B = frames.shape[0]
        cond = self.condition(frames)
        x = torch.randn(B, self.t_mel, self.n_mels, device=frames.device)
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((B,), i * dt, device=frames.device)
            v = self.decoder(x, t, cond)
            if cfg_scale != 1.0:
                v_uncond = self.decoder(x, t, torch.zeros_like(cond))
                v = v_uncond + cfg_scale * (v - v_uncond)
            x = x + dt * v
        return x * self.mel_std + self.mel_mean
