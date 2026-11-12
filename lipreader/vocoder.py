"""Pretrained HiFi-GAN vocoder (UNIVERSAL_V1): log-mel → waveform.

The Generator below is copied from https://github.com/jik876/hifi-gan (models.py, config_v1),
MIT License, Copyright (c) 2020 Jungil Kong. Mels must come from lipreader.audio.extract_mel,
which reproduces HiFi-GAN's mel function.

    pixi run download-vocoder      # fetches UNIVERSAL_V1/g_02500000 (~55 MB) once
"""
import os
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import weight_norm, remove_weight_norm

from lipreader.paths import HIFIGAN_DIR

HIFIGAN_DRIVE_FOLDER = "https://drive.google.com/drive/folders/1-eEYTB5Av9jNql0WGBlRoi-WH2J7bp5Y"
HIFIGAN_FILE = "UNIVERSAL_V1/g_02500000"
HIFIGAN_CKPT = os.path.join(HIFIGAN_DIR, "g_02500000")

# config_v1.json
UPSAMPLE_RATES = [8, 8, 2, 2]
UPSAMPLE_KERNEL_SIZES = [16, 16, 4, 4]
UPSAMPLE_INITIAL_CHANNEL = 512
RESBLOCK_KERNEL_SIZES = [3, 7, 11]
RESBLOCK_DILATION_SIZES = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
LRELU_SLOPE = 0.1


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


class ResBlock1(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=d,
                               padding=get_padding(kernel_size, d)))
            for d in dilation
        ])
        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
            for _ in dilation
        ])

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = c1(F.leaky_relu(x, LRELU_SLOPE))
            xt = c2(F.leaky_relu(xt, LRELU_SLOPE))
            x = xt + x
        return x

    def remove_weight_norm(self):
        for layer in list(self.convs1) + list(self.convs2):
            remove_weight_norm(layer)


class Generator(nn.Module):
    def __init__(self, n_mels=80):
        super().__init__()
        self.num_kernels = len(RESBLOCK_KERNEL_SIZES)
        self.num_upsamples = len(UPSAMPLE_RATES)
        self.conv_pre = weight_norm(Conv1d(n_mels, UPSAMPLE_INITIAL_CHANNEL, 7, 1, padding=3))

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(UPSAMPLE_RATES, UPSAMPLE_KERNEL_SIZES)):
            self.ups.append(weight_norm(ConvTranspose1d(
                UPSAMPLE_INITIAL_CHANNEL // (2 ** i), UPSAMPLE_INITIAL_CHANNEL // (2 ** (i + 1)),
                k, u, padding=(k - u) // 2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = UPSAMPLE_INITIAL_CHANNEL // (2 ** (i + 1))
            for k, d in zip(RESBLOCK_KERNEL_SIZES, RESBLOCK_DILATION_SIZES):
                self.resblocks.append(ResBlock1(ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))

    def forward(self, x):
        """(B, n_mels, T_mel) log-mel → (B, 1, T_mel * 256) waveform in [-1, 1]."""
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = self.ups[i](F.leaky_relu(x, LRELU_SLOPE))
            xs = sum(self.resblocks[i * self.num_kernels + j](x) for j in range(self.num_kernels))
            x = xs / self.num_kernels
        x = self.conv_post(F.leaky_relu(x))
        return torch.tanh(x)

    def remove_weight_norm(self):
        for layer in self.ups:
            remove_weight_norm(layer)
        for block in self.resblocks:
            block.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


def download_hifigan(dest: str = HIFIGAN_CKPT) -> str:
    """Download the UNIVERSAL_V1 generator weights from the authors' Google Drive folder."""
    import gdown

    if os.path.exists(dest):
        print(f"HiFi-GAN already downloaded: {dest}")
        return dest
    files = gdown.download_folder(HIFIGAN_DRIVE_FOLDER, skip_download=True, quiet=True)
    match = [f for f in files if f.path == HIFIGAN_FILE]
    if not match:
        raise RuntimeError(f"{HIFIGAN_FILE} not found in {HIFIGAN_DRIVE_FOLDER}")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    gdown.download(id=match[0].id, output=dest, quiet=False)
    return dest


def load_hifigan(device: torch.device, ckpt_path: str = HIFIGAN_CKPT) -> Generator:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"{ckpt_path} not found. Run `pixi run download-vocoder` first.")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)  # torch.nn.utils.weight_norm deprecation
        model = Generator()
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state["generator"])
        model.remove_weight_norm()
    return model.eval().to(device)


@torch.no_grad()
def vocode(vocoder: Generator, log_mel: np.ndarray) -> np.ndarray:
    """(T_mel, n_mels) log-mel → 1-D float32 waveform at 22.05 kHz."""
    device = next(vocoder.parameters()).device
    mel = torch.as_tensor(log_mel, dtype=torch.float32, device=device).T.unsqueeze(0)
    return vocoder(mel).squeeze().cpu().numpy().astype(np.float32)


if __name__ == "__main__":
    download_hifigan()
