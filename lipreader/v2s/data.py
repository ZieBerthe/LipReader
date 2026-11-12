"""Data for the lip-to-speech model: per-clip log-mels next to the cached mouth crops.

    pixi run v2s-prep     # writes data/mels/<speaker>/<clip>.npy and data/mels/stats.npz

Mouth crops come from the existing cache (`pixi run cache`); mels follow audio.extract_mel,
which matches the pretrained HiFi-GAN.
"""
import os
import multiprocessing as mp

import numpy as np
import torch
from torch.utils.data import Subset
from tqdm import tqdm

from lipreader.audio import extract_mel
from lipreader.data.dataset import LipReadingDataset, MAX_MOUTH_FRAMES
from lipreader.data.grid import find_samples
from lipreader.paths import DATA_ROOT, MEL_ROOT

T_MEL = 256                      # mel frames per clip (≈2.97 s at 22.05 kHz / hop 256)
MEL_PAD_VALUE = float(np.log(1e-5))  # log-mel of silence
SPLIT_SEED = 67
VAL_FRACTION = 0.1
STATS_PATH = os.path.join(MEL_ROOT, "stats.npz")


def mel_path_for(video_path: str) -> str:
    speaker = os.path.basename(os.path.dirname(video_path))
    clip = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(MEL_ROOT, speaker, clip + ".npy")


def split_indices(n: int, seed: int = SPLIT_SEED):
    """Deterministic train/val split over the clips returned by find_samples()."""
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    n_val = max(1, round(VAL_FRACTION * n))
    return sorted(perm[n_val:]), sorted(perm[:n_val])


def load_mel_stats(path: str = STATS_PATH):
    """Per-mel-bin mean and std over the training clips, as float32 arrays of shape (n_mels,)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run `pixi run v2s-prep` first.")
    stats = np.load(path)
    return stats["mean"].astype(np.float32), stats["std"].astype(np.float32)


def fit_length(x: torch.Tensor, length: int, pad_value: float = 0.0) -> torch.Tensor:
    """Trim or pad the first (time) dimension to exactly `length`."""
    x = x[:length]
    if x.shape[0] < length:
        pad = torch.full((length - x.shape[0], *x.shape[1:]), pad_value, dtype=x.dtype)
        x = torch.cat([x, pad])
    return x


class V2SDataset(LipReadingDataset):
    """Cached mouth crops (75, H, W) paired with the clip's raw log-mel (T_MEL, 80)."""

    def __init__(self, corpus_root: str = DATA_ROOT):
        super().__init__(corpus_root=corpus_root, use_cache=True)

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        mel = torch.from_numpy(np.load(mel_path_for(item['video_path'])))
        return {
            'mouth_frames': fit_length(item['mouth_frames'], MAX_MOUTH_FRAMES),
            'mel': fit_length(mel, T_MEL, MEL_PAD_VALUE),
            'video_path': item['video_path'],
        }


def train_val_datasets(max_clips: int = None):
    """Return (train, val) subsets. With max_clips, train on the first clips and validate on them too
    (an overfitting sanity check)."""
    dataset = V2SDataset()
    train_idx, val_idx = split_indices(len(dataset))
    if max_clips:
        train_idx = val_idx = train_idx[:max_clips]
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


# ---------------------------------------------------------------------------
#  Preprocessing
# ---------------------------------------------------------------------------

def _extract_one(video_path: str):
    out = mel_path_for(video_path)
    if not os.path.exists(out):
        os.makedirs(os.path.dirname(out), exist_ok=True)
        np.save(out, extract_mel(video_path))
    return out


def prep_mels(corpus_root: str = DATA_ROOT, workers: int = None):
    videos = [video for _, video, _ in find_samples(corpus_root)]
    workers = workers or max(1, (os.cpu_count() or 2) // 2)
    print(f"Extracting log-mels for {len(videos)} clips with {workers} workers → {MEL_ROOT}")
    with mp.get_context("spawn").Pool(workers) as pool:
        paths = list(tqdm(pool.imap(_extract_one, videos, chunksize=8), total=len(videos)))

    # Normalisation statistics from the training split only
    train_idx, _ = split_indices(len(paths))
    mels = np.concatenate([np.load(paths[i]) for i in train_idx])
    mean, std = mels.mean(axis=0), mels.std(axis=0) + 1e-5
    np.savez(STATS_PATH, mean=mean, std=std)
    lengths = [np.load(p, mmap_mode='r').shape[0] for p in paths]
    print(f"Saved stats → {STATS_PATH}")
    print(f"Mel frames per clip: min {min(lengths)}, max {max(lengths)} (model uses {T_MEL})")


if __name__ == "__main__":
    prep_mels()
