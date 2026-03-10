"""Preprocess data for TTS with video style embedding.

For each video in the GRID corpus:
  1. Extract audio → log-mel spectrogram → mel.npy
  2. Extract mouth crop frames (via MediaPipe) → frames.npy
  3. Parse alignment file → text string

Outputs:
    tts/prepped/
        manifest.jsonl          one JSON record per sample
        tts_mel_config.json     mel-spectrogram parameters
        <video_id>/
            mel.npy             (T_mel, n_mels) float32
            frames.npy          (T_video, H, W) float32  (normalised mouth crops)

Usage (from project root):
    python tts/prep.py
    python tts/prep.py --max_samples 100   # quick test with fewer files
"""

import os
import sys
import json
import glob
import argparse
import contextlib

# ── path setup so we can import from the project root ──
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import librosa
from tqdm import tqdm

# Reuse mouth-crop extraction from the lip-reader
from load_data import load_video, VIDEO_FPS

# ── Mel spectrogram configuration ─────────────────────────────────────────
MEL_SR        = 22050
MEL_N_FFT     = 1024
MEL_HOP       = 256
MEL_WIN       = 1024
MEL_N_MELS    = 80
MEL_FMIN      = 0
MEL_FMAX      = 8000


@contextlib.contextmanager
def suppress_stderr():
    """Redirect stderr to devnull (silences C++ / MediaPipe warnings)."""
    fd = sys.stderr.fileno()
    old = os.dup(fd)
    with open(os.devnull, "w") as devnull:
        os.dup2(devnull.fileno(), fd)
        try:
            yield
        finally:
            os.dup2(old, fd)
            os.close(old)


# ── Helpers ────────────────────────────────────────────────────────────────

def extract_mel(video_path: str) -> np.ndarray:
    """Extract log-mel spectrogram from the audio track of a video.

    Returns:
        (T_mel, n_mels) float32 array
    """
    y, _ = librosa.load(video_path, sr=MEL_SR)
    mel = librosa.feature.melspectrogram(
        y=y, sr=MEL_SR,
        n_fft=MEL_N_FFT, hop_length=MEL_HOP, win_length=MEL_WIN,
        n_mels=MEL_N_MELS, fmin=MEL_FMIN, fmax=MEL_FMAX,
    )
    log_mel = np.log(np.maximum(mel, 1e-5))        # log-mel
    return log_mel.astype(np.float32).T             # (T_mel, n_mels)


def extract_text(alignment_path: str) -> str:
    """Return the spoken sentence from a GRID .align file (no 'sil')."""
    words = []
    with open(alignment_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3 and parts[2] != "sil":
                words.append(parts[2])
    return " ".join(words)


# ── Main ───────────────────────────────────────────────────────────────────

GRID_CORPUS_ROOT = '/Data/grid_corpus'


def main():
    parser = argparse.ArgumentParser(description="Preprocess GRID data for TTS")
    parser.add_argument("--corpus_root", default=GRID_CORPUS_ROOT)
    parser.add_argument("--output_dir",  default=os.path.join(ROOT, "tts", "prepped"))
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of samples (for quick testing)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Discover all speaker folders and collect (video, align) pairs
    speaker_dirs = sorted(glob.glob(os.path.join(args.corpus_root, 's*_processed')))
    if not speaker_dirs:
        raise FileNotFoundError(f"No s*_processed folders in {args.corpus_root}")

    video_paths = []
    align_lookup = {}  # video_path → align_path
    for spk_dir in speaker_dirs:
        align_dir = os.path.join(spk_dir, 'align')
        for v in sorted(glob.glob(os.path.join(spk_dir, '*.mpg'))):
            vid_id = os.path.splitext(os.path.basename(v))[0]
            align_path = os.path.join(align_dir, vid_id + '.align')
            video_paths.append(v)
            align_lookup[v] = align_path

    if args.max_samples:
        video_paths = video_paths[: args.max_samples]

    print(f"Found {len(video_paths)} videos across {len(speaker_dirs)} speakers")

    # Save mel config so train / infer scripts stay consistent
    mel_config = dict(
        sr=MEL_SR, n_fft=MEL_N_FFT, hop_length=MEL_HOP,
        win_length=MEL_WIN, n_mels=MEL_N_MELS, fmin=MEL_FMIN, fmax=MEL_FMAX,
    )
    config_path = os.path.join(args.output_dir, "tts_mel_config.json")
    with open(config_path, "w") as f:
        json.dump(mel_config, f, indent=2)
    print(f"Saved mel config → {config_path}")

    manifest_path = os.path.join(args.output_dir, "manifest.jsonl")
    skipped = 0

    with open(manifest_path, "w") as mf:
        for video_path in tqdm(video_paths, desc="Preprocessing"):
            vid_id = os.path.splitext(os.path.basename(video_path))[0]
            align_path = align_lookup[video_path]
            if not os.path.exists(align_path):
                skipped += 1
                continue

            sample_dir = os.path.join(args.output_dir, vid_id)
            os.makedirs(sample_dir, exist_ok=True)

            # ---- mel spectrogram ----
            try:
                mel = extract_mel(video_path)
            except Exception as e:
                print(f"  [skip] mel failed for {vid_id}: {e}")
                skipped += 1
                continue
            mel_path = os.path.join(sample_dir, "mel.npy")
            np.save(mel_path, mel)

            # ---- mouth crop frames ----
            try:
                with suppress_stderr():
                    frames_tensor = load_video(video_path)      # (T, H, W) normalised
                frames_np = frames_tensor.numpy().astype(np.float32)
            except Exception as e:
                print(f"  [skip] frames failed for {vid_id}: {e}")
                skipped += 1
                continue
            frames_path = os.path.join(sample_dir, "frames.npy")
            np.save(frames_path, frames_np)

            # ---- text ----
            text = extract_text(align_path)

            # ---- manifest entry ----
            entry = dict(
                id=vid_id,
                text=text,
                mel_path=mel_path,
                frames_path=frames_path,
                n_mel_frames=int(mel.shape[0]),
                n_video_frames=int(frames_np.shape[0]),
            )
            mf.write(json.dumps(entry) + "\n")

    print(f"\nDone.  manifest → {manifest_path}")
    print(f"  processed : {len(video_paths) - skipped}")
    print(f"  skipped   : {skipped}")


if __name__ == "__main__":
    main()
