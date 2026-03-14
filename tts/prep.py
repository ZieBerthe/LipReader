"""Preprocess data for TTS with video style embedding.

For each video in the GRID corpus:
  1. Extract audio → log-mel spectrogram → mel.npy
  2. Extract mouth crop frames (via MediaPipe) → frames.npy
  3. Parse alignment file → text string

Outputs:
    /Data/tts_prepped/
        manifest.jsonl          one JSON record per sample
        tts_mel_config.json     mel-spectrogram parameters
        <speaker>/<video_id>/
            mel.npy             (T_mel, n_mels) float32
            frames.npy          (T_video, H, W) float32  (normalised mouth crops)

Usage (from project root):
    python tts/prep.py
    python tts/prep.py --max_samples 100   # quick test with fewer files
    python tts/prep.py --workers 8         # parallel processing (default: auto)
"""

import os
import sys
import json
import glob
import argparse
import contextlib
import multiprocessing as mp
from functools import partial

# ── path setup so we can import from the project root ──
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
from tqdm import tqdm

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
    import librosa  # import inside function for multiprocessing pickling
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


# ── Worker function for multiprocessing ────────────────────────────────────

def _process_one(args_tuple):
    """Process a single video sample. Returns a manifest dict or None on failure.

    Designed to run in a worker process (all imports happen inside).
    """
    video_path, align_path, output_dir, speaker_name = args_tuple
    vid_id = os.path.splitext(os.path.basename(video_path))[0]

    if not os.path.exists(align_path):
        return None

    sample_dir = os.path.join(output_dir, speaker_name, vid_id)
    mel_path    = os.path.join(sample_dir, "mel.npy")
    frames_path = os.path.join(sample_dir, "frames.npy")

    # ---- Skip if already cached ----
    if os.path.exists(mel_path) and os.path.exists(frames_path):
        # Rebuild manifest entry from existing files
        try:
            mel = np.load(mel_path)
            frames = np.load(frames_path)
            text = extract_text(align_path)
            return dict(
                id=vid_id,
                speaker=speaker_name,
                text=text,
                mel_path=mel_path,
                frames_path=frames_path,
                n_mel_frames=int(mel.shape[0]),
                n_video_frames=int(frames.shape[0]),
            )
        except Exception:
            pass  # re-process if cache is corrupt

    os.makedirs(sample_dir, exist_ok=True)

    # ---- mel spectrogram ----
    try:
        mel = extract_mel(video_path)
    except Exception as e:
        return f"mel:{vid_id}:{e}"
    np.save(mel_path, mel)

    # ---- mouth crop frames ----
    try:
        from load_data import load_video  # import per-worker for MediaPipe
        with suppress_stderr():
            frames_tensor = load_video(video_path)      # (T, H, W) normalised
        frames_np = frames_tensor.numpy().astype(np.float32)
    except Exception as e:
        return f"frames:{vid_id}:{e}"
    np.save(frames_path, frames_np)

    # ---- text ----
    text = extract_text(align_path)

    return dict(
        id=vid_id,
        speaker=speaker_name,
        text=text,
        mel_path=mel_path,
        frames_path=frames_path,
        n_mel_frames=int(mel.shape[0]),
        n_video_frames=int(frames_np.shape[0]),
    )


# ── Main ───────────────────────────────────────────────────────────────────

GRID_CORPUS_ROOT = '/Data/grid_corpus'
DEFAULT_OUTPUT   = '/Data/tts_prepped'


def main():
    parser = argparse.ArgumentParser(description="Preprocess GRID data for TTS")
    parser.add_argument("--corpus_root", default=GRID_CORPUS_ROOT)
    parser.add_argument("--output_dir",  default=DEFAULT_OUTPUT)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of samples (for quick testing)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: cpu_count / 2)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Discover all speaker folders and collect (video, align, speaker) tuples ----
    speaker_dirs = sorted(glob.glob(os.path.join(args.corpus_root, 's*_processed')))
    if not speaker_dirs:
        raise FileNotFoundError(f"No s*_processed folders in {args.corpus_root}")

    work_items = []  # (video_path, align_path, output_dir, speaker_name)
    for spk_dir in speaker_dirs:
        speaker_name = os.path.basename(spk_dir)  # e.g. s1_processed
        align_dir = os.path.join(spk_dir, 'align')
        for v in sorted(glob.glob(os.path.join(spk_dir, '*.mpg'))):
            vid_id = os.path.splitext(os.path.basename(v))[0]
            align_path = os.path.join(align_dir, vid_id + '.align')
            work_items.append((v, align_path, args.output_dir, speaker_name))

    if args.max_samples:
        work_items = work_items[: args.max_samples]

    # Count already-cached
    cached_count = sum(
        1 for v, a, o, s in work_items
        if os.path.exists(os.path.join(o, s, os.path.splitext(os.path.basename(v))[0], "mel.npy"))
        and os.path.exists(os.path.join(o, s, os.path.splitext(os.path.basename(v))[0], "frames.npy"))
    )

    print(f"Found {len(work_items)} videos across {len(speaker_dirs)} speakers")
    print(f"  Already cached: {cached_count}  |  To process: {len(work_items) - cached_count}")

    # ---- Save mel config so train / infer scripts stay consistent ----
    mel_config = dict(
        sr=MEL_SR, n_fft=MEL_N_FFT, hop_length=MEL_HOP,
        win_length=MEL_WIN, n_mels=MEL_N_MELS, fmin=MEL_FMIN, fmax=MEL_FMAX,
    )
    config_path = os.path.join(args.output_dir, "tts_mel_config.json")
    with open(config_path, "w") as f:
        json.dump(mel_config, f, indent=2)
    print(f"Saved mel config → {config_path}")

    # ---- Parallel processing ----
    n_workers = args.workers
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 4) // 2)
    # Clamp to something reasonable (MediaPipe is memory-hungry)
    n_workers = min(n_workers, 16)
    print(f"Processing with {n_workers} workers …")

    manifest_path = os.path.join(args.output_dir, "manifest.jsonl")
    processed = 0
    skipped = 0
    errors = []

    if n_workers <= 1:
        # ---- Sequential fallback ----
        results = []
        for item in tqdm(work_items, desc="Preprocessing"):
            results.append(_process_one(item))
    else:
        # ---- Multiprocessing with imap for progress bar ----
        # Use 'spawn' context to avoid MediaPipe fork issues
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=n_workers) as pool:
            results = list(tqdm(
                pool.imap(_process_one, work_items, chunksize=4),
                total=len(work_items),
                desc="Preprocessing",
            ))

    # ---- Write manifest from results ----
    with open(manifest_path, "w") as mf:
        for result in results:
            if result is None:
                skipped += 1
            elif isinstance(result, str):
                # Error string
                skipped += 1
                errors.append(result)
            else:
                mf.write(json.dumps(result) + "\n")
                processed += 1

    print(f"\nDone.  manifest → {manifest_path}")
    print(f"  processed : {processed}")
    print(f"  skipped   : {skipped}")
    if errors:
        print(f"  errors ({len(errors)}):")
        for e in errors[:20]:
            print(f"    {e}")
        if len(errors) > 20:
            print(f"    … and {len(errors) - 20} more")


if __name__ == "__main__":
    main()
