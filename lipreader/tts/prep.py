"""Preprocess data for TTS with video style embedding.

For each video in the GRID corpus:
  1. Extract audio → log-mel spectrogram → mel.npy
  2. Extract mouth crop frames (via MediaPipe) → frames.npy
  3. Parse alignment file → text string

Outputs:
    data/tts_prepped/            (paths.TTS_PREP_ROOT)
        manifest.jsonl          one JSON record per sample
        tts_mel_config.json     mel-spectrogram parameters
        <speaker>/<video_id>/
            mel.npy             (T_mel, n_mels) float32
            frames.npy          (T_video, H, W) float32  (normalised mouth crops)

Usage:
    pixi run tts-prep
    pixi run tts-prep --max_samples 100   # quick test with fewer files
    pixi run tts-prep --workers 8         # parallel processing (default: auto)
"""

import os
import json
import argparse
import multiprocessing as mp

import numpy as np
from tqdm import tqdm

from lipreader.audio import MEL_CONFIG, extract_mel
from lipreader.data.grid import find_samples, read_transcript
from lipreader.paths import DATA_ROOT, TTS_PREP_ROOT
from lipreader.utils import suppress_stderr


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
            text = read_transcript(align_path)
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
        from lipreader.data.video import load_video  # import per-worker for MediaPipe
        with suppress_stderr():
            frames_tensor = load_video(video_path)      # (T, H, W) normalised
        frames_np = frames_tensor.numpy().astype(np.float32)
    except Exception as e:
        return f"frames:{vid_id}:{e}"
    np.save(frames_path, frames_np)

    # ---- text ----
    text = read_transcript(align_path)

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



def main():
    parser = argparse.ArgumentParser(description="Preprocess GRID data for TTS")
    parser.add_argument("--corpus_root", default=DATA_ROOT)
    parser.add_argument("--output_dir",  default=TTS_PREP_ROOT)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of samples (for quick testing)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: cpu_count / 2)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Discover all speaker folders and collect (video, align, speaker) tuples ----
    samples = find_samples(args.corpus_root)
    speaker_dirs = sorted({s for s, _, _ in samples})

    # (video_path, align_path, output_dir, speaker_name)
    work_items = [(v, a, args.output_dir, s) for s, v, a in samples]

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
    mel_config = MEL_CONFIG
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
