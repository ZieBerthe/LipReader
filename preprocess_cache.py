#!/usr/bin/env python3
"""
Preprocess all videos and cache extracted mouth frames to disk.
Run this ONCE before training to speed up data loading.
"""
import os
import glob
import torch
import numpy as np
from tqdm import tqdm
from load_data import load_data

GRID_CORPUS_ROOT = '/Data/grid_corpus'
CACHE_ROOT = '/Data/grid_corpus_cache'


def preprocess_and_cache(corpus_root, cache_root):
    """Process all speaker videos and save mouth frames + alignment data to cache."""

    speaker_dirs = sorted(glob.glob(os.path.join(corpus_root, 's*_processed')))
    if not speaker_dirs:
        raise FileNotFoundError(f"No s*_processed folders found in {corpus_root}")

    total_videos = 0
    total_cached = 0

    for speaker_dir in speaker_dirs:
        speaker_name = os.path.basename(speaker_dir)
        align_dir = os.path.join(speaker_dir, 'align')
        cache_dir = os.path.join(cache_root, speaker_name)
        os.makedirs(cache_dir, exist_ok=True)

        video_paths = sorted(glob.glob(os.path.join(speaker_dir, '*.mpg')))
        total_videos += len(video_paths)

        # Count already-cached
        already_cached = sum(
            1 for v in video_paths
            if os.path.exists(os.path.join(cache_dir, os.path.splitext(os.path.basename(v))[0] + '.pt'))
        )
        if already_cached == len(video_paths):
            total_cached += already_cached
            continue  # nothing to do for this speaker

        print(f"\n[{speaker_name}] {len(video_paths)} videos ({already_cached} already cached)")

        for video_path in tqdm(video_paths, desc=f"  {speaker_name}"):
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            cache_path = os.path.join(cache_dir, f"{video_name}.pt")

            if os.path.exists(cache_path):
                total_cached += 1
                continue

            align_path = os.path.join(align_dir, f"{video_name}.align")

            try:
                mouth_frames, char_indices, frame_word_labels, word_timings = load_data(
                    video_path, align_path
                )
                torch.save({
                    'mouth_frames': mouth_frames,
                    'char_indices': char_indices,
                    'frame_word_labels': frame_word_labels,
                    'word_timings': word_timings,
                    'video_path': video_path,
                }, cache_path)
                total_cached += 1

            except Exception as e:
                print(f"\n  Error processing {video_name}: {e}")
                continue

    # Summary
    cache_size_mb = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fnames in os.walk(cache_root) for f in fnames
    ) / 1024**2
    print(f"\n✓ Preprocessing complete! {total_cached}/{total_videos} videos cached.")
    print(f"  Total cache size: {cache_size_mb:.1f} MB")


if __name__ == "__main__":
    preprocess_and_cache(GRID_CORPUS_ROOT, CACHE_ROOT)
