"""Preprocess all videos once and cache mouth crops + alignment data to disk (.pt per clip).

Training reads from this cache, which is much faster than running MediaPipe every epoch.
"""
import os

import torch
from tqdm import tqdm

from lipreader.paths import DATA_ROOT, CACHE_ROOT
from lipreader.data.grid import find_samples, load_sample


def preprocess_and_cache(corpus_root, cache_root):
    """Process all speaker videos and save mouth frames + alignment data to cache."""

    samples = find_samples(corpus_root)
    by_speaker = {}
    for speaker_name, video_path, align_path in samples:
        by_speaker.setdefault(speaker_name, []).append((video_path, align_path))

    total_videos = 0
    total_cached = 0

    for speaker_name, items in by_speaker.items():
        cache_dir = os.path.join(cache_root, speaker_name)
        os.makedirs(cache_dir, exist_ok=True)

        video_paths = [v for v, _ in items]
        align_for = {v: a for v, a in items}
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

            align_path = align_for[video_path]

            try:
                mouth_frames, char_indices, frame_word_labels, word_timings = load_sample(
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
    preprocess_and_cache(DATA_ROOT, CACHE_ROOT)
