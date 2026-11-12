"""GRID corpus: locating samples on disk and parsing alignment files."""
import os
import re
import glob
from typing import List, Tuple

import torch

from lipreader.paths import DATA_ROOT
from lipreader.data.vocab import encode_chars

ALIGN_SAMPLE_RATE = 25000  # alignment timestamps are in 1/25000 s
VIDEO_FPS = 25

_SPEAKER_DIR = re.compile(r"^s\d+(_processed)?$")


def find_samples(corpus_root: str = DATA_ROOT) -> List[Tuple[str, str, str]]:
    """Return sorted (speaker, video_path, align_path) tuples for every .mpg found.

    Two GRID layouts are supported:
      * download layout:   <root>/s1/*.mpg  +  <root>/alignments/s1/*.align
      * processed layout:  <root>/s1_processed/*.mpg  +  <root>/s1_processed/align/*.align
    The align path is returned even if the file is missing; callers decide what to do.
    """
    speaker_dirs = sorted(
        d for d in glob.glob(os.path.join(corpus_root, "s*"))
        if os.path.isdir(d) and _SPEAKER_DIR.match(os.path.basename(d))
    )
    if not speaker_dirs:
        raise FileNotFoundError(
            f"No speaker folders (s1/, s2/ … or s1_processed/ …) found in {corpus_root}. "
            f"Run `pixi run download` or set LIPREADER_DATA."
        )

    samples = []
    for spk_dir in speaker_dirs:
        speaker = os.path.basename(spk_dir)
        if os.path.isdir(os.path.join(spk_dir, "align")):
            align_dir = os.path.join(spk_dir, "align")
        else:
            align_dir = os.path.join(corpus_root, "alignments", speaker)
        for video in sorted(glob.glob(os.path.join(spk_dir, "*.mpg"))):
            vid_id = os.path.splitext(os.path.basename(video))[0]
            samples.append((speaker, video, os.path.join(align_dir, vid_id + ".align")))
    return samples


def read_transcript(alignment_path: str) -> str:
    """Return the spoken sentence of an alignment file, without silences."""
    with open(alignment_path) as f:
        words = [parts[2] for parts in (line.split() for line in f) if len(parts) >= 3]
    return " ".join(w for w in words if w != "sil")


def load_alignment(
    alignment_path: str,
    num_frames: int = 75,
) -> Tuple[List[int], torch.Tensor, List[Tuple[str, float, float]]]:
    """Load alignment file and return text + per-frame word timing labels.

    The alignment file format (GRID corpus):
        start_sample end_sample word
        e.g. "23750 29500 bin"

    Timestamps are in units of 1/25000 s.  At 25 fps the conversion is
        frame_idx = timestamp * VIDEO_FPS / ALIGN_SAMPLE_RATE  (= timestamp / 1000)

    Returns:
        char_indices:      Encoded character indices for CTC loss
        frame_word_labels: (num_frames,) tensor — word position index per frame
                           0 = silence, 1 = first word, 2 = second word, …
        word_timings:      List of (word, start_sec, end_sec) tuples
    """
    with open(alignment_path, 'r') as f:
        lines = f.readlines()

    segments = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 3:
            segments.append((int(parts[0]), int(parts[1]), parts[2]))

    # ---- Character sequence (for CTC) ----
    tokens = ''
    for start, end, word in segments:
        if word != 'sil':
            tokens += ' ' + word
    char_indices = encode_chars(list(tokens))

    # ---- Per-frame word labels & word timing list ----
    word_timings: List[Tuple[str, float, float]] = []
    frame_word_labels = torch.zeros(num_frames, dtype=torch.long)
    word_idx = 0  # 0 = silence

    for start, end, word in segments:
        start_sec = start / ALIGN_SAMPLE_RATE
        end_sec = end / ALIGN_SAMPLE_RATE

        if word != 'sil':
            word_idx += 1
            word_timings.append((word, start_sec, end_sec))

        start_frame = int(start * VIDEO_FPS / ALIGN_SAMPLE_RATE)
        end_frame = int(end * VIDEO_FPS / ALIGN_SAMPLE_RATE)
        start_frame = max(0, min(start_frame, num_frames))
        end_frame = max(0, min(end_frame, num_frames))

        label = word_idx if word != 'sil' else 0
        frame_word_labels[start_frame:end_frame] = label

    return char_indices, frame_word_labels, word_timings


def load_sample(video_path: str, alignment_path: str):
    """Load mouth crops and alignment for one GRID clip.

    Returns:
        mouth_frames:      (T, H, W) tensor of normalised mouth crops
        char_indices:      Encoded character indices for CTC
        frame_word_labels: (T,) tensor of per-frame word indices (0 = silence)
        word_timings:      List of (word, start_sec, end_sec) tuples
    """
    from lipreader.data.video import load_video  # lazy: imports MediaPipe

    mouth_frames = load_video(video_path)
    char_indices, frame_word_labels, word_timings = load_alignment(
        alignment_path, num_frames=mouth_frames.shape[0]
    )
    return mouth_frames, char_indices, frame_word_labels, word_timings


if __name__ == "__main__":
    speaker, video, align = find_samples()[0]
    mouth_frames, char_indices, frame_word_labels, word_timings = load_sample(video, align)
    print(f'Sample: {video}')
    print(f'Mouth frames shape: {tuple(mouth_frames.shape)}')
    print(f'Transcript: {read_transcript(align)!r}')
    print(f'Character indices: {char_indices}')
    print(f'Frame word labels: {frame_word_labels.tolist()}')
    for word, start, end in word_timings:
        print(f'  {word}: {start:.3f}s - {end:.3f}s')
