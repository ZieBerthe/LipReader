"""PyTorch dataset over GRID clips (cached mouth crops + transcripts) and its collate fn."""
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from lipreader.paths import DATA_ROOT, CACHE_ROOT, OUTPUT_DIR
from lipreader.data.grid import find_samples, load_sample
from lipreader.data.vocab import decode_indices

# Global maximum dimensions for padding
MAX_MOUTH_FRAMES = 75   # Maximum number of video frames
MAX_TEXT_LENGTH = 50    # Maximum text sequence length (longest in GRID is 42)
MAX_WORDS = 10          # Maximum number of words per sentence (GRID has 6)


class LipReadingDataset(Dataset):
    def __init__(self, corpus_root=DATA_ROOT, cache_root=None, use_cache=True):
        """
        Args:
            corpus_root: GRID root with speaker folders (see grid.find_samples for layouts).
            cache_root:  Root directory for cached preprocessed data.
                         Defaults to paths.CACHE_ROOT (<data>/cache)
            use_cache:   Whether to use cached data (much faster!)
        """
        self.corpus_root = corpus_root
        self.video_paths = []
        self.align_paths = []
        self.cache_paths = []

        if cache_root is None:
            cache_root = CACHE_ROOT
        self.cache_root = cache_root

        samples = find_samples(corpus_root)
        for speaker_name, v, align_path in samples:
            vname = os.path.splitext(os.path.basename(v))[0]
            self.video_paths.append(v)
            self.align_paths.append(align_path)
            self.cache_paths.append(os.path.join(cache_root, speaker_name, f"{vname}.pt"))
        num_speakers = len({s for s, _, _ in samples})

        self.use_cache = use_cache

        if self.use_cache:
            cached_count = sum(1 for p in self.cache_paths if os.path.exists(p))
            if cached_count < len(self.video_paths):
                print(f"⚠ Warning: Only {cached_count}/{len(self.video_paths)} videos cached.")
                print(f"  Run `pixi run cache` to cache all videos for faster loading.")
            else:
                print(f"✓ All {cached_count} videos cached.")

        print(f"Dataset: {len(self.video_paths)} videos from {num_speakers} speakers")

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        cache_path = self.cache_paths[idx]
        
        # Try to load from cache first
        if self.use_cache and os.path.exists(cache_path):
            data = torch.load(cache_path)
            return {
                'mouth_frames': data['mouth_frames'],
                'char_indices': torch.tensor(data['char_indices'], dtype=torch.long) if not isinstance(data['char_indices'], torch.Tensor) else data['char_indices'],
                'frame_word_labels': data['frame_word_labels'],
                'word_timings': data['word_timings'],
                'video_path': video_path,
            }
        
        # Fallback to on-the-fly processing — skip bad videos gracefully
        align_path = self.align_paths[idx]
        try:
            mouth_frames, char_indices, frame_word_labels, word_timings = load_sample(
                video_path, align_path
            )
        except Exception:
            # Video is broken (no face detected, etc.) — return a random sample instead
            alt_idx = random.randint(0, len(self.video_paths) - 1)
            return self.__getitem__(alt_idx)
        return {
            'mouth_frames': mouth_frames,                                       # (T, H, W)
            'char_indices': torch.tensor(char_indices, dtype=torch.long),       # (L,)
            'frame_word_labels': frame_word_labels,                             # (T,)
            'word_timings': word_timings,                                       # list of (word, start, end)
            'video_path': video_path,
        }


def collate_fn(batch):
    """Collate and pad variable-length sequences to global max sizes."""
    mouth_seqs = [item['mouth_frames'] for item in batch]
    text_seqs = [item['char_indices'] for item in batch]
    timing_seqs = [item['frame_word_labels'] for item in batch]

    # Truncate anything that exceeds max lengths
    mouth_seqs = [s[:MAX_MOUTH_FRAMES] for s in mouth_seqs]
    text_seqs = [s[:MAX_TEXT_LENGTH] for s in text_seqs]
    timing_seqs = [s[:MAX_MOUTH_FRAMES] for s in timing_seqs]

    mouth_lens = [seq.shape[0] for seq in mouth_seqs]
    text_lens = [seq.shape[0] for seq in text_seqs]

    # Pad mouth frames: (B, MAX_MOUTH_FRAMES, H, W)
    mouth_padded = torch.stack([
        torch.nn.functional.pad(seq, (0, 0, 0, 0, 0, MAX_MOUTH_FRAMES - seq.shape[0]))
        for seq in mouth_seqs
    ])

    # Pad text: (B, MAX_TEXT_LENGTH)
    text_padded = torch.stack([
        torch.nn.functional.pad(seq, (0, MAX_TEXT_LENGTH - seq.shape[0]))
        for seq in text_seqs
    ])

    # Pad frame word labels: (B, MAX_MOUTH_FRAMES)
    timing_padded = torch.stack([
        torch.nn.functional.pad(seq, (0, MAX_MOUTH_FRAMES - seq.shape[0]))
        for seq in timing_seqs
    ])

    return {
        'mouth_frames': mouth_padded,
        'mouth_lengths': torch.tensor(mouth_lens),
        'char_indices': text_padded,
        'text_lengths': torch.tensor(text_lens),
        'frame_word_labels': timing_padded,
        'word_timings': [item['word_timings'] for item in batch],
        'video_paths': [item['video_path'] for item in batch],
    }


if __name__ == "__main__":
    import imageio

    dataset = LipReadingDataset()
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    batch = next(iter(loader))

    print('Batch mouth frames shape:', tuple(batch['mouth_frames'].shape))
    print('Batch char indices shape:', tuple(batch['char_indices'].shape))
    for i, path in enumerate(batch['video_paths']):
        text = decode_indices(batch['char_indices'][i].tolist())
        print(f'\n{path}\n  Text: {text!r}')
        for word, start, end in batch['word_timings'][i]:
            print(f'    {word}: {start:.3f}s - {end:.3f}s')

    # Save a GIF of the first clip's mouth crops to check the cropping visually
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    gif_path = os.path.join(OUTPUT_DIR, 'mouth_preview.gif')
    frames = batch['mouth_frames'][0].numpy()
    frames = ((frames - frames.min()) / (frames.max() - frames.min()) * 255).astype(np.uint8)
    imageio.mimsave(gif_path, list(frames), fps=10)
    print(f'\nSaved mouth-crop preview to {gif_path}')
