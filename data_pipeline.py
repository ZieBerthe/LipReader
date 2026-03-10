import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from matplotlib import pyplot as plt
import imageio
from load_data import load_data, decode_indices, encode_chars, char_to_num, num_to_char


# ---- Data root (absolute path – do NOT auto-detect) ----
GRID_CORPUS_ROOT = '/Data/grid_corpus'


# Custom Dataset for LipReading (text + word timing prediction)
class LipReadingDataset(Dataset):
    def __init__(self, corpus_root=GRID_CORPUS_ROOT, cache_root=None, use_cache=True):
        """
        Args:
            corpus_root: Root directory containing s*_processed/ speaker folders.
                         Each folder has .mpg files and an align/ subfolder with .align files.
            cache_root:  Root directory for cached preprocessed data.
                         Defaults to <corpus_root>/../grid_corpus_cache
            use_cache:   Whether to use cached data (much faster!)
        """
        self.corpus_root = corpus_root
        self.video_paths = []
        self.align_paths = []
        self.cache_paths = []

        if cache_root is None:
            cache_root = os.path.join(os.path.dirname(corpus_root), 'grid_corpus_cache')
        self.cache_root = cache_root

        # Discover all speaker folders
        speaker_dirs = sorted(glob.glob(os.path.join(corpus_root, 's*_processed')))
        if not speaker_dirs:
            raise FileNotFoundError(f"No s*_processed folders found in {corpus_root}")

        for speaker_dir in speaker_dirs:
            speaker_name = os.path.basename(speaker_dir)  # e.g. s1_processed
            align_dir = os.path.join(speaker_dir, 'align')
            cache_dir = os.path.join(cache_root, speaker_name)

            videos = sorted(glob.glob(os.path.join(speaker_dir, '*.mpg')))
            for v in videos:
                vname = os.path.splitext(os.path.basename(v))[0]
                self.video_paths.append(v)
                self.align_paths.append(os.path.join(align_dir, f"{vname}.align"))
                self.cache_paths.append(os.path.join(cache_dir, f"{vname}.pt"))

        self.use_cache = use_cache

        if self.use_cache:
            cached_count = sum(1 for p in self.cache_paths if os.path.exists(p))
            if cached_count < len(self.video_paths):
                print(f"⚠ Warning: Only {cached_count}/{len(self.video_paths)} videos cached.")
                print(f"  Run 'python preprocess_cache.py' to cache all videos for faster loading.")
            else:
                print(f"✓ All {cached_count} videos cached.")

        print(f"Dataset: {len(self.video_paths)} videos from {len(speaker_dirs)} speakers")

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
            mouth_frames, char_indices, frame_word_labels, word_timings = load_data(
                video_path, align_path
            )
        except Exception:
            # Video is broken (no face detected, etc.) — return a random cached sample
            import random
            alt_idx = random.randint(0, len(self.video_paths) - 1)
            return self.__getitem__(alt_idx)
        return {
            'mouth_frames': mouth_frames,                                       # (T, H, W)
            'char_indices': torch.tensor(char_indices, dtype=torch.long),       # (L,)
            'frame_word_labels': frame_word_labels,                             # (T,)
            'word_timings': word_timings,                                       # list of (word, start, end)
            'video_path': video_path,
        }


# Global maximum dimensions for padding
MAX_MOUTH_FRAMES = 75   # Maximum number of video frames
MAX_TEXT_LENGTH = 50     # Maximum text sequence length (longest in GRID is 42)
MAX_WORDS = 10           # Maximum number of words per sentence (GRID has 6)


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
    dataset = LipReadingDataset(corpus_root=GRID_CORPUS_ROOT)
    total_len = len(dataset)
    train_len = int(0.7 * total_len)
    test_len = total_len - train_len
    train_set, test_set = random_split(
        dataset, [train_len, test_len],
        generator=torch.Generator().manual_seed(42)
    )
    batch_size = 10
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    batch = next(iter(train_loader))
    frames = batch['mouth_frames'].numpy()
    alignments = batch['char_indices'].numpy()

    print('Batch mouth frames shape:', frames.shape)
    print('Batch alignments shape:', alignments.shape)
    print('Batch frame word labels shape:', batch['frame_word_labels'].shape)
    print('Batch video paths:', batch['video_paths'])

    first_video_frames = frames[0]
    print('First video shape:', first_video_frames.shape)

    # Save GIF of first video
    gif_path = './animation.gif'
    gif_frames = [((f - f.min()) / (f.max() - f.min()) * 255).astype(np.uint8)
                  for f in first_video_frames]
    imageio.mimsave(gif_path, gif_frames, fps=10)
    print(f'Saved GIF to {gif_path}')

    # Print text and word timings for each sample in the batch
    for i in range(min(batch_size, len(batch['video_paths']))):
        char_indices = batch['char_indices'][i].numpy()
        text = ''.join([num_to_char[idx] for idx in char_indices if idx != 0])
        print(f'\nVideo: {batch["video_paths"][i]}')
        print(f'  Text: {text}')
        print(f'  Frame labels (first 20): {batch["frame_word_labels"][i][:20].tolist()}')
        print(f'  Word timings:')
        for word, start, end in batch['word_timings'][i]:
            print(f'    {word}: {start:.3f}s - {end:.3f}s')
