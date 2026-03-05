import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from matplotlib import pyplot as plt
import imageio
from load_data import load_data, decode_indices, encode_chars, char_to_num, num_to_char


# Custom Dataset for LipReading (text + word timing prediction)
class LipReadingDataset(Dataset):
    def __init__(self, video_dir, align_dir):
        self.video_paths = sorted(glob.glob(os.path.join(video_dir, '*.mpg')))
        self.align_paths = [
            os.path.join(align_dir, os.path.splitext(os.path.basename(v))[0] + '.align')
            for v in self.video_paths
        ]

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        align_path = self.align_paths[idx]
        mouth_frames, char_indices, frame_word_labels, word_timings = load_data(
            video_path, align_path
        )
        return {
            'mouth_frames': mouth_frames,                                       # (T, H, W)
            'char_indices': torch.tensor(char_indices, dtype=torch.long),       # (L,)
            'frame_word_labels': frame_word_labels,                             # (T,)
            'word_timings': word_timings,                                       # list of (word, start, end)
            'video_path': video_path,
        }


# Global maximum dimensions for padding
MAX_MOUTH_FRAMES = 75   # Maximum number of video frames
MAX_TEXT_LENGTH = 40     # Maximum text sequence length
MAX_WORDS = 10           # Maximum number of words per sentence (GRID has 6)


def collate_fn(batch):
    """Collate and pad variable-length sequences to global max sizes."""
    mouth_seqs = [item['mouth_frames'] for item in batch]
    text_seqs = [item['char_indices'] for item in batch]
    timing_seqs = [item['frame_word_labels'] for item in batch]

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
    video_dir = 'data/s1'
    align_dir = 'data/alignments/s1'
    dataset = LipReadingDataset(video_dir, align_dir)
    total_len = len(dataset)
    train_len = min(450, int(0.9 * total_len))
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
