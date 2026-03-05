import os
import glob
import soundfile as sf
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from matplotlib import pyplot as plt
import imageio
from load_data import load_data, decode_indices, encode_chars, char_to_num, num_to_char, reconstruct_audio_from_stft, reconstruct_audio_from_magnitude_only
from audio_config import get_audio_config

# Custom Dataset for LipReading
class LipReadingDataset(Dataset):
    def __init__(self, video_dir, align_dir, audio_config=None):
        self.video_paths = sorted(glob.glob(os.path.join(video_dir, '*.mpg')))
        self.align_paths = [
            os.path.join(align_dir, os.path.splitext(os.path.basename(v))[0] + '.align')
            for v in self.video_paths
        ]
        self.audio_config = audio_config if audio_config is not None else get_audio_config()

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        align_path = self.align_paths[idx]
        mouth_frames, audio_stft_mag, char_indices = load_data(
            video_path, align_path, self.audio_config
        )
        return {
            'mouth_frames': mouth_frames,  # (T, H, W)
            'audio_stft_magnitude': audio_stft_mag,  # (A, F) - STFT magnitude
            'char_indices': torch.tensor(char_indices, dtype=torch.long),
            'video_path': video_path,
        }

# Global maximum lengths - ensures all batches have same output size
MAX_MOUTH_FRAMES = 75  # Maximum number of video frames
MAX_TEXT_LENGTH = 40   # Maximum text sequence length
MAX_AUDIO_FRAMES = 300 # Maximum audio STFT frames

# Collate function for batching and padding
def collate_fn(batch):
    mouth_seqs = [item['mouth_frames'] for item in batch]
    text_seqs = [item['char_indices'] for item in batch]
    audio_mag_seqs = [item['audio_stft_magnitude'] for item in batch]
    
    # Get actual lengths before padding
    mouth_lens = [seq.shape[0] for seq in mouth_seqs]
    text_lens = [seq.shape[0] for seq in text_seqs]
    audio_lens = [seq.shape[0] for seq in audio_mag_seqs]
    
    # Pad to GLOBAL max length (not batch-specific max)
    # This ensures all batches have the same tensor size
    mouth_padded = torch.stack([
        torch.nn.functional.pad(seq, (0, 0, 0, 0, 0, MAX_MOUTH_FRAMES - seq.shape[0]))
        for seq in mouth_seqs
    ])
    
    text_padded = torch.stack([
        torch.nn.functional.pad(seq, (0, MAX_TEXT_LENGTH - seq.shape[0]))
        for seq in text_seqs
    ])
    
    audio_mag_padded = torch.stack([
        torch.nn.functional.pad(seq, (0, 0, 0, MAX_AUDIO_FRAMES - seq.shape[0]))
        for seq in audio_mag_seqs
    ])
    
    return {
        'mouth_frames': mouth_padded,  # (B, MAX_MOUTH_FRAMES, H, W)
        'mouth_lengths': torch.tensor(mouth_lens),
        'char_indices': text_padded,  # (B, MAX_TEXT_LENGTH)
        'text_lengths': torch.tensor(text_lens),
        'audio_stft_magnitude': audio_mag_padded,  # (B, MAX_AUDIO_FRAMES, F)
        'audio_lengths': torch.tensor(audio_lens),
        'video_paths': [item['video_path'] for item in batch]
    }

if __name__ == "__main__":
    # Load global audio config
    audio_config = get_audio_config()
    print(f"Audio config: {audio_config}\n")
    
    video_dir = 'data/s1'
    align_dir = 'data/alignments/s1'
    dataset = LipReadingDataset(video_dir, align_dir, audio_config)
    total_len = len(dataset)
    train_len = min(450, int(0.9 * total_len))
    test_len = total_len - train_len
    train_set, test_set = random_split(dataset, [train_len, test_len], generator=torch.Generator().manual_seed(42))
    batch_size = 10
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Get one batch (like .as_numpy_iterator().next())
    batch = next(iter(train_loader))
    frames = batch['mouth_frames'].numpy()  # (B, T, H, W)
    alignments = batch['char_indices'].numpy()  # (B, L)

    print('Batch mouth frames shape:', frames.shape)
    print('Batch alignments shape:', alignments.shape)
    print('Batch video paths:', batch['video_paths'])

    # Get first video in batch, first frame
    first_video_frames = frames[0]  # (T, H, W)
    print('First video shape:', first_video_frames.shape)

    # Save GIF of first video
    gif_path = './animation.gif'
    # Normalize for imageio (uint8)
    gif_frames = [((f - f.min()) / (f.max() - f.min()) * 255).astype(np.uint8) for f in first_video_frames]
    imageio.mimsave(gif_path, gif_frames, fps=10)
    print(f'Saved GIF to {gif_path}')
    
    # Print the audio STFT spectrogram shapes
    audio_stft_mag = batch['audio_stft_magnitude'].numpy()  # (B, A, F)
    print('Batch audio STFT magnitude shape:', audio_stft_mag.shape)
    
    # Restore the original text from the char indices
    for i in range(batch_size):
        char_indices = batch['char_indices'][i].numpy()
        text = ''.join([num_to_char[idx] for idx in char_indices if idx != 0])  # Skip padding
        print(f'Video: {batch["video_paths"][i]}')
        print(f'Text: {text}')
    
    # Reconstruct the audio waveform from STFT magnitude (Griffin-Lim)
    stft_mag = batch['audio_stft_magnitude'][0].numpy()  # (A, F)
    
    reconstructed_audio = reconstruct_audio_from_magnitude_only(
        stft_mag,
        magnitude_mean=audio_config['mag_mean'],
        magnitude_std=audio_config['mag_std'],
        sr=audio_config['sr'],
        hop_length=audio_config['hop_length'],
        n_fft=audio_config['n_fft'],
        win_length=audio_config['win_length'],
        n_iter=60
    )
    print('Reconstructed audio shape:', reconstructed_audio.shape)
    sf.write('reconstructed_audio_pipeline.wav', reconstructed_audio, audio_config['sr'])
    print('Saved reconstructed_audio_pipeline.wav (uses Griffin-Lim)')
    
    # Show a specific frame (e.g., frame 35)
    frame_idx = 13 if first_video_frames.shape[0] > 35 else 0
    plt.imshow(first_video_frames[frame_idx], cmap='gray')
    plt.title(f'Frame {frame_idx} of first video')
    plt.show()
    # print data type, shape, batch size, feature dimensions, labels, and label shapes
    print('Data type of mouth frames:', batch['mouth_frames'].dtype)
    print('Shape of mouth frames:', batch['mouth_frames'].shape)
    print('Batch size:', batch['mouth_frames'].shape[0])
    print('Feature dimensions (H, W):', batch['mouth_frames'].shape[2], batch['mouth_frames'].shape[3])
    print('Data type of char indices:', batch['char_indices'].dtype)
    print('Shape of char indices:', batch['char_indices'].shape)
    print('Data type of audio STFT magnitude:', batch['audio_stft_magnitude'].dtype)
    print('Shape of audio STFT magnitude:', batch['audio_stft_magnitude'].shape)
    print('Audio feature dimensions (A, F):', batch['audio_stft_magnitude'].shape[1], batch['audio_stft_magnitude'].shape[2])
    