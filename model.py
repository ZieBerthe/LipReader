import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
os.environ['GLOG_minloglevel'] = '3'  # Suppress mediapipe verbose logging
os.environ['GLOG_logtostderr'] = '0'  # Don't log to stderr
os.environ['GLOG_stderrthreshold'] = '3'  # Only show FATAL errors

import warnings
import logging
warnings.filterwarnings('ignore')
logging.getLogger('absl').setLevel(logging.ERROR)
logging.getLogger('mediapipe').setLevel(logging.ERROR)

# Suppress C++ level warnings from mediapipe
import contextlib

@contextlib.contextmanager
def suppress_stderr():
    """Context manager to suppress stderr at file descriptor level (works for C++ code)."""
    stderr_fd = sys.stderr.fileno()
    # Save a copy of the original stderr fd
    with open(os.devnull, 'w') as devnull:
        old_stderr = os.dup(stderr_fd)
        try:
            # Redirect stderr to devnull
            os.dup2(devnull.fileno(), stderr_fd)
            yield
        finally:
            # Restore stderr
            os.dup2(old_stderr, stderr_fd)
            os.close(old_stderr)

import glob
import soundfile as sf
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from matplotlib import pyplot as plt
import imageio

# Try to import tqdm, fall back to a dummy if not available
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Dummy tqdm for when it's not installed
    class tqdm:
        def __init__(self, iterable, **kwargs):
            self.iterable = iterable
            self.desc = kwargs.get('desc', '')
            self.total = len(iterable) if hasattr(iterable, '__len__') else '?'
            self.current = 0
            print(f"\n{self.desc}")
        
        def __iter__(self):
            for i, item in enumerate(self.iterable, 1):
                self.current = i
                yield item
        
        def set_postfix(self, ordered_dict):
            # Print progress every 20 batches
            if self.current % 20 == 0:
                postfix_str = ', '.join([f"{k}={v}" for k, v in ordered_dict.items()])
                print(f"  Batch {self.current}/{self.total}: {postfix_str}")
        
        @staticmethod
        def write(s):
            print(s)

from load_data import load_data, decode_indices, encode_chars, char_to_num, num_to_char, reconstruct_audio_from_stft, reconstruct_audio_from_magnitude_only
from audio_config import get_audio_config
from data_pipeline import LipReadingDataset, collate_fn, MAX_MOUTH_FRAMES, MAX_TEXT_LENGTH, MAX_AUDIO_FRAMES 
import torch.nn as nn
import torch.optim as optim


class LipReadingModel(nn.Module):
    def __init__(self, num_chars, audio_feature_dim):
        super(LipReadingModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv3d(1, 128, kernel_size= 3, padding=1),  # (B, 128, 75, 50, 100)
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            
            nn.Conv3d(128, 256, kernel_size=3, padding=1),  # (B, 256, 75, 25, 50)
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),  #don't pool over time dimension

            nn.Conv3d(256, 512, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),  # (B, 512, 75, 12, 25)
        )
        self.feature_size = 512 * 12 * 25
        # self.rnn = nn.LSTM(input_size=32 * 8 * 8, hidden_size=128, num_layers=2, batch_first=True)
        # self.fc = nn.Linear(128, num_chars)
        self.bilstm1 = nn.LSTM(input_size=self.feature_size, hidden_size=256, num_layers=1, batch_first=True, bidirectional=True)
        self.bilstm2 = nn.LSTM(input_size=512, hidden_size=256, num_layers=1, batch_first=True, bidirectional=True)
        self.text_head = nn.Linear(512, num_chars)
        self.audio_upsampler = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(), nn.Dropout(0.4))
        self.audio_lstm = nn.LSTM(input_size=512, hidden_size=256, num_layers=1, batch_first=True, bidirectional=True)

        self.audio_head = nn.Linear(512, audio_feature_dim)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
    def forward(self, x):
        batch_size, seq_len, H, W = x.size()
        x = x.unsqueeze(1)  # Add channel dimension: (B, 1, T, H, W)
        features = self.cnn(x)  # (B, 512, T, H', W')
        B, C, T, H_new, W_new = features.size()
        features = features.permute(0, 2, 1, 3, 4)  # (10, 75, 512, 12, 25)
        features = features.contiguous().view(B, T, -1)  # (10, 75, 153600)
        
        # Shared BiLSTM encoding
        lstm_out, _ = self.bilstm1(features)  # (10, 75, 512)
        lstm_out = self.dropout1(lstm_out)
        
        lstm_out, _ = self.bilstm2(lstm_out)  # (10, 75, 512)
        lstm_out = self.dropout2(lstm_out)
        
        # Dual prediction heads
        text_logits = self.text_head(lstm_out)  # (10, 75, num_chars)
        audio_features = self.audio_upsampler(lstm_out)  # (B, 75, 512)
        
        # Step 2: Upsample temporal dimension 75 -> 300 (4x)
        # Permute to (B, C, T) for interpolation
        audio_features = audio_features.transpose(1, 2)  # (B, 512, 75)
        audio_features_upsampled = nn.functional.interpolate(
            audio_features,
            size=300,  # Target temporal dimension
            mode='linear',
            align_corners=False
        )  # (B, 512, 300)
        audio_features_upsampled = audio_features_upsampled.transpose(1, 2)  # (B, 300, 512)
        
        # Step 3: Refine with audio-specific LSTM
        audio_refined, _ = self.audio_lstm(audio_features_upsampled)  # (B, 300, 512)
        
        # Step 4: Predict audio magnitude
        audio_pred = self.audio_head(audio_refined)  # (B, 300, 101)
        return text_logits, audio_pred
def train_step(model, batch, criterion_ctc, criterion_audio, optimizer):
    mouth_frames = batch['mouth_frames']  # (10, 75, 100, 200)
    text_targets = batch['char_indices']  # (10, 40)
    audio_targets = batch['audio_stft_magnitude']  # (10, 300, 101)
    
    mouth_lengths = batch['mouth_lengths']
    text_lengths = batch['text_lengths']
    audio_lengths = batch['audio_lengths']
    
    # Forward pass
    text_logits, audio_pred = model(mouth_frames)  # (10, 75, num_chars), (10, 300, 101)
    
    # Text loss (CTC)
    log_probs = nn.functional.log_softmax(text_logits, dim=2)
    log_probs = log_probs.transpose(0, 1)
    
    text_loss = criterion_ctc(
        log_probs,
        text_targets,
        mouth_lengths,
        text_lengths
    )
    
    # Audio loss (now direct, no interpolation!)
    mask = torch.arange(300, device=audio_targets.device).unsqueeze(0) < audio_lengths.unsqueeze(1)
    mask = mask.unsqueeze(2)  # (10, 300, 1)
    
    
    audio_loss = criterion_audio(audio_pred * mask, audio_targets * mask)
    
    
    # Combined loss
    #making the model focus on word for the moment
    total_loss = text_loss + 0.1 * audio_loss
    # total_loss = text_loss + 0.5 * audio_loss
    
    optimizer.zero_grad()
    total_loss.backward()
    
    # Gradient clipping to prevent exploding gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    
    return total_loss.item(), text_loss.item(), audio_loss.item()       

def train(model, train_loader, criterion_ctc, criterion_audio, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_loader:
            loss, text_loss, audio_loss = train_step(model, batch, criterion_ctc, criterion_audio, optimizer)
            total_loss += loss
            if num_epochs % 10 == 0:
                print(f"Batch loss: {loss:.4f} (Text: {text_loss:.4f}, Audio: {audio_loss:.4f})")
            # save model checkpoint at some point if needed
            if epoch % 20 == 0:
                torch.save(model.state_dict(), f"lipreading_model_epoch{epoch}.pth")
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

def ctc_greedy_decode(logits, blank_id=0):
    """Greedy CTC decoder: collapse repeats and remove blanks."""
    pred_ids = torch.argmax(logits, dim=-1)  # (B, T)
    results = []
    for seq in pred_ids:
        out = []
        prev = None
        for t in seq.tolist():
            if t != blank_id and t != prev:
                out.append(t)
            prev = t
        results.append(out)
    return results


def ids_to_text(ids, num_to_char_dict):
    return "".join(num_to_char_dict.get(i, "?") for i in ids)


class SilentDataLoader:
    """Wrapper for DataLoader that suppresses stderr during batch loading."""
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self._iterator = None
    
    def __iter__(self):
        """Iterate through batches while suppressing MediaPipe C++ warnings."""
        # Get the iterator from the dataloader
        iterator = iter(self.dataloader)
        
        while True:
            try:
                # Suppress stderr while fetching the next batch
                with suppress_stderr():
                    batch = next(iterator)
                yield batch
            except StopIteration:
                break
    
    def __len__(self):
        return len(self.dataloader)


def run_inference_examples(model, loader, audio_config, device, num_examples=3):
    """Run inference on a few samples and save outputs."""
    model.eval()
    print("\n" + "="*60)
    print("INFERENCE EXAMPLES")
    print("="*60)
    
    count = 0
    with torch.no_grad():
        for batch in loader:
            if count >= num_examples:
                break
                
            mouth = batch['mouth_frames'].to(device)
            text_targets = batch['char_indices']
            text_lengths = batch['text_lengths']

            # Run inference on the batch
            text_logits, audio_pred = model(mouth)  # (B, 75, num_chars), (B, 300, 101)
            decoded_batch = ctc_greedy_decode(text_logits, blank_id=0)
            
            # Check if model is outputting same prediction for all samples in batch
            unique_predictions = set(tuple(pred) for pred in decoded_batch)
            if len(unique_predictions) == 1 and len(decoded_batch) > 1:
                print(f"\n⚠️  WARNING: Model predicted identical text for all {len(decoded_batch)} samples in this batch!")
                print(f"   This indicates the model hasn't learned to differentiate inputs yet.\n")

            # Process each sample in the batch
            for i in range(mouth.size(0)):
                if count >= num_examples:
                    break

                # Ground truth
                gt_ids = text_targets[i][:text_lengths[i]].tolist()
                gt_text = ids_to_text(gt_ids, num_to_char)
                
                # Prediction
                pred_text = ids_to_text(decoded_batch[i], num_to_char)

                print(f"\n--- Example {count+1} ---")
                print(f"  Video         : {batch['video_paths'][i]}")
                print(f"  Ground truth  : '{gt_text}'")
                print(f"  Predicted text: '{pred_text}'")
                print(f"  Pred IDs      : {decoded_batch[i][:15]}")
                
                if pred_text == gt_text:
                    print(f"  ✓ EXACT MATCH!")
                else:
                    correct_chars = sum(1 for a, b in zip(pred_text, gt_text) if a == b)
                    print(f"  Char accuracy : {correct_chars}/{len(gt_text)} ({100*correct_chars/max(len(gt_text),1):.1f}%)")

                # Save predicted audio
                mag = audio_pred[i].cpu().numpy()
                wav = reconstruct_audio_from_magnitude_only(
                    mag,
                    magnitude_mean=audio_config['mag_mean'],
                    magnitude_std=audio_config['mag_std'],
                    sr=audio_config['sr'],
                    hop_length=audio_config['hop_length'],
                    win_length=audio_config['win_length'],
                    n_fft=audio_config['n_fft'],
                    n_iter=60,
                )
                out_path = f"generated_audio_example_{count+1}.wav"
                sf.write(out_path, wav, audio_config['sr'])
                print(f"  Saved audio   : {out_path}")
                
                count += 1


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')

    # ---- GPU Memory Management ----
    # Clear GPU cache from previous runs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        # Set environment variable to reduce fragmentation
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        print("✓ GPU cache cleared")

    # ---- Config ----
    print(f'{torch.cuda.is_available()}')
    print(f"PyTorch version: {torch.__version__}")
    print(f"{torch.version.cuda}")
    DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    VIDEO_DIR  = 'data/s1'
    ALIGN_DIR  = 'data/alignments/s1'
    BATCH_SIZE = 5# Reduced from 10 to avoid OOM errors
    NUM_EPOCHS = 20
    LR         = 1e-3
    AUDIO_DIM  = 101

    print(f"Device: {DEVICE}")
    audio_config = get_audio_config()
    print(f"Audio config: sr={audio_config['sr']}, n_fft={audio_config['n_fft']}, hop={audio_config['hop_length']}")

    # ---- Dataset ----
    print("Loading dataset (suppressing MediaPipe warnings)...")
    with suppress_stderr():
        dataset = LipReadingDataset(VIDEO_DIR, ALIGN_DIR, audio_config)
    print(f"✓ Dataset loaded")
    
    # # Use only half the data for faster testing
    # subset_size = len(full_dataset) // 2
    # dataset = torch.utils.data.Subset(full_dataset, range(subset_size))
    # print(f"Using {subset_size}/{len(full_dataset)} samples (half of dataset)")
    
    total   = len(dataset)
    train_n = max(1, int(0.9 * total))
    test_n  = total - train_n
    train_set, test_set = random_split(dataset, [train_n, test_n], generator=torch.Generator().manual_seed(4))
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    # Wrap loaders to suppress MediaPipe warnings
    train_loader = SilentDataLoader(train_loader)
    test_loader = SilentDataLoader(test_loader)
    
    print(f"Dataset: {total} samples  →  {train_n} train / {test_n} test")

    # ---- Model ----
    num_chars = len(char_to_num) + 1  # +1 for CTC blank (index 0)
    model     = LipReadingModel(num_chars=num_chars, audio_feature_dim=AUDIO_DIM).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    # ---- Load checkpoint if it exists ----
    checkpoint_path = "lipreading_final.pth"
    if os.path.exists(checkpoint_path):
        print(f"\n✓ Found checkpoint file: {checkpoint_path}")
        try:
            # Check file size first
            file_size = os.path.getsize(checkpoint_path)
            print(f"  Checkpoint file size: {file_size:,} bytes")
            
            if file_size > 0:
                state_dict = torch.load(checkpoint_path, map_location=DEVICE)
                model.load_state_dict(state_dict)
                print(f"✓ Successfully loaded model weights. Continuing training...\n")
            else:
                print(f"✗ Checkpoint file is empty (0 bytes). Training from scratch.\n")
        except Exception as e:
            print(f"✗ Error loading checkpoint: {e}")
            print(f"✗ Checkpoint may be corrupted. Training from scratch.\n")
    else:
        print(f"\n✗ No checkpoint found at {checkpoint_path}. Training from scratch.\n")

    optimizer     = optim.Adam(model.parameters(), lr=LR)
    criterion_ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    criterion_mse = nn.MSELoss()

    # ---- Training ----
    print("\n" + "="*60)
    print(f"TRAINING  ({NUM_EPOCHS} epochs)")
    print("="*60)

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        epoch_loss = epoch_text_loss = epoch_audio_loss = 0.0
        num_batches = len(train_loader)

        # Progress bar for batches
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}", 
                   ncols=120, leave=True)
        
        for batch_idx, batch in enumerate(pbar, 1):
            # Move tensors to device
            batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            loss, tloss, aloss = train_step(
                model, batch, criterion_ctc, criterion_mse, optimizer
            )
            epoch_loss       += loss
            epoch_text_loss  += tloss
            epoch_audio_loss += aloss

            # Update progress bar with current metrics
            pbar.set_postfix({
                'total': f'{loss:.4f}',
                'text': f'{tloss:.4f}',
                'audio': f'{aloss:.4f}'
            })

        avg       = epoch_loss       / num_batches
        avg_text  = epoch_text_loss  / num_batches
        avg_audio = epoch_audio_loss / num_batches
        
        tqdm.write(f"\n{'='*60}")
        tqdm.write(f"Epoch {epoch}/{NUM_EPOCHS} Summary:")
        tqdm.write(f"  Avg Total Loss : {avg:.4f}")
        tqdm.write(f"  Avg Text Loss  : {avg_text:.4f}")
        tqdm.write(f"  Avg Audio Loss : {avg_audio:.4f}")
        
        # Quick training sample check to see if model is learning
        if epoch % 2 == 0:
            model.eval()
            with torch.no_grad():
                sample_batch = next(iter(train_loader))
                sample_batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                               for k, v in sample_batch.items()}
                logits, _ = model(sample_batch['mouth_frames'])
                decoded = ctc_greedy_decode(logits, blank_id=0)
                pred_text = ids_to_text(decoded[0][:30], num_to_char)
                gt_text = ids_to_text(sample_batch['char_indices'][0][:sample_batch['text_lengths'][0]].cpu().tolist(), num_to_char)
                tqdm.write(f"  [Sample] GT: '{gt_text[:40]}' | Pred: '{pred_text[:40]}'")
            model.train()
        tqdm.write(f"{'='*60}\n")

        # Save checkpoint only on last epoch to avoid disk quota issues
        if epoch == NUM_EPOCHS:
            ckpt = f"lipreading_final.pth"
            try:
                torch.save(model.state_dict(), ckpt)
                print(f"   Final checkpoint saved → {ckpt}")
            except Exception as e:
                print(f"   Warning: Could not save checkpoint (disk quota?): {e}")

    # ---- Inference examples ----
    run_inference_examples(model, test_loader, audio_config, DEVICE, num_examples=3)