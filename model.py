import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'
os.environ['GLOG_logtostderr'] = '0'
os.environ['GLOG_stderrthreshold'] = '3'

import warnings
import logging
warnings.filterwarnings('ignore')
logging.getLogger('absl').setLevel(logging.ERROR)
logging.getLogger('mediapipe').setLevel(logging.ERROR)

import contextlib


@contextlib.contextmanager
def suppress_stderr():
    """Context manager to suppress stderr at file descriptor level (works for C++ code)."""
    stderr_fd = sys.stderr.fileno()
    with open(os.devnull, 'w') as devnull:
        old_stderr = os.dup(stderr_fd)
        try:
            os.dup2(devnull.fileno(), stderr_fd)
            yield
        finally:
            os.dup2(old_stderr, stderr_fd)
            os.close(old_stderr)


import glob
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import  tqdm


from load_data import load_video, decode_indices, encode_chars, char_to_num, num_to_char, VIDEO_FPS
from data_pipeline import LipReadingDataset, collate_fn, MAX_MOUTH_FRAMES, MAX_TEXT_LENGTH, MAX_WORDS
import torch.nn as nn
import torch.optim as optim


class LipReadingModel(nn.Module):
    """Lip reading model that predicts text (CTC only).

    Outputs:
        text_logits: (B, T, num_chars) — for CTC decoding
    """

    def __init__(self, num_chars, max_word_slots=MAX_WORDS):
        super(LipReadingModel, self).__init__()
        # CNN sized for ~21k samples (31 speakers)
        self.cnn = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),

            nn.Conv3d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm3d(96),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),  # (B, 96, T, 12, 25)
        )
        # Collapse spatial dims: (B, 96, T, 12, 25) → (B, 96, T, 2, 2)
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 2, 2))
        self.feature_size = 96 * 2 * 2  # = 384

        # 2-layer BiLSTM
        self.bilstm = nn.LSTM(
            input_size=self.feature_size, hidden_size=128,
            num_layers=2, batch_first=True, bidirectional=True,
            dropout=0.3,
        )

        # CTC text prediction head
        self.text_head = nn.Linear(256, num_chars)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        B, T, H, W = x.size()
        x = x.unsqueeze(1)  # (B, 1, T, H, W)

        features = self.cnn(x)  # (B, 96, T, 12, 25)
        features = self.spatial_pool(features)                          # (B, 96, T, 2, 2)
        _, C, T_out, H_new, W_new = features.size()
        features = features.permute(0, 2, 1, 3, 4)                     # (B, T, C, H', W')
        features = features.contiguous().view(B, T_out, -1)             # (B, T, 384)

        lstm_out, _ = self.bilstm(features)   # (B, T, 256)
        lstm_out = self.dropout(lstm_out)

        text_logits = self.text_head(lstm_out)      # (B, T, num_chars)

        return text_logits


# ---------------------------------------------------------------------------
#  Training
# ---------------------------------------------------------------------------

def train_step(model, batch, criterion_ctc, optimizer, epoch=1):
    """Single training step: CTC loss for text."""
    mouth_frames = batch['mouth_frames']
    text_targets = batch['char_indices']
    mouth_lengths = batch['mouth_lengths']
    text_lengths = batch['text_lengths']

    text_logits = model(mouth_frames)

    # ---- CTC loss for text ----
    log_probs = nn.functional.log_softmax(text_logits, dim=2).transpose(0, 1)
    loss = criterion_ctc(log_probs, text_targets, mouth_lengths, text_lengths)

    optimizer.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return loss.item(), grad_norm.item()


# ---------------------------------------------------------------------------
#  Decoding helpers
# ---------------------------------------------------------------------------

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


def compute_word_accuracy(pred_text, gt_text):
    """Compute word-level accuracy (forgiving to single char mismatches).
    
    Args:
        pred_text: Predicted text string
        gt_text: Ground truth text string
    
    Returns:
        word_accuracy: Fraction of words predicted correctly
        exact_match: Boolean, True if texts match exactly
    """
    pred_words = pred_text.lower().split()
    gt_words = gt_text.lower().split()
    
    if len(gt_words) == 0:
        return 1.0 if len(pred_words) == 0 else 0.0, len(pred_words) == 0
    
    correct = sum(1 for p, g in zip(pred_words, gt_words) if p == g)
    word_accuracy = correct / len(gt_words)
    exact_match = pred_text.lower() == gt_text.lower()
    
    return word_accuracy, exact_match


# ---------------------------------------------------------------------------
#  Data loader wrapper
# ---------------------------------------------------------------------------

class SilentDataLoader:
    """Wrapper for DataLoader that suppresses stderr during batch loading."""

    def __init__(self, dataloader):
        self.dataloader = dataloader

    def __iter__(self):
        iterator = iter(self.dataloader)
        while True:
            try:
                with suppress_stderr():
                    batch = next(iterator)
                yield batch
            except StopIteration:
                break

    def __len__(self):
        return len(self.dataloader)


# ---------------------------------------------------------------------------
#  Inference
# ---------------------------------------------------------------------------
def run_single_test_video(model, video_path, device):
    """Run inference on a single video and print predicted text."""
    model.eval()
    with torch.no_grad():
        # Load and preprocess video
        mouth_frames = load_video(video_path)  # (T, H, W)
        mouth_tensor = torch.tensor(mouth_frames).unsqueeze(0).to(device)  # (1, T, H, W)

        # Run model
        text_logits = model(mouth_tensor)
        decoded = ctc_greedy_decode(text_logits, blank_id=0)[0]
        pred_text = ids_to_text(decoded, num_to_char)

        print(f"\nVideo: {video_path}")
        print(f"Predicted text: '{pred_text}'")
def run_inference_examples(model, loader, device, num_examples=3):
    """Run inference on a few samples and display text predictions with word-level accuracy."""
    model.eval()
    print("\n" + "=" * 60)
    print("INFERENCE EXAMPLES")
    print("=" * 60)

    count = 0
    total_word_acc = 0.0
    total_exact_match = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in loader:
            if count >= num_examples:
                break

            mouth = batch['mouth_frames'].to(device)
            text_targets = batch['char_indices']
            text_lengths = batch['text_lengths']

            text_logits = model(mouth)
            print(f"text_logits shape: {text_logits.shape}")
            decoded_batch = ctc_greedy_decode(text_logits, blank_id=0)

            # Check for collapsed predictions
            unique_predictions = set(tuple(pred) for pred in decoded_batch)
            if len(unique_predictions) == 1 and len(decoded_batch) > 1:
                print(f"\n⚠️  WARNING: Model predicted identical text for all "
                      f"{len(decoded_batch)} samples in this batch!")

            for i in range(mouth.size(0)):
                if count >= num_examples:
                    break

                # Ground truth text
                gt_ids = text_targets[i][:text_lengths[i]].tolist()
                gt_text = ids_to_text(gt_ids, num_to_char)

                # Predicted text
                pred_text = ids_to_text(decoded_batch[i], num_to_char)

                # Word accuracy
                word_acc, exact_match = compute_word_accuracy(pred_text, gt_text)
                total_word_acc += word_acc
                total_exact_match += int(exact_match)
                total_samples += 1

                print(f"\n--- Example {count + 1} ---")
                print(f"  Video         : {batch['video_paths'][i]}")
                print(f"  Ground truth  : '{gt_text}'")
                print(f"  Predicted text: '{pred_text}'")

                if exact_match:
                    print(f"  ✓ EXACT MATCH!")
                else:
                    print(f"  Word accuracy : {100 * word_acc:.1f}%")

                count += 1
    
    if total_samples > 0:
        avg_word_acc = total_word_acc / total_samples
        exact_match_rate = total_exact_match / total_samples
        print(f"\n" + "=" * 60)
        print(f"Average word accuracy  : {100 * avg_word_acc:.1f}%")
        print(f"Exact match rate       : {100 * exact_match_rate:.1f}%")
        print(f"=" * 60)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')

    # ---- GPU Memory Management ----
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        print("✓ GPU cache cleared")

    # ---- Config ----
    print(f'CUDA available: {torch.cuda.is_available()}')
    print(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")

    DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CORPUS_ROOT = '/Data/grid_corpus'
    BATCH_SIZE = 16
    NUM_EPOCHS = 150 
    LR         = 1e-3  # Slightly higher for faster initial learning

    print(f"Device: {DEVICE}")

    # ---- Dataset ----
    print("Loading dataset (suppressing MediaPipe warnings)...")
    with suppress_stderr():
        dataset = LipReadingDataset(corpus_root=CORPUS_ROOT, use_cache=True)
    print(f"✓ Dataset loaded")

    total   = len(dataset)
    train_n = int(0.7 * total)
    test_n  = total - train_n
    train_set, test_set = random_split(
        dataset, [train_n, test_n],
        generator=torch.Generator().manual_seed(67),
    )
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # Wrap loaders to suppress MediaPipe warnings
    train_loader = SilentDataLoader(train_loader)
    test_loader  = SilentDataLoader(test_loader)

    print(f"Dataset: {total} samples  →  {train_n} train / {test_n} test")

    # ---- Model ----
    num_chars = len(char_to_num) + 1  # +1 for CTC blank (index 0)
    model     = LipReadingModel(num_chars=num_chars).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    # ---- Load checkpoint if it exists ----
    checkpoint_path = "lipreading_best.pth"
    if os.path.exists(checkpoint_path):
        print(f"\n✓ Found checkpoint file: {checkpoint_path}")
        try:
            file_size = os.path.getsize(checkpoint_path)
            print(f"  Checkpoint file size: {file_size:,} bytes")
            if file_size > 0:
                state_dict = torch.load(checkpoint_path, map_location=DEVICE)
                model.load_state_dict(state_dict, strict=False)
                print(f"✓ Loaded model weights (strict=False for architecture changes).\n")
            else:
                print(f"✗ Checkpoint file is empty. Training from scratch.\n")
        except Exception as e:
            print(f"✗ Error loading checkpoint: {e}")
            print(f"  Training from scratch.\n")
    else:
        print(f"\n✗ No checkpoint found. Training from scratch.\n")

    optimizer        = optim.Adam(model.parameters(), lr=LR)
    scheduler        = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )
    criterion_ctc    = nn.CTCLoss(blank=0, zero_infinity=True)

    # # ---- Training ----
    # print("\n" + "=" * 60)
    # print(f"TRAINING  ({NUM_EPOCHS} epochs)")
    # print("=" * 60)

    # best_val_loss = float('inf')
    # patience_counter = 0
    
    # for epoch in range(1, NUM_EPOCHS + 1):
    #     model.train()
    #     epoch_loss = 0.0
    #     epoch_grad_norm = 0.0
    #     num_batches = len(train_loader)

    #     pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}",
    #                 ncols=120, leave=True)

    #     for batch_idx, batch in enumerate(pbar, 1):
    #         batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
    #                  for k, v in batch.items()}

    #         try:
    #             loss, grad_norm = train_step(
    #                 model, batch, criterion_ctc, optimizer, epoch
    #             )
    #         except Exception as e:
    #             tqdm.write(f"  [SKIP] Batch {batch_idx} error: {e}")
    #             continue

    #         epoch_loss      += loss
    #         epoch_grad_norm += grad_norm

    #         pbar.set_postfix({
    #             'loss': f'{loss:.4f}',
    #             'grad': f'{grad_norm:.3f}',
    #         })

    #     avg_loss = epoch_loss     / max(num_batches, 1)
    #     avg_grad = epoch_grad_norm / max(num_batches, 1)

    #     tqdm.write(f"\n{'=' * 60}")
    #     tqdm.write(f"Epoch {epoch}/{NUM_EPOCHS} Summary:")
    #     tqdm.write(f"  Avg Loss        : {avg_loss:.4f}")
    #     tqdm.write(f"  Avg Grad Norm   : {avg_grad:.4f}")

    #     # Validation loss
    #     model.eval()
    #     val_loss = 0.0
    #     val_batches = 0
    #     with torch.no_grad():
    #         for val_batch in test_loader:
    #             val_batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
    #                         for k, v in val_batch.items()}
    #             try:
    #                 v_text_logits = model(val_batch['mouth_frames'])
                    
    #                 v_log_probs = nn.functional.log_softmax(v_text_logits, dim=2).transpose(0, 1)
    #                 v_loss = criterion_ctc(
    #                     v_log_probs, val_batch['char_indices'], 
    #                     val_batch['mouth_lengths'], val_batch['text_lengths']
    #                 ).item()
                    
    #                 val_loss += v_loss
    #                 val_batches += 1
    #             except Exception:
    #                 continue
        
    #     val_loss /= max(val_batches, 1)
        
    #     tqdm.write(f"  Val Loss        : {val_loss:.4f}")
        
    #     # Learning rate scheduling
    #     scheduler.step(val_loss)
    #     current_lr = optimizer.param_groups[0]['lr']
    #     tqdm.write(f"  Learning Rate   : {current_lr:.6f}")
        
    #     # Early stopping
    #     if val_loss < best_val_loss:
    #         best_val_loss = val_loss
    #         patience_counter = 0
    #         # Save best model
    #         torch.save(model.state_dict(), "lipreading_best.pth")
    #         tqdm.write(f"  ✓ New best model saved!")
    #     else:
    #         patience_counter += 1
    #         if patience_counter >= 10:
    #             tqdm.write(f"\n  Early stopping triggered (no improvement for 10 epochs)")
    #             tqdm.write(f"  Best validation loss: {best_val_loss:.4f}")
    #             break

    #     # Quick training sample check
    #     if epoch % 5 == 0:
    #         model.eval()
    #         with torch.no_grad():
    #             sample_batch = next(iter(train_loader))
    #             sample_batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
    #                             for k, v in sample_batch.items()}
    #             logits = model(sample_batch['mouth_frames'])

    #             # Text sample
    #             decoded = ctc_greedy_decode(logits, blank_id=0)
    #             pred_text = ids_to_text(decoded[0][:30], num_to_char)
    #             gt_text = ids_to_text(
    #                 sample_batch['char_indices'][0][:sample_batch['text_lengths'][0]].cpu().tolist(),
    #                 num_to_char,
    #             )

    #             # Word accuracy on sample
    #             word_acc, exact_match = compute_word_accuracy(pred_text, gt_text)

    #             tqdm.write(f"  [Sample] GT: '{gt_text[:40]}' | Pred: '{pred_text[:40]}'")
    #             tqdm.write(f"  [Sample] Word accuracy: {100 * word_acc:.1f}%")
    #         model.train()
    #     tqdm.write(f"{'=' * 60}\n")

    #     # Save periodic checkpoint
    #     if epoch % 10 == 0 or epoch == NUM_EPOCHS:
    #         ckpt = f"lipreading_epoch_{epoch}.pth"
    #         try:
    #             torch.save(model.state_dict(), ckpt)
    #             tqdm.write(f"  Checkpoint saved → {ckpt}")
    #         except Exception as e:
    #             tqdm.write(f"  Warning: Could not save checkpoint: {e}")
    
    # # Load best model for inference
    # print(f"\nLoading best model for inference...")
    # if os.path.exists("lipreading_best.pth"):
    #     model.load_state_dict(torch.load("lipreading_best.pth", map_location=DEVICE))
    #     print(f"✓ Loaded best model (val loss: {best_val_loss:.4f})")

    # ---- Inference examples ----
    run_inference_examples(model, test_loader, DEVICE, num_examples=3)
    # run_single_test_video(model, "data/testons.mpg", DEVICE)
