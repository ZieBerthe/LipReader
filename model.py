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


from load_data import decode_indices, encode_chars, char_to_num, num_to_char, VIDEO_FPS
from data_pipeline import LipReadingDataset, collate_fn, MAX_MOUTH_FRAMES, MAX_TEXT_LENGTH, MAX_WORDS
import torch.nn as nn
import torch.optim as optim


class LipReadingModel(nn.Module):
    """Lip reading model that predicts text (CTC) and per-frame word timing.

    Outputs:
        text_logits:   (B, T, num_chars)       — for CTC decoding
        timing_logits: (B, T, max_words + 1)   — per-frame word index classification
    """

    def __init__(self, num_chars, max_word_slots=MAX_WORDS):
        super(LipReadingModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv3d(1, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),

            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),

            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),  # (B, 512, T, 12, 25)
        )
        self.feature_size = 512 * 12 * 25

        self.bilstm1 = nn.LSTM(
            input_size=self.feature_size, hidden_size=256,
            num_layers=1, batch_first=True, bidirectional=True,
        )
        self.bilstm2 = nn.LSTM(
            input_size=512, hidden_size=256,
            num_layers=1, batch_first=True, bidirectional=True,
        )

        # Head 1: CTC text prediction
        self.text_head = nn.Linear(512, num_chars)

        # Head 2: Per-frame word index classification
        #   class 0 = silence,  1..max_word_slots = word positions
        self.timing_head = nn.Linear(512, max_word_slots + 1)

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, x):
        B, T, H, W = x.size()
        x = x.unsqueeze(1)  # (B, 1, T, H, W)

        features = self.cnn(x)  # (B, 512, T, H', W')
        _, C, T_out, H_new, W_new = features.size()
        features = features.permute(0, 2, 1, 3, 4)                     # (B, T, C, H', W')
        features = features.contiguous().view(B, T_out, -1)             # (B, T, feature_size)

        lstm_out, _ = self.bilstm1(features)   # (B, T, 512)
        lstm_out = self.dropout1(lstm_out)

        lstm_out, _ = self.bilstm2(lstm_out)   # (B, T, 512)
        lstm_out = self.dropout2(lstm_out)

        text_logits = self.text_head(lstm_out)      # (B, T, num_chars)
        timing_logits = self.timing_head(lstm_out)   # (B, T, max_words+1)

        return text_logits, timing_logits


# ---------------------------------------------------------------------------
#  Training
# ---------------------------------------------------------------------------

def train_step(model, batch, criterion_ctc, criterion_timing, optimizer):
    """Single training step: CTC loss for text + CrossEntropy for word timing."""
    mouth_frames = batch['mouth_frames']
    text_targets = batch['char_indices']
    timing_targets = batch['frame_word_labels']
    mouth_lengths = batch['mouth_lengths']
    text_lengths = batch['text_lengths']

    text_logits, timing_logits = model(mouth_frames)

    # ---- CTC loss for text ----
    log_probs = nn.functional.log_softmax(text_logits, dim=2).transpose(0, 1)
    text_loss = criterion_ctc(log_probs, text_targets, mouth_lengths, text_lengths)

    # ---- CrossEntropy loss for per-frame word timing ----
    # Only compute loss on valid (non-padded) frames
    B, T, C = timing_logits.shape
    mask = torch.arange(T, device=mouth_frames.device).unsqueeze(0) < mouth_lengths.unsqueeze(1)

    timing_loss = nn.functional.cross_entropy(
        timing_logits[mask],          # (valid_frames, num_classes)
        timing_targets[mask],         # (valid_frames,)
    )

    total_loss = text_loss + timing_loss

    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return total_loss.item(), text_loss.item(), timing_loss.item()


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


def extract_word_timings(frame_preds, fps=VIDEO_FPS):
    """Convert per-frame word-index predictions to timed segments.

    Args:
        frame_preds: (T,) tensor of word indices per frame (0 = silence)
        fps: Video frame rate

    Returns:
        List of (word_index, start_sec, end_sec) tuples (silence excluded)
    """
    segments = []
    current_word = frame_preds[0].item()
    start_frame = 0

    for i in range(1, len(frame_preds)):
        if frame_preds[i].item() != current_word:
            if current_word != 0:
                segments.append((current_word, start_frame / fps, i / fps))
            current_word = frame_preds[i].item()
            start_frame = i

    # Last segment
    if current_word != 0:
        segments.append((current_word, start_frame / fps, len(frame_preds) / fps))

    return segments


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

def run_inference_examples(model, loader, device, num_examples=3):
    """Run inference on a few samples and display text + timing predictions."""
    model.eval()
    print("\n" + "=" * 60)
    print("INFERENCE EXAMPLES")
    print("=" * 60)

    count = 0
    with torch.no_grad():
        for batch in loader:
            if count >= num_examples:
                break

            mouth = batch['mouth_frames'].to(device)
            text_targets = batch['char_indices']
            text_lengths = batch['text_lengths']
            gt_word_timings = batch['word_timings']

            text_logits, timing_logits = model(mouth)
            decoded_batch = ctc_greedy_decode(text_logits, blank_id=0)
            timing_preds = torch.argmax(timing_logits, dim=-1)  # (B, T)

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

                # Predicted word timings
                pred_timings = extract_word_timings(timing_preds[i])

                # Ground truth word timings
                gt_timings = gt_word_timings[i]

                print(f"\n--- Example {count + 1} ---")
                print(f"  Video         : {batch['video_paths'][i]}")
                print(f"  Ground truth  : '{gt_text}'")
                print(f"  Predicted text: '{pred_text}'")

                if pred_text == gt_text:
                    print(f"  ✓ EXACT MATCH!")
                else:
                    correct_chars = sum(1 for a, b in zip(pred_text, gt_text) if a == b)
                    print(f"  Char accuracy : {correct_chars}/{len(gt_text)} "
                          f"({100 * correct_chars / max(len(gt_text), 1):.1f}%)")

                print(f"  GT word timings:")
                for word, start, end in gt_timings:
                    print(f"    '{word}': {start:.3f}s - {end:.3f}s")

                print(f"  Predicted word timings:")
                for word_idx, start, end in pred_timings:
                    print(f"    word_{word_idx}: {start:.3f}s - {end:.3f}s")

                count += 1


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
    VIDEO_DIR  = 'data/s1'
    ALIGN_DIR  = 'data/alignments/s1'
    BATCH_SIZE = 15
    NUM_EPOCHS = 20
    LR         = 1e-3

    print(f"Device: {DEVICE}")

    # ---- Dataset ----
    print("Loading dataset (suppressing MediaPipe warnings)...")
    with suppress_stderr():
        dataset = LipReadingDataset(VIDEO_DIR, ALIGN_DIR)
    print(f"✓ Dataset loaded")

    total   = len(dataset)
    train_n = max(1, int(0.9 * total))
    test_n  = total - train_n
    train_set, test_set = random_split(
        dataset, [train_n, test_n],
        generator=torch.Generator().manual_seed(4),
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
    checkpoint_path = "lipreading_final.pth"
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
    criterion_ctc    = nn.CTCLoss(blank=0, zero_infinity=True)
    criterion_timing = nn.CrossEntropyLoss()

    # ---- Training ----
    print("\n" + "=" * 60)
    print(f"TRAINING  ({NUM_EPOCHS} epochs)")
    print("=" * 60)

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        epoch_loss = epoch_text_loss = epoch_timing_loss = 0.0
        num_batches = len(train_loader)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}",
                    ncols=120, leave=True)

        for batch_idx, batch in enumerate(pbar, 1):
            batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            loss, tloss, timing_loss = train_step(
                model, batch, criterion_ctc, criterion_timing, optimizer
            )
            epoch_loss        += loss
            epoch_text_loss   += tloss
            epoch_timing_loss += timing_loss

            pbar.set_postfix({
                'total': f'{loss:.4f}',
                'text': f'{tloss:.4f}',
                'timing': f'{timing_loss:.4f}',
            })

        avg        = epoch_loss        / num_batches
        avg_text   = epoch_text_loss   / num_batches
        avg_timing = epoch_timing_loss / num_batches

        tqdm.write(f"\n{'=' * 60}")
        tqdm.write(f"Epoch {epoch}/{NUM_EPOCHS} Summary:")
        tqdm.write(f"  Avg Total Loss  : {avg:.4f}")
        tqdm.write(f"  Avg Text Loss   : {avg_text:.4f}")
        tqdm.write(f"  Avg Timing Loss : {avg_timing:.4f}")

        # Quick training sample check
        if epoch % 2 == 0:
            model.eval()
            with torch.no_grad():
                sample_batch = next(iter(train_loader))
                sample_batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                                for k, v in sample_batch.items()}
                logits, timing_logits = model(sample_batch['mouth_frames'])

                # Text sample
                decoded = ctc_greedy_decode(logits, blank_id=0)
                pred_text = ids_to_text(decoded[0][:30], num_to_char)
                gt_text = ids_to_text(
                    sample_batch['char_indices'][0][:sample_batch['text_lengths'][0]].cpu().tolist(),
                    num_to_char,
                )

                # Timing accuracy
                timing_preds = torch.argmax(timing_logits, dim=-1)
                timing_gt = sample_batch['frame_word_labels']
                n_frames = sample_batch['mouth_lengths'][0].item()
                n_correct = (timing_preds[0, :n_frames] == timing_gt[0, :n_frames]).sum().item()

                tqdm.write(f"  [Sample] GT: '{gt_text[:40]}' | Pred: '{pred_text[:40]}'")
                tqdm.write(f"  [Sample] Timing accuracy: {n_correct}/{n_frames} "
                           f"({100 * n_correct / max(n_frames, 1):.1f}%)")
            model.train()
        tqdm.write(f"{'=' * 60}\n")

        # Save checkpoint on last epoch
        if epoch == NUM_EPOCHS:
            ckpt = "lipreading_final.pth"
            try:
                torch.save(model.state_dict(), ckpt)
                print(f"   Final checkpoint saved → {ckpt}")
            except Exception as e:
                print(f"   Warning: Could not save checkpoint: {e}")

    # ---- Inference examples ----
    run_inference_examples(model, test_loader, DEVICE, num_examples=3)
