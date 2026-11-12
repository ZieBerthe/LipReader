"""Train the CTC lip reader on cached GRID mouth crops.

    pixi run train-lipreader
Resumes from checkpoints/lipreading_best.pth if it exists.
"""
import os
import warnings
import logging

warnings.filterwarnings('ignore')
logging.getLogger('absl').setLevel(logging.ERROR)
logging.getLogger('mediapipe').setLevel(logging.ERROR)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from lipreader.device import get_device
from lipreader.paths import DATA_ROOT, CHECKPOINT_DIR, LIPREADING_CKPT
from lipreader.utils import suppress_stderr
from lipreader.data.vocab import NUM_CLASSES, num_to_char
from lipreader.data.dataset import LipReadingDataset, collate_fn
from lipreader.lipreading.model import LipReadingModel
from lipreader.lipreading.decoding import ctc_greedy_decode, ids_to_text, compute_word_accuracy


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
    # CTC isn't implemented on MPS, so the loss runs on CPU; gradients still
    # flow back to the model on the GPU (the loss itself is cheap).
    log_probs = nn.functional.log_softmax(text_logits, dim=2).transpose(0, 1)
    loss = criterion_ctc(log_probs.cpu(), text_targets.cpu(), mouth_lengths.cpu(), text_lengths.cpu())

    optimizer.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return loss.item(), grad_norm.item()


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
        from lipreader.data.video import load_video  # lazy: imports MediaPipe
        mouth_frames = load_video(video_path)  # (T, H, W)
        mouth_tensor = mouth_frames.unsqueeze(0).to(device)  # (1, T, H, W)

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

def main():
    # ---- Config ----
    DEVICE     = get_device()
    print(f"PyTorch version: {torch.__version__}")
    if DEVICE.type == 'mps':
        torch.mps.empty_cache()
    elif DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
    # 16 GB unified memory on the Mac: batch 8 already swaps (~15x slower per sample)
    BATCH_SIZE = 4 if DEVICE.type == 'mps' else 16
    NUM_EPOCHS = 150 
    LR         = 1e-3  # Slightly higher for faster initial learning

    print(f"Device: {DEVICE}")

    # ---- Dataset ----
    print("Loading dataset (suppressing MediaPipe warnings)...")
    with suppress_stderr():
        dataset = LipReadingDataset(corpus_root=DATA_ROOT, use_cache=True)
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
    model     = LipReadingModel(num_chars=NUM_CLASSES).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    # ---- Load checkpoint if it exists ----
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_path = LIPREADING_CKPT
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

    # ---- Training ----
    print("\n" + "=" * 60)
    print(f"TRAINING  ({NUM_EPOCHS} epochs)")
    print("=" * 60)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        epoch_grad_norm = 0.0
        num_batches = len(train_loader)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}",
                    ncols=120, leave=True)

        for batch_idx, batch in enumerate(pbar, 1):
            batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            try:
                loss, grad_norm = train_step(
                    model, batch, criterion_ctc, optimizer, epoch
                )
            except Exception as e:
                tqdm.write(f"  [SKIP] Batch {batch_idx} error: {e}")
                continue

            epoch_loss      += loss
            epoch_grad_norm += grad_norm

            pbar.set_postfix({
                'loss': f'{loss:.4f}',
                'grad': f'{grad_norm:.3f}',
            })

        avg_loss = epoch_loss     / max(num_batches, 1)
        avg_grad = epoch_grad_norm / max(num_batches, 1)

        tqdm.write(f"\n{'=' * 60}")
        tqdm.write(f"Epoch {epoch}/{NUM_EPOCHS} Summary:")
        tqdm.write(f"  Avg Loss        : {avg_loss:.4f}")
        tqdm.write(f"  Avg Grad Norm   : {avg_grad:.4f}")

        # Validation loss
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for val_batch in test_loader:
                val_batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                            for k, v in val_batch.items()}
                try:
                    v_text_logits = model(val_batch['mouth_frames'])

                    v_log_probs = nn.functional.log_softmax(v_text_logits, dim=2).transpose(0, 1)
                    v_loss = criterion_ctc(  # CPU: CTC isn't implemented on MPS
                        v_log_probs.cpu(), val_batch['char_indices'].cpu(),
                        val_batch['mouth_lengths'].cpu(), val_batch['text_lengths'].cpu()
                    ).item()

                    val_loss += v_loss
                    val_batches += 1
                except Exception:
                    continue

        val_loss /= max(val_batches, 1)

        tqdm.write(f"  Val Loss        : {val_loss:.4f}")

        # Learning rate scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        tqdm.write(f"  Learning Rate   : {current_lr:.6f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), LIPREADING_CKPT)
            tqdm.write(f"  ✓ New best model saved!")
        else:
            patience_counter += 1
            if patience_counter >= 10:
                tqdm.write(f"\n  Early stopping triggered (no improvement for 10 epochs)")
                tqdm.write(f"  Best validation loss: {best_val_loss:.4f}")
                break

        # Quick training sample check
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                sample_batch = next(iter(train_loader))
                sample_batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                                for k, v in sample_batch.items()}
                logits = model(sample_batch['mouth_frames'])

                # Text sample
                decoded = ctc_greedy_decode(logits, blank_id=0)
                pred_text = ids_to_text(decoded[0][:30], num_to_char)
                gt_text = ids_to_text(
                    sample_batch['char_indices'][0][:sample_batch['text_lengths'][0]].cpu().tolist(),
                    num_to_char,
                )

                # Word accuracy on sample
                word_acc, exact_match = compute_word_accuracy(pred_text, gt_text)

                tqdm.write(f"  [Sample] GT: '{gt_text[:40]}' | Pred: '{pred_text[:40]}'")
                tqdm.write(f"  [Sample] Word accuracy: {100 * word_acc:.1f}%")
            model.train()
        tqdm.write(f"{'=' * 60}\n")

        # Save periodic checkpoint
        if epoch % 10 == 0 or epoch == NUM_EPOCHS:
            ckpt = os.path.join(CHECKPOINT_DIR, f"lipreading_epoch_{epoch}.pth")
            try:
                torch.save(model.state_dict(), ckpt)
                tqdm.write(f"  Checkpoint saved → {ckpt}")
            except Exception as e:
                tqdm.write(f"  Warning: Could not save checkpoint: {e}")

    # Load best model for inference
    print(f"\nLoading best model for inference...")
    if os.path.exists(LIPREADING_CKPT):
        model.load_state_dict(torch.load(LIPREADING_CKPT, map_location=DEVICE))
        print(f"✓ Loaded best model (val loss: {best_val_loss:.4f})")

    # ---- Inference examples ----
    run_inference_examples(model, test_loader, DEVICE, num_examples=3)
    # run_single_test_video(model, os.path.join(DATA_ROOT, "testons.mpg"), DEVICE)


if __name__ == "__main__":
    main()
