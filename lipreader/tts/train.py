"""Train the video-style TTS on the manifest produced by `pixi run tts-prep`.

Loss: L1(mel_pred, mel_gt) + L1(mel_postnet, mel_gt)

Usage:
    pixi run tts-train
    pixi run tts-train --epochs 40 --batch_size 8
"""
import os
import sys
import json
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

from lipreader.device import get_device
from lipreader.paths import TTS_PREP_ROOT, TTS_CKPT_DIR
from lipreader.data.vocab import encode_chars
from lipreader.tts.model import VideoStyleTTS

# Training hyper-parameters
INITIAL_LR   = 1e-6

# ---------------------------------------------------------------------------
#  Dataset
# ---------------------------------------------------------------------------

class TTSDataset(Dataset):
    """Load (text, frames, mel) triples from the manifest produced by prep.py."""

    def __init__(self, manifest_path: str):
        self.samples = []
        with open(manifest_path) as f:
            for line in f:
                self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        mel    = np.load(s["mel_path"]).astype(np.float32)       # (T_mel, n_mels)
        frames = np.load(s["frames_path"]).astype(np.float32)    # (T_vid, H, W)
        text   = s["text"]
        char_ids = encode_chars(list(text))                      # list[int]
        return dict(
            mel=torch.from_numpy(mel),
            frames=torch.from_numpy(frames),
            text_ids=torch.tensor(char_ids, dtype=torch.long),
            text=text,
            vid_id=s["id"],
        )


def tts_collate_fn(batch):
    """Pad variable-length sequences to the maximum in the batch."""
    mels       = [b["mel"]      for b in batch]
    frames_lst = [b["frames"]   for b in batch]
    texts      = [b["text_ids"] for b in batch]

    mel_lens   = torch.tensor([m.shape[0] for m in mels])
    frame_lens = torch.tensor([f.shape[0] for f in frames_lst])
    text_lens  = torch.tensor([t.shape[0] for t in texts])

    T_mel_max  = mel_lens.max().item()
    T_frame_max = frame_lens.max().item()
    T_text_max = text_lens.max().item()

    n_mels = mels[0].shape[1]
    H, W   = frames_lst[0].shape[1], frames_lst[0].shape[2]

    mel_padded   = torch.zeros(len(batch), T_mel_max, n_mels)
    frame_padded = torch.zeros(len(batch), T_frame_max, H, W)
    text_padded  = torch.zeros(len(batch), T_text_max, dtype=torch.long)

    for i in range(len(batch)):
        mel_padded[i,   :mel_lens[i]]   = mels[i]
        frame_padded[i, :frame_lens[i]] = frames_lst[i]
        text_padded[i,  :text_lens[i]]  = texts[i]

    return dict(
        mel=mel_padded,              # (B, T_mel, n_mels)
        mel_lengths=mel_lens,        # (B,)
        frames=frame_padded,         # (B, T_vid, H, W)
        frame_lengths=frame_lens,    # (B,)
        text_ids=text_padded,        # (B, T_text)
        text_lengths=text_lens,      # (B,)
        texts=[b["text"] for b in batch],
        vid_ids=[b["vid_id"] for b in batch],
    )


# ---------------------------------------------------------------------------
#  Training
# ---------------------------------------------------------------------------

def masked_l1(pred, target, lengths):
    """L1 loss computed only on valid (non-padded) mel frames."""
    B, T, C = pred.shape
    mask = (
        torch.arange(T, device=pred.device)
        .unsqueeze(0).expand(B, -1)
        < lengths.unsqueeze(1)
    ).unsqueeze(-1).float()                                  # (B, T, 1)
    loss = (torch.abs(pred - target) * mask).sum() / mask.sum() / C
    return loss


def train_one_epoch(model, loader, optimizer, device, epoch, num_epochs):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{num_epochs}", ncols=120, leave=True)

    for batch in pbar:
        mel_gt       = batch["mel"].to(device)
        mel_lens     = batch["mel_lengths"].to(device)
        frames       = batch["frames"].to(device)
        frame_lens   = batch["frame_lengths"].to(device)
        text_ids     = batch["text_ids"].to(device)
        text_lens    = batch["text_lengths"].to(device)

        mel_pred, mel_post = model(text_ids, text_lens, frames, frame_lens, mel_lens)

        loss_pred = masked_l1(mel_pred, mel_gt, mel_lens)
        loss_post = masked_l1(mel_post, mel_gt, mel_lens)
        loss = loss_pred + loss_post

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_loss = 0.0
    for batch in loader:
        mel_gt     = batch["mel"].to(device)
        mel_lens   = batch["mel_lengths"].to(device)
        frames     = batch["frames"].to(device)
        frame_lens = batch["frame_lengths"].to(device)
        text_ids   = batch["text_ids"].to(device)
        text_lens  = batch["text_lengths"].to(device)

        mel_pred, mel_post = model(text_ids, text_lens, frames, frame_lens, mel_lens)
        loss = masked_l1(mel_post, mel_gt, mel_lens)
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train TTS with video style embedding")
    parser.add_argument("--manifest", default=os.path.join(TTS_PREP_ROOT, "manifest.jsonl"))
    parser.add_argument("--epochs",   type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--ckpt_dir", default=TTS_CKPT_DIR)
    parser.add_argument("--resume",   default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    if not os.path.exists(args.manifest):
        print(f"Manifest not found: {args.manifest}")
        print("Run  pixi run tts-prep  first.")
        sys.exit(1)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    # ---- Load mel config (n_mels) ----
    mel_cfg_path = os.path.join(os.path.dirname(args.manifest), "tts_mel_config.json")
    with open(mel_cfg_path) as f:
        mel_cfg = json.load(f)
    n_mels = mel_cfg["n_mels"]
    print(f"Mel config: {mel_cfg}")

    # ---- Dataset / loaders ----
    dataset = TTSDataset(args.manifest)
    total = len(dataset)
    train_n = int(0.7 * total)
    test_n  = total - train_n
    train_set, test_set = random_split(
        dataset, [train_n, test_n],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              collate_fn=tts_collate_fn, num_workers=0)
    test_loader  = DataLoader(test_set,  batch_size=args.batch_size, shuffle=False,
                              collate_fn=tts_collate_fn, num_workers=0)
    print(f"Dataset: {total} samples → {train_n} train / {test_n} test")

    # ---- Model ----
    model = VideoStyleTTS(n_mels=n_mels).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=INITIAL_LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    start_epoch = 1
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        for param_group in optimizer.param_groups:
            param_group["lr"] = INITIAL_LR
        start_epoch = ckpt.get("epoch", 0) + 1

        print(f"Resumed from {args.resume} (epoch {start_epoch - 1})")

    # ---- Training loop ----
    print(f"\n{'=' * 60}")
    print(f"TRAINING  ({args.epochs} epochs, batch={args.batch_size}, lr={INITIAL_LR})")
    print(f"{'=' * 60}\n")

    if not (args.resume and os.path.isfile(args.resume)):
        best_val = float("inf")
    else :
        best_val = ckpt.get("val_loss", float("inf"))

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, args.epochs)
        val_loss   = validate(model, test_loader, device)
        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        tqdm.write(
            f"  Epoch {epoch:3d} | train {train_loss:.4f} | val {val_loss:.4f} | lr {lr_now:.2e}"
        )

        # Save best
        if val_loss < best_val:
            best_val = val_loss
            best_path = os.path.join(args.ckpt_dir, "tts_best.pth")
            torch.save(dict(
                epoch=epoch, model=model.state_dict(),
                optimizer=optimizer.state_dict(), val_loss=val_loss,
                mel_config=mel_cfg,
            ), best_path)
            tqdm.write(f"  ✓ Best model saved → {best_path}  (val {val_loss:.4f})")

#        # Periodic checkpoint
        # if epoch % 10 == 0 or epoch == args.epochs:
        #     path = os.path.join(args.ckpt_dir, f"tts_epoch_{epoch}.pth")
        #     torch.save(dict(
        #         epoch=epoch, model=model.state_dict(),
        #         optimizer=optimizer.state_dict(), val_loss=val_loss,
        #         mel_config=mel_cfg,
        #     ), path)
        #     tqdm.write(f"  Checkpoint → {path}")

    print(f"\nTraining complete.  Best val loss: {best_val:.4f}")


if __name__ == "__main__":
    main()
