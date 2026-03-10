"""Train a text-to-speech model conditioned on a video style embedding.

Architecture (non-autoregressive):
    TextEncoder   : char embedding → Conv1D → BiLSTM → (B, T_text, 256)
    VideoStyleEnc : per-frame 2-D CNN → mean pool → FC → style vector (128)
    MelDecoder    : interpolate text features to mel length, concat style,
                    Conv1D stack → mel prediction (B, T_mel, n_mels)
    PostNet       : Conv1D residual refinement of mel

Loss: L1(mel_pred, mel_gt) + L1(mel_postnet, mel_gt)

Usage (from project root):
    python tts/train_tts_video_style.py
    python tts/train_tts_video_style.py --epochs 40 --batch_size 8
"""

import os
import sys
import json
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

from load_data import char_to_num, num_to_char, vocab, encode_chars, VIDEO_FPS

# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------
VOCAB_SIZE   = len(vocab) + 1          # +1 for padding/blank at index 0
MAX_TEXT_LEN = 50                       # characters (GRID sentences are short)
MAX_VIDEO_FRAMES = 75
MAX_MEL_FRAMES   = 350                  # ~4 s at 22 050 Hz / hop 256

# Model hyper-parameters
ENC_DIM      = 256
STYLE_DIM    = 128
DEC_HIDDEN   = 512
POSTNET_CH   = 256
POSTNET_LAYERS = 3

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
#  Model components
# ---------------------------------------------------------------------------

class TextEncoder(nn.Module):
    """Character embedding → 3 × Conv1D + BN + ReLU → BiLSTM."""

    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=ENC_DIM, out_dim=ENC_DIM):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        convs = []
        for _ in range(3):
            convs += [
                nn.Conv1d(embed_dim, embed_dim, kernel_size=5, padding=2),
                nn.BatchNorm1d(embed_dim),
                nn.ReLU(),
                nn.Dropout(0.5),
            ]
        self.convs = nn.Sequential(*convs)

        self.lstm = nn.LSTM(
            embed_dim, out_dim // 2,
            num_layers=1, batch_first=True, bidirectional=True,
        )

    def forward(self, text_ids, text_lengths):
        """
        Args:
            text_ids:     (B, T_text)  long
            text_lengths: (B,)
        Returns:
            enc_out: (B, T_text, out_dim)
        """
        x = self.embedding(text_ids)                      # (B, T, E)
        x = self.convs(x.transpose(1, 2)).transpose(1, 2) # (B, T, E)

        # Pack → LSTM → unpack
        packed = nn.utils.rnn.pack_padded_sequence(
            x, text_lengths.cpu().clamp(min=1), batch_first=True, enforce_sorted=False,
        )
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        return out                                         # (B, T_text, out_dim)


class VideoStyleEncoder(nn.Module):
    """Per-frame 2-D CNN → BiLSTM → attention pool → style vector.

    Instead of simple mean-pooling (which loses all temporal dynamics),
    we run a BiLSTM over the per-frame CNN features and use a small
    attention layer to produce a single global style vector that captures
    speaking rhythm, pace and emphasis patterns from the video.
    """

    def __init__(self, style_dim=STYLE_DIM):
        super().__init__()
        cnn_out = 128

        # Spatial feature extraction (per frame)
        self.cnn = nn.Sequential(
            nn.Conv2d(1,  32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, cnn_out, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),                # → (B*T, 128, 1, 1)
        )

        # Temporal modelling: captures rhythm / speaking rate
        self.lstm = nn.LSTM(
            input_size=cnn_out,
            hidden_size=style_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # Attention pooling over time (learns *which* frames matter most)
        self.attn = nn.Sequential(
            nn.Linear(style_dim, style_dim),
            nn.Tanh(),
            nn.Linear(style_dim, 1),            # score per frame
        )

        self.out_proj = nn.Sequential(
            nn.Linear(style_dim, style_dim),
            nn.ReLU(),
        )

    def forward(self, frames, frame_lengths):
        """
        Args:
            frames:        (B, T_vid, H, W)
            frame_lengths: (B,)
        Returns:
            style: (B, style_dim)
        """
        B, T, H, W = frames.shape

        # ---- Per-frame spatial features ----
        x = frames.reshape(B * T, 1, H, W)
        x = self.cnn(x).squeeze(-1).squeeze(-1)          # (B*T, 128)
        x = x.view(B, T, -1)                             # (B, T, 128)

        # ---- Temporal modelling (BiLSTM) ----
        packed = nn.utils.rnn.pack_padded_sequence(
            x, frame_lengths.cpu().clamp(min=1),
            batch_first=True, enforce_sorted=False,
        )
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(   # (B, T, style_dim)
            lstm_out, batch_first=True, total_length=T,
        )

        # ---- Attention pooling (learns which frames are important) ----
        mask = (
            torch.arange(T, device=frames.device)
            .unsqueeze(0).expand(B, -1)
            < frame_lengths.unsqueeze(1)
        )                                                 # (B, T) bool

        scores = self.attn(lstm_out).squeeze(-1)          # (B, T)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # (B, T, 1)

        pooled = (lstm_out * weights).sum(dim=1)          # (B, style_dim)

        return self.out_proj(pooled)                      # (B, style_dim)


class MelDecoder(nn.Module):
    """Conv1D stack: (enc_dim + style_dim) → n_mels."""

    def __init__(self, in_dim=ENC_DIM + STYLE_DIM, n_mels=80, hidden=DEC_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, hidden, 5, padding=2), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Conv1d(hidden, hidden, 5, padding=2), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Conv1d(hidden, hidden, 5, padding=2), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Conv1d(hidden, n_mels, 1),
        )

    def forward(self, x):
        """x: (B, T_mel, in_dim) → (B, T_mel, n_mels)"""
        return self.net(x.transpose(1, 2)).transpose(1, 2)


class PostNet(nn.Module):
    """Residual Conv1D refinement of the mel output."""

    def __init__(self, n_mels=80, hidden=POSTNET_CH, n_layers=POSTNET_LAYERS):
        super().__init__()
        layers = []
        in_ch = n_mels
        for i in range(n_layers - 1):
            layers += [
                nn.Conv1d(in_ch, hidden, 5, padding=2),
                nn.BatchNorm1d(hidden),
                nn.Tanh(),
                nn.Dropout(0.1),
            ]
            in_ch = hidden
        layers.append(nn.Conv1d(hidden, n_mels, 5, padding=2))
        self.net = nn.Sequential(*layers)

    def forward(self, mel):
        """mel: (B, T, n_mels) → residual (B, T, n_mels)"""
        return self.net(mel.transpose(1, 2)).transpose(1, 2)


# ---------------------------------------------------------------------------
#  Full model
# ---------------------------------------------------------------------------

class VideoStyleTTS(nn.Module):
    """Text-to-mel with video style conditioning.

    Forward:
        1. Encode text → (B, T_text, enc_dim)
        2. Encode video → (B, style_dim)
        3. Interpolate text features up to mel length
        4. Concat style vector at every mel frame
        5. Decode → mel prediction
        6. PostNet → mel refinement (residual)
    """

    def __init__(self, n_mels=80):
        super().__init__()
        self.text_encoder    = TextEncoder()
        self.style_encoder   = VideoStyleEncoder()
        self.mel_decoder     = MelDecoder(in_dim=ENC_DIM + STYLE_DIM, n_mels=n_mels)
        self.post_net        = PostNet(n_mels=n_mels)

    def forward(self, text_ids, text_lengths, frames, frame_lengths, mel_lengths):
        """
        All lengths are 1-D (B,) int tensors.
        Returns mel_pred, mel_postnet — both (B, T_mel_max, n_mels).
        """
        # 1. Text encoding
        enc_out = self.text_encoder(text_ids, text_lengths)  # (B, T_text, enc_dim)

        # 2. Video style
        style = self.style_encoder(frames, frame_lengths)    # (B, style_dim)

        # 3. Up-sample text features to mel length (linear interpolation)
        T_mel_max = mel_lengths.max().item()
        enc_up = F.interpolate(
            enc_out.transpose(1, 2),                         # (B, enc_dim, T_text)
            size=T_mel_max, mode="linear", align_corners=False,
        ).transpose(1, 2)                                    # (B, T_mel_max, enc_dim)

        # 4. Broadcast style to every mel frame & concat
        style_exp = style.unsqueeze(1).expand(-1, T_mel_max, -1)  # (B, T_mel, style_dim)
        dec_in = torch.cat([enc_up, style_exp], dim=-1)           # (B, T_mel, enc_dim+style_dim)

        # 5. Mel prediction
        mel_pred = self.mel_decoder(dec_in)                  # (B, T_mel, n_mels)

        # 6. PostNet refinement (residual)
        mel_postnet = mel_pred + self.post_net(mel_pred)

        return mel_pred, mel_postnet

    @torch.no_grad()
    def infer(self, text_ids, text_lengths, frames, frame_lengths, mel_length: int):
        """Inference helper – generates mel of a given length."""
        mel_lengths = torch.tensor([mel_length], device=text_ids.device)
        _, mel_post = self.forward(
            text_ids, text_lengths, frames, frame_lengths, mel_lengths,
        )
        return mel_post                                      # (1, mel_length, n_mels)


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
    parser.add_argument("--manifest", default=os.path.join(ROOT, "tts", "prepped", "manifest.jsonl"))
    parser.add_argument("--epochs",   type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr",       type=float, default=1e-3)
    parser.add_argument("--ckpt_dir", default=os.path.join(ROOT, "tts", "checkpoints"))
    parser.add_argument("--resume",   default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    if not os.path.exists(args.manifest):
        print(f"Manifest not found: {args.manifest}")
        print("Run  python tts/prep.py  first.")
        sys.exit(1)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    start_epoch = 1
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"Resumed from {args.resume} (epoch {start_epoch - 1})")

    # ---- Training loop ----
    print(f"\n{'=' * 60}")
    print(f"TRAINING  ({args.epochs} epochs, batch={args.batch_size}, lr={args.lr})")
    print(f"{'=' * 60}\n")

    best_val = float("inf")

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

        # Periodic checkpoint
        if epoch % 10 == 0 or epoch == args.epochs:
            path = os.path.join(args.ckpt_dir, f"tts_epoch_{epoch}.pth")
            torch.save(dict(
                epoch=epoch, model=model.state_dict(),
                optimizer=optimizer.state_dict(), val_loss=val_loss,
                mel_config=mel_cfg,
            ), path)
            tqdm.write(f"  Checkpoint → {path}")

    print(f"\nTraining complete.  Best val loss: {best_val:.4f}")


if __name__ == "__main__":
    main()
