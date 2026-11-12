"""Train the lip-to-speech flow-matching model.

    pixi run v2s-train                          # full run on s1 (90/10 split)
    pixi run v2s-train --init-encoder           # start the encoder from checkpoints/lipreading_best.pth
    pixi run v2s-train --max-clips 8 --epochs 300 --sample-every 50   # overfitting sanity check

Checkpoints: checkpoints/v2s/{best,last}.pth. Audio samples: outputs/v2s/.
"""
import os
import argparse
import warnings

warnings.filterwarnings("ignore")

import torch
import soundfile as sf
from torch.utils.data import DataLoader
from tqdm import tqdm

from lipreader.audio import MEL_SR
from lipreader.device import get_device
from lipreader.paths import V2S_CKPT_DIR, LIPREADING_CKPT, OUTPUT_DIR
from lipreader.v2s.data import train_val_datasets, load_mel_stats
from lipreader.v2s.model import LipToSpeech

SAMPLE_DIR = os.path.join(OUTPUT_DIR, "v2s")


@torch.no_grad()
def evaluate(model, loader, device):
    """Mean flow-matching loss with fixed noise, so epochs are comparable."""
    model.eval()
    generator = torch.Generator().manual_seed(0)
    total, count = 0.0, 0
    for batch in loader:
        frames, mel = batch["mouth_frames"].to(device), batch["mel"].to(device)
        total += model.loss(frames, mel, generator=generator).item() * mel.shape[0]
        count += mel.shape[0]
    return total / max(count, 1)


def write_samples(model, dataset, vocoder, device, tag, n=2, steps=32, cfg_scale=2.0):
    from lipreader.vocoder import vocode

    model.eval()
    items = [dataset[i] for i in range(min(n, len(dataset)))]
    frames = torch.stack([item["mouth_frames"] for item in items]).to(device)
    mels = model.sample(frames, steps=steps, cfg_scale=cfg_scale).cpu().numpy()
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    for item, mel in zip(items, mels):
        clip = os.path.splitext(os.path.basename(item["video_path"]))[0]
        path = os.path.join(SAMPLE_DIR, f"{tag}_{clip}.wav")
        sf.write(path, vocode(vocoder, mel), MEL_SR)
        tqdm.write(f"  sample → {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=None, help="default: 4 on MPS, 16 otherwise")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--init-encoder", nargs="?", const=LIPREADING_CKPT, default=None,
                        help="initialise the visual encoder from a lip-reader checkpoint")
    parser.add_argument("--resume", default=None, help="checkpoint to resume from (e.g. checkpoints/v2s/last.pth)")
    parser.add_argument("--max-clips", type=int, default=None, help="train and validate on the first N clips only")
    parser.add_argument("--sample-every", type=int, default=10, help="write audio samples every N epochs (0 = never)")
    parser.add_argument("--steps", type=int, default=32, help="Euler steps when sampling")
    parser.add_argument("--cfg", type=float, default=2.0, help="classifier-free guidance scale when sampling")
    args = parser.parse_args()

    device = get_device()
    batch_size = args.batch_size or (4 if device.type == "mps" else 16)
    print(f"Device: {device}  |  batch size: {batch_size}")

    train_set, val_set = train_val_datasets(args.max_clips)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=len(train_set) > batch_size)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    print(f"Clips: {len(train_set)} train / {len(val_set)} val")

    model = LipToSpeech()
    model.set_mel_stats(*load_mel_stats())
    if args.init_encoder:
        model.init_encoder(args.init_encoder)
        print(f"Encoder initialised from {args.init_encoder}")
    model.to(device)
    n_enc = sum(p.numel() for p in model.encoder.parameters())
    n_dec = sum(p.numel() for p in model.decoder.parameters())
    print(f"Parameters: encoder {n_enc:,}  |  flow decoder {n_dec:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    start_epoch, best_val = 1, float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch, best_val = ckpt["epoch"] + 1, ckpt.get("best_val", float("inf"))
        print(f"Resumed from {args.resume} (epoch {ckpt['epoch']})")

    vocoder = None
    if args.sample_every:
        try:
            from lipreader.vocoder import load_hifigan
            vocoder = load_hifigan(device)
        except FileNotFoundError as e:
            print(f"No audio samples during training: {e}")

    os.makedirs(V2S_CKPT_DIR, exist_ok=True)
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", ncols=100)
        for batch in pbar:
            loss = model.loss(batch["mouth_frames"].to(device), batch["mel"].to(device))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = total / max(len(train_loader), 1)
        val_loss = evaluate(model, val_loader, device)
        is_best = val_loss < best_val
        best_val = min(best_val, val_loss)
        tqdm.write(f"Epoch {epoch:4d} | train {train_loss:.4f} | val {val_loss:.4f}" + ("  ✓ best" if is_best else ""))

        ckpt = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch,
                    val_loss=val_loss, best_val=best_val, args=vars(args))
        torch.save(ckpt, os.path.join(V2S_CKPT_DIR, "last.pth"))
        if is_best:
            torch.save(ckpt, os.path.join(V2S_CKPT_DIR, "best.pth"))

        if vocoder is not None and epoch % args.sample_every == 0:
            write_samples(model, val_set, vocoder, device, tag=f"epoch{epoch:04d}",
                          steps=args.steps, cfg_scale=args.cfg)

    print(f"Done. Best val loss {best_val:.4f}; checkpoints in {V2S_CKPT_DIR}")


if __name__ == "__main__":
    main()
