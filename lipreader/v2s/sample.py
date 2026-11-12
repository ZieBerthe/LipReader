"""Generate speech from lip video with a trained lip-to-speech model.

    pixi run v2s-sample                         # 4 validation clips → outputs/v2s/
    pixi run v2s-sample --video data/s1/bbaf2n.mpg
    pixi run v2s-sample --copy-synthesis        # also vocode the ground-truth mels (vocoder sanity check;
                                                # works without a trained model)
"""
import os
import argparse
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
import soundfile as sf

from lipreader.audio import MEL_SR
from lipreader.device import get_device
from lipreader.paths import V2S_CKPT_DIR, OUTPUT_DIR
from lipreader.data.dataset import MAX_MOUTH_FRAMES
from lipreader.v2s.data import train_val_datasets, fit_length
from lipreader.v2s.model import LipToSpeech
from lipreader.vocoder import load_hifigan, vocode


def load_model(path, device):
    ckpt = torch.load(path, map_location=device)
    model = LipToSpeech()
    model.load_state_dict(ckpt["model"])
    print(f"Loaded {path} (epoch {ckpt['epoch']}, val loss {ckpt['val_loss']:.4f})")
    return model.to(device).eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", default=os.path.join(V2S_CKPT_DIR, "best.pth"))
    parser.add_argument("--video", default=None, help="generate for this .mpg instead of validation clips")
    parser.add_argument("--num", type=int, default=4, help="number of validation clips")
    parser.add_argument("--steps", type=int, default=32, help="Euler steps")
    parser.add_argument("--cfg", type=float, default=2.0, help="classifier-free guidance scale")
    parser.add_argument("--copy-synthesis", action="store_true", help="also write vocoded ground-truth mels")
    parser.add_argument("--out-dir", default=os.path.join(OUTPUT_DIR, "v2s"))
    args = parser.parse_args()

    device = get_device()
    os.makedirs(args.out_dir, exist_ok=True)
    vocoder = load_hifigan(device)
    model = load_model(args.ckpt, device) if os.path.exists(args.ckpt) else None
    if model is None and not args.copy_synthesis:
        parser.error(f"{args.ckpt} not found; train with `pixi run v2s-train` first")

    def save(name, mel):
        path = os.path.join(args.out_dir, f"{name}.wav")
        sf.write(path, vocode(vocoder, mel), MEL_SR)
        print(f"  → {path}")

    if args.video:
        from lipreader.data.video import load_video  # lazy: imports MediaPipe
        frames = fit_length(load_video(args.video), MAX_MOUTH_FRAMES)[None].to(device)
        clip = os.path.splitext(os.path.basename(args.video))[0]
        save(f"{clip}_generated", model.sample(frames, args.steps, args.cfg)[0].cpu().numpy())
        return

    _, val_set = train_val_datasets()
    errors = []
    for i in range(min(args.num, len(val_set))):
        item = val_set[i]
        clip = os.path.splitext(os.path.basename(item["video_path"]))[0]
        print(clip)
        gt_mel = item["mel"].numpy()
        if args.copy_synthesis:
            save(f"{clip}_groundtruth", gt_mel)
        if model is not None:
            frames = item["mouth_frames"][None].to(device)
            mel = model.sample(frames, args.steps, args.cfg)[0].cpu().numpy()
            save(f"{clip}_generated", mel)
            errors.append(np.abs(mel - gt_mel).mean())
    if errors:
        print(f"Mean log-mel L1 vs ground truth: {np.mean(errors):.3f}")


if __name__ == "__main__":
    main()
