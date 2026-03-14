"""TTS inference: text + video → wav.

Modes:
    1. Provide text + video path   (frames extracted on the fly)
    2. Provide text + frames .npy  (pre-extracted mouth crops)
    3. Auto-read text from video   (runs the lip-reading model first)

The mel spectrogram is inverted to audio with Griffin-Lim (no separate
vocoder training needed).

Usage (from project root):
    python tts/infer.py --text "bin blue at f two now" --video data/s1/bbaf2n.mpg
    python tts/infer.py --text "bin blue at f two now" --frames tts/prepped/bbaf2n/frames.npy
    python tts/infer.py --video data/s1/bbaf2n.mpg      # auto lip-read for text
"""

import os
import sys
import json
import argparse
import contextlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import torch
import librosa
import soundfile as sf

from load_data import (
    load_video, char_to_num, num_to_char, vocab,
    encode_chars, VIDEO_FPS,
)
from tts.train_tts_video_style import VideoStyleTTS, VOCAB_SIZE


@contextlib.contextmanager
def suppress_stderr():
    fd = sys.stderr.fileno()
    old = os.dup(fd)
    with open(os.devnull, "w") as devnull:
        os.dup2(devnull.fileno(), fd)
        try:
            yield
        finally:
            os.dup2(old, fd)
            os.close(old)


# ---------------------------------------------------------------------------
#  Lip-reading helper (optional – used when --text is not provided)
# ---------------------------------------------------------------------------

def lip_read_text(video_frames: torch.Tensor, device: torch.device) -> str:
    """Run the trained lip-reading model to get text from video frames.

    Falls back to a placeholder if the lip-reading checkpoint is missing.
    """
    from model import LipReadingModel, ctc_greedy_decode, ids_to_text
    from data_pipeline import MAX_MOUTH_FRAMES

    num_chars = len(char_to_num) + 1  # +1 for CTC blank (index 0)
    lr_model = LipReadingModel(num_chars=num_chars).to(device)

    ckpt_path = os.path.join(ROOT, "lipreading_best.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Lip-reading checkpoint not found at {ckpt_path}.  "
            "Provide --text explicitly or train the lip-reader first."
        )

    state = torch.load(ckpt_path, map_location=device)
    lr_model.load_state_dict(state, strict=False)
    lr_model.eval()

    # Pad to MAX_MOUTH_FRAMES
    T = video_frames.shape[0]
    if T < MAX_MOUTH_FRAMES:
        pad = torch.zeros(MAX_MOUTH_FRAMES - T, *video_frames.shape[1:])
        video_frames = torch.cat([video_frames, pad], dim=0)
    video_frames = video_frames[:MAX_MOUTH_FRAMES]

    x = video_frames.unsqueeze(0).to(device)               # (1, T, H, W)
    with torch.no_grad():
        # text_logits, _ = lr_model(x)
        text_logits = lr_model(x)                          # (1, L, num_chars)
    decoded = ctc_greedy_decode(text_logits, blank_id=0)
    text = ids_to_text(decoded[0], num_to_char)
    return text


# ---------------------------------------------------------------------------
#  Mel → waveform (Griffin-Lim)
# ---------------------------------------------------------------------------

def mel_to_wav(log_mel: np.ndarray, mel_cfg: dict, n_iter: int = 60) -> np.ndarray:
    """Invert a log-mel spectrogram to a waveform using Griffin-Lim.

    Args:
        log_mel:  (T_mel, n_mels) – model output (log scale)
        mel_cfg:  dict with sr, n_fft, hop_length, win_length, fmin, fmax
        n_iter:   Griffin-Lim iterations
    Returns:
        1-D float32 waveform
    """
    mel_linear = np.exp(log_mel).T                         # (n_mels, T_mel)

    wav = librosa.feature.inverse.mel_to_audio(
        mel_linear,
        sr=mel_cfg["sr"],
        n_fft=mel_cfg["n_fft"],
        hop_length=mel_cfg["hop_length"],
        win_length=mel_cfg["win_length"],
        fmin=mel_cfg["fmin"],
        fmax=mel_cfg["fmax"],
        n_iter=n_iter,
    )
    return wav.astype(np.float32)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="TTS inference with video style")
    parser.add_argument("--text",    default=None, help="Text to synthesise (if omitted, lip-reader is used)")
    parser.add_argument("--video",   default=None, help="Path to .mpg video (extracts frames on the fly)")
    parser.add_argument("--frames",  default=None, help="Path to pre-extracted frames .npy")
    parser.add_argument("--ckpt",    default=os.path.join(ROOT, "tts", "checkpoints", "tts_best.pth"),
                        help="TTS model checkpoint")
    parser.add_argument("--gl_iter", type=int, default=60, help="Griffin-Lim iterations")
    args = parser.parse_args()
    video_name = os.path.splitext(os.path.basename(args.video))[0] if args.video else "output"
    parser.add_argument("--out",     default=os.path.join(ROOT, "tts", f"output{video_name}.wav"),
                        help="Output wav path")
    args = parser.parse_args()

    if args.video is None and args.frames is None:
        parser.error("Provide at least one of --video or --frames")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Load checkpoint & mel config ----
    if not os.path.isfile(args.ckpt):
        print(f"Checkpoint not found: {args.ckpt}")
        print("Train the TTS model first:  python tts/train_tts_video_style.py")
        sys.exit(1)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    mel_cfg = ckpt["mel_config"]
    n_mels  = mel_cfg["n_mels"]
    print(f"Mel config: {mel_cfg}")

    # ---- Build & load model ----
    model = VideoStyleTTS(n_mels=n_mels).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print("Model loaded.")

    # ---- Load video frames ----
    if args.frames is not None:
        frames_np = np.load(args.frames).astype(np.float32)
        frames = torch.from_numpy(frames_np)
        print(f"Loaded pre-extracted frames: {frames.shape}")
    else:
        print("Extracting mouth frames from video …")
        with suppress_stderr():
            frames = load_video(args.video)                 # (T, H, W)
        print(f"Extracted {frames.shape[0]} frames.")

    # ---- Get text ----
    if args.text is not None:
        text = args.text
    else:
        print("No --text provided → running lip-reading model …")
        text = lip_read_text(frames, device)
    print(f"Text: \"{text}\"")

    # ---- Prepare tensors ----
    char_ids = encode_chars(list(text))
    text_ids = torch.tensor(char_ids, dtype=torch.long).unsqueeze(0).to(device)  # (1, L)
    text_len = torch.tensor([len(char_ids)], device=device)

    frames_in = frames.unsqueeze(0).to(device)              # (1, T, H, W)
    frame_len = torch.tensor([frames.shape[0]], device=device)

    # Estimate mel length from video duration
    video_dur_sec = frames.shape[0] / VIDEO_FPS
    mel_length = int(video_dur_sec * mel_cfg["sr"] / mel_cfg["hop_length"])
    print(f"Video duration: {video_dur_sec:.2f} s  →  mel length: {mel_length}")

    # ---- Inference ----
    with torch.no_grad():
        mel_out = model.infer(text_ids, text_len, frames_in, frame_len, mel_length)
    mel_np = mel_out.squeeze(0).cpu().numpy()               # (T_mel, n_mels)

    # ---- Mel → wav (Griffin-Lim) ----
    print("Running Griffin-Lim vocoder …")
    wav = mel_to_wav(mel_np, mel_cfg, n_iter=args.gl_iter)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    sf.write(args.out, wav, mel_cfg["sr"])
    print(f"Saved: {args.out}  ({len(wav) / mel_cfg['sr']:.2f} s, {mel_cfg['sr']} Hz)")


if __name__ == "__main__":
    main()
