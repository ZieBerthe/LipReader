"""TTS inference: silent video → lip-read text → video-informed pretrained TTS → audio.

Pipeline:
    1. Video → mouth crops → LipReadingModel → predicted text + word timings
    2. Video frames → prosody extraction (energy, pauses, speaking rate)
    3. Text + prosody → pretrained Coqui TTS → generated audio
    4. Time-stretch to match video duration + insert pauses

Usage (from project root):
    python tts/infer.py --video data/s1/bbaf2n.mpg
    python tts/infer.py --video data/s1/bbaf2n.mpg --text "bin blue at f two now"
    python tts/infer.py --video data/s1/bbaf2n.mpg --tts_model tts_models/en/ljspeech/vits
"""

import os
import sys
import argparse
import contextlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import torch
import librosa
import soundfile as sf
from scipy.ndimage import uniform_filter1d

from load_data import (
    load_video, char_to_num, num_to_char, vocab,
    encode_chars, VIDEO_FPS,
)
from model import (
    LipReadingModel, ctc_greedy_decode, ids_to_text,
    extract_word_timings,
)
from data_pipeline import MAX_MOUTH_FRAMES, MAX_WORDS


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
#  Video prosody extraction
# ---------------------------------------------------------------------------

def mouth_openness(frames: np.ndarray) -> np.ndarray:
    """Per-frame mouth openness from greyscale mouth crops.

    Uses std-dev of the centre region: open mouth → darker interior → higher
    contrast → higher std.

    Args:
        frames: (T, H, W) normalised mouth crops
    Returns:
        (T,) float array in [0, 1]
    """
    T, H, W = frames.shape
    h0, h1 = int(H * 0.3), int(H * 0.7)
    w0, w1 = int(W * 0.3), int(W * 0.7)
    centre = frames[:, h0:h1, w0:w1]
    openness = np.std(centre, axis=(1, 2))
    rng = openness.max() - openness.min()
    if rng > 1e-6:
        openness = (openness - openness.min()) / rng
    return openness


def movement_energy(frames: np.ndarray, smooth: int = 5) -> np.ndarray:
    """Smoothed frame-to-frame pixel difference → speech energy proxy.

    Args:
        frames: (T, H, W)
        smooth: uniform-filter window (frames)
    Returns:
        (T,) float array in [0, 1]
    """
    T = frames.shape[0]
    diffs = np.zeros(T)
    for t in range(1, T):
        diffs[t] = np.mean(np.abs(frames[t] - frames[t - 1]))
    diffs[0] = diffs[1] if T > 1 else 0.0
    energy = uniform_filter1d(diffs, size=smooth)
    rng = energy.max() - energy.min()
    if rng > 1e-6:
        energy = (energy - energy.min()) / rng
    return energy


def detect_pauses(
    openness: np.ndarray,
    threshold: float = 0.15,
    min_frames: int = 3,
) -> list:
    """Find intervals where the mouth stays closed.

    Returns:
        List of (start_sec, end_sec) tuples.
    """
    closed = openness < threshold
    pauses = []
    start = None
    for i, c in enumerate(closed):
        if c and start is None:
            start = i
        elif not c and start is not None:
            if i - start >= min_frames:
                pauses.append((start / VIDEO_FPS, i / VIDEO_FPS))
            start = None
    if start is not None and len(closed) - start >= min_frames:
        pauses.append((start / VIDEO_FPS, len(closed) / VIDEO_FPS))
    return pauses


class VideoProsody:
    """Bundle all prosody cues extracted from video frames."""

    def __init__(self, frames_np: np.ndarray, word_timings: list):
        self.frames = frames_np
        self.word_timings = word_timings
        self.n_frames = frames_np.shape[0]
        self.duration = self.n_frames / VIDEO_FPS

        self.openness = mouth_openness(frames_np)
        self.energy = movement_energy(frames_np)
        self.pauses = detect_pauses(self.openness)
        self.word_durs = [end - start for _, start, end in word_timings]

    def speaking_rate_wps(self) -> float:
        """Words per second (from timing head predictions)."""
        if not self.word_durs or self.duration < 0.01:
            return 3.0
        return len(self.word_durs) / max(sum(self.word_durs), 0.1)

    def mean_energy_for_interval(self, start_sec: float, end_sec: float) -> float:
        f0 = max(0, int(start_sec * VIDEO_FPS))
        f1 = min(self.n_frames, int(end_sec * VIDEO_FPS))
        if f1 <= f0:
            return 0.5
        return float(np.mean(self.energy[f0:f1]))

    def summary(self) -> dict:
        return dict(
            duration=round(self.duration, 2),
            n_words=len(self.word_timings),
            speaking_rate=round(self.speaking_rate_wps(), 2),
            n_pauses=len(self.pauses),
            mean_energy=round(float(np.mean(self.energy)), 3),
            pauses=[(round(s, 3), round(e, 3)) for s, e in self.pauses],
            word_durations=[round(d, 3) for d in self.word_durs],
        )


# ---------------------------------------------------------------------------
#  Step 1 — Lip-reading: video → text + word timings
# ---------------------------------------------------------------------------

def lip_read(
    video_frames: torch.Tensor,
    device: torch.device,
    ckpt_path: str,
) -> tuple:
    """Run LipReadingModel → (text, word_timings).

    Returns:
        text: predicted sentence
        word_timings: list of (word_idx, start_sec, end_sec)
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Lip-reading checkpoint not found: {ckpt_path}\n"
            "Train with `python model.py` or supply --text."
        )

    num_chars = len(char_to_num) + 1  # +1 CTC blank
    lr_model = LipReadingModel(num_chars=num_chars).to(device)
    state = torch.load(ckpt_path, map_location=device)
    lr_model.load_state_dict(state, strict=False)
    lr_model.eval()

    T = video_frames.shape[0]
    vf = video_frames.clone()
    if T < MAX_MOUTH_FRAMES:
        pad = torch.zeros(MAX_MOUTH_FRAMES - T, *vf.shape[1:])
        vf = torch.cat([vf, pad], dim=0)
    vf = vf[:MAX_MOUTH_FRAMES]

    x = vf.unsqueeze(0).to(device)  # (1, T, H, W)
    with torch.no_grad():
        text_logits, timing_logits = lr_model(x)

    decoded = ctc_greedy_decode(text_logits, blank_id=0)
    text = ids_to_text(decoded[0], num_to_char)

    timing_preds = torch.argmax(timing_logits, dim=-1)[0]
    valid = min(T, MAX_MOUTH_FRAMES)
    word_timings = extract_word_timings(timing_preds[:valid], fps=VIDEO_FPS)

    return text, word_timings


# ---------------------------------------------------------------------------
#  Step 2 — Prosody-informed synthesis with Coqui TTS
# ---------------------------------------------------------------------------

def _build_ssml(text: str, prosody: VideoProsody) -> str:
    """Build SSML with per-word rate/volume and pauses from video prosody.

    Coqui VITS and some models support a subset of SSML; for models that
    don't, we fall back to plain text + post-hoc manipulation.
    """
    words = text.strip().split()
    if not prosody.word_timings:
        return text  # no timing info → plain text

    parts = ['<speak>']
    avg_dur = 0.3  # typical GRID word duration

    for i, word in enumerate(words):
        # find matching timing entry (word_idx is 1-based in extract_word_timings)
        match = [wt for wt in prosody.word_timings if wt[0] == i + 1]

        # check for pause before this word
        if match:
            _, w_start, w_end = match[0]
            for p_start, p_end in prosody.pauses:
                if abs(p_end - w_start) < 0.12:
                    ms = int((p_end - p_start) * 1000)
                    parts.append(f'<break time="{ms}ms"/>')
                    break

            dur = w_end - w_start
            rate_pct = int((avg_dur / max(dur, 0.05)) * 100)
            rate_pct = max(50, min(200, rate_pct))

            e = prosody.mean_energy_for_interval(w_start, w_end)
            vol = "loud" if e > 0.65 else ("soft" if e < 0.25 else "medium")

            parts.append(
                f'<prosody rate="{rate_pct}%" volume="{vol}">{word}</prosody> '
            )
        else:
            parts.append(f'{word} ')

    parts.append('</speak>')
    return ''.join(parts)


def synthesise(
    text: str,
    prosody: VideoProsody,
    out_path: str,
    model_name: str = None,
    speaker_wav: str | None = None,
) -> tuple:
    """Synthesise audio with Piper TTS, using prosody cues from video.

    Piper supports length_scale (speaking rate) and sentence_silence
    as direct parameters, making it ideal for video-informed prosody.

    Returns:
        (wav, sr)
    """
    import wave
    import struct
    from piper import PiperVoice

    # Default voice path
    if model_name is None:
        model_name = os.path.join(
            ROOT, "tts", "piper_voices",
            "en-us-lessac-medium.onnx"
        )

    voice = PiperVoice.load(model_name)

    # ── Video-informed speaking rate ──
    # length_scale > 1.0 = slower, < 1.0 = faster
    # Default ~3.5 wps for English; adjust relative to video
    video_wps = prosody.speaking_rate_wps()
    default_wps = 3.5
    length_scale = default_wps / max(video_wps, 0.5)
    length_scale = max(0.5, min(2.0, length_scale))  # clamp

    # ── Video-informed pause duration ──
    # Average pause duration from video (seconds)
    if prosody.pauses:
        avg_pause = np.mean([e - s for s, e in prosody.pauses])
    else:
        avg_pause = 0.2
    sentence_silence = float(min(avg_pause, 0.8))

    print(f"  Piper length_scale: {length_scale:.2f} (video wps: {video_wps:.1f})")
    print(f"  Piper sentence_silence: {sentence_silence:.2f}s")

    # ── Synthesise ──
    # Piper outputs raw PCM via an iterator
    audio_chunks = []
    for audio_bytes in voice.synthesize_stream_raw(
        text,
        length_scale=length_scale,
        sentence_silence=sentence_silence,
    ):
        audio_chunks.append(audio_bytes)

    raw_audio = b"".join(audio_chunks)

    # Piper outputs 16-bit mono PCM at the model's sample rate
    sr = voice.config.sample_rate
    samples = struct.unpack(f"<{len(raw_audio)//2}h", raw_audio)
    wav = np.array(samples, dtype=np.float32) / 32768.0

    # Save
    sf.write(out_path, wav, sr)
    print(f"  Piper output: {len(wav)/sr:.2f}s, {sr} Hz")

    return wav, sr


# ---------------------------------------------------------------------------
#  Step 3 — Post-hoc prosody alignment
# ---------------------------------------------------------------------------

def insert_pauses(
    wav: np.ndarray,
    sr: int,
    prosody: VideoProsody,
    text: str,
) -> np.ndarray:
    """Insert silence at detected pause positions (proportional to video).

    We split the audio roughly into word-length chunks and insert silence
    between them where the video shows closed mouth.
    """
    words = text.strip().split()
    n_words = len(words)
    if n_words < 2 or not prosody.pauses or not prosody.word_timings:
        return wav

    total_audio = len(wav)
    # crude per-word split (uniform, since we don't have forced alignment of TTS)
    chunk_len = total_audio // n_words

    result_parts = []
    for i in range(n_words):
        start = i * chunk_len
        end = (i + 1) * chunk_len if i < n_words - 1 else total_audio
        result_parts.append(wav[start:end])

        # check if there's a video pause after word i+1
        match = [wt for wt in prosody.word_timings if wt[0] == i + 1]
        if match:
            _, _, w_end = match[0]
            for p_start, p_end in prosody.pauses:
                if abs(p_start - w_end) < 0.15:
                    silence_samples = int((p_end - p_start) * sr)
                    silence_samples = min(silence_samples, int(0.5 * sr))  # cap 500 ms
                    result_parts.append(np.zeros(silence_samples, dtype=np.float32))
                    break

    return np.concatenate(result_parts)


def match_duration(
    wav: np.ndarray,
    sr: int,
    target_sec: float,
) -> np.ndarray:
    """Time-stretch (pitch-preserving) to match video duration."""
    audio_sec = len(wav) / sr
    if audio_sec < 0.05 or target_sec < 0.05:
        return wav
    rate = audio_sec / target_sec
    if abs(rate - 1.0) < 0.05:
        return wav
    rate = max(0.5, min(2.0, rate))
    return librosa.effects.time_stretch(wav, rate=rate).astype(np.float32)


def apply_energy_envelope(
    wav: np.ndarray,
    sr: int,
    prosody: VideoProsody,
) -> np.ndarray:
    """Scale amplitude of the TTS waveform to follow the video energy contour.

    Resamples the per-frame energy curve to the audio sample rate, then
    applies it as a multiplicative envelope (blended with uniform to avoid
    silence in low-energy regions).
    """
    n_audio = len(wav)
    n_frames = len(prosody.energy)
    if n_frames < 2 or n_audio < 2:
        return wav

    # resample energy to audio sample rate
    x_frames = np.linspace(0, 1, n_frames)
    x_audio = np.linspace(0, 1, n_audio)
    envelope = np.interp(x_audio, x_frames, prosody.energy)

    # blend: 60% envelope + 40% flat (prevents crushing quiet parts)
    envelope = 0.4 + 0.6 * envelope
    return (wav * envelope).astype(np.float32)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Silent video → lip-read → video-informed TTS → wav"
    )
    parser.add_argument("--video", required=True,
                        help="Path to .mpg video")
    parser.add_argument("--text", default=None,
                        help="Override lip-reading with explicit text")
    parser.add_argument("--lr_ckpt",
                        default=os.path.join(ROOT, "lipreading_final.pth"),
                        help="Lip-reading model checkpoint")
    parser.add_argument("--tts_model",
                        default="tts_models/en/ljspeech/tacotron2-DDC",
                        help="Coqui TTS model identifier")
    parser.add_argument("--speaker_wav", default=None,
                        help="Reference wav for voice cloning (XTTS only)")
    parser.add_argument("--out",
                        default=os.path.join(ROOT, "tts", "output.wav"),
                        help="Output wav path")
    parser.add_argument("--no_stretch", action="store_true",
                        help="Skip time-stretching to video duration")
    parser.add_argument("--no_envelope", action="store_true",
                        help="Skip energy envelope from video")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── 1. Extract mouth frames ──────────────────────────────────────────
    print("\n[1/4] Extracting mouth frames …")
    with suppress_stderr():
        frames = load_video(args.video)                     # (T, H, W)
    video_dur = frames.shape[0] / VIDEO_FPS
    print(f"  {frames.shape[0]} frames, {video_dur:.2f} s")

    # ── 2. Lip-read text + timings ───────────────────────────────────────
    print("\n[2/4] Predicting text …")
    if args.text is not None:
        text = args.text
        # still run timing head for prosody
        _, word_timings = lip_read(frames, device, args.lr_ckpt)
        print(f"  Text (provided):  \"{text}\"")
    else:
        text, word_timings = lip_read(frames, device, args.lr_ckpt)
        print(f"  Text (predicted): \"{text}\"")

    if not text.strip():
        print("ERROR: empty text — provide --text or retrain the lip-reader.")
        sys.exit(1)

    print(f"  Word timings ({len(word_timings)}):")
    for widx, start, end in word_timings:
        print(f"    word_{widx}: {start:.3f}s – {end:.3f}s")

    # ── 3. Extract video prosody ─────────────────────────────────────────
    print("\n[3/4] Extracting video prosody …")
    frames_np = frames.numpy().astype(np.float32)
    prosody = VideoProsody(frames_np, word_timings)
    s = prosody.summary()
    print(f"  Speaking rate : {s['speaking_rate']} words/s")
    print(f"  Pauses        : {s['n_pauses']}  {s['pauses']}")
    print(f"  Mean energy   : {s['mean_energy']}")
    print(f"  Word durations: {s['word_durations']}")

    # ── 4. Synthesise + apply prosody ────────────────────────────────────
    print(f"\n[4/4] Synthesising with Coqui TTS ({args.tts_model}) …")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    wav, sr = synthesise(
        text, prosody, args.out,
        model_name=args.tts_model,
        speaker_wav=args.speaker_wav,
    )
    print(f"  Raw TTS: {len(wav)/sr:.2f} s, {sr} Hz")

    # Insert pauses matching video
    wav = insert_pauses(wav, sr, prosody, text)
    print(f"  After pause insertion: {len(wav)/sr:.2f} s")

    # Time-stretch to video duration
    if not args.no_stretch:
        wav = match_duration(wav, sr, video_dur)
        print(f"  After time-stretch: {len(wav)/sr:.2f} s (target: {video_dur:.2f} s)")

    # Apply energy envelope from video
    if not args.no_envelope:
        wav = apply_energy_envelope(wav, sr, prosody)
        print(f"  Applied video energy envelope")

    # Save
    sf.write(args.out, wav, sr)
    print(f"\n✓ Saved: {args.out}  ({len(wav)/sr:.2f} s, {sr} Hz)")


if __name__ == "__main__":
    main()