"""Audio features shared by every speech-generation model: log-mel extraction and inversion.

The mel follows HiFi-GAN's `meldataset.mel_spectrogram` exactly (config_v1), so generated mels
can be vocoded by the pretrained HiFi-GAN in vocoder.py: magnitude STFT (power=1), reflect
padding of (n_fft - hop) / 2 on each side, center=False, Slaney mel filters, natural log
clamped at 1e-5.
"""
import numpy as np

# Mel spectrogram configuration (saved alongside prepared data as tts_mel_config.json)
MEL_SR = 22050
MEL_N_FFT = 1024
MEL_HOP = 256
MEL_WIN = 1024
MEL_N_MELS = 80
MEL_FMIN = 0
MEL_FMAX = 8000

MEL_CONFIG = dict(
    sr=MEL_SR, n_fft=MEL_N_FFT, hop_length=MEL_HOP,
    win_length=MEL_WIN, n_mels=MEL_N_MELS, fmin=MEL_FMIN, fmax=MEL_FMAX,
)


def extract_mel(audio_path: str) -> np.ndarray:
    """Load the audio track of a file (e.g. a GRID .mpg) and return its log-mel spectrogram.

    Returns:
        (T_mel, n_mels) float32 array
    """
    import librosa  # imported lazily so multiprocessing workers stay light

    y, _ = librosa.load(audio_path, sr=MEL_SR)
    return wav_to_mel(y)


def wav_to_mel(y: np.ndarray) -> np.ndarray:
    """Waveform at MEL_SR in [-1, 1] → (T_mel, n_mels) HiFi-GAN-compatible log-mel."""
    import librosa

    pad = (MEL_N_FFT - MEL_HOP) // 2
    y = np.pad(y, (pad, pad), mode="reflect")
    mel = librosa.feature.melspectrogram(
        y=y, sr=MEL_SR,
        n_fft=MEL_N_FFT, hop_length=MEL_HOP, win_length=MEL_WIN,
        n_mels=MEL_N_MELS, fmin=MEL_FMIN, fmax=MEL_FMAX,
        center=False, power=1.0,
    )
    log_mel = np.log(np.maximum(mel, 1e-5))
    return log_mel.astype(np.float32).T


def mel_to_wav(log_mel: np.ndarray, mel_cfg: dict = MEL_CONFIG, n_iter: int = 60) -> np.ndarray:
    """Invert a log-mel spectrogram to a waveform with Griffin-Lim.

    Args:
        log_mel:  (T_mel, n_mels) log-scale mel
        mel_cfg:  dict with sr, n_fft, hop_length, win_length, fmin, fmax
        n_iter:   Griffin-Lim iterations
    Returns:
        1-D float32 waveform
    """
    import librosa

    wav = librosa.feature.inverse.mel_to_audio(
        np.exp(log_mel).T,
        sr=mel_cfg["sr"],
        n_fft=mel_cfg["n_fft"],
        hop_length=mel_cfg["hop_length"],
        win_length=mel_cfg["win_length"],
        fmin=mel_cfg["fmin"],
        fmax=mel_cfg["fmax"],
        power=1.0,  # extract_mel produces a magnitude mel
        n_iter=n_iter,
    )
    return wav.astype(np.float32)
