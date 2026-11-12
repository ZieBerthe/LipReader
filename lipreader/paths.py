"""Where everything lives on disk.

All data lives under DATA_ROOT (default: <repo>/data, filled by `pixi run download`).
Override it with the LIPREADER_DATA environment variable, e.g.
    LIPREADER_DATA=/Volumes/ext/grid pixi run train-lipreader
"""
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Static assets shipped with the repo
ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")
FACE_LANDMARKER_MODEL = os.path.join(ASSETS_DIR, "face_landmarker.task")

# Data (gitignored)
DATA_ROOT = os.environ.get("LIPREADER_DATA", os.path.join(PROJECT_ROOT, "data"))
CACHE_ROOT = os.path.join(DATA_ROOT, "cache")              # mouth-crop cache (.pt), see data/cache.py
TTS_PREP_ROOT = os.path.join(DATA_ROOT, "tts_prepped")     # mel + frames for TTS, see tts/prep.py
MEL_ROOT = os.path.join(DATA_ROOT, "mels")                 # HiFi-GAN log-mels per clip, see v2s/data.py

# Training outputs (gitignored)
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
LIPREADING_CKPT = os.path.join(CHECKPOINT_DIR, "lipreading_best.pth")
TTS_CKPT_DIR = os.path.join(CHECKPOINT_DIR, "tts")
V2S_CKPT_DIR = os.path.join(CHECKPOINT_DIR, "v2s")          # lip-to-speech flow model
HIFIGAN_DIR = os.path.join(CHECKPOINT_DIR, "hifigan")       # pretrained vocoder weights
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")         # generated audio, previews
