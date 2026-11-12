"""Central place for data/checkpoint locations and GRID corpus discovery.

All data lives under DATA_ROOT (default: <repo>/data, filled by download_data.py).
Override it with the LIPREADER_DATA environment variable, e.g.
    LIPREADER_DATA=/Volumes/ext/grid pixi run python model.py
"""
import os
import re
import glob
from typing import List, Tuple

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.environ.get("LIPREADER_DATA", os.path.join(PROJECT_ROOT, "data"))

CACHE_ROOT = os.path.join(DATA_ROOT, "cache")              # preprocess_cache.py output (.pt)
TTS_PREP_ROOT = os.path.join(DATA_ROOT, "tts_prepped")     # tts/prep.py output
LIPREADING_CKPT = os.path.join(PROJECT_ROOT, "lipreading_best.pth")

_SPEAKER_DIR = re.compile(r"^s\d+(_processed)?$")


def find_samples(corpus_root: str = DATA_ROOT) -> List[Tuple[str, str, str]]:
    """Return sorted (speaker, video_path, align_path) tuples for every .mpg found.

    Two GRID layouts are supported:
      * download_data.py layout:  <root>/s1/*.mpg  +  <root>/alignments/s1/*.align
      * processed layout:         <root>/s1_processed/*.mpg  +  <root>/s1_processed/align/*.align
    The align path is returned even if the file is missing; callers decide what to do.
    """
    speaker_dirs = sorted(
        d for d in glob.glob(os.path.join(corpus_root, "s*"))
        if os.path.isdir(d) and _SPEAKER_DIR.match(os.path.basename(d))
    )
    if not speaker_dirs:
        raise FileNotFoundError(
            f"No speaker folders (s1/, s2/ … or s1_processed/ …) found in {corpus_root}. "
            f"Run `pixi run python download_data.py` or set LIPREADER_DATA."
        )

    samples = []
    for spk_dir in speaker_dirs:
        speaker = os.path.basename(spk_dir)
        if os.path.isdir(os.path.join(spk_dir, "align")):
            align_dir = os.path.join(spk_dir, "align")
        else:
            align_dir = os.path.join(corpus_root, "alignments", speaker)
        for video in sorted(glob.glob(os.path.join(spk_dir, "*.mpg"))):
            vid_id = os.path.splitext(os.path.basename(video))[0]
            samples.append((speaker, video, os.path.join(align_dir, vid_id + ".align")))
    return samples
