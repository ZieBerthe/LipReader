# LipReader

Reconstructing the words and voice of a person talking in a silent video, on the
[GRID corpus](https://zenodo.org/records/3625687).

Three models:

- **Lip-to-speech v2** (video → speech, `lipreader/v2s/`): the lip reader's 3D-CNN + BiLSTM encodes
  the mouth crops, a small Transformer trained with **conditional flow matching** generates the
  log-mel spectrogram (32 Euler steps, classifier-free guidance), and a pretrained **HiFi-GAN**
  turns it into audio. Background:
  [docs/research/lip-to-speech-flow-matching.md](docs/research/lip-to-speech-flow-matching.md).
- **Lip reader** (video → text): MediaPipe mouth crops → 3D-CNN + BiLSTM → CTC over characters.
- **Video-style TTS** (text + video → speech): text encoder + mouth-video style embedding → mel
  spectrogram → Griffin-Lim.

## Setup

Dependencies are managed with [pixi](https://pixi.sh) (Linux x86-64 and Apple Silicon):

```bash
pixi install
pixi run download     # GRID speaker s1 (~400 MB) into data/
pixi run cache        # extract mouth crops once (~5 min for s1, ~6 GB) into data/cache/
```

Set `LIPREADER_DATA=/path/to/grid` to keep the data somewhere else. Both the downloaded layout
(`s1/*.mpg` + `alignments/s1/*.align`) and the `s1_processed/{*.mpg,align/}` layout are recognised.

## Usage

### Lip-to-speech v2

```bash
pixi run download-vocoder        # pretrained HiFi-GAN UNIVERSAL_V1 → checkpoints/hifigan/ (55 MB)
pixi run v2s-prep                # HiFi-GAN log-mels + normalisation stats → data/mels/ (~1 min for s1)
pixi run v2s-sample --copy-synthesis   # sanity check: ground-truth mels through HiFi-GAN
pixi run v2s-train               # → checkpoints/v2s/{best,last}.pth, audio samples in outputs/v2s/
pixi run v2s-sample              # generate validation clips (or --video data/s1/bbaf2n.mpg)
```

Useful `v2s-train` flags: `--init-encoder` (start from `checkpoints/lipreading_best.pth`),
`--resume checkpoints/v2s/last.pth`, `--max-clips 4` (overfitting check), `--sample-every N`,
`--steps` / `--cfg` for sampling. Split: 90/10 of the clips, fixed seed.

### Other models

| Command | What it does |
|---|---|
| `pixi run preview` | Print a batch and save a mouth-crop GIF to `outputs/` |
| `pixi run train-lipreader` | Train the CTC lip reader (resumes from `checkpoints/lipreading_best.pth`) |
| `pixi run tts-prep` | Extract mels + mouth frames for the TTS into `data/tts_prepped/` |
| `pixi run tts-train` | Train the video-style TTS (`--epochs`, `--batch_size`, `--resume`) |
| `pixi run tts-infer --video data/s1/bbaf2n.mpg` | Lip-read a clip and synthesise it to `outputs/` (`--text` to skip lip reading) |

Training picks the device automatically: Apple MPS, then CUDA, then CPU. On a 16 GB Mac the lip
reader uses batch size 4 (8 already swaps); on CUDA it uses 16.

## Layout

```
lipreader/
  paths.py            where data, checkpoints, logs and outputs live
  device.py           MPS / CUDA / CPU selection
  audio.py            HiFi-GAN-compatible log-mel extraction, Griffin-Lim inversion
  vocoder.py          pretrained HiFi-GAN (download, load, mel → wav)
  utils.py            small helpers (stderr silencing for MediaPipe)
  data/
    download.py       fetch GRID s1
    grid.py           find clips, parse alignments, load one sample
    video.py          MediaPipe mouth cropping
    vocab.py          character vocabulary (index 0 = CTC blank / padding)
    dataset.py        LipReadingDataset + collate_fn
    cache.py          precompute mouth crops to data/cache/
  v2s/
    data.py           mel prep, train/val split, dataset (mouth crops + mel)
    model.py          FlowDecoder (Transformer) + LipToSpeech (encoder, flow loss, Euler sampler)
    train.py          training loop
    sample.py         generation + copy-synthesis
  lipreading/
    model.py          3D-CNN + BiLSTM CTC network (encode() is reused by v2s)
    decoding.py       greedy CTC decoding, word accuracy
    train.py          training loop
  tts/
    model.py          VideoStyleTTS
    prep.py           build the TTS manifest (mels + frames)
    train.py          training loop
    infer.py          text/video → wav
assets/               MediaPipe face landmarker model
docs/research/        lip-to-speech flow-matching study and its source notes
```

Generated folders (gitignored): `data/`, `checkpoints/`, `logs/`, `outputs/`.

## Notes for Apple Silicon (MPS)

- `AdaptiveAvgPool3d` isn't available on MPS, so the lip reader pools with an equivalent
  hand-written 2×2 average (`lipreading/model.py`).
- CTC loss has no MPS kernel, so it's computed on CPU; gradients still flow to the GPU model.
