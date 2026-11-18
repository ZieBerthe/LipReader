# From Silent Video to Speech: Two-Stage Lipreading + Video-Conditioned TTS

A practical pipeline that turns **silent talking-face video into audible speech**, built in two modular stages:

1. **Lipreading (VSR)**: mouth-region frames → text, with a 3D-CNN + BiLSTM + CTC model.
2. **Video-conditioned TTS**: predicted text + a speaking-style embedding extracted from the same video → mel spectrogram → waveform (Griffin-Lim).

Splitting the problem makes supervision simpler and lets you tell recognition errors apart from synthesis errors.

**Authors:** Berthé Zié Aziz, Dreesen Bonet Nicholas (March 2026)
**Dataset:** [GRID Audio-Visual Speech Corpus](https://zenodo.org/records/3625687)

---

## Pipeline

```
Silent video ──► MediaPipe mouth crops (100×200, ≤75 frames)
                    │
        ┌───────────┴───────────────────────────────┐
        ▼                                           ▼
 STAGE 1: Lipreading                        Video style encoder
 3D-CNN → BiLSTM → CTC head                 2D-CNN → BiLSTM → attention pooling
        │ (greedy CTC decoding)                     │ 128-D style vector
        ▼                                           │
   Predicted text ──► Text encoder ──┐              │
                      (Conv1D+BiLSTM)├─► Mel decoder + PostNet ──► Griffin-Lim ──► Audio
                                     └──────────────┘
                         STAGE 2: video-conditioned TTS
```

### Stage 1: CTC lipreading
- Input `(B, 1, T, 100, 200)` grayscale mouth crops, padded/truncated to 75 frames.
- 3 × Conv3D blocks (32 → 64 → 96 channels, 3×3×3, BatchNorm, ReLU, MaxPool3D(1,2,2)), adaptive average pooling to 2×2, giving 384-D per-frame features.
- 2-layer BiLSTM (hidden 128, dropout 0.3), then a linear head over characters + CTC blank.
- CTC loss; greedy decoding (argmax, collapse repeats, drop blank).

### Stage 2: video-conditioned non-autoregressive TTS
- **Text encoder**: embedding (vocab 80, dim 256) → 3 × Conv1D (k=5) → BiLSTM (128).
- **Video style encoder**: per-frame 2D-CNN (3 blocks) → BiLSTM (128) → attention pooling → 128-D style vector.
- **Mel decoder**: text features interpolated to the mel length, concatenated with the repeated style vector (384-D), then 3 × Conv1D (hidden 512) and a final Conv1D to 80 mel bins.
- **PostNet**: 3 × Conv1D residual refinement, `M̂_post = M̂ + PostNet(M̂)`.
- **Loss**: masked L1 on both the decoder output and the PostNet output.
- **Vocoder**: Griffin-Lim, 60 iterations (no trained vocoder).

Audio features: 22.05 kHz, 80 mel bins, hop length 256.

---

## Training setup

| | Lipreading | TTS |
|---|---|---|
| Optimizer | Adam, lr 1e-3, betas (0.9, 0.999) | Adam, lr 1e-6, ×0.5 every 20 epochs |
| Batch size | 16 | 1 (variable mel/video lengths) |
| Regularization | Grad clipping (max norm 1.0), early stopping on val loss | Up to 50 epochs, best checkpoint by val L1 |

Data split: fixed 70/30 random train/test split, seed 67, on the cached GRID data.

---

## Results (GRID test set)

**Lipreading**

| Metric | Value |
|---|---|
| Word accuracy | 90.25 % |
| Character accuracy | 96.70 % |
| Sentence accuracy | 51.20 % |

**TTS (Griffin-Lim reconstruction)**

| Metric | Value |
|---|---|
| Log Spectral Distance | 28.16 dB |
| Mel Cepstral Distortion | 222.21 dB |
| SI-SDR | −5.75 dB |
| Duration ratio | 1.000 |
| Spectral convergence | 0.5901 |

**Where errors come from**
- Colors, digits and command words: 95–100 % accuracy.
- Short prepositions are the main failure: **"at" 60.3 %, "in" 58.7 %**, mostly confused with each other.
- Hardest letters: x (17.5 %), n (30 %), a (30 %), p (40 %), d (42.5 %). Fricatives and dental stops are visually ambiguous.
- Errors propagate: the TTS faithfully speaks whatever text Stage 1 predicts (e.g. "at" → "in").

---

## Installation

> TODO: fill in with your actual environment.

```bash
git clone <REPO_URL>
cd <REPO_NAME>
pip install -r requirements.txt
```

Main dependencies (from the method): PyTorch, MediaPipe, librosa (or equivalent, for mel/Griffin-Lim).

## Data preparation

1. Download GRID (video + word alignments) from [Zenodo](https://zenodo.org/records/3625687).
2. Extract mouth crops with MediaPipe, normalize to 100×200, cache as tensors.
3. Pre-compute log-mel spectrograms (22.05 kHz, 80 mels, hop 256).
4. Tokenize text character by character from the alignment files.

```bash
# TODO: replace with your real scripts
python preprocess.py --grid_root <PATH_TO_GRID> --out <CACHE_DIR>
```

## Usage

```bash
# Train stage 1 (lipreading)
python train_lipreading.py --data <CACHE_DIR>

# Train stage 2 (video-conditioned TTS)
python train_tts.py --data <CACHE_DIR>

# End-to-end inference: silent video -> wav
python infer.py --video <input.mp4> --out <output.wav>
```

Inference runs in one pass: MediaPipe frames → Stage 1 text → Stage 2 mel → Griffin-Lim waveform.

## Repository structure

> TODO: adapt to your actual layout.

```
.
├── preprocess.py
├── train_lipreading.py
├── train_tts.py
├── infer.py
├── models/
├── evaluation/
└── report_gen_ai.pdf
```

---

## Limitations and future work

- Greedy CTC decoding can over-collapse ambiguous segments.
- Griffin-Lim adds spectral distortion (metallic artifacts, smeared transients).
- Short function words ("at", "in") dominate residual errors.

Planned: beam-search CTC with language-model constraints, confidence-aware TTS conditioning, a learned neural vocoder, and joint end-to-end training of text and acoustic outputs.

## References

1. Assael et al., *LipNet: End-to-end sentence-level lipreading*, arXiv:1611.01599, 2016.
2. Chung & Zisserman, *Lip Reading in the Wild*, ACCV Workshops, 2016.
3. GRID Audio-Visual Speech Corpus, https://zenodo.org/records/3625687
4. Graves et al., *Connectionist Temporal Classification*, ICML 2006.
5. Shen et al., *Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions*, ICASSP 2018.
6. van den Oord et al., *WaveNet*, arXiv:1609.03499, 2016.
7. Griffin & Lim, *Signal estimation from modified short-time Fourier transform*, IEEE TASSP, 1984.

## License

> TODO: choose a license.
