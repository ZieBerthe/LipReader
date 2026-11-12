# Feasibility: lip-video-to-speech (visual encoder + diffusion/flow-matching mel generator) on GRID, trained on an Apple M5 MacBook (16 GB) with PyTorch MPS

Research date: 2026-11-12. Budget-limited session (about 16 tool calls), so several items are marked as **unverified** or as **prior knowledge**. Claims marked that way were not confirmed against a source in this session.

Project context (from the brief, not researched): speaker s1 is downloaded (1000 clips, 75 frames at 25 fps, about 3 s each, 100x200 grayscale mouth crops from MediaPipe). The audio track in the .mpg files is 44.1 kHz. The existing 1.15M-param 3D-CNN+BiLSTM CTC model takes about 2.85 s per batch of 4 on MPS (about 9.5 min/epoch over 700 clips), and batch 8 swaps memory. Ops already found unsupported on MPS: AdaptiveAvgPool3d, adaptive pooling with non-divisible sizes, and CTC loss (run on CPU).

---

## 1. GRID corpus facts and standard lip-to-speech splits/metrics

### Takeaway
GRID has 34 talkers with 1000 sentences each (34,000 total), and talker 21 has no video. Zenodo distributes it under CC BY 4.0 (about 16.2 GB). The standard lip-to-speech benchmark is "GRID-4S": speakers s1, s2, s4 and s29, each split 90/5/5 at random (3600/200/200 clips, about 3 h of training audio). Published systems reach roughly STOI 0.69–0.76, ESTOI 0.50–0.61 and PESQ 1.7–2.0 there. Speaker s1 alone (about 0.75 h) is enough for a single-speaker proof of concept. Matching the literature requires downloading s2, s4 and s29.

### Cited Findings
- **Speakers and size:** 34 talkers (18 male, 16 female) with 1,000 sentences each, 34,000 in total. — [Zenodo record 3625687](https://zenodo.org/records/3625687); [Sheffield GRID page](https://spandh.dcs.shef.ac.uk/gridcorpus/)
- **Missing video:** talker 21 has no video (the Sheffield page calls it a "technical oversight"). Only audio and alignments exist for that talker, and s21.zip is absent from Zenodo. — [Sheffield](https://spandh.dcs.shef.ac.uk/gridcorpus/); [Zenodo](https://zenodo.org/records/3625687)
- **Audio:** downsampled to 25 kHz and endpointed, with the raw 50 kHz versions also provided. Per talker, the 25 kHz audio is about 100 MB and the 50 kHz audio about 300 MB. On Zenodo, audio_25k.zip is 2.6 GB for all talkers. — [Sheffield](https://spandh.dcs.shef.ac.uk/gridcorpus/); [Zenodo](https://zenodo.org/records/3625687)
- **Video:** normal quality is 360x288 and high quality is 720x576, both .mpg. Normal video is about 480 MB per talker, and high video about 1.2 GB per part (two parts). The Zenodo per-speaker archives range from about 355 MB (s14) to about 492 MB (s4). — [Sheffield](https://spandh.dcs.shef.ac.uk/gridcorpus/); [Zenodo](https://zenodo.org/records/3625687)
- **Total download:** 16.2 GB on Zenodo. — [Zenodo](https://zenodo.org/records/3625687)
- **Word alignments:** alignments.zip is 20.1 MB on Zenodo (about 190 KB per talker on Sheffield). — [Zenodo](https://zenodo.org/records/3625687); [Sheffield](https://spandh.dcs.shef.ac.uk/gridcorpus/)
- **Licence:** Zenodo lists CC BY 4.0. The Sheffield page says only "freely available for research use." — [Zenodo](https://zenodo.org/records/3625687); [Sheffield](https://spandh.dcs.shef.ac.uk/gridcorpus/)
- **Example sentence:** "put red at G9 now." Neither page lists the full word lists in the fetched text. — [Zenodo](https://zenodo.org/records/3625687)
- **GRID-4S split:** two male speakers (s1, s2) and two female speakers (s4, s29), split per speaker 90% train, 5% validation, 5% test.
  - That gives 3,600 / 200 / 200 clips (about 2.98 / 0.17 / 0.17 h).
  - Audio is 16 kHz, mels are at 100 Hz, and video features are at 25 Hz.
  - The setup is "commonly used for lip-to-speech evaluation."
  - — [Niu et al., "On the Audio-visual Synchronization for Lip-to-Speech Synthesis", arXiv 2303.00502](https://arxiv.org/html/2303.00502)
- **Unseen-speaker split differs between tasks.** For lip *reading*, the Assael et al. split tests on speakers 1, 2, 20 and 22. Do not confuse it with GRID-4S. — [arXiv 2403.16071 (search snippet)](https://arxiv.org/pdf/2403.16071)
- **Lip2Wav GRID numbers:** STOI 0.731, ESTOI 0.535, PESQ 1.772 and WER 14.08% for 4 speakers. The search snippet does not say which ASR produced the WER. — [Lip2Wav, arXiv 2005.08209](https://arxiv.org/pdf/2005.08209)
- **GRID-4S state-of-the-art table** (Table 5 of 2303.00502):

  | Method | STOI | ESTOI | PESQ |
  |---|---|---|---|
  | E2E-V2AResNet | 0.627 | – | 2.030 |
  | Yadav et al. | 0.724 | 0.540 | 1.932 |
  | VCA-GAN | 0.724 | 0.609 | 2.008 |
  | Lip2Wav | 0.731 | 0.535 | 1.722 |
  | Kim et al. | 0.738 | 0.579 | 1.984 |
  | SLTS | 0.757 | 0.588 | 1.931 |

  Lip2Wav's PESQ is 1.722 here but 1.772 in the Lip2Wav snippet. Treat this as a transcription discrepancy between sources. — [arXiv 2303.00502](https://arxiv.org/html/2303.00502)
- **WER numbers from the same paper (Table 2):**

  | Model | STOI | ESTOI | PESQ | WER, Whisper-medium | WER, Kaldi ASR trained on GRID |
  |---|---|---|---|---|---|
  | VCA-GAN | 0.688 | 0.500 | 1.917 | 23.67% | 8.50% |
  | SLTS | 0.703 | 0.525 | 1.932 | 12.83% | 2.92% |

  — [arXiv 2303.00502](https://arxiv.org/html/2303.00502)
- **Reference compute in the literature:** a Conformer (S) model trained up to 50k iterations at batch size 32 took about 1 day per model on one RTX 2080 Ti. VCA-GAN was trained up to 70k iterations at batch size 24. — [arXiv 2303.00502](https://arxiv.org/html/2303.00502)
- **Speaker-independent results are far worse:** one vocoder-based method reports PESQ 1.23, ESTOI 0.227 and WER 51.6%. This comes from a secondary review page, so it is low confidence. — [SSI archive review](https://nao-ki-mura.pages.dev/paper/vocoder-based-speech-synthesis-from-silent-videos)
- **Diffusion lip-to-speech exists, but on other data.** LipVoicer (ICLR 2024) is a video-conditioned diffusion model with ASR classifier guidance. It is evaluated on LRS2/LRS3, not GRID. Code: github.com/yochaiye/LipVoicer. — [arXiv 2306.03258](https://arxiv.org/abs/2306.03258)

### Inferences
- s1 alone gives about 900 training clips (about 0.75 h), roughly a quarter of the GRID-4S training set. Reporting s1-only numbers is reasonable for a student project, but they are **not directly comparable** to the GRID-4S table. To be comparable, download s2, s4 and s29 (about 0.45 GB of normal-quality video each on Zenodo, plus audio) and use the 90/5/5 split.
- The user's current 700-clip training split (70%) differs from the 90/5/5 convention. Re-splitting per speaker at 90/5/5 is cheap and makes the numbers comparable.
- 2303.00502 trains at 16 kHz audio. Resampling the 44.1 kHz .mpg audio (or the 25/50 kHz official WAVs) to 16 kHz matches the literature, and the official WAVs are cleaner and endpointed.
- Target to aim for on GRID-4S: STOI of about 0.70 or more, ESTOI of about 0.50 or more, PESQ of about 1.9.

### Gaps
- **Not confirmed from a source this session (prior knowledge, unverified):**
  - The grammar has six slots: command (bin, lay, place, set), colour (blue, green, red, white), preposition (at, by, in, with), letter (A–Z minus W), digit (zero–nine) and adverb (again, now, please, soon). That gives 51 words.
  - Video is 25 fps.
  - The jasagrid.pdf on Zenodo (Cooke et al., JASA 2006) is the primary source for both and should be checked.
- The exact unseen-speaker lip-to-speech split (for example a 33-speaker train/test split as used in VCA-GAN/SVTS) could not be verified. The VCA-GAN PDF could not be text-extracted here.
- No SVTS GRID numbers were found.
- Which ASR Lip2Wav used for its 14.08% WER is unconfirmed.

---

## 2. Evaluation tooling (STOI/ESTOI, PESQ, WER, speaker similarity, MOS proxies) on Mac

### Takeaway
STOI/ESTOI and PESQ are standard and cheap, and they are what GRID papers report. Whisper-based WER is **unreliable on GRID**. Whisper-medium makes homophone errors on GRID's constrained vocabulary (single letters and digits), and a Kaldi ASR trained on GRID gives much lower and more discriminative WER. A student project should either constrain or normalize Whisper output to the GRID grammar, or train a small GRID audio ASR.

### Cited Findings
- 2303.00502 reports two WERs: w-WER from Whisper (medium) and k-WER from a Kaldi ASR trained on the GRID training set. They added the second "because Whisper produces homophone errors on GRID's constrained vocabulary." The two are very different (for example SLTS: 12.83% vs 2.92%). — [arXiv 2303.00502](https://arxiv.org/html/2303.00502)
- On GRID, WER is computed by comparing ASR transcripts of the synthesized speech against the provided ground-truth text. Ground-truth and vocoded speech are also evaluated as references. — ["Let There Be Sound", arXiv 2308.15256 (search snippet)](https://arxiv.org/pdf/2308.15256)
- WER = (S + D + I) / N. — [arXiv 2403.16071 (snippet)](https://arxiv.org/pdf/2403.16071)

### Inferences
- **pystoi (STOI/ESTOI):** pure NumPy/SciPy (prior knowledge), so it runs on macOS arm64 with no GPU. Note that STOI resamples internally to 10 kHz.
- **pesq (ITU-T P.862 wrapper):** a C extension (prior knowledge, unverified that arm64 wheels exist). It may compile from source on macOS with Xcode command-line tools. It needs 8 or 16 kHz input, which is another reason to work at 16 kHz.
- **Whisper on Mac** runs via openai-whisper (CPU/MPS), whisper.cpp (Metal) or mlx-whisper (prior knowledge, versions unverified). Evaluating about 200 test clips of 3 s each with medium or large-v3 should take minutes, not hours.
  - To make Whisper WER meaningful on GRID, map each transcript to the nearest valid GRID sentence. Normalise "see"/"sea" to "c", "two"/"to" to "2", and so on, or do slot-wise nearest-word matching.
  - Alternatively, use the `initial_prompt` with the vocabulary.
  - Always report the ground-truth-audio WER under the same ASR as the ceiling.
- **A cheaper and more defensible WER:** train a small audio-only GRID ASR (CTC over mels) on the 4-speaker training split. The user already has a working CTC pipeline. This replicates the paper's "k-WER" idea.
- **Speaker similarity** (for example ECAPA/WavLM-SV cosine) and **MOS proxies** (UTMOS, DNSMOS) are optional for GRID-4S papers, which mostly report STOI/ESTOI/PESQ/WER. All are PyTorch or ONNX inference and should run on CPU on a Mac (unverified).

### Gaps
- No primary source was checked for pesq/pystoi/UTMOS/DNSMOS macOS-arm64 wheel availability. The user should simply `pip install` them and test.
- No paper reporting Whisper-large-v3 WER on GRID ground-truth audio was found.

---

## 3. PyTorch MPS status (2.8–2.10) and MLX as an alternative; M5 GPU

### Takeaway
MPS training of a small transformer or conv model works, but it has rough edges:
- `scaled_dot_product_attention` on MPS is reportedly a naive matmul-softmax-matmul, not a flash kernel.
- bf16 behaviour depends on the macOS and PyTorch version.
- conv3d historically had gaps, and third-party Metal kernels exist for conv3d and STFT.

The M5's GPU "Neural Accelerators" (dedicated matmul units) give large gains in *MLX*: 3.3–4x prefill vs M4, which requires macOS 26.2 or later. No training benchmark was found, and no evidence was found that PyTorch MPS uses them yet.

### Cited Findings
- PyTorch 2.9.0 release notes: "PyTorch MPS is only supported on MacOS-14 or later." — [PyTorch 2.9.0 release notes (newreleases mirror)](https://newreleases.io/project/pypi/torch/release/2.9.0)
- **Attention:** a third-party package states that PyTorch's MPS backend routes `scaled_dot_product_attention` to a naive matmul-softmax-matmul graph, and offers a faster MPSGraph-based replacement. This is a claim by an interested party. — [mps-sdpa on PyPI](https://pypi.org/project/mps-sdpa/)
- **bf16 on MPS:**
  - A PyTorch PR enabled bfloat16 on macOS 14 by adding the MPSGraph enums. — [PR #119641 mirror](https://app.semanticdiff.com/gh/strisunshinewentingwang/pytorch/commit/4ee8aac4323c41cd26310217db9eb7a8a65b9035)
  - A DeepSeek-OCR commit reports bf16 working on MPS *without* autocast on macOS 26.0.1 with PyTorch 2.9.0, and disables autocast on MPS. — [HF commit](https://huggingface.co/deepseek-ai/DeepSeek-OCR/commit/6178e01bfd05ea92839d10bbd54358809387f976)
  - An ExecuTorch Whisper report says bf16 failed in decoder SDPA on macOS 26.1 because of stricter MPS dtype requirements. — [executorch-examples #115](https://github.com/meta-pytorch/executorch-examples/issues/115)
  - An older bf16 TypeError report exists (PyTorch 2.1.2). — [pytorch #141864](https://github.com/pytorch/pytorch/issues/141864)
- **STFT/ISTFT:** a third-party `mps-spectro` package provides MPS STFT/ISTFT via custom Metal kernels (`torch.fft.irfft` on MPS plus a fused overlap-add kernel). It requires PyTorch 2.10 or later, and its benchmarks are self-reported. No official MPS stft issue was found. — [mps-spectro](https://awesome.ecosyste.ms/projects/github.com%2Fssmall256%2Fmps-spectro)
- **conv3d:**
  - MPS historically lacked `aten::slow_conv3d_forward` (NotImplementedError reports). — [pytorch issue #77764 via redirect](https://github-redirect.dependabot.com/pytorch/pytorch/issues/77764)
  - A third-party `mps-conv3d` Metal kernel claims forward and backward support plus fp16. — [mps-conv3d on PyPI](https://pypi.org/project/mps-conv3d/)
  - Conv3d on the arm64 *CPU* was about 4x slower in a 2022 report. — [PyTorch forum](https://discuss.pytorch.org/t/slow-arm64-conv3d-on-apple-silicon-cpu/168925)
- **Matmul:** there are reports of MPS matmul/slicing performance problems. — [pytorch #122123](https://github.com/pytorch/pytorch/issues/122123)
- **M5 Neural Accelerators (Apple ML Research, 2025-11-19):**
  - The GPU gains "dedicated matrix-multiplication operations." MLX uses them via Metal 4 TensorOps / Metal Performance Primitives, and this **requires macOS 26.2 or later**.
  - M5 memory bandwidth is 153 GB/s vs 120 GB/s on M4.
  - Speedups vs M4 on 24 GB MacBook Pros: prefill (TTFT) 3.33–4.06x, token generation 1.19–1.27x, and FLUX-dev-4bit image generation more than 3.8x.
  - No training benchmarks and no TFLOPS figure are given.
  - — [Apple ML Research: Exploring LLMs with MLX and the Neural Accelerators in the M5 GPU](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)
- MacStories independently measured M5 prefill gains that exceed Apple's claims (a 16k-token prompt took 118 s on M4 and 38 s on M5), with generation about 1.5x faster. Testing was on a pre-release MLX. — [MacStories](https://www.macstories.net/stories/ipad-pro-m5-neural-benchmarks-mlx/)
- Early on, "full support for the Neural Accelerators isn't in MLX yet." — [Creative Strategies](https://creativestrategies.com/?p=10332)
- A newsletter claims 3.2x faster ViT training in MLX vs PyTorch MPS on M5, with no methodology. **Unverified.** — [businessanalytics substack](https://businessanalytics.substack.com/p/apple-m5-delivers-4x-ai-power-with)
- A third-party "PyTorch 2.15" MPS SDPA claim surfaced in search. The version number is implausible as of this writing, so it should be **ignored**. — [hyper.ai](https://hyper.ai/en/stories/2e40320aeb5e29e969bbcf57d24ad552)

### Inferences
- **For a 20–50M-param 1D transformer/conformer over mel frames:**
  - Plain fp32 on MPS is the safest choice.
  - Try bf16 weights and activations directly (without autocast) only after a correctness check on macOS 26.x.
  - Avoid relying on `torch.compile` for MPS. Its status was not verified, and it should be treated as experimental.
- **Keep STFT/mel extraction off the training loop:** precompute log-mels on CPU (torchaudio/librosa) once and store them. This sidesteps any MPS stft/complex gaps.
- **Vocoder:** use a pretrained HiFi-GAN or BigVGAN at inference only, on CPU or MPS. No complex ops are needed during training.
- The user's existing 3D-CNN is slow on MPS (2.85 s per batch of 4). That is consistent with conv3d being a weak spot on MPS, and possibly with on-the-fly video decoding in the dataloader. **Pre-extracting visual features once removes conv3d from the training loop entirely.**
- **MLX is the only path to use M5 Neural Accelerators today**, as far as found. Porting a small flow-matching transformer to MLX is feasible (mlx-examples has transformer and diffusion examples, per prior knowledge), but it costs engineering time. For a student project, starting with PyTorch MPS and switching only if throughput is unacceptable is the lower-risk option.
- The M5 base chip's bandwidth (153 GB/s) is about 6x below a T4's ~320 GB/s and far below an A100. For small-batch training dominated by memory traffic and kernel launch overhead, expect a laptop GPU to be several times slower than a T4. This is an inference, not measured.

### Gaps
- No primary PyTorch 2.8/2.9/2.10 release-note text on MPS SDPA, autocast or torch.compile was retrieved.
- No measured tokens/s or TFLOPS for PyTorch-MPS training on M4/M5 was found. M5 base GPU TFLOPS is not published in the Apple source.
- Whether PyTorch MPS (MPSGraph) uses the M5 Neural Accelerators is unknown. No source was found.

---

## 4. Pretrained visual speech encoders on Mac and feature caching

### Takeaway
Run the pretrained encoder **once**, offline, to cache per-frame features. Training then never touches the encoder or conv3d. Auto-AVSR is the easiest choice (standalone PyTorch repo with a feature-extraction tutorial and slimmed dependencies). AV-HuBERT depends on legacy fairseq, which is painful on modern Python and Apple Silicon. If needed, run AV-HuBERT extraction once on Colab/Kaggle and download the features. Caching is tiny: well under 1 GB even for 4 speakers.

### Cited Findings
- **Auto-AVSR:**
  - The repository includes a tutorial for using the models to "crop mouth ROIs or extract visual speech features."
  - It lists a 2025-01-06 update that reduced package dependencies.
  - Visual-only LRS3 checkpoints are available. One model-zoo entry is about 891 MB at 19.1% WER, and the READMEs list different checkpoints.
  - — [mpc001/auto_avsr](https://github.com/mpc001/auto_avsr); [AV-LLM/AVSR mirror](https://github.com/AV-LLM/AVSR); [mpc001 VSR for Multiple Languages](https://www.github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages)
- The VSR repos expect RetinaFace or MediaPipe tracking for mouth ROI preprocessing. — [mpc001 VSR repo](https://www.github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages)
- **AV-HuBERT:** an issue reports `ImportError: cannot import name 'metrics' from 'fairseq'` (a bundled fairseq import problem, on Linux). — [av_hubert issue #70](https://github.com/facebookresearch/av_hubert/issues/70)
- **fairseq on macOS:** fairseq 0.10.2 failed to install with Python 3.9 on macOS. MacPorts carries a py310-fairseq port. fairseq2 (a different project, not usable for AV-HuBERT) installs via pip on Apple Silicon. — [fairseq #3535](https://github.com/facebookresearch/fairseq/issues/3535); [MacPorts py310-fairseq](https://ports.macports.org/port/py310-fairseq/); [fairseq2 install docs](https://facebookresearch.github.io/fairseq2/stable/getting_started/installation/index.html)

### Inferences
- **Preprocessing mismatch (important):** Auto-AVSR and AV-HuBERT were trained on **88x88 (or 96x96) grayscale mouth ROIs, cropped and aligned to a reference face**, at 25 fps (prior knowledge, unverified here). The user's 100x200 MediaPipe crops will need re-cropping or resizing to the encoder's expected ROI and normalisation, or features will be degraded. GRID is 25 fps, which matches.
- **Storage estimate for cached features**, with T = 75 frames per clip:

  | Data | Calculation | Size |
  |---|---|---|
  | s1, 768-d fp16 | 1000 × 75 × 768 × 2 B | ≈ 115 MB |
  | s1, 1024-d fp16 | 1000 × 75 × 1024 × 2 B | ≈ 154 MB |
  | GRID-4S (4000 clips), 768-d fp16 | | ≈ 0.46 GB |
  | GRID-4S (4000 clips), 1024-d fp16 | | ≈ 0.61 GB |
  | Target log-mels, 4000 clips, fp32 (16 kHz, hop 160, 80 bins) | 300 frames × 80 bins × 4 B per clip | ≈ 0.38 GB |

  Everything fits in RAM, so the dataloader becomes trivial.
- **Encoder extraction time:** a ~100–300M-param visual frontend + transformer forward pass over 4000 three-second clips is a one-off job. It should take roughly tens of minutes to a couple of hours on MPS or CPU, if the conv3d frontend runs. If MPS conv3d is a problem, run this on CPU overnight or on a free Colab/Kaggle T4. This is an estimate.
- **Fine-tuning vs freezing:** with frozen cached features, the generator learns only the mapping to audio. That is the realistic option on 16 GB. Fine-tuning the encoder end-to-end on a Mac is not realistic.
- **Simpler alternative:** use the user's own trained lip reader's BiLSTM hidden states as features (already GRID-specific). It is weaker in principle but has zero setup cost. This is a good ablation.

### Gaps
- RAVEn/BRAVEn repos were not checked for Mac usability. The assumption is plain PyTorch, but this is unverified.
- No report of AV-HuBERT successfully running on Apple Silicon was found. No Hugging Face transformers port of AV-HuBERT was verified in this session.
- The exact output dimension and frame rate of Auto-AVSR's visual features (about 768-d at 25 Hz is assumed) is unverified.

---

## 5. Memory/compute budget for a 20–50M-param flow-matching mel generator on 16 GB

### Takeaway
With precomputed visual features and mels, a 30M-param conditional flow-matching transformer on about 300 mel frames fits comfortably in 16 GB at batch 16–32 (the estimate is about 1–5 GB). Wall-clock is the real constraint. A back-of-envelope estimate is about 0.5–2 s/step at batch 32 on MPS, so 100k steps is about 14–55 hours, which can be split across nights. That is about 2–5x slower than the literature's "1 day on a 2080 Ti for 50k steps." Flow matching converges in fewer steps than DDPM-style diffusion is commonly claimed to, so a budget of 30k–100k steps is reasonable. All numbers are estimates, not measurements.

### Cited Findings
- **Literature reference point:** 50k iterations at batch 32 took about 1 day per model on an RTX 2080 Ti for a GRID-4S Conformer-S lip-to-speech model. — [arXiv 2303.00502](https://arxiv.org/html/2303.00502)
- **M5 memory bandwidth:** 153 GB/s. — [Apple ML Research](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)
- **Observed baseline (from the user's brief):** 1.15M-param 3D-CNN+BiLSTM at 2.85 s per batch of 4 on MPS. This is dominated by conv3d over 75×100×200 video, not by parameter count.

### Inferences
These are estimates, with the arithmetic shown so they can be checked.
- **Sequence length:**
  - At 16 kHz with hop 160, 3 s is 300 mel frames.
  - At 22.05 kHz with hop 256, it is about 258 frames.
  - Visual features at 25 Hz (75 frames) are upsampled by 4x to 100 Hz, or cross-attended.
- **Training compute:** about 6 × params × tokens.

  | | Calculation | FLOPs per step |
  |---|---|---|
  | 30M params, T = 300, batch 32 | 6 × 30e6 × 300 × 32 | ≈ 1.7e12 |
  | 50M params, same | | ≈ 2.9e12 |

  Attention adds a few percent at T = 300. If MPS reaches 1–3 TFLOPS sustained in fp32 (assumed; the M5 base GPU's peak is not documented in sources found), that is about 0.6–3 s/step.

  | Steps | Wall-clock at ~0.6–3 s/step |
  |---|---|
  | 100k | ≈ 17–80 h |
  | 200k | ≈ 1.5–7 days |

  Kernel-launch overhead on small models can make MPS slower than this FLOP estimate, so benchmark 100 steps before committing.
- **Memory for 30M params, fp32 with AdamW:**

  | Item | Size |
  |---|---|
  | Weights, gradients and two Adam moments | 30M × 16 B ≈ 0.5 GB |
  | Activations (12 layers, d = 512, T = 300, B = 32, ~20 floats per token per layer stored, fp32) | 32 × 300 × 512 × 20 × 12 × 4 B ≈ 3.8 GB |
  | Total | ≈ 4–5 GB |

  This fits alongside macOS within 16 GB unified memory. The current "batch 8 swaps" problem comes from raw-video 3D-conv activations (75×100×200 per clip) and does not apply here.
  - Use B = 16 if the model is d = 768 or larger.
  - Gradient checkpointing can halve activations if needed.
- **Model sizing:** for about 3 h of single-domain GRID-4S data (or 0.75 h for s1 alone), 20–30M params is plenty. A larger model mostly overfits.
- **Recipe that keeps compute small:**
  1. Precompute log-mels and frozen visual features.
  2. Train a 1D transformer or conformer with conditional flow matching (OT-CFM / rectified flow) predicting an 80-bin mel at 100 Hz.
  3. Add a speaker embedding (4 speakers).
  4. Sample with 10–32 Euler/midpoint steps.
  5. Vocode with a pretrained universal HiFi-GAN or BigVGAN at 16 or 22.05 kHz, with no vocoder training.
- **Even cheaper baseline:** a non-generative regression model (L1 on mels) with the same encoder. It trains in a fraction of the time and gives the flow-matching model a comparison point. Most published GRID numbers come from regression/GAN models.
- **Expected outcome:** STOI/ESTOI often favour regression models, while generative models improve naturalness. A flow-matching model may not beat VCA-GAN on STOI. This is plausible but unverified.

### Gaps
- No measured PyTorch-MPS training throughput for a comparable model on M4/M5 was found. The user must benchmark it.
- No published M5 GPU TFLOPS figure (fp32/fp16) was found.

---

## 6. Cloud fallback (Colab, Kaggle free GPUs)

### Takeaway
Kaggle's free tier is commonly reported as about 30 GPU-hours/week (P100 16 GB or 2×T4) with 12-hour sessions. Colab free gives a T4 without a published quota. Either is enough for one-off feature extraction (for example running AV-HuBERT under Linux) and for a few long training runs. The Mac remains fine for development, debugging, evaluation and small runs.

### Cited Findings
- **Kaggle sessions:** 12 h for CPU/GPU and 9 h for TPU, per Kaggle docs as read in September 2026.
  - The weekly GPU quota is not stated in the docs. About 30 h/week is the commonly reported figure, and it "floats."
  - The hardware is a P100 (16 GB) or 2×T4.
  - Linking Colab Pro/Pro+ reportedly adds 15/30 h/week on Kaggle.
  - These details come from third-party aggregations. — [gpuperhour: Free Cloud GPUs in 2026](https://gpuperhour.com/blog/free-cloud-gpus-and-credits); [Awesome Free Compute 2026](https://amitpandeytiktok.github.io/awesome-free-compute-2026/compute.html)
- **Colab free:** typically a T4, but Google names no GPU model, weekly quota or idle timeout, and limits fluctuate. Sessions run about 12 h maximum with about a 90-minute idle disconnect.
  - Google's September 2026 AI plan update adds priority accelerator access for paid plans, not more free quota.
  - — [hivenet Colab guide](https://www.hivenet.com/post/google-colaboratory-gpu-complete-guide-to-free-cloud-gpu-access-and-limitations); [pasqualepillitteri.it on Colab + Google AI plans](https://pasqualepillitteri.it/en/news/17594/colab-google-ai-plans-premium-gpu)

### Inferences
- A T4 (about 65 TFLOPS fp16 tensor-core peak and about 320 GB/s, per prior knowledge and unverified) with AMP should run the 30M flow-matching model several times faster than M5 MPS fp32.
- 30 Kaggle hours/week would cover about 100k steps per week, given 12-hour session limits and checkpoint/resume.
- The cached-feature design makes migration trivial: upload about 1 GB of .npy/.pt files as a Kaggle Dataset.

### Gaps
- Official current Kaggle/Colab quotas were not verified on kaggle.com or colab.research.google.com.

---

## Realistic plan scope (synthesis for the report writer; inferences)
1. **Data (1–2 days).** Download s2, s4 and s29 (normal video and 25 kHz audio from Zenodo, CC BY 4.0) to get GRID-4S. Resample audio to 16 kHz. Split 90/5/5 per speaker. Precompute 80-bin log-mels (hop 10 ms).
2. **Features (1–2 days).** Re-crop mouth ROIs to the Auto-AVSR convention. Extract frozen Auto-AVSR visual features once (MPS, CPU, or a Colab/Kaggle T4 if conv3d is an issue) and cache them in fp16 (about 0.5 GB). Optionally, also cache the user's own lip-reader BiLSTM states as a baseline feature.
3. **Baselines (2–3 days).** Train an L1 mel-regression conformer (about 10–20M params) and vocode with pretrained HiFi-GAN or BigVGAN. Set up evaluation:
   - pystoi and pesq
   - Whisper WER with GRID-grammar normalisation, plus WER on ground-truth audio
   - optionally, a small GRID audio-CTC ASR for a "k-WER"-style metric
4. **Flow matching (1–2 weeks).**
   - Model: a conditional flow-matching transformer of about 20–30M params, batch 16–32, fp32 on MPS.
   - Budget: 30k–100k steps (about 0.5–3 days of wall-clock per run on the Mac, estimated).
   - Move long runs to Kaggle if a benchmark shows more than about 1.5 s/step.
   - Ablations: number of ODE steps, classifier-free guidance on the visual condition, and speaker embedding on/off.
5. **Out of scope on 16 GB:** end-to-end encoder fine-tuning, unseen-speaker generalisation (it needs all 33 speakers' video, about 16 GB of download and much more compute), and training a vocoder.
