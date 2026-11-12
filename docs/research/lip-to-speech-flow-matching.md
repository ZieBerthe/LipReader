# Give the lip reader a flow-matching voice

Replacing the GRID cascade with a VLA-style lip-to-speech model is feasible on the M5 MacBook, and it is the direction the field has already taken. My recommendation has four parts. First, a **frozen, speech-specific visual encoder** whose features are cached once to disk. Second, a small **discrete content head** that predicts self-supervised speech units, which plays the role pi0.5's FAST tokens play. Third, a **~20M-parameter Matcha-style conditional flow-matching decoder** that generates an 80-bin mel from frame-aligned visual features in about 10 Euler steps. Fourth, a **pretrained BigVGAN-v2 or HiFi-GAN vocoder** in place of Griffin-Lim. Every strong lip-to-speech system of 2025–26 (V2SFlow, From Faces to Voices, SLD-L2S, HiCoDiT) already follows this pattern: a pretrained lip encoder feeding a flow-matching or diffusion generator. A general-purpose VLM is the wrong backbone, because lip reading needs 25 fps mouth detail and VLMs sample video at 1–2 fps. The new model will clearly win on audio quality, lip sync and prosody. It will not make homophenes go away. In GRID's letter slot, b/p, d/t and f/v look identical and are drawn at random, so no model can recover them reliably from the lips. The realistic goal is to turn a mumbled blend into a crisp, honest guess and to measure that error explicitly. The two practical blockers are data and compute. Only speaker s1 is downloaded, and a 50k–150k-step run is a multi-night job on MPS that needs benchmarking first. Neither is serious.

## Every strong 2025–26 lip-to-speech model is already VLA-shaped

The field moved through four stages. First came regression and GAN models on GRID (Lip2Wav, VCA-GAN, SVTS). Then came models with intermediate linguistic targets such as text, CTC or self-supervised speech units (Intelligible-L2S, ReVISE, LTBS). Diffusion arrived in 2023–24 (LipVoicer, DiffV2S), and flow matching or codec-latent diffusion in 2025–26. In the current generation, **frozen AV-HuBERT-Large lip features are upsampled to the acoustic frame rate and condition a flow-matching generator over mel or codec latents**. V2SFlow uses a rectified-flow Transformer over 80-bin mel with HiFi-GAN ([V2SFlow](https://arxiv.org/abs/2411.19486)). From Faces to Voices (FTV) uses an OT-CFM U-Net decoder ([FTV](https://arxiv.org/html/2503.16956v1)). SLD-L2S generates continuous X-Codec latents in 10 function evaluations ([SLD-L2S](https://arxiv.org/html/2602.11477)). That is structurally pi0: a pretrained perception model, then a small expert that turns noise into a continuous output chunk.

The mapping is close. pi0 attaches a **~300M-parameter action expert** to a PaliGemma VLM. The two share only self-attention, under a blockwise-causal mask. The expert flow-matches a 50-step action chunk and integrates it in **10 Euler steps**, with timesteps drawn from a Beta schedule that emphasises noise ([pi0](https://arxiv.org/html/2410.24164v4)). pi0.5's Knowledge Insulation adds a stop-gradient between the expert and the backbone and keeps a discrete FAST-token loss on the backbone. Physical Intelligence notes that the stop-gradient alone was not enough ([Physical Intelligence](https://www.pi.website/research/knowledge_insulation)).

| pi0 component | Lip-to-speech counterpart |
|---|---|
| PaliGemma image+text prefix tokens | 75 × D cached lip-encoder features (25 fps) |
| Proprioceptive state block | Speaker-ID embedding (constant, cacheable) |
| Noisy action chunk (H=50) | Noisy mel chunk (~258 × 80 for a 3 s clip) |
| 10 Euler steps, Beta τ-schedule | ~10 Euler steps, logit-normal t-schedule |
| Stop-gradient + FAST auxiliary tokens | Frozen encoder + HuBERT-unit / CTC auxiliary head |

The research turned up no paper that explicitly frames video-to-speech as a VLA analogue. That gives the project a small but real framing contribution.

The analogy breaks at one point: **the "V" should not be a general VLM**. Omni models sample video at 1–2 fps, which HumanOmni-Speaker calls "often ineffective" for visual speech recognition. It finds that about 16 fps is the minimum for basic lip reading and 25 fps is best ([HumanOmni-Speaker](https://arxiv.org/html/2603.21664)). Even LLM-decoder lip readers gain mainly from lexical context, not better visual understanding ([Jain & Harte, ICASSP 2026](https://arxiv.org/abs/2509.14880)). GRID's grammar is fixed and its letter slot is random, so lexical context is worth almost nothing there. The visual side therefore needs a speech-specific encoder trained on thousands of hours of mouth crops. A CLIP- or SigLIP-style encoder is worth running only as an ablation ("generic vs speech features").

## Recommended build: frozen lip encoder, unit head, 20M-parameter flow expert

The recommended model has five components, listed below in order of data flow. Each choice is the lowest-friction one supported by evidence from the notes.

| Stage | Choice | Why |
|---|---|---|
| Visual backbone (frozen, run once) | **Auto-AVSR visual encoder**, with USR 2.0 / BRAVEn Base+ as the self-supervised alternative; your own CTC BiLSTM states as the zero-cost baseline | Apache-2.0 code, slim dependencies, documented feature-extraction tutorial ([auto_avsr](https://github.com/mpc001/auto_avsr)); generalised best zero-shot to GRID among tested models ([AVSRBench](https://arxiv.org/html/2609.10366v1)) |
| Adapter + content head | 2–4-layer Transformer/Conformer (d≈256) over 75 frames; auxiliary cross-entropy on HuBERT k-means units, plus optional character CTC | Units supply phonetic content without words; used by FTV, V2SFlow and LTBS |
| Speech expert | Matcha-style OT-CFM 1D U-Net (~18M) or 8-layer DiT (d=512, adaLN-Zero); conditioning interpolated to mel length and concatenated | Proven aligned-conditioning recipes ([Matcha-TTS](https://arxiv.org/html/2309.03199v2), [V2SFlow](https://arxiv.org/html/2411.19486v2)) |
| Guidance and sampling | Conditioning dropout p=0.1, CFG γ≈2, logit-normal t, 8–10 Euler steps | V2SFlow found γ=2 best; FTV's 10 steps match 1000 |
| Vocoder | **BigVGAN-v2 22 kHz 80-band fmax-8k** (or HiFi-GAN V1) | Your TTS already uses this exact mel config |

**Backbone.** Auto-AVSR is the pragmatic first choice. It is plain PyTorch with a feature-extraction tutorial, and its supervised CTC/attention training makes its features already "phonetically shaped." In a September 2026 zero-shot benchmark on all of GRID, it scored 66.5% visual-only WER, against 83.8% for AV-HuBERT ([AVSRBench](https://arxiv.org/html/2609.10366v1)). Those numbers mostly reflect language-decoder mismatch ("the model hallucinates phrases to satisfy its language decoder"), not weak features. The decoder is discarded here anyway.

USR 2.0 is the closest fit to "learn rich representations instead of words." It is MIT-licensed, ships self-supervised encoder-only checkpoints from Base to Huge, and uses MediaPipe landmarks by default, as your pipeline does ([usr2](https://github.com/ahaliassos/usr2)). AV-HuBERT is the encoder the literature uses most. It is the worst fit for this machine: its repo was archived in September 2024, it needs Python 3.8 and a pinned fairseq fork, and its license is non-commercial ([av_hubert](https://github.com/facebookresearch/av_hubert)). If you want it, extract its features once on Colab and download them.

All of these encoders expect landmark-aligned 88–96 px grayscale mouth ROIs at 25 fps ([BRAVEn](https://arxiv.org/html/2404.02098v1)). Your 100×200 crops must be re-cropped with each encoder's own script, or the frozen features will degrade.

**Content head.** This is the piece that answers "learn representations of the words instead of predicting the words." Do not force a transcript as the interface. Do give the encoder a discrete, frame-level target that captures *what sound* is being made: k-means clusters of HuBERT features computed from the GRID audio. LTBS predicts HuBERT layer-12 k-means-200 units with cross-entropy and needs no text. It argues that plain L1/L2 reconstruction gives "blurry and over-smoothed synthetic speech" ([LTBS](https://ar5iv.labs.arxiv.org/html/2308.15256)). FTV predicts 1,000 units and loses clearly when its content hierarchy is removed: UTMOS falls from 4.031 to 3.737 and WER rises from 30.45 to 33.64 ([FTV](https://arxiv.org/html/2503.16956v1)). The flow decoder is then conditioned on the adapter's continuous features plus the unit embeddings or soft posteriors. A character-CTC head, as in AlignDiT's auxiliary CTC ([AlignDiT](https://arxiv.org/html/2504.20629v2)), is a cheap extra that also lets the model still emit text.

**Speech expert.** Matcha-TTS's decoder is the closest off-the-shelf design. It is an **18.2M-parameter** OT-CFM 1D U-Net that maps a mel-length conditioning sequence `mu` plus noise to a mel. At 4 steps it keeps 2.15% WER and MOS 3.77, against 2.09% / 3.84 at 10 steps. It used 4.8 GiB at batch 32 in fp16 ([Matcha-TTS](https://arxiv.org/html/2309.03199v2)). Replace its text encoder and alignment search with "interpolate 75 visual frames to T_mel." Lip video is already time-aligned, so this is simpler than text-to-speech. Keep Matcha's prior loss ‖mu − mel‖: it gives you a free regression baseline inside the same model.

V2SFlow's decoder is the proven lip-to-speech variant: 8 layers, d=512, adaLN-Zero, logit-normal timesteps and channel-concatenated, interpolated conditions. **Classifier-free guidance matters** there. At γ=2, UTMOS is 3.780, against 3.365 without guidance ([V2SFlow](https://arxiv.org/html/2411.19486v2)). Few steps are enough: FTV reports UTMOS 4.031 at 10 steps versus 3.993 at 1000 ([FTV](https://arxiv.org/html/2503.16956v1)). A literal pi0-style variant is worth one ablation: joint attention over [visual tokens; noisy mel tokens] under a blockwise mask. With synchronous video, frame-aligned concatenation is cheaper and more data-efficient, so it should be the main system.

**Vocoder.** The existing TTS already computes mels at 22.05 kHz, n_fft 1024, hop 256, 80 bins and fmax 8000 (`tts/prep.py`). That is exactly the HiFi-GAN V1 configuration ([HiFi-GAN config](https://raw.githubusercontent.com/jik876/hifi-gan/master/config_v1.json)) and NVIDIA's `bigvgan_v2_22khz_80band_fmax8k_256x` (112M parameters, CUDA kernel off by default) ([BigVGAN](https://github.com/NVIDIA/BigVGAN)). Neither has an STFT in its forward pass, so both run on MPS. Avoid Vocos as the first choice, because it ends in an iSTFT head ([Vocos config](https://huggingface.co/charactr/vocos-mel-24khz/raw/main/config.yaml)).

The one trap is the mel function itself. Log base, Slaney vs HTK filters and the clamp floor must match the vocoder's `meldataset.py` exactly. Before training anything, check that a ground-truth mel passed through the vocoder sounds right.

**Speaker identity.** For s1 alone, no speaker conditioning is needed. For the four-speaker GRID-4S setup, a learned speaker-ID table injected by adaLN is enough. LTBS shows that removing such a table raises WER from 34.71 to 42.33 on Lip2Wav ([LTBS](https://ar5iv.labs.arxiv.org/html/2308.15256)).

## Homophenes become a sampling decision, but they do not disappear

Homophenes are the current model's main failure mode, and the generative model does not remove the underlying information gap. Roughly half of English words have a visually identical counterpart ([Sensors 2021](https://www.mdpi.com/1424-8220/21/23/7890/htm)). With small viseme inventories, a language model can no longer correct them ([arXiv 1909.07147](https://arxiv.org/pdf/1909.07147)).

GRID is the worst case for context. Its six-slot grammar (command, colour, preposition, letter A–Z without W, digit, adverb) draws the letter independently of the rest of the sentence ([LipNet](https://arxiv.org/pdf/1611.01599)). In that slot, b/p, d/t and f/v differ only in voicing, which is invisible on the lips. **No lip reader, text pipeline or flow model can reliably recover them**, short of learning subtle cues such as closure duration. The current model's residual errors probably sit close to this floor rather than reflecting a weak network. That is an inference: no study in the notes measures GRID's visually indistinguishable fraction.

The generator changes *how* the error sounds. The two pipelines handle ambiguity differently:

| Pipeline | What it does at an ambiguous frame | Resulting audio |
|---|---|---|
| Current cascade | Commits to one character | A cleanly spoken wrong letter, with the wrong timing |
| Regression (L1) model | Averages the possible sounds | A blurred, in-between sound; LTBS's "over-smoothed" speech |
| Unguided diffusion | Under-articulates | "Smooth" mels that sound natural but are hard to understand |

The unguided-diffusion row comes from LipVoicer. Without guidance, its spectrograms showed exactly this behaviour, and LRS3 WER rose from **21.4% to 86.2%** when ASR guidance was switched off ([LipVoicer](https://arxiv.org/html/2306.03258)).

A well-trained flow model should, in principle, commit to one mode. Deterministic few-step sampling and strong CFG can still drift toward averages where the conditioning separates the classes poorly. No paper has measured "commit vs average" on homophenes, so this has to be tested rather than assumed.

That is why the recommended design keeps a **discrete content bottleneck**. Predicting HuBERT units, or characters through CTC, turns the ambiguity into an explicit categorical posterior, such as 50/50 over the units for /b/ and /p/. You can then decide what to do with it: take the argmax, sample it, or report it. It also makes homophene errors measurable as per-slot letter accuracy.

Two further levers exist, and neither is worth much on GRID. One is LipVoicer-style ASR or text guidance at sampling time. It is the strongest single intelligibility lever on LRS3, and LipVoicer even publishes GRID settings: CFG 2.6, ASR guidance 0.7 from step 300 of 400 ([LipVoicer](https://arxiv.org/html/2306.03258v2)). On GRID, though, your lip reader is already near-perfect except on homophenes, so guidance would mostly enforce the same wrong letter more loudly. The other lever is a larger pretrained encoder. Encoders trained on about 3,000 hours ([BRAVEn](https://arxiv.org/html/2404.02098v1)) might pick up voicing-timing cues that a 1.15M-parameter model trained on 700 clips cannot. That is the most interesting open question the project could answer: compare per-letter accuracy on b/p/d/t/f/v across the encoder ablations.

## Feasible on the M5 once raw video leaves the training loop

The current 2.85 s per batch of 4, and the swapping at batch 8, come from running conv3d over raw 75×100×200 video every step. The recommended design runs the encoder **once**, offline, and trains only on cached tensors. At 1024-dim fp16, s1's features take about 154 MB, and all of GRID-4S about 0.6 GB. Mels for 4,000 clips take about 0.4 GB. Everything fits in RAM, and MPS gaps such as AdaptiveAvgPool3d drop out of the training loop. If the encoder's conv3d front-end will not run on MPS, extract on CPU overnight or on a free Kaggle/Colab GPU. Kaggle's free tier is commonly reported at about 30 GPU-hours a week in 12-hour sessions ([gpuperhour](https://gpuperhour.com/blog/free-cloud-gpus-and-credits)). Compute mels on CPU during preprocessing, as your TTS already does, so no STFT runs on MPS. If you keep a CTC auxiliary head, its loss will still run on CPU, as now.

The training cost is moderate, but it is an estimate. Published references used much harder data. Matcha trained for 500k updates on two RTX 3090s ([Matcha-TTS](https://arxiv.org/html/2309.03199v2)). SLD-L2S trained for 150k iterations in about 5 days on one H100, using 439 h of LRS3 ([SLD-L2S](https://arxiv.org/html/2602.11477)). On GRID-4S, a Conformer lip-to-speech model trained 50k iterations at batch 32 in about a day on an RTX 2080 Ti ([SLTS](https://arxiv.org/html/2303.00502)).

For a 20–30M-parameter model at batch 16–32 with about 258 mel frames per clip, the arithmetic is as follows. Memory comes to roughly 4–5 GB in fp32 with AdamW. Speed is roughly **0.3–3 s per step**, so 50k–150k steps means one to several nights. Nobody has published a measured MPS training throughput for a comparable model, so **benchmark 200 steps before committing**. Move long runs to Kaggle if a step takes more than about 1.5 s. Train in fp32 and in eager mode: bf16 behaviour on MPS varies across macOS and PyTorch versions, and torch.compile on MPS is unproven. The M5's GPU Neural Accelerators are currently reachable only through MLX on macOS 26.2+ ([Apple ML Research](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)). An MLX port is a later optimisation, not a prerequisite.

Data is the bigger gap. Speaker s1 gives about 0.75 h of audio, which is enough for a single-speaker proof of concept. Standard GRID lip-to-speech results, however, use **GRID-4S**: speakers s1, s2, s4 and s29, split 90/5/5 per speaker into 3,600/200/200 clips ([SLTS](https://arxiv.org/html/2303.00502)). The corpus is CC BY 4.0 on Zenodo, and each extra speaker's normal-quality video is about 0.45 GB ([Zenodo](https://zenodo.org/records/3625687)). Downloading three more speakers and re-splitting at 90/5/5 instead of 700/300 makes your numbers comparable to the published table.

The targets on that table are STOI ≈0.72–0.76, ESTOI ≈0.54–0.61 and PESQ ≈1.9–2.0 (VCA-GAN 0.724 / 0.609 / 2.008; SLTS 0.757 / 0.588 / 1.931) ([SLTS](https://arxiv.org/html/2303.00502)). No diffusion or flow-matching paper reports these GRID metrics, so your result would be among the first.

Evaluation needs care. Whisper makes homophone errors on GRID's letters and digits: SLTS scored 12.83% WER with Whisper-medium but 2.92% with a GRID-trained Kaldi ASR ([SLTS](https://arxiv.org/html/2303.00502)). Train a small audio-only CTC ASR on GRID mels, which your existing CTC code nearly is already, or snap Whisper output to the GRID grammar. Always report the ground-truth-audio WER under the same ASR. STOI is offset-sensitive: in SLTS's re-run, an alignment search raised VCA-GAN's STOI from 0.688 to 0.732.

The work can be staged so that every step yields something usable. First, **swap Griffin-Lim for BigVGAN on the existing TTS**. It is a few hours of work and probably the largest single audible improvement. Then download the extra speakers, re-crop the mouths and cache encoder features and mels (2–4 days). Next, train the regression baseline through Matcha's prior loss (2–3 days). Finally, add flow matching, CFG and the unit head, and run the encoder ablations (1–2 weeks). End-to-end encoder fine-tuning, unseen-speaker generalisation and vocoder training are out of scope on 16 GB.

## What the new model wins and loses against the current cascade

The current pipeline is the textbook cascade failure, with a weak vocoder added on top. Two published comparisons quantify the cascade's weaknesses:

| Comparison | Cascade (lip reader + TTS) | Video-aligned model |
|---|---|---|
| Hegde et al., LRS3: LSE-C (lip-sync confidence) | 1.77 | 7.89 |
| Hegde et al., LRS3: human sync MOS | 1.01 | 3.82 |
| Hegde et al., LRS3: intelligibility MOS | 3.61 | 3.49 |
| WYS, LRS2: WER, XTTS-v2 / CosyVoice cascade vs WYS | 31.4 / 29.4 | 20.1 |
| WYS, LRS2: LSE-C, same comparison | 3.12 / 3.71 | 7.50 |

Sources: [Hegde et al.](https://ar5iv.labs.arxiv.org/html/2403.01087); [WYS](https://arxiv.org/html/2610.01012v1).

Hegde's cascade slightly beat the video-aligned model on intelligibility MOS. Its authors still conclude that a cascade "cannot infer the speed, prosody, and accents of speakers just from the text input." On constrained GRID, a direct model already matches a lip reader's content accuracy: VCA-GAN's generated speech scores 4.8% WER against LipNet's 5.6% ([VCA-GAN](https://ar5iv.labs.arxiv.org/html/2204.01726)). So going direct costs little in intelligibility.

| Dimension | Current cascade | Proposed lip-to-speech flow model |
|---|---|---|
| Audio quality | Griffin-Lim on an L1-regressed mel: buzzy, over-smoothed | Flow-matched mel + neural vocoder; naturalness near ground truth in the literature (V2SFlow MOS 4.60 vs GT 4.42) |
| Timing and lip sync | Duration guessed from text | Inherited frame-by-frame from video |
| Prosody / emphasis | Lost at the text bottleneck | Carried through continuous visual features |
| Homophenes | Hard wrong commitment, spoken clearly | Same information floor; explicit posterior via unit head; risk of mumbled blends without it |
| Text output | Native | Only through an optional CTC head |
| Supervision | Needs transcripts | Needs only paired audio and video (units come from audio) |
| Models to maintain | Two (lip reader and TTS) | One generator, a frozen encoder and an off-the-shelf vocoder |
| Training cost | Small, but conv3d-bound on MPS | Larger model, but trains on cached features; multi-night runs |
| Evaluation | Character error rate is direct | Needs a GRID-aware ASR, STOI/ESTOI/PESQ and offset alignment |
| Failure risks | Error propagation | CFG hallucinating fluent wrong words, mel/vocoder mismatch, unnormalised mels |

V2SFlow's MOS figures are from [V2SFlow](https://arxiv.org/html/2411.19486v2).

On LRS3, text-guided systems still lead on WER (LipVoicer about 21%, against roughly 28–31% for purely visual flow models), while flow models lead on naturalness, with UTMOS around 3.7–4.2 ([SLD-L2S](https://arxiv.org/html/2602.11477)). The generator raises quality, not content accuracy. Expect STOI to favour regression models and naturalness to favour the flow model. Whether your flow model beats VCA-GAN's STOI on GRID-4S is genuinely uncertain.

## Conclusion

The VLA intuition holds, with one correction. The useful transfer from robotics is not "put a big VLM in front." It is pi0's division of labour: a frozen, task-specific perception encoder; a small flow-matching expert producing a continuous chunk in about 10 steps; and Knowledge Insulation's lesson that a discrete auxiliary objective should sit beside the continuous one. In lip-to-speech, that discrete objective, speech units or CTC, is exactly where homophenes become visible and measurable instead of being smeared into the mel. Building the model this way turns your main failure mode from a hidden defect into an experimental question. Can a large pretrained lip encoder see voicing cues that a small GRID-trained model cannot?

On GRID, intelligibility is already near its visual ceiling. The new model's value is therefore everything around the words: real audio quality, sync and prosody, plus a clean testbed for the encoder ablations. Start by replacing Griffin-Lim with BigVGAN and caching features, since both pay off whether or not the full flow model succeeds. Treat any GRID-4S STOI/ESTOI/PESQ result from a flow-matching model as a contribution in its own right, because the literature has none.
