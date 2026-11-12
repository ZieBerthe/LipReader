# Prior Work on Lip-to-Speech / Video-to-Speech Synthesis (focus: diffusion and flow matching), as of Nov 2026

Notes on comparability (apply to every table below): papers use different ASR models for WER (Google API, Whisper, Auto-AVSR, etc.), different vocoders, different test subsets and different speaker-identity settings (audio reference vs. video-only). The same baseline often gets noticeably different numbers in different papers (examples in Q3). Read cross-paper numbers as rough rankings, not exact comparisons.

---

## Q1. What are the main L2S/V2S lineages, and which of the named models actually exist?

### Takeaway
The field went through four stages. (1) Regression and GAN models that predict mel or spectral features on constrained data such as GRID and TCD-TIMIT: Vid2Speech, Ephrat et al., Lip2Wav, VCA-GAN, SVTS. (2) Models that bring in intermediate linguistic targets such as text, CTC or self-supervised speech units: MT-L2S, Intelligible-L2S, ReVISE, LTBS. (3) Diffusion models from 2023–24: LipVoicer, DiffV2S. (4) Flow matching and latent or codec diffusion from 2025–26: V2SFlow, FTV, AlignDiT, LipDiffuser, SLD-L2S, HiCoDiT, Watch Your Speech, LipSody. Almost every strong recent system conditions on AV-HuBERT visual features. Most of these come from KAIST (Joon Son Chung's MM lab) and KAIST IVL (Yong Man Ro's lab).

### Cited Findings

**Early regression / GAN lineage (mostly GRID, TCD-TIMIT, LRW)**
- Vid2Speech, Ephrat et al., Lip2AudSpec, the 1D GAN-based model (Vougioukas et al.), Lip2Wav, the VAE-based model and the vocoder-based model (Michelsanti et al.) are the standard GRID baselines. VCA-GAN's constrained-speaker table compares all of them (numbers in Q3) — [VCA-GAN, Kim et al., NeurIPS 2021](https://ar5iv.labs.arxiv.org/html/2204.01726)
- **Lip2Wav** (Prajwal et al., CVPR 2020) is a speaker-specific sequence-to-sequence lip-to-mel model that "learns individual speaking styles". It also introduced the Lip2Wav dataset (five lecturers, about 20 h each) — [Lip2Wav paper](https://arxiv.org/pdf/2005.08209); dataset description from [LTBS](https://ar5iv.labs.arxiv.org/html/2308.15256)
- **VCA-GAN** (NeurIPS 2021) treats the mel-spectrogram as an image and trains it with a 2D GAN. It uses a 3D-conv + ResNet-18 visual encoder, three generators with visual context attention, multi-scale discriminators, a contrastive synchronization loss, and Griffin-Lim with a postnet that maps mel to linear spectrogram — [VCA-GAN](https://ar5iv.labs.arxiv.org/html/2204.01726)
- **SVTS** (Mira et al., Interspeech 2022) is a feedforward video-to-spectrogram predictor plus a pretrained neural vocoder. It reports SOTA on GRID and LRW and claims the first intelligible results on LRS3 — [arXiv 2205.02058](https://arxiv.org/abs/2205.02058v1); [ISCA archive](https://www.isca-archive.org/interspeech_2022/schoburgcarrillodemira22_interspeech.html)
- **SLTS** (Niu & Mak, "On the Audio-visual Synchronization for L2S", 2023) has a ResNet-18 frame encoder, a Conformer decoder producing 100 Hz mel and a UnivNet vocoder. It adds data and self-synchronization modules, and it shows that STOI/ESTOI are very sensitive to small audio-video offsets — [arXiv 2303.00502](https://arxiv.org/html/2303.00502v1)

**Text / unit-supervised lineage**
- **MT / Lip2Speech multi-task** (Kim et al., ICASSP 2023) is "Lip-to-Speech Synthesis in the Wild with Multi-task Learning". It uses text supervision as an auxiliary task, so it cannot train on unlabeled data — [arXiv 2302.08841](https://arxiv.org/pdf/2302.08841); the text-requirement note comes from [LTBS](https://ar5iv.labs.arxiv.org/html/2308.15256)
- **Intelligible L2S with Speech Units** (Choi, Kim, Ro, Interspeech 2023) has two stages: (1) a multi-target Lip2Speech model predicts mel plus HuBERT-Base layer-6 k-means-200 units, (2) a multi-input vocoder takes mel, units and a speaker embedding. Its visual front end is optionally AV-HuBERT Large — [GitHub](https://github.com/choijeongsoo/lip2speech-unit); [arXiv 2305.19603](https://arxiv.org/html/2305.19603)
- **ReVISE** (Hsu et al., Meta, CVPR 2023) does "pseudo-AVSR" from AV-HuBERT to discrete self-supervised units, then "pseudo-TTS" from units to waveform. The authors call it "the first high-quality model for in-the-wild video-to-speech synthesis" on LRS3 — [arXiv 2212.11377](https://arxiv.org/abs/2212.11377v1); [CVPR paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Hsu_ReVISE_Self-Supervised_Speech_Resynthesis_With_Visual_Input_for_Universal_and_CVPR_2023_paper.pdf)
- **LTBS / "Let There Be Sound"** (Kim, Kim, Chung, AAAI 2024) uses a 3D-conv + ResNet-18 + Conformer encoder and a variance decoder. That decoder predicts HuBERT-Large layer-12 k-means-200 units with cross-entropy, plus pYIN pitch and energy, and a Conformer decoder then produces a coarse mel. A **normalizing-flow post-net** (NLL-trained, *not* flow matching) refines the mel, and a Fre-GAN vocoder produces the waveform. Speaker identity comes from a learned speaker-embedding table — [LTBS](https://ar5iv.labs.arxiv.org/html/2308.15256)
- **Hegde et al., "Towards Accurate L2S in-the-Wild"** (ACM MM 2023) is a two-stage system. A pretrained lip-to-text model (VTP) produces noisy text, then a *visual* TTS attends from visual features (queries) to phoneme embeddings (keys/values), which aligns text to video time without a duration predictor. A BigVGAN vocoder produces audio — [ar5iv 2403.01087](https://ar5iv.labs.arxiv.org/html/2403.01087)
- **Lip2Vec** (ICCV 2023) is a *VSR* (lip-reading) method, not V2S. A prior network maps visual latents to audio (wav2vec-style) latents, and a frozen ASR decodes them to text (26% WER on LRS3). It is relevant only as a "map video features to audio features" idea — [CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Djilali_Lip2Vec_Efficient_and_Robust_Visual_Speech_Recognition_via_Latent-to-Latent_Visual_ICCV_2023_paper.html); [arXiv 2308.06112](https://www.arxiv.org/abs/2308.06112)

**Diffusion lineage (2023–2025)**
- **LipVoicer** (Yemini et al., arXiv 2306.03258, v2 Mar 2024) is a DDPM with a DiffWave-style backbone over 80-bin mel (400 steps). It conditions on a lip-video embedding and a ResNet-18 face embedding, with classifier-free guidance. At inference, text predicted by a lip-reader (Auto-AVSR, Ma et al. 2023) is injected through **classifier guidance from a modified ASR** acting on the noisy mel. A DiffWave vocoder produces the waveform — [arXiv 2306.03258](https://arxiv.org/html/2306.03258)
- **DiffV2S** (Choi, Hong, Ro, ICCV 2023) is a diffusion model generating mel. It is conditioned on visual features and on a **vision-guided speaker embedding**, extracted from video only with a self-supervised pretrained model plus prompt tuning — [arXiv 2308.07787](https://arxiv.org/abs/2308.07787)
- **LipDiffuser** (arXiv 2505.11391, v3 Oct 2025) is an EDM-style score-based diffusion model built on a magnitude-preserving ADM U-Net (~205M params) over 80-bin mel at 16 kHz, with 32 deterministic second-order sampling steps. BRAVEn visual features are interpolated to the mel frame rate and injected per frame through magnitude-preserving FiLM. A WeSpeaker embedding from an enrollment utterance supplies speaker identity, and HiFi-GAN produces the waveform — [arXiv 2505.11391](https://arxiv.org/html/2505.11391)
- **LipSody** (Lee, Oh, Lee, SNU, ICASSP 2026, arXiv 2602.01908) builds on LipVoicer (DDPM, T=400, CFG). It adds a separately trained prosody predictor that maps face identity, lip content and EmoCLIP emotion to frame-wise pitch (FCPE) and energy, which then condition the diffusion model. Output is mel, vocoded by HiFi-GAN — [arXiv 2602.01908](https://arxiv.org/html/2602.01908)

**Flow-matching lineage (2024–2026)**
- **V2SFlow** (Choi, Kim, Li, Chung, Liu, KAIST + Microsoft, ICASSP 2025, arXiv 2411.19486) decomposes speech into three components, each predicted from video: content (mHuBERT layer-11 k-means-1000 units, 50 Hz), pitch (VQ-VAE units of YAAPT F0, 12.5 Hz) and a global speaker embedding. A **Transformer rectified-flow-matching decoder** generates 100 Hz 80-bin mel, and HiFi-GAN produces the waveform — [arXiv 2411.19486](https://arxiv.org/abs/2411.19486); [GitHub](https://github.com/kaistmm/V2SFlow)
- **FTV, "From Faces to Voices"** (Kim, Choi, Kim, Jung, Chung, CVPR 2025 Highlight, arXiv 2503.16956) has a hierarchical visual encoder with three stages. The content stage uses an AV-HuBERT-Large weighted layer sum and predicts k-means-1000 units. The timbre stage uses ArcFace face embeddings and predicts time-averaged HuBERT layer-1 features. The prosody stage uses a facial-expression encoder and predicts pitch and energy. An **OT-CFM flow-matching decoder** (U-Net with 1D residual convolutions plus Transformer blocks) generates mel, and HiFi-GAN produces the waveform — [arXiv 2503.16956](https://arxiv.org/html/2503.16956v1); [project page](https://mm.kaist.ac.kr/projects/faces2voices/)
- **AlignDiT** (Choi et al., KAIST/POSTECH, ACM MM 2025, arXiv 2504.20629) is a DiT (18 blocks, 768-d) trained with conditional flow matching over mel, plus an auxiliary CTC loss. It is mainly built for dubbing / automated dialogue replacement (inputs: text, video and reference audio), and is also evaluated on V2S using lip-reader-predicted text — [arXiv 2504.20629v2](https://arxiv.org/html/2504.20629v2); [project page](https://mm.kaist.ac.kr/projects/AlignDiT)
- **SLD-L2S** (Liang et al., IACAS/Zhipu, arXiv 2602.11477, Feb 2026, preprint) uses *reparameterized* flow matching: the network predicts the clean latent x1 instead of the velocity. It generates **continuous latents of a pretrained X-Codec-hubert codec** (16 kHz), and the codec decoder produces the waveform directly, so no mel vocoder is needed. The backbone is 12 "Diffusion Convolution Blocks" (convolutional replacements for DiT attention and MLP) over 8 parallel subspaces of AV-HuBERT-Large features. It uses 10 NFE — [arXiv 2602.11477](https://arxiv.org/html/2602.11477)
- **HiCoDiT** (Ye et al., Fudan, arXiv 2604.15923, Apr 2026) uses **masked discrete diffusion** (SEDD) over RVQ tokens of the MaskGCT codec (12 levels × 1024). Low-level tokens (levels 1–2) are conditioned on AV-HuBERT lip features and ArcFace identity, and high-level tokens (levels 3–12) on facial-expression (Poster V2) prosody. It uses 64 Euler steps. (One search snippet called it CVPR 2026; the arXiv page I read lists no venue, so this is unverified) — [arXiv 2604.15923](https://arxiv.org/html/2604.15923); [GitHub](https://github.com/Jiaxin-Ye/HiCoDiT)
- **Watch Your Speech (WYS)** (Lee, Oh, Han, BMVC 2026, arXiv 2610.01012, Oct 2026) uses **conditional flow matching with CFG** and a DiffWave backbone that predicts the velocity field over mel. The visual front end is 3D-CNN + ShuffleNet + TCN on 96×96 lip crops. A frozen BERT with LoRA encodes text, and the text is fused with video through self-attention plus bidirectional cross-attention. Text is ground truth in training and **Auto-AVSR lip-reader output** at inference. It uses 100 Euler steps with ω=2 — [arXiv 2610.01012](https://arxiv.org/html/2610.01012v1)

**Related or adjacent (not pure V2S; most need a text script)**
- **EmoDubber** (Cong et al., CVPR 2025) and **FlowDubber** (Cong et al., ACM MM 2025, LLM semantics plus flow-matching voice enhancement) are *movie dubbing* models that need text input. HiCoDiT uses EmoDubber as a baseline — [HiCoDiT related work](https://arxiv.org/html/2604.15923); [FlowDubber arXiv 2505.01263](https://arxiv.org/pdf/2505.01263)
- **DiFlowDubber** (arXiv 2603.14267, 2026) applies discrete flow matching to video dubbing. I saw only its search listing — [arXiv 2603.14267](https://arxiv.org/html/2603.14267v1)
- **VSSFlow** (arXiv 2509.24773, v4 Mar 2026) is a unified flow-based model for video-conditioned sound and speech. I saw only its listing — [arXiv 2509.24773](https://arxiv.org/html/2509.24773v4)
- **NaturalL2S** (Liang et al., arXiv 2502.12002, Feb 2025) is non-generative and end-to-end. A frozen AV-HuBERT feeds FFT pitch and content generators, a DDSP synthesizer produces a coarse signal, and a conditional HiFi-GAN refines it. It needs no reference speaker embedding — [arXiv 2502.12002](https://arxiv.org/abs/2502.12002v1)
- **Uni-Dubbing** (codec residual tokens) and **LightL2S** are cited by SLD-L2S as recent L2S works. I did not read them — [SLD-L2S related work](https://arxiv.org/html/2602.11477)

### Inferences
- The strongest current recipe is: frozen or fine-tuned **AV-HuBERT-Large lip features**, upsampled to the acoustic frame rate, conditioning a **flow-matching (or EDM-diffusion) generator over mel or codec latents**, followed by a pretrained neural vocoder or codec decoder. V2SFlow, FTV, AlignDiT, SLD-L2S, HiCoDiT and LipDiffuser all follow this pattern.
- This matches the robotics VLA analogy the student has in mind: a pretrained perception backbone (AV-HuBERT, or VSR encoders such as BRAVEn or Auto-AVSR) plus a flow-matching "action head" that outputs mel or latent frames.

### Gaps
- **"VoiceDiT"** and **"LipFlow"**: I found no primary source for either as a V2S model. Treat them as unverified or nonexistent under those names.
- Uni-Dubbing, LightL2S, SyncVoice, VisualTTS, DiFlowDubber and VSSFlow appeared only in search listings or citations; I did not read their details.
- I could not machine-read the original PDFs of Vid2Speech, Ephrat et al. and Mira et al.'s "End-to-end V2S with GANs". Their GRID numbers here come from VCA-GAN's and SLTS's comparison tables.

---

## Q2. Which models use diffusion or flow matching, what do they generate, and how is the visual conditioning injected?

### Takeaway
Most diffusion and flow-matching V2S models generate **80-bin mel at 16 kHz** and vocode it with HiFi-GAN or DiffWave. The 2026 work moves to **codec latents** (SLD-L2S: continuous X-Codec latents; HiCoDiT: discrete RVQ tokens). Visual conditioning is almost always **frame-aligned**: lip features are upsampled to the mel frame rate and then added, concatenated or FiLM-ed per frame. Global signals (speaker, face identity, emotion) enter through AdaLN, FiLM or broadcast addition. Text, when used, enters through cross-attention (AlignDiT, WYS) or ASR classifier guidance (LipVoicer). CFG with about 10–20% condition dropout is standard.

### Cited Findings

| Model (date, venue) | Generator | Target generated | Visual encoder | Injection mechanism | Guidance | Sampling steps |
|---|---|---|---|---|---|---|
| LipVoicer (2023, arXiv v2 2024) | DDPM, DiffWave-style | 80-bin mel → DiffWave vocoder | lip-reading-architecture lip embedding + ResNet-18 face embedding | conditioning embeddings into the denoiser | CFG (cond. dropout 0.2) + **ASR classifier guidance** using lip-reader text (from step t=270 of 400 on LRS3) | 400 — [arXiv](https://arxiv.org/html/2306.03258) |
| DiffV2S (ICCV 2023) | diffusion | mel | visual features + vision-guided speaker embedding | conditioning on both | — | 1000 (as listed by [SLD-L2S](https://arxiv.org/html/2602.11477) and [FTV](https://arxiv.org/html/2503.16956v1)) — [arXiv](https://arxiv.org/abs/2308.07787) |
| V2SFlow (ICASSP 2025) | Transformer rectified flow | 100 Hz 80-bin mel → HiFi-GAN | AV-HuBERT Large (25 Hz) | visual → predicted content units + pitch units + speaker embedding, which condition the RFM decoder | — | 30 NFE (per [SLD-L2S table](https://arxiv.org/html/2602.11477)) — [GitHub](https://github.com/kaistmm/V2SFlow) |
| FTV (CVPR 2025) | OT-CFM, U-Net (1D residual conv + Transformer, Snake-beta) | mel → HiFi-GAN | AV-HuBERT Large (weighted layer sum), ArcFace, expression encoder | hierarchical encoder outputs a frame-level μ: unit embeddings, timbre and pitch/energy embeddings **added** to the hidden sequence; μ is the decoder's condition | CFG (dropout 0.1, β=0.7), Euler | 10 (≈ same quality as 1000) — [arXiv](https://arxiv.org/html/2503.16956v1) |
| AlignDiT (ACM MM 2025) | DiT + CFM (+CTC aux) | 100 Hz 80-bin mel → HiFi-GAN | video features aligned to 100 fps | audio and video **concatenated channel-wise**; text through **cross-attention** in each block | **multimodal CFG** with separate s_text=5, s_video=2; modality dropout 0.2; Euler with sway sampling | not stated — [arXiv](https://arxiv.org/html/2504.20629v2) |
| LipDiffuser (2025) | EDM score-based, MP-ADM U-Net (~205M) | 80-bin mel → HiFi-GAN | BRAVEn (88×88 grey lip ROI, 25 fps, 1024-d) | features interpolated to mel rate, **per-frame MP-FiLM**; WeSpeaker enrollment embedding | — | 32 (2nd-order deterministic) — [arXiv](https://arxiv.org/html/2505.11391) |
| SLD-L2S (Feb 2026) | reparameterized FM (predicts x1) with conv DiCB blocks | **continuous X-Codec latents** → codec decoder | AV-HuBERT Large, upsampled by transposed conv, split into 8 subspaces | features are the backbone input; timestep and GE2E speaker embedding via **AdaLN-SOLA**; aux WavLM "SLM" loss and HuBERT semantic loss | — | 10 — [arXiv](https://arxiv.org/html/2602.11477) |
| HiCoDiT (Apr 2026) | masked discrete diffusion (SEDD) | **discrete RVQ tokens** (MaskGCT codec, 12×1024) | AV-HuBERT Large, ArcFace, Poster V2 | lip features **concatenated channel-wise** with masked low-level tokens; identity and emotion via **dual-scale AdaLN** | compositional predictor-free guidance (10% drop per condition) | 64 Euler — [arXiv](https://arxiv.org/html/2604.15923) |
| LipSody (ICASSP 2026) | DDPM (LipVoicer-based), DiffWave | mel → HiFi-GAN | 3D-conv lip encoder, ResNet-18 face, EmoCLIP | predicted frame-wise pitch and energy added as extra conditions | CFG | 400 — [arXiv](https://arxiv.org/html/2602.01908) |
| WYS (BMVC 2026) | CFM, DiffWave backbone | mel → HiFi-GAN | 3D-CNN + ShuffleNet + TCN (VSR-style) | lip and text **bidirectional cross-attention** fusion, concatenated with a ResNet-18 face speaker embedding | CFG ω=2 (null prob 0.1) | 100 Euler — [arXiv](https://arxiv.org/html/2610.01012v1) |
| LTBS (AAAI 2024) | regression + **normalizing-flow post-net** (not diffusion or FM) | mel → Fre-GAN | 3D-conv + ResNet-18 + Conformer | variance adaptor (units, pitch, energy) | — | — — [ar5iv](https://ar5iv.labs.arxiv.org/html/2308.15256) |

- **Flow matching needs few steps.** FTV reports LRS3 UTMOS 4.031 at 10 steps vs 3.993 at 1000 steps, and WER 30.45 vs 30.37 — [FTV](https://arxiv.org/html/2503.16956v1)
- **Guidance is crucial for intelligibility in LipVoicer.** With ASR guidance off (w2=0), LRS3 WER rises from 21.4% to 86.2%. Disabling CFG raises WER to 30.1%. Ground-truth text instead of lip-reader text gives 5.4% WER. Large ASR-guidance weights hurt quality and synchronization — [LipVoicer](https://arxiv.org/html/2306.03258)
- **Architecture matters for timbre.** In SLD-L2S, replacing its convolutional DiCB with a parameter-matched DiT drops SECS from 0.804 to 0.618 — [SLD-L2S](https://arxiv.org/html/2602.11477)
- **Ablating FTV's hierarchy hurts.** Without it, UTMOS falls from 4.031 to 3.737 and WER rises from 30.45 to 33.64. Without the prosody stage, WER rises to 35.03 — [FTV](https://arxiv.org/html/2503.16956v1)
- **AlignDiT fusion choice.** Channel-wise concatenation of frame-aligned video was chosen for audio-video, and cross-attention for text (preferred over early fusion or text-prefix) — [AlignDiT](https://arxiv.org/html/2504.20629v2)

### Inferences
- For a GRID student project, the simplest strong design is FTV / V2SFlow-lite: lip features (their existing 3D-CNN+BiLSTM encoder, or AV-HuBERT) are upsampled to the mel rate and concatenated with the noisy mel in a 1D U-Net or DiT trained with OT-CFM. Add a speaker embedding by AdaLN or addition, train with CFG dropout of about 0.1, sample in about 10 Euler steps, and vocode with a pretrained HiFi-GAN.
- An auxiliary content loss on intermediate features is a cheap way to reuse their existing text/CTC lip-reader as an intelligibility signal inside a direct model. Options are a CTC loss as in AlignDiT, or speech-unit prediction as in FTV, V2SFlow and LTBS.

### Gaps
- DiffV2S's exact backbone and injection details: I only read its abstract.
- NFE for AlignDiT is not stated.
- LipVoicer's total NFE including the classifier-guidance cost is not explicitly reported.

---

## Q3. Reported results on GRID and LRS3/LRS2

### Takeaway
**GRID is nearly saturated for seen speakers, and no diffusion or flow-matching paper reports standard GRID STOI/ESTOI/PESQ tables.** The best GRID numbers are from GAN and regression models (STOI ~0.73–0.76, ESTOI ~0.54–0.61, PESQ ~1.9–2.0, WER ~5–14% depending on the ASR). The only diffusion- or flow-related GRID results are LipVoicer's MOS-only appendix and LTBS (normalizing-flow post-net, MOS and WER). On LRS3, flow-matching models (FTV, V2SFlow, SLD-L2S, HiCoDiT) lead on naturalness (UTMOS ~3.7–4.2, often above ground truth). Text-guided models (LipVoicer, WYS, AlignDiT with lip-reader text) lead on WER (~19–24%), while purely visual models sit at ~28–31%.

### GRID protocols (from VCA-GAN and SLTS)
- **Constrained-speaker (GRID-4S):** speakers s1, s2 (male), s4, s29 (female). In SLTS the per-speaker split is random 90/5/5, giving 3600/200/200 clips — [VCA-GAN](https://ar5iv.labs.arxiv.org/html/2204.01726); [SLTS](https://arxiv.org/html/2303.00502v1)
- **Unseen-speaker:** 15 train / 8 val / 10 test speakers, following Vougioukas et al. and Michelsanti et al. — [VCA-GAN](https://ar5iv.labs.arxiv.org/html/2204.01726)
- **Multi-speaker (seen, 33 speakers):** all 33 speakers in train and test, following the LipNet protocol. LTBS instead uses an 80/10/10 split sampled equally from all 33 speakers — [VCA-GAN](https://ar5iv.labs.arxiv.org/html/2204.01726); [LTBS](https://ar5iv.labs.arxiv.org/html/2308.15256)
- GRID has a 51-word vocabulary, about 50 min per speaker, and near-zero AV offsets (within ±1 frame) — [LTBS](https://ar5iv.labs.arxiv.org/html/2308.15256); [SLTS](https://arxiv.org/html/2303.00502v1)

### GRID constrained-speaker (4 speakers) — VCA-GAN Table 2 (WER via Google ASR, following Lip2Wav)
| Method | STOI | ESTOI | PESQ | WER |
|---|---|---|---|---|
| Vid2Speech | 0.491 | 0.335 | 1.734 | 44.92% |
| Ephrat et al. | 0.659 | 0.376 | 1.825 | 27.83% |
| Lip2AudSpec | 0.513 | 0.352 | 1.673 | 32.51% |
| 1D GAN-based | 0.564 | 0.361 | 1.684 | 26.64% |
| Lip2Wav | 0.731 | 0.535 | 1.772 | 14.08% |
| VAE-based | 0.724 | 0.540 | 1.932 | – |
| Vocoder-based | 0.648 | 0.455 | 1.900 | 23.33% |
| VCA-GAN | 0.724 | 0.609 | 2.008 | 12.25% (5.83% with their pretrained ASR) |
Source: [VCA-GAN](https://ar5iv.labs.arxiv.org/html/2204.01726)

### GRID-4S SOTA comparison — SLTS Table 5
| Method | STOI | ESTOI | PESQ |
|---|---|---|---|
| E2E-V2AResNet | 0.627 | – | 2.030 |
| Yadav et al. | 0.724 | 0.540 | 1.932 |
| VCA-GAN | 0.724 | 0.609 | 2.008 |
| Lip2Wav | 0.731 | 0.535 | 1.722 |
| Kim et al. | 0.738 | 0.579 | 1.984 |
| SLTS | 0.757 | 0.588 | 1.931 |
Source: [SLTS](https://arxiv.org/html/2303.00502v1). Lip2Wav PESQ is listed as 1.722 here but 1.772 in VCA-GAN, probably a transcription difference. In SLTS's re-run, an alignment frontend that searches ±300 ms raises VCA-GAN's STOI from 0.688 to 0.732, which shows how offset-sensitive these metrics are.

### GRID unseen-speaker — VCA-GAN Table 5
| Method | STOI | ESTOI | PESQ | WER |
|---|---|---|---|---|
| GAN-based (1D) | 0.445 | 0.188 | 1.240 | 38.51% |
| Vocoder-based | 0.537 | 0.227 | 1.230 | 37.97% |
| VCA-GAN | 0.569 | 0.336 | 1.372 | 23.45% |
Source: [VCA-GAN](https://ar5iv.labs.arxiv.org/html/2204.01726)

### GRID multi-speaker (33 seen)
- VCA-GAN reaches STOI 0.736, ESTOI 0.604, PESQ 1.961, CER 1.8%, WER 4.8%, which matches or beats the LipNet lip reader (CER 2.0%, WER 5.6%) — [VCA-GAN](https://ar5iv.labs.arxiv.org/html/2204.01726)
- LTBS Table 1 (33 speakers, 80/10/10):

| Method | Nat. MOS | Intel. MOS | WER | CER |
|---|---|---|---|---|
| Ground truth | 4.82 | 4.82 | 12.20 | 7.03 |
| Vocoded GT | 4.74 | 4.79 | 12.44 | 7.18 |
| VCA-GAN | 3.46 | 4.10 | 17.62 | 9.55 |
| SVTS | 3.35 | 3.97 | 23.30 | 13.12 |
| MT (text multi-task) | 2.42 | 3.08 | 30.56 | 18.14 |
| LTBS | 4.46 | 4.63 | 17.07 | 9.17 |

Ground truth itself scores 12.2% WER with their ASR, so GRID WER mostly measures the ASR's robustness to GRID's unusual grammar — [LTBS](https://ar5iv.labs.arxiv.org/html/2308.15256)
- **LipVoicer on GRID** (Appendix F, 50 samples from 4 unseen speakers, MOS only): intelligibility 3.75, naturalness 3.98, quality 3.60, sync 3.82, vs VCA-GAN 3.29, 3.23, 2.98, 3.68 — [LipVoicer](https://arxiv.org/html/2306.03258)

### LRS3 — flow and diffusion era (purely visual unless noted)
FTV's table (WER via its ASR; video-driven speaker identity for LTBS, DiffV2S and FTV; audio-driven for SVTS and Intelligible):

| Method | Steps | UTMOS | DNSMOS | WER | SECS (GE2E) |
|---|---|---|---|---|---|
| Ground truth | – | 3.545 | 2.582 | 2.29 | – |
| SVTS | – | 1.283 | 1.860 | 84.98 | – |
| Intelligible L2S | – | 2.702 | 2.395 | 29.60 | – |
| LTBS | – | 2.417 | 2.361 | 84.08 | 0.609 |
| DiffV2S | 1000 | 3.058 | 2.558 | 41.07 | 0.621 |
| **FTV** | 10 | 4.031 | 2.789 | 30.45 | 0.650 |

LRS2 (zero-shot, trained on LRS3): FTV UTMOS 3.921 / WER 39.37 vs DiffV2S 2.945 / 54.86. MOS on LRS3: FTV naturalness 4.49 vs GT 4.54 and DiffV2S 2.97; intelligibility 4.01 vs GT 4.84 — [FTV](https://arxiv.org/html/2503.16956v1)

SLD-L2S's table (WER via Auto-AVSR; * = reference-audio speaker embedding):

| Method | NFE | UTMOS | WER | SECS | MOS-Nat | MOS-Intel |
|---|---|---|---|---|---|---|
| GT | – | 3.57 | 0.71 | – | 4.36 | 4.48 |
| Intelligible* | 1 | 2.70 | 27.73 | 0.761 | 2.82 | 3.37 |
| DiffV2S | 1000 | 3.07 | 35.81 | 0.627 | 2.95 | 2.98 |
| LipVoicer | 400 | 2.45 | 20.62 | 0.588 | 2.51 | 2.97 |
| NaturalL2S | 1 | 3.66 | 30.37 | 0.628 | 3.54 | 3.43 |
| V2SFlow* | 30 | 3.69 | 27.55 | 0.851 | 3.88 | 3.69 |
| SLD-L2S* | 10 | 4.21 | 30.22 | 0.804 | 4.17 | 3.65 |
Source: [SLD-L2S](https://arxiv.org/html/2602.11477). On LRS2 (unseen), WER is 24.92 for LipVoicer, 34.17 for V2SFlow and 39.54 for SLD-L2S.

HiCoDiT's table (trained on VoxCeleb2, zero-shot on LRS3; "A" = audio reference used):

| Method | WER | DNSMOS | UTMOS | SpkSim |
|---|---|---|---|---|
| GT | 2.29 | 3.29 | 3.57 | 1.0 |
| EmoDubber (A, needs text) | 41.52 | 2.95 | 2.83 | 0.605 |
| DiffV2S | 41.07 | 2.56 | 3.06 | – |
| LTBS | 84.00 | 2.36 | 2.42 | – |
| AlignDiT (V2S) | 31.37 | 3.24 | 3.76 | 0.560 |
| FTV | 30.37 | 3.22 | 3.99 | 0.598 |
| HiCoDiT (video only) | 29.41 | 3.50 | 3.84 | 0.568 |
| HiCoDiT (+audio ref) | 28.98 | 3.44 | 3.80 | 0.672 |
Source: [HiCoDiT](https://arxiv.org/html/2604.15923). In A/B tests HiCoDiT was preferred over AlignDiT 57.0% vs 38.1% and over FTV 52.1% vs 41.8%.

WYS's table on LRS3 (text-conditioned; STOI here is STOI-Net, a non-intrusive estimate):

| Method | WER | LSE-C | LSE-D | STOI-Net | DNSMOS |
|---|---|---|---|---|---|
| GT | 1.01 | 7.34 | 6.90 | 0.931 | 3.30 |
| VCA-GAN | 90.68 | 4.55 | 9.13 | 0.633 | 2.27 |
| SVTS | 75.66 | 6.28 | 7.94 | 0.707 | 2.42 |
| DiffV2S | 38.09 | 6.28 | 7.94 | 0.921 | 3.22 |
| IntelligibleL2S | 50.01 | 6.78 | 7.38 | 0.884 | 2.87 |
| LipVoicer | 22.24 | 5.70 | 8.45 | 0.921 | 3.18 |
| V2SFlow-V | 29.61 | 7.01 | 7.45 | 0.933 | 3.29 |
| FTV | 29.14 | 7.10 | 7.26 | 0.935 | 3.17 |
| AlignDiT | 29.67 | 6.96 | 7.35 | 0.933 | 3.20 |
| WYS | 23.73 | 7.01 | 7.26 | 0.937 | 3.24 |
On LRS2, WYS gets WER 20.13 vs LipVoicer 18.90, and the best LSE-C (7.50) and LSE-D (6.81). Source: [WYS](https://arxiv.org/html/2610.01012v1)

- **AlignDiT V2S** (LRS3, with lip-reader text): WER 19.51, AVSync 0.688, spkSIM 0.508, vs LipVoicer 21.16 / 0.524 / 0.094, Intelligible 27.43 / 0.675 / 0.316 and DiffV2S 35.21 / 0.608 / 0.115 — [AlignDiT](https://arxiv.org/html/2504.20629v2)
- **LipVoicer's own LRS3 results:** WER 21.4%, STOI-Net 0.92, DNSMOS 3.11, vs Lip2Speech 57.4%, SVTS 82.4% and VCA-GAN 90.6%. On LRS2, WER 17.8%. Human MOS on LRS3 is about 3.4–3.6 vs about 2.2 for the best baseline. LipVoicer's **intrusive** scores are lower than some baselines (LRS3 PESQ 1.10 vs 1.31 for Lip2Speech), and the authors argue intrusive metrics are unsuitable for L2S — [LipVoicer](https://arxiv.org/html/2306.03258)
- **Intrusive metrics for the pre-diffusion unit model** (Intelligible L2S, LRS3, AV-HuBERT variant): STOI 0.578, ESTOI 0.393, PESQ 1.31, WER 29.8%. On LRS2: 0.585, 0.412, 1.34, 35.7% — [GitHub README](https://github.com/choijeongsoo/lip2speech-unit)
- **LipSody vs LipVoicer (LRS3):** WER 0.225 vs 0.221 (reconstructed LipVoicer), naturalness MOS 3.47 vs 3.36 (not significant), and significantly better prosody metrics (GF0 25.15 vs 28.87) — [LipSody](https://arxiv.org/html/2602.01908)
- **LipDiffuser (LRS3 + CHiME3 noise at −10 dB, video plus enrollment):** DNSMOS 3.64, NISQA 4.57, WER 0.38, SpkSim 0.63, vs LipVoicer 3.17 / 3.53 / 0.36 / 0.15 and DiffV2S 3.17 / 3.48 / 0.51 / 0.16 — [LipDiffuser](https://arxiv.org/html/2505.11391)

**Cross-paper inconsistencies to be aware of:**
- DiffV2S LRS3 WER is reported as 41.07 ([FTV](https://arxiv.org/html/2503.16956v1)), 35.81 ([SLD-L2S](https://arxiv.org/html/2602.11477)), 38.09 ([WYS](https://arxiv.org/html/2610.01012v1)) and 35.21 ([AlignDiT](https://arxiv.org/html/2504.20629v2)).
- LipVoicer WER ranges from 20.6 to 22.2 across these papers.
- FTV WER is 30.45 in its own paper and 29.14 in WYS.
- V2SFlow differs between the audio-reference variant (27.55, SECS 0.851) and the visual-only variant "V2SFlow-V" (29.61).

### Inferences
- On LRS3, purely visual flow-matching models (FTV, V2SFlow-V, AlignDiT-V2S, HiCoDiT, SLD-L2S) cluster at **~28–31% WER with UTMOS ~3.8–4.2**. Text-guided models (LipVoicer, WYS, AlignDiT with lip-reader text) reach **~19–24% WER**, but with lower naturalness (LipVoicer) or no clear naturalness gain. The intelligibility ceiling is set by visual ambiguity (homophenes), not by the generator.
- On GRID the generator choice matters less for intelligibility: seen-speaker V2S already reaches lip-reader-level WER (VCA-GAN 4.8% vs LipNet 5.6%). The student's main win over Griffin-Lim TTS would be **audio quality and naturalness**, which flow matching plus a neural vocoder delivers. If the project reports results on GRID, it would be among the first published flow-matching results there.
- The student should report both intrusive (STOI/ESTOI/PESQ, for comparison with the GRID literature) and non-intrusive metrics (UTMOS/DNSMOS, ASR WER, LSE-C/D, for comparison with the diffusion and flow literature). Because GRID STOI is offset-sensitive, an alignment search is worth applying, as SLTS shows.

### Gaps
- No diffusion or flow-matching paper found with STOI/ESTOI/PESQ on GRID's standard protocols.
- No per-protocol GRID numbers for SVTS, Lip2Speech-MT or Intelligible L2S were retrieved directly (only LTBS's re-runs).
- WYS numbers on LRS2 for FTV and AlignDiT, and LRS3 speaker similarity, are not reported.

---

## Q4. Two-stage (video → text → TTS) vs direct (video → speech)

### Takeaway
A pure cascade (lip-reader, then an off-the-shelf TTS) gives clean audio and reasonable content but **fails badly on lip synchronization and prosody**. It cannot know timing, speed or emphasis from text alone, and it inherits lip-reader errors. Direct models keep timing and prosody but struggle with intelligibility because of homophenes. The best current systems are **hybrids**: they generate directly from video and use text (or speech units, or CTC) only as guidance or an auxiliary signal. Examples are LipVoicer's ASR guidance, WYS's text cross-attention, AlignDiT's text with multimodal CFG, and Hegde's visual TTS.

### Cited Findings
- **Hegde et al. (ACM MM 2023), LRS3.** A "Lip-to-Text + TTS" cascade scores STOI 0.16, ESTOI 0.01, PESQ 0.42 and LSE-C 1.77, vs 0.58 / 0.37 / 1.39 / 7.89 for their video-aligned visual TTS. Human MOS sync is 1.01 for the cascade vs 3.82, but intelligibility is slightly *higher* for the cascade (3.61 vs 3.49). The authors say the cascade "cannot infer the speed, prosody, and accents of speakers just from the text input" — [Hegde et al.](https://ar5iv.labs.arxiv.org/html/2403.01087)
- **Hegde et al., text-source ablation.** A worse lip-reader gives worse speech: VTP (22.6% L2T-WER) is best, Deep Lip Reading (51.3) and AV-HuBERT (46.1) are worse, and ground-truth text is best of all — [Hegde et al.](https://ar5iv.labs.arxiv.org/html/2403.01087)
- **WYS (BMVC 2026), LRS2.** A lip-reading + TTS cascade with XTTS-v2 or CosyVoice gets WER 31.4 / 29.4 and LSE-C 3.12 / 3.71, vs WYS at 20.1 WER and 7.50 LSE-C. The cascade's DNSMOS is high (3.38 / 3.25 vs 3.09). Its audio is fine, but sync and content are worse — [WYS](https://arxiv.org/html/2610.01012v1)
- **LipVoicer.** Text guidance is "crucial": WER goes from 86.2% without ASR guidance to 21.4% with it. Quality scales with lip-reader accuracy (5.4% WER with ground-truth text, 21.4% with Ma et al. 2023, 38.1% with Ma et al. 2022). Strong text guidance can hurt naturalness and sync. The authors also flag misuse risk from injecting false text — [LipVoicer](https://arxiv.org/html/2306.03258)
- **Text-supervised training has limits.** Multi-task L2S needs transcripts, so it cannot train on Lip2Wav, and it had the lowest GRID naturalness in LTBS's comparison (MOS 2.42 vs LTBS 4.46). Self-supervised speech units give the disambiguation without text labels — [LTBS](https://ar5iv.labs.arxiv.org/html/2308.15256)
- **Off-the-shelf TTS gives the worst lip alignment** when overlaid on video in AV speech-to-speech translation (Amazon), because it has no notion of lip synchrony — [Amazon AV2AV sync paper](https://assets.amazon.science/f9/96/a66cc5774d11ab3d8f5ff643d451/improving-lip-synchrony-in-direct-audio-visual-speech-to-speech-translation.pdf) (snippet-level only)
- **On constrained GRID, a direct model's intelligibility already matches a lip reader.** VCA-GAN (direct) gets WER 4.8% vs LipNet's 5.6% — [VCA-GAN](https://ar5iv.labs.arxiv.org/html/2204.01726)

### Inferences
- The student's current pipeline is the exact "cascade" failure case. It also uses a weak TTS (Griffin-Lim), so it loses on both quality and sync.
- A recommended hybrid is a direct flow-matching generator conditioned on frame-level lip features, with their existing CTC lip-reader reused in one of two ways: (a) as a frozen encoder whose features condition the generator, or (b) as an auxiliary CTC head on the generator, as in AlignDiT. On GRID, LipVoicer-style ASR or text guidance is optional, because the vocabulary is tiny and seen-speaker lip-reading is near-perfect.

### Gaps
- No controlled study compares cascade and direct on GRID specifically.
- Hegde's cascade STOI/ESTOI numbers are extremely low (0.16 / 0.01), probably because the TTS output is unaligned in time. That says more about the metrics penalising misalignment than about quality.

---

## Q5. Public code, pretrained weights and training compute

### Takeaway
Only a few diffusion and flow V2S models have usable code and weights. **V2SFlow** (code plus LRS3 encoder, decoder and HiFi-GAN checkpoints) and **Intelligible L2S** (code plus LRS3/LRS2 checkpoints) are the most reusable. LipVoicer, HiCoDiT and WYS have GitHub repos (weights not verified). FTV, AlignDiT, LTBS, DiffV2S and LipSody have demo pages only. Compute is modest: most models train on 1–4 GPUs, typically about 60 H100-hours up to about 5 H100-days, or 4 consumer GPUs for 350k–1M steps.

### Cited Findings
| Model | Code | Weights | Training compute / data |
|---|---|---|---|
| V2SFlow | [github.com/kaistmm/V2SFlow](https://github.com/kaistmm/V2SFlow) (PyTorch 2.2.2) | yes: encoder, RFM decoder, HiFi-GAN (LRS3) on Google Drive | encoders trained following the IntelligibleL2S repo; GPUs not stated on repo — [GitHub](https://github.com/kaistmm/V2SFlow) |
| Intelligible L2S | [github.com/choijeongsoo/lip2speech-unit](https://github.com/choijeongsoo/lip2speech-unit) (fairseq, PyTorch 1.13.1) | yes: LRS3 and LRS2 two-stage checkpoints; no GRID | — |
| LipVoicer | [github.com/yochaiye/LipVoicer](https://github.com/yochaiye/LipVoicer) (from paper) | not verified | 4× RTX 2080 Ti, 1M mini-batches of 16, Adam 2e-4 — [paper](https://arxiv.org/html/2306.03258) |
| FTV | none (demo: [mm.kaist.ac.kr/projects/faces2voices](https://mm.kaist.ac.kr/projects/faces2voices/)) | – | 4× A5000, batch 64, 350k steps, LRS3 only — [paper](https://arxiv.org/html/2503.16956v1) |
| AlignDiT | none found (demo only) | – | LRS3 + LibriSpeech audio pretraining 500k steps, then multimodal fine-tuning 400k; GPUs not stated — [paper](https://arxiv.org/html/2504.20629v2) |
| SLD-L2S | not provided | – | **1× H100, about 5 days**, 150k iterations, batch 16, LRS3 only — [paper](https://arxiv.org/html/2602.11477) |
| HiCoDiT | [github.com/Jiaxin-Ye/HiCoDiT](https://github.com/Jiaxin-Ye/HiCoDiT) | not verified | VoxCeleb2 (261.5 h), batch 32, 200k iterations; GPUs not stated — [paper](https://arxiv.org/html/2604.15923) |
| LipDiffuser | "upon acceptance" | – | **about 60 H100 GPU-hours**, 32M samples, batch 256, plus LibriSpeech pretraining — [paper](https://arxiv.org/html/2505.11391) |
| WYS | [github.com/gunwoo5034/Watch-your-Speech](https://github.com/gunwoo5034/Watch-your-Speech) | not verified | 1× RTX PRO 6000, 1M iterations, batch 64, 153M params — [paper](https://arxiv.org/html/2610.01012v1) |
| LipSody | none (demo only) | – | not reported — [paper](https://arxiv.org/html/2602.01908) |
| LTBS | none (demo only) | – | 4× A5000, batch 64, 400 epochs on GRID — [paper](https://ar5iv.labs.arxiv.org/html/2308.15256) |
| DiffV2S | none found | – | – — [arXiv](https://arxiv.org/abs/2308.07787) |
| VCA-GAN | official code exists (LTBS used it as a baseline) | – | Titan RTX — [VCA-GAN](https://ar5iv.labs.arxiv.org/html/2204.01726); [LTBS](https://ar5iv.labs.arxiv.org/html/2308.15256) |
| Hegde et al. | project page [cvit.iiit.ac.in/…/ms-l2s-itw](http://cvit.iiit.ac.in/research/projects/cvit-projects/ms-l2s-itw); VTP at [github.com/prajwalkr/vtp](https://github.com/prajwalkr/vtp) | – | 1× 2080 Ti, about 900k steps — [paper](https://ar5iv.labs.arxiv.org/html/2403.01087) |

### Inferences
- GRID is about 30 h in total, so a mel flow-matching decoder of about 50–150M parameters on GRID should be trainable on a single GPU or an Apple-Silicon MPS machine in days, not weeks. This is an extrapolation from the LRS3 compute figures above (LRS3 is about 400+ h).
- **V2SFlow's released HiFi-GAN** (16 kHz, 80-bin, 100 Hz hop) and its RFM decoder code are the most direct starting points. Matching that mel configuration lets the student reuse the pretrained vocoder.

### Gaps
- Whether the LipVoicer, HiCoDiT and WYS repos include pretrained weights was not verified.
- Wall-clock training time is missing for most papers.
- The VCA-GAN official repo URL was not retrieved.

---

## Q6. How do models handle speaker identity?

### Takeaway
There are three strategies. (1) A **reference-audio speaker embedding** (GE2E, WeSpeaker or Resemblyzer from an enrollment clip) gives the best similarity: Intelligible, V2SFlow-A, SLD-L2S, LipDiffuser, HiCoDiT+A, Hegde. (2) **Face-derived identity**, video-only: DiffV2S's vision-guided embedding, ArcFace in FTV and HiCoDiT, a ResNet-18 face embedding in LipVoicer, LipSody and WYS. This is more practical but gives lower similarity. (3) A **learned speaker-ID embedding table** for closed-set data: LTBS on GRID. For GRID's 33 known speakers, an embedding table or an enrollment-audio embedding is the natural choice.

### Cited Findings
- **Reference audio.** SLD-L2S uses a 256-d GE2E embedding from a reference utterance, and audio-reference systems get much higher SECS (V2SFlow* 0.851, SLD-L2S* 0.804, Intelligible* 0.761) than video-only ones (DiffV2S 0.627, LipVoicer 0.588) — [SLD-L2S](https://arxiv.org/html/2602.11477)
- **Reference audio.** LipDiffuser's WeSpeaker enrollment embedding gives SpkSim 0.63 vs 0.15–0.16 for LipVoicer and DiffV2S — [LipDiffuser](https://arxiv.org/html/2505.11391)
- **Reference audio.** HiCoDiT's SpkSim rises from 0.568 (face only) to 0.672 when an audio reference is added — [HiCoDiT](https://arxiv.org/html/2604.15923)
- **Face-derived (video only).** DiffV2S extracts the speaker embedding from video only, using a self-supervised model plus prompt tuning, motivated by the fact that reference audio is often unavailable at inference — [DiffV2S](https://arxiv.org/abs/2308.07787)
- **Face-derived (video only).** FTV fuses ArcFace embeddings and predicts time-averaged HuBERT layer-1 features as a timbre target. Removing face ID raises RMSE-F0 from 39.0 to 40.8 and lowers SECS from 0.650 to 0.640 — [FTV](https://arxiv.org/html/2503.16956v1)
- **Face-derived (video only).** HiCoDiT aligns ArcFace features to GE2E speech embeddings with an ℓ1 loss — [HiCoDiT](https://arxiv.org/html/2604.15923)
- **Face-derived (video only).** LipVoicer, LipSody and WYS use a ResNet-18 embedding of a single face frame. Removing LipVoicer's face embedding raises WER from 21.4% to 22.9% — [LipVoicer](https://arxiv.org/html/2306.03258); [LipSody](https://arxiv.org/html/2602.01908); [WYS](https://arxiv.org/html/2610.01012v1)
- **Embedding table (closed set).** LTBS uses a learned speaker-embedding table on GRID and Lip2Wav. Removing it on Lip2Wav raises WER from 34.71 to 42.33 and reduces naturalness CMOS by 0.48 — [LTBS](https://ar5iv.labs.arxiv.org/html/2308.15256)
- **Reference clip (visual TTS).** Hegde et al. add a 256-d embedding from a random 1-second clip of the target speaker — [Hegde et al.](https://ar5iv.labs.arxiv.org/html/2403.01087)
- **No speaker embedding.** NaturalL2S deliberately uses no reference speaker embedding, for practicality — [NaturalL2S](https://arxiv.org/abs/2502.12002v1)
- **Prosody is a separate axis from identity.** LipSody predicts pitch and energy from face, lip and emotion, and with oracle prosody WER drops from 0.225 to 0.210 and GF0 error from 25.15 to 16.98 — [LipSody](https://arxiv.org/html/2602.01908)

### Inferences
- For the GRID project:
  - On the seen-speaker protocol (33 or 4 speakers), use a learned speaker-ID embedding through AdaLN or addition (LTBS-style). It is trivial and effective.
  - For unseen-speaker generalization, use a GE2E or WeSpeaker embedding from a reference clip, or an ArcFace face embedding, the FTV/HiCoDiT video-only option.
- An explicit pitch and energy predictor (FTV, LTBS, LipSody, V2SFlow pitch units) consistently helps prosody and is cheap to add.

### Gaps
- No paper found that compares the three identity strategies on GRID directly.
