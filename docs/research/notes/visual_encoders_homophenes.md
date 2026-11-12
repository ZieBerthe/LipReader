# Visual speech encoders for lip-to-speech generation, and homophenes (research notes, Nov 2026)

Scope: which pretrained or from-scratch visual encoders could be the "vision encoder" for a GRID lip-video-to-speech diffusion or flow-matching generator on an Apple M5 with 16 GB and PyTorch MPS, and what the literature says about homophenes (visemes).

Legend: "[UNVERIFIED]" marks claims from background knowledge that I did not confirm against a primary source in this session. The report writer should treat these as leads, not facts.

---

## Q1. Pretrained / self-supervised visual speech encoders: specs, weights, licenses, dependencies, L2S use

### Takeaway
Three encoder families have public weights and a proven record in lip-to-speech (L2S):
- **AV-HuBERT.** The de-facto L2S frontend, used by SLD-L2S 2026, "From Faces to Voices" 2025 and Choi et al. 2023. Its weights are non-commercial and it depends on an archived fairseq fork.
- **Auto-AVSR.** Apache-2.0 code, light dependencies, and plain PyTorch.
- **The Haliassos RAVEn → BRAVEn → USR → USR 2.0 line.** MIT license, PyTorch Lightning/Hydra, with self-supervised encoder-only checkpoints from Base to Huge.

For an M5 with 16 GB, a Base-size encoder used frozen (or with features precomputed once) is the realistic choice. Large models are feasible for inference only.

### Cited Findings
**AV-HuBERT (Meta, 2022)**
- The official repo requires Python 3.8 and fairseq as a git submodule pinned to commit afc77bd, installed with `pip install --editable`. The repo was archived (read-only) on 1 Sep 2024. AV-HuBERT features default to 25 Hz label rate (MFCC 100 Hz) — [facebookresearch/av_hubert](https://github.com/facebookresearch/av_hubert)
- Checkpoints come in Base and Large sizes, pretrained either on LRS3 only or on LRS3 + VoxCeleb2 (English). Variants include self-supervised only, VSR-finetuned (LRS3 30 h or 433 h), "Large + self-training", and noise-augmented AVSR models — [AV-HuBERT model page](https://facebookresearch.github.io/av_hubert/)
- The license allows non-commercial research only. It prohibits commercial, military, surveillance and biometric uses, and the restrictions extend to derivative works and data the software produces — [AV-HuBERT model page](https://facebookresearch.github.io/av_hubert/)
- A third-party HF mirror (vumichien/AV-HuBERT) hosts the original 1.93 GB `model.pt`. It is a pickle that imports fairseq classes, so fairseq or a manual state-dict conversion is still needed. nguyenvulebinh/AV-HuBERT is a separate MuAViC-trained seq2seq model in safetensors that uses a custom class. I found no official `transformers` port — [vumichien/AV-HuBERT](https://huggingface.co/vumichien/AV-HuBERT/tree/main), [nguyenvulebinh/AV-HuBERT](https://huggingface.co/nguyenvulebinh/AV-HuBERT)
- One search-result summary quoted a work reporting a 160M-param Base and a 477M-param Large AV-HuBERT-style model. This conflicts with commonly cited sizes of about 103M and 325M [UNVERIFIED]. The writer should not state exact parameter counts without checking the AV-HuBERT paper (arXiv 2201.02184).
- Input format: 88×88 grayscale mouth ROI at 25 fps, landmark-aligned (dlib or RetinaFace + FAN), 96×96 crop with a random or center 88×88 crop [UNVERIFIED for AV-HuBERT specifically]. The AV-HuBERT-based L2S paper below does use 25 fps, grayscale, RetinaFace + FAN landmarks.
- **Used for L2S:**
  - SLD-L2S (arXiv 2602.11477, Feb 2026) uses the "visual frontend of AV-Hubert Large" (LRS3 + VoxCeleb2) with latent flow matching into an audio-codec latent space. It reports WER 30.22 on LRS3 and 39.54 on LRS2. No code is released — [SLD-L2S](https://arxiv.org/html/2602.11477)
  - "From Faces to Voices" (KAIST, arXiv 2503.16956, Mar 2025) uses a learnable weighted sum of all AV-HuBERT Large layers. It takes 25 fps grayscale lip crops via RetinaFace + FAN and uses an OT-CFM (Matcha-TTS-style) mel decoder with a HiFi-GAN vocoder. It reports LRS3 WER 30.45 with 10 steps (30.37 with 1000 steps) and LRS2 WER 39.37 — [From Faces to Voices](https://arxiv.org/html/2503.16956v1)
  - "Intelligible L2S with Speech Units" (Interspeech 2023) has an "Ours (Proposed + AV-HuBERT)" configuration. Code is at github.com/choijeongsoo/lip2speech-unit — [project page](https://choijeongsoo.github.io/lip2speech-unit). A secondary summary reports LRS3 WER improving from 65.8% to 29.8% with AV-HuBERT plus speech units [secondary source, not verified in the PDF] — [arXiv 2305.19603](https://arxiv.org/pdf/2305.19603)

**Auto-AVSR (Ma et al., ICASSP 2023; mpc001/auto_avsr)**
- Model zoo:
  - VSR checkpoints are about 250M params. LRS3 VSR WER is 36.0% (LRS3 438 h), 24.6% (LRS3 + Vox2, 1759 h) and 20.3% (LRS2 + LRS3 + Vox2 + AVSpeech, 3291 h).
  - ASR checkpoints are 243M params, with WER of 2.0% and 1.0%.
  - Code is Apache-2.0. The weights "may have their own licenses... derived from the dataset used."
  - Install is `pip install torch torchvision torchaudio pytorch-lightning sentencepiece av`. Dependencies were reduced in a 2025-01-06 update.

  Source: [mpc001/auto_avsr](https://github.com/mpc001/auto_avsr)
- These are supervised VSR models (CTC/attention), not self-supervised encoders. Their front-end + conformer encoder can still serve as a feature extractor [inference].
- LipVoicer (ICLR 2024) uses a Ma et al. (2023) lip reader for LRS2/LRS3, with WER 14.6% and 19.1% — [LipVoicer](https://arxiv.org/html/2306.03258v2)

**RAVEn / BRAVEn (Haliassos et al.; ICLR 2023 / ICASSP 2024)**
- Video input is a 96×96 grayscale mouth crop with random 88×88 crops and horizontal flips as augmentation. The encoder is a 2D ResNet-18 with a 3D stem, followed by a Transformer. Transformer parameters are 41M (Base), 93M (Base+) and 328M (Large) — [BRAVEn arXiv 2404.02098](https://arxiv.org/html/2404.02098v1)
- Pretraining data: LRS3 433 h, VoxCeleb2-En 1,326 h, AVSpeech-En 1,323 h (about 3k h total; the paper is internally inconsistent, 3,082 vs 3,052).
- VSR WER:
  - Low-resource (30 h labeled): Base 43.4% vs RAVEn's 47.0%, Base+ 35.1%, Large 24.8%, and Large with self-training + LM 20.0%.
  - High-resource: Large 23.6%, or 20.1% with self-training + LM.

  Source: [BRAVEn](https://arxiv.org/html/2404.02098v1)
- Code is at github.com/ahaliassos/raven. The BRAVEn paper does not link weights directly — [BRAVEn](https://arxiv.org/html/2404.02098v1)

**USR (NeurIPS 2024) and USR 2.0 (ICLR 2026)**
- USR trains a single model for ASR, VSR and AVSR, built on BRAVEn-style pretraining. Base with LRS3 pretraining reports VSR 36.0%, ASR 3.2% and AVSR 3.0% WER. Code and models are at github.com/ahaliassos/usr — [USR arXiv 2411.02256](https://arxiv.org/html/2411.02256v1)
- USR 2.0 ("Pay Attention to CTC: Fast and Robust Pseudo-Labelling for Unified Speech Recognition", arXiv 2602.19316, ICLR 2026):
  - License is MIT.
  - Self-supervised encoder-only checkpoints: Base (LRS3), Base+ (LRS3 + Vox2), Large (LRS3 + Vox2) and Huge (LRS2 + LRS3 + Vox2 + AVS). Fine-tuned checkpoints are also provided.
  - LRS3 VSR WER: low-resource Base 36.2, Base+ 26.4, Large 23.7; high-resource Base+ 24.8, Large 21.5, Huge 17.6.
  - Dependencies: PyTorch, torchaudio, torchvision, PyTorch Lightning, Hydra, MediaPipe (the default landmark detector), vendored ESPnet and FFmpeg. RetinaFace + FAN is an optional alternative that requires CUDA.

  Source: [ahaliassos/usr2](https://github.com/ahaliassos/usr2)

**LLM-coupled VSR (relevant to the "VLM backbone" idea)**
- VSP-LLM (Yeo, Han, Kim, Ro; EMNLP 2024) maps a self-supervised visual speech model's features into a frozen LLM's input space, fine-tuned with (Q)LoRA. It averages features that share a discrete "visual speech unit" (deduplication), which shortens sequences by about 50% with minimal loss. It explicitly motivates the LLM as context modeling to resolve homophenes. An aggregator lists LRS3 WER 25.4; code is at sally-sh/vsp-llm — [VSP-LLM arXiv 2402.15151](https://arxiv.org/html/2402.15151v2)
- "From Hype to Insight" (Jain & Harte, ICASSP 2026; arXiv 2509.14880, v3 Jan 2026) finds that LLM decoders in VSR "refine contextual reasoning rather than visual features" and that gains are "primarily from lexical rather than semantic processing". Larger decoders give limited improvement. Their best result is Llama-2-13B at 24.7% WER on LRS3 and 47.0% on WildVSR. They conclude that stronger visual encoders are needed — [arXiv 2509.14880](https://arxiv.org/abs/2509.14880)

**Others named in the brief: not researched in this session**
- VATLM, LiRA, u-HuBERT, AV-data2vec, SyncVSR, Whisper-Flamingo and V2SFlow: see Gaps.
- V2SFlow (ICASSP 2025, rectified-flow video-to-speech) has LRS3 WER 27.55 as reported in SLD-L2S's comparison table — [SLD-L2S](https://arxiv.org/html/2602.11477)

### Inferences
- **Encoder ranking for this project:**
  1. **USR 2.0 / BRAVEn Base or Base+ self-supervised encoder.** It is MIT-licensed, uses a plain-PyTorch ResNet-18 + Transformer, and Base is only about 41M Transformer params. It fits easily on 16 GB and has no fairseq dependency.
  2. **Auto-AVSR VSR encoder.** Apache code, minimal dependencies, 250M total. It is supervised, so its features are already "text-shaped", which helps intelligibility.
  3. **AV-HuBERT Large.** It is the most proven in L2S papers. It needs fairseq (archived, Python 3.8) or a manual state-dict port, and its license is non-commercial (fine for a student project).
- Because all of these take 25 fps grayscale mouth ROIs, the GRID pipeline (25 fps) matches natively. The main work is landmark-based ROI extraction matching each repo's own preprocessing script.
- **Practical tip:** precompute frozen encoder features for all of GRID once (on CPU or MPS) and cache them to disk. Training the flow-matching head then never touches the big encoder. This sidesteps most MPS operator-coverage issues and memory limits.
- In the 2025–2026 L2S papers, about 30% WER on LRS3 is typical for generated speech from AV-HuBERT features. Only LipVoicer, which relies on a VSR transcript plus ASR guidance, goes lower (20.62).

### Gaps
- **Not researched this session:** VATLM, LiRA, u-HuBERT, AV-data2vec, SyncVSR and Whisper-Flamingo. From background knowledge [UNVERIFIED]:
  - u-HuBERT and VATLM are fairseq-based, AV-HuBERT-derived models.
  - LiRA (2021) is an older self-supervised visual encoder from Ma et al.
  - AV-data2vec is a Meta fairseq model; I am unsure whether its weights are public.
  - SyncVSR (Interspeech 2024) adds audio-token sync supervision.
  - Whisper-Flamingo injects AV-HuBERT visual features into Whisper via gated cross-attention, so it also needs AV-HuBERT.
- I could not confirm exact AV-HuBERT parameter counts or its exact input crop spec from a primary source.
- **MPS compatibility:** I found no specific report of AV-HuBERT or fairseq on Apple Silicon. General reports note old fairseq (0.10.x) install failures on macOS with newer Python, and MPS op gaps such as int64 cumsum and bfloat16 — [fairseq issue 3535](https://github.com/facebookresearch/fairseq/issues/3535), [pytorch issue 125254](https://www.github.com/pytorch/pytorch/issues/125254). `PYTORCH_ENABLE_MPS_FALLBACK=1` is a known workaround [UNVERIFIED in sources fetched].
- MediaPipe (USR 2.0's default detector) runs on macOS CPU, which suits Apple Silicon. RetinaFace + FAN in USR 2.0 is marked CUDA-only — [usr2](https://github.com/ahaliassos/usr2)

---

## Q2. Are general-purpose VLMs / CLIP-style features useful for lip reading?

### Takeaway
I found no evidence that generic VLMs or CLIP/SigLIP features are competitive for lip reading. The 2026 evidence points the other way:
- Omni models sample video at 1–2 fps, which is "often ineffective" for VSR. About 16 fps is the minimum for basic lip reading, and 25 fps is best.
- Even dedicated LLM-decoder AVSR models collapse on out-of-domain visual input.

A speech-specific visual encoder is the sensible backbone. The "VLA analogy" works best as a speech-specific encoder feeding a generative head, optionally with an LLM or text prior.

### Cited Findings
- HumanOmni-Speaker (Alibaba Tongyi; arXiv 2603.21664, v3 Jul 2026) states that existing omni models use sparse visual sampling (1–2 fps) and are "often ineffective when faced with VSR and AVSR tasks". It reports no visual-only WER for Qwen2.5-Omni, Gemini or GPT-4o; its Qwen2.5-Omni numbers, 3.47/3.63 on LRS2/LRS3, are audio-only. It finds that 2 fps is too low for lip reading, 16 fps enables basic reading and 25 fps gives the best viseme modeling. Its own model reaches VSR 29.8/33.4 on LRS2/LRS3, which it calls comparable to Auto-AVSR. It claims to be "the first Omni model to support end-to-end lip-reading natively" without lip cropping — [HumanOmni-Speaker](https://arxiv.org/html/2603.21664)
- A 2025 speaker-invariant lipreading paper argues that coarse, video-level contrastive (CLIP-like) alignment "fail[s] to achieve the frame-wise alignment needed for continuous lip movements" [search-snippet level; the paper was not fetched] — [arXiv 2506.07572](https://arxiv.org/pdf/2506.07572)
- LLM decoders improve VSR mainly through lexical context, not better visual understanding — [Jain & Harte, ICASSP 2026](https://arxiv.org/abs/2509.14880)
- A measurement-reading VLM benchmark shows that fine-grained visual reading is hard for VLMs: the best model, Gemini 2.5 Pro, scores 30.3%. This is indirect evidence only, since the benchmark does not test lip reading — [MeasureBench arXiv 2510.26865](https://arxiv.org/html/2510.26865v1)

### Inferences
- Generic image encoders (CLIP/SigLIP) are trained for semantic, image-level content at 224–448 px whole frames. Lip reading needs about 25 fps temporal resolution of small mouth-shape changes, so CLIP features very likely discard most of the visemic signal [inference, not measured].
- A VLM backbone would also cost far more memory (7B+ params), with no reported benefit on 16 GB.
- If the student wants the "VLA" flavour, the closest literature analog is VSP-LLM: a speech-specific SSL visual encoder → projector → LLM. Its text output (or hidden states) could condition the speech generator.
- A small 2D/3D CNN or SigLIP would be an interesting ablation ("generic vs speech-specific features"). It should not be the main system.

### Gaps
- I found no published table of GPT-4o, Gemini or Qwen-VL visual-only WER on LRS3 or GRID, and no direct CLIP-vs-AV-HuBERT VSR comparison. Searched twice without success.

---

## Q3. Transfer of LRS3/VoxCeleb-pretrained encoders to GRID; preprocessing

### Takeaway
Zero-shot, LRS3-trained VSR models do badly on GRID: Auto-AVSR 66.5% and AV-HuBERT 83.8% video-only WER in a Sep 2026 benchmark. The cause is mostly vocabulary, grammar and language-decoder mismatch ("hallucinates phrases to satisfy its language decoder"). This does not prove the visual features themselves are poor.

For L2S on GRID, the frozen encoder's features plus a GRID-trained head (or light fine-tuning) is the expected recipe. A from-scratch L2S model on GRID already reaches about 17% generated-speech WER.

### Cited Findings
- AVSRBench (Jain & Harte, arXiv 2609.10366, 9 Sep 2026) ran official checkpoints zero-shot on all of GRID (34 speakers, 32,895 utterances, 23.0 h). It used 96×96 mouth crops via RetinaFace + 68-point landmarks and 16 kHz audio. GRID WER (VO / AO / AV):

  | Model | VO | AO | AV |
  |---|---|---|---|
  | Auto-AVSR | 66.53 | 12.57 | 21.47 |
  | AV-HuBERT | 83.80 | 42.42 | 42.98 |
  | Llama-AVSR | 116.39 | 42.26 | 84.14 |

  The authors attribute this to the unseen six-word command grammar and to homophenes: "To compensate for visual ambiguity, the model hallucinates phrases to satisfy its language decoder." Auto-AVSR generalized best — [AVSRBench](https://arxiv.org/html/2609.10366v1)
- AV-HuBERT significantly outperforms prior lip-reading models, especially when labeled data is scarce. Fine-tuning on 30 h of LRS3 is competitive [search summary of AV-HuBERT-related papers] — [AV-HuBERT repo](https://www.github.com/facebookresearch/av_hubert)
- **Encoder-compatible preprocessing:**
  - BRAVEn/USR: 96×96 grayscale mouth crop, 88×88 random crop for training — [BRAVEn](https://arxiv.org/html/2404.02098v1)
  - USR 2.0: MediaPipe landmarks by default — [usr2](https://github.com/ahaliassos/usr2)
  - AV-HuBERT-based L2S: 25 fps, RetinaFace + FAN landmarks, grayscale — [From Faces to Voices](https://arxiv.org/html/2503.16956v1)
- HumanOmni-Speaker finds that 25 fps is best for viseme modeling, so GRID's native 25 fps is a good match — [HumanOmni-Speaker](https://arxiv.org/html/2603.21664)

### Inferences
- **What the poor zero-shot WERs mean.** They mainly reflect the decoders' language priors, which cannot produce "bin blue at f two now"-style strings, and they mix in homophene confusions. An L2S head trained on GRID with frozen pretrained features avoids the LM mismatch. The pretrained visual features should still carry useful lip-shape information.
- **Resolution and framing.** GRID's low resolution and frontal framing may not match LRS3's in-the-wild TED faces. Once mouth ROIs are resized to 88/96 px grayscale, the mismatch is mainly one of quality, not format.
- **Preprocessing.** Use the encoder repo's own crop script (landmark-aligned to a mean face, then mouth crop) so feature statistics match pretraining. The student's current crop pipeline may differ (e.g., fixed-box crops), and differences matter for frozen features [inference].
- **Experiment to propose.**
  - (a) A frozen SSL encoder plus a linear CTC probe on GRID. This measures phoneme or letter separability, e.g., b vs p vs d.
  - (b) The same probe on the student's own 3D-CNN features.
  - (c) Light fine-tuning (last layers or LoRA) if memory allows.

### Gaps
- I found no published fine-tuning study of AV-HuBERT, BRAVEn or USR on GRID, and no GRID L2S paper that uses a pretrained SSL visual encoder. The recent L2S flow papers (SLD-L2S, From Faces to Voices) evaluate only on LRS2/LRS3.

---

## Q4. Homophenes: how much is ambiguous, and how L2S/VSR models handle it

### Takeaway
The phoneme-to-viseme mapping is many-to-one (roughly 3:1). Homophene statistics vary widely by viseme definition: "about half of English words" have a homopheme, and the often-repeated "only about 30% of speech is lip-readable" is folklore, not a measured result.

Literature strategies for ambiguity:
- Context and LLM language modeling (VSP-LLM).
- Global visual context attention (VCA-GAN).
- Self-supervised speech units or HuBERT clusters as content targets (Choi 2023, Kim 2024/2025).
- Text/ASR classifier guidance at sampling time (LipVoicer).

Text guidance is the strongest single lever: removing ASR guidance in LipVoicer raises LRS3 WER from 21.4% to 86.2%. Without it, a diffusion model produces natural-sounding but much less intelligible, "smooth" speech.

### Cited Findings
- **How common homophenes are:**
  - "Roughly half of the words in the English language are homophemes" (e.g., "time"/"some") — [Sensors 2021, 21(23):7890](https://www.mdpi.com/1424-8220/21/23/7890/htm)
  - English has about three times as many phonemes as visemes, so "it is often claimed that only 30% of speech can be lip read" (a commonly repeated claim, not a measurement) — [EPFL Graph concept page](https://graphsearch.epfl.ch/en/concept/315084)
  - Viseme sets with fewer units create more viseme strings that map to multiple words. Reporting viseme error hides this — [Bear & Harvey, arXiv 1710.01142](https://arxiv.org/pdf/1710.01142)
  - Homophenes become noticeable below about 11 visual units, where a language model can no longer correct them — [arXiv 1909.07147](https://arxiv.org/pdf/1909.07147)
  - The visual dictionary is smaller than the acoustic one because homophones merge — [arXiv 1805.02924](https://arxiv.org/pdf/1805.02924)
- **Ambiguity makes L2S one-to-many.**
  - Homophenes ("bit"/"pit") make L2S one-to-many rather than one-to-one. Kim et al. (VCA-GAN) use global visual context in intermediate generator layers to resolve the viseme-to-phoneme ambiguity — [VCA-GAN arXiv 2204.01726](https://arxiv.org/pdf/2204.01726)
  - SLD-L2S frames visual-to-audio as "an inherently ill-posed, one-to-many problem" and uses flow matching plus semantic and speech-language-model losses to preserve content — [SLD-L2S](https://arxiv.org/html/2602.11477)
- **Text/ASR guidance (LipVoicer, ICLR 2024).**
  - A VSR model transcribes the silent video. A diffusion MelGen is conditioned on the video and guided by a pretrained ASR classifier toward that transcript. Rationale: "several phonemes can be attributed to the same lip movement sequence," and the text removes the uncertainty.
  - On LRS3, setting the ASR guidance weight to 0 raises WER from 21.4% to 86.2%. Unguided spectrograms show "smooth transitions" and sound more natural but less intelligible.
  - GRID settings: CFG weight 2.6, ASR guidance weight 0.7, guidance starting at step 300 of 400.
  - GRID results are human MOS only (intelligibility 3.75 vs VCA-GAN 3.29, GT 4.43).

  Source: [LipVoicer](https://arxiv.org/html/2306.03258v2)
- **Self-supervised speech units instead of text.**
  - "Let There Be Sound" (Kim, Kim, Chung; arXiv 2308.15256) predicts HuBERT-Large layer-12 K-means (200 cluster) indices from video with cross-entropy, so no text is needed.
  - It states that L1/L2 reconstruction gives "blurry and over-smoothed synthetic speech" and adds a flow-based post-net.
  - It uses a from-scratch 3D-conv + ResNet18 + Conformer video encoder with a speaker embedding.
  - **GRID results** (80/10/10 split, 33 speakers):

    | | WER | CER |
    |---|---|---|
    | Ground truth | 12.20 | 7.03 |
    | Vocoded | 12.44 | — |
    | VCA-GAN | 17.62 | — |
    | SVTS | 23.30 | — |
    | Multi-task (text) | 30.56 | — |
    | Ours | 17.07 | 9.17 |

    MOS is 4.46 for naturalness and 4.63 for intelligibility. These WERs come from an ASR on the generated audio; the ASR scores 12.2% even on ground truth.

  Source: [Let There Be Sound](https://arxiv.org/html/2308.15256v2)
- Choi, Kim, Ro (Interspeech 2023) predict both mel and discrete speech units. A multi-input vocoder uses the units to produce a clear waveform "even from blurry and noisy mel-spectrogram". Content supervision needs no text — [project page](https://choijeongsoo.github.io/lip2speech-unit)
- "From Faces to Voices" (2025) predicts HuBERT-Large 1,000-unit content tokens from lips with label-smoothed cross-entropy and an auxiliary masked conv block. A flow-matching (OT-CFM) mel decoder with CFG 0.7 follows. It uses no text or ASR guidance — [From Faces to Voices](https://arxiv.org/html/2503.16956v1)
- LLM context for homophenes: VSP-LLM — [arXiv 2402.15151](https://arxiv.org/html/2402.15151v2). However, LLM gains are mostly lexical — [arXiv 2509.14880](https://arxiv.org/abs/2509.14880)
- In VSR on GRID, decoder language priors make models "hallucinate phrases" to resolve visual ambiguity — [AVSRBench](https://arxiv.org/html/2609.10366v1)

### Inferences
- **GRID-specific: some homophenes are irreducible.** The GRID grammar is command / color / preposition / letter / digit / adverb, with letters A–Z except W — [LipNet](https://arxiv.org/pdf/1611.01599); grammar recalled from that paper and not re-verified here.
  - In the letter slot, "b"/"p" differ only by voicing (invisible on the lips), "d"/"t" likewise, and "f"/"v" differ only by voicing. Many letters share the "-ee" rhyme.
  - Each letter is uniformly random given the rest of the sentence, so **no language model or context can disambiguate it**.
  - Any generator, text-based or direct, can only guess there, unless subtle visual cues (aspiration timing, closure duration) are learned.
  - This likely explains the student's residual errors: they are near the information-theoretic floor, not a model deficiency [inference].
- **Direct video→speech vs a text pipeline.**
  - A text pipeline commits to a discrete choice, wrong or right. A conditional generative model trained with a regression loss averages and produces a blurred, in-between sound — the "over-smoothed" observation in Let There Be Sound.
  - A well-trained diffusion or flow model, sampled stochastically, should in principle sample one mode, i.e., commit to a plausible phoneme, sometimes the wrong one. Deterministic ODE sampling with few steps, or strong CFG, can still drift toward mode-averaging in poorly separated regions.
  - LipVoicer's "smooth transitions" without guidance, and its 86% WER, suggest that unguided diffusion often under-articulates rather than cleanly committing.

  This is my synthesis; no paper measured "commit vs average" for homophenes specifically.
- **Most promising design for GRID.** Add a discrete content bottleneck:
  - Either CTC over phonemes/letters (the student already has this), or HuBERT units predicted by cross-entropy.
  - Condition the flow-matching decoder on these (soft posteriors or sampled tokens).
  - Optionally add LipVoicer-style ASR guidance.

  This makes the ambiguity explicit (a categorical posterior over b/p) instead of hiding it in continuous mel space. It also lets the evaluation report how often homophene slots are wrong.

### Gaps
- I found no reliable, primary-source percentage of GRID-specific phoneme confusions that are visually indistinguishable. Classic viseme tables (e.g., Fisher 1968, Lander) were not retrieved.
- I found no paper that measures whether diffusion/flow L2S averages vs commits on homophenes.
- LipVoicer's GRID WER is not reported; only MOS is.

---

## Q5. Lightweight from-scratch alternatives on GRID

### Takeaway
On GRID, from-scratch models are already strong:
- VSR WER is about 1–2% on seen speakers and about 8.6–11.4% on unseen speakers.
- From-scratch L2S (3D-conv + ResNet18 + Conformer) gives about 17% ASR-measured WER on generated speech, against a 12.2% ground-truth floor.

A pretrained SSL encoder is therefore more of a robustness, speaker-generalization and "rich features" bet than a necessity. It should be compared against a from-scratch ResNet-18 + Conformer/TCN baseline.

### Cited Findings
- LipNet's unseen-speaker WER is 11.4% (speakers 1, 2, 20 and 22 held out) — [LipNet arXiv 1611.01599](https://arxiv.org/pdf/1611.01599)
- The following figures are reported via a 2026 NAVIR preprint summary table [secondary, unverified against originals] — [NAVIR arXiv 2609.24391](https://arxiv.org/pdf/2609.24391):
  - A hybrid CTC/attention conformer with mutual-information regularisation (Wu et al., 2024) reaches 1.83% WER overlapped and 10.21% unseen.
  - HLR-Net reaches 9.7% unseen.
  - A 3D-2D-CNN-BLSTM-HMM reports 1.3% seen and 8.6% unseen.
- A landmark-guided cross-speaker lip reading paper (2024) targets the unseen-speaker gap — [arXiv 2403.16071](https://arxiv.org/pdf/2403.16071)
- The from-scratch L2S encoder in "Let There Be Sound" (3D conv + ResNet18 + Conformer + speaker embedding) achieves GRID WER 17.07 / CER 9.17 on generated speech — [Let There Be Sound](https://arxiv.org/html/2308.15256v2)
- The BRAVEn/RAVEn front-end itself is a ResNet-18 (3D stem) + Transformer. A from-scratch model with the same architecture gives a clean ablation against the pretrained one — [BRAVEn](https://arxiv.org/html/2404.02098v1)

### Inferences
- **Recommended ablation ladder:**
  - (A) The student's existing 3D-CNN + BiLSTM, as features or via CTC posteriors.
  - (B) A from-scratch ResNet-18 (3D stem) + small Conformer/TCN.
  - (C) Frozen BRAVEn/USR-Base or Auto-AVSR encoder features.
  - (D) Frozen AV-HuBERT Large features, if fairseq can be made to work or the weights can be ported.

  The same flow-matching head would sit on top of each.
- **Metrics:**
  - Whisper/ASR WER on generated audio.
  - Per-slot letter accuracy (the homophene slot).
  - STOI/ESTOI.
  - Unseen-speaker split performance, where pretraining should help most.
- **Memory.** On 16 GB unified memory, (B) and (C)-Base are trainable end-to-end. (D) should be frozen with cached features.

### Gaps
- I found no GRID study comparing pretrained SSL encoders to from-scratch encoders for L2S or VSR.
- The NAVIR table numbers are second-hand.
- I did not verify the Ma et al. "Towards practical lipreading" (ResNet-18 + MS-TCN) GRID numbers.
