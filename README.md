# CSE425 Music Generation Project

## FILE STRUCTURE

```
cse425_music/
├── data/
│   ├── maestro/               ← downloaded dataset
│   └── processed/
│       ├── train.npy          ← piano-roll windows (N, 128, 88)
│       ├── validation.npy
│       ├── test.npy
│       ├── tokens_train.npy   ← REMI token sequences
│       ├── tokens_validation.npy
│       ├── tokens_test.npy
│       └── vocab_size.npy
├── outputs/
│   ├── task1_ae/              ← 5 generated MIDI files
│   ├── task2_vae/             ← 8 multi-genre + 8 interpolation MIDIs + heatmap
│   ├── task3_transformer/     ← 10 long compositions
│   ├── task4_rlhf/            ← 10 RLHF-tuned compositions + survey data
│   └── baselines/
│       ├── random/            ← 5 random baseline MIDIs
│       └── markov/            ← 5 Markov chain baseline MIDIs
├── checkpoints/               ← best model weights (.pt)
├── plots/
│   ├── eda_duration_and_pitch.png           ← Duration histogram + pitch distribution
│   ├── task1_loss_curve.png
│   ├── task2_loss_curves.png
│   ├── task3_perplexity.png
│   ├── task4_rl_curves.png
│   ├── task4_before_after_comparison.png    ← Before/After RLHF comparison
│   ├── comparison_table.png                 ← 6 models × 4 metrics table
│   └── human_scores_bar_chart.png           ← Human evaluation scores
└── *.py                                      
```

---

## DELIVERABLES CHECKLIST

### Task 1: LSTM Autoencoder
- [x] `plots/task1_loss_curve.png` — reconstruction loss curve
- [x] `outputs/task1_ae/generated_01.mid` through `generated_05.mid` — 5 MIDI samples
- [x] `checkpoints/task1_ae.pt` — trained model

### Task 2: Variational Autoencoder
- [x] `plots/task2_loss_curves.png` — total + KL/recon separate plots
- [x] `outputs/task2_vae/multigenre_01.mid` … `multigenre_08.mid` — 8 multi-genre samples
- [x] `outputs/task2_vae/interpolation_01_*.mid` … `interpolation_08_*.mid` — 8 latent interpolations
- [x] `outputs/task2_vae/interpolation_heatmap.png` — piano-roll heatmap visualization of 8-step interpolation
- [x] `checkpoints/task2_vae.pt`

### Task 3: Transformer Generator
- [x] `plots/task3_perplexity.png` — perplexity per epoch
- [x] `outputs/task3_transformer/composition_01.mid` … `composition_10.mid` — 10 long compositions
- [x] `checkpoints/task3_transformer.pt`
- [x] Console output table: baseline comparison with perplexity values

### Task 4: RLHF (REINFORCE)
- [x] `outputs/task4_rlhf/human_survey.json` — filled survey with ≥10 participants (20 samples rated)
- [x] `outputs/task4_rlhf/survey_samples/*.mid` — 20 samples rated by humans
- [x] `checkpoints/reward_model.pt` — trained reward model
- [x] `plots/task4_rl_curves.png` — reward, loss, KL divergence per RL iteration
- [x] `plots/task4_before_after_comparison.png` — before/after metrics visualization
- [x] `outputs/task4_rlhf/rlhf_composition_01.mid` … `rlhf_composition_10.mid`
- [x] Before/After comparison table (printed to console + chart)

### Exploratory Data Analysis (EDA)
- [x] `plots/eda_duration_and_pitch.png` — duration histogram + pitch class distribution
  - Duration: mean piece length from MAESTRO training set
  - Pitch: frequency of 12 pitch classes (C, C#, D, …, B)

### Evaluation Metrics & Results
- [x] `plots/comparison_table.png` — formatted table visualization (6 models × 4 metrics)
- [x] `plots/human_scores_bar_chart.png` — mean human ratings with error bars (Task 4 RLHF)
- [x] Baseline models: Random Generator, Markov Chain
- [x] Task models: Task 1 (LSTM AE), Task 2 (VAE), Task 3 (Transformer), Task 4 (RLHF)
- [x] Metrics: Rhythm Diversity, Repetition Ratio, Pitch Histogram Similarity, File Count
- [x] MIDI validity check: 78/79 valid (≥50 notes, ≥5 seconds)

---

## LISTENING TO The MIDI FILES

Used Windows Media Player

## COMMON PROBLEMS & FIXES

### CUDA out of memory
Reduce `BATCH_SIZE` from 32 → 16 in the relevant script.

### All generated MIDI files are silent
- Lower the generation threshold: change `threshold=0.3` → `threshold=0.1`
- Check the loss curve — if it plateaued from epoch 1, the pos_weight needs tuning

### miditok import error (Task 3)
```bash
pip install miditok symusic
```

### MAESTRO files raise parsing errors
Normal — a handful of files are malformed. The scripts skip them automatically.

### Loss is NaN
- Gradient explosion: already handled by `clip_grad_norm_(max_norm=1.0)`
- Double sigmoid: do NOT add `torch.sigmoid()` to model output — the loss function does it internally

### Task 3 generates gibberish
- Verify causal mask is active (it is by default in the code)
- Try lowering `TEMPERATURE` to 0.8 for more coherent output
- Train for more epochs if perplexity is still high after 40

---

## ARCHITECTURE SUMMARY

### Task 1 — LSTM Autoencoder
```
Input (B,128,88) → LSTM Encoder → z (B,64) → LSTM Decoder → Logits (B,128,88)
Loss: Focal BCE (γ=2, pos_weight=20) handles 97% sparsity
```

### Task 2 — VAE
```
Input → LSTM Encoder → µ,logσ² → reparameterize → z → LSTM Decoder → Logits
Loss: Focal BCE + β·KL   (β annealed 0→1 over 30 epochs)
Generation: z ~ N(0,I) → Decoder
Interpolation: z_α = (1-α)µ₁ + α·µ₂
```

### Task 3 — Transformer
```
Tokens[:-1] → Embedding + PositionalEncoding → 4× (MaskedSelfAttn + FFN) → Linear → Logits
Loss: CrossEntropy   Metric: Perplexity = exp(avg cross-entropy)
Generation: seed token → top-k sampling (k=50, T=1.0) → 1024 tokens
```

### Task 4 — RLHF 
```
Task 3 Transformer (frozen θ₀) → Generate sequences X ~ pθ(·)
Reward Model: trained on human feedback r ∈ [1,5] → normalized [-1,1]
Policy Loss: E[r · ∇θ log pθ(X)] - λ·KL(pθ || pθ₀)
Generation: seed token → top-k sampling (k=50, T=1.0) → RLHF-tuned 1024 tokens
```


## METRIC REFERENCE

| Metric | Formula | Good range |
|--------|---------|-----------|
| Pitch Hist Sim | Σ\|p_i - q_i\| | Lower = better (0=identical) |
| Rhythm Diversity | unique_durations / total_notes | Higher = more variety |
| Repetition Ratio | repeated 4-grams / total 4-grams | 0.1–0.5 = coherent |
| Perplexity | exp(avg cross-entropy) | Lower = better |
| Human Score | mean rating 1–5 | Higher = better |
