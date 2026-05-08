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


## MEMBER CONTRIBUTION
```
Team Member 1 (22201039_Zarin Tasnim Vega)

Contribution: Piano-Roll Representations, Autoencoders and Generative Latent Spaces

  - Data Pipeline: Implemented the pretty_midi piano-roll extraction, including
    the sparsity filtering logic (discarding windows with <2% active cells) and
    EDA visualizations (duration and pitch histograms).
  - Task 1 (LSTM Autoencoder): Designed the Encoder/Decoder architecture.
    Identified the piano-roll sparsity problem and successfully implemented the
    Focal Loss function (with w^+ = 20) to prevent the model from collapsing to
    silence.
  - Task 2 (Variational Autoencoder): Implemented the reparameterization trick
    and the KL divergence loss. Diagnosed posterior collapse and implemented KL
    Annealing (30-epoch warmup) to fix it.
  - Experiments & Visuals: Coded the latent space interpolation experiment
    (generating intermediate samples z_\alpha) and generated the AE/VAE loss
    curves and interpolation heatmaps.
  - Report Writing: Authored the Abstract, Introduction, Dataset & Preprocessing
    (Piano-roll section), Methods (Tasks 1 & 2), and the corresponding Results
    and Discussion sections regarding sparsity, focal loss, and posterior
    collapse.

Team Member 2 (22201040_Rifah Tasnim Labonno)

Contribution: Token Sequences, Autoregressive Modeling and Reinforcement Learning

  - Data Pipeline & Baselines: Implemented the miditok REMI tokenization
    pipeline. Coded the two baseline models (Random Note Generator and Markov
    Chain) for comparative evaluation.
  - Task 3 (Transformer Generator): Built the GPT-style decoder-only
    Transformer. Implemented the absolute positional encodings, the crucial
    upper-triangular causal mask, and the temperature/top-k sampling generation
    loops.
  - Task 4 (RLHF): Built the entire RLHF pipeline. Created the proxy reward
    feature extractor, trained the Reward Model, and implemented the REINFORCE
    policy gradient algorithm with batch normalization and the reference-policy
    KL penalty to prevent reward hacking.
  - Evaluation & Visuals: Authored the comprehensive metrics.py script
    (calculating Pitch Similarity, Rhythm Diversity, Repetition Ratio, and
    Perplexity). Conducted the Human Listening Survey and generated the RLHF
    training curves and final comparison tables.
  - Report Writing: Authored the Related Work, Methods (Tasks 3 & 4, Baselines,
    Evaluation Metrics), Results (Tasks 3 & 4, Quantitative Comparison), and the
    corresponding Discussion (Reward hacking, causal masking) and Conclusion.

```

## DRIVE LINK OF MIDI OUTPUTS
```

https://drive.google.com/drive/folders/1waeRpQ2LSkVVgfQk7hSo0nj7RdD44KFZ?usp=sharing

```

## DRIVE LINK OF REPORT PDF
```
https://drive.google.com/file/d/1LfPqkn6W8hnGhOd4LYKyME-YAvyqrbwQ/view?usp=sharing

```
```
