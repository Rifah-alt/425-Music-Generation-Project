"""
TASK 4: Reinforcement Learning from Human Feedback (RLHF)
        REINFORCE Policy Gradient Fine-Tuning of Task 3 Transformer

Theory basis (from linked articles):
  Policy   : pθ(X) = the Task 3 Transformer — our "agent" that generates music
  Action   : each autoregressively sampled token is one action
  State    : the sequence of tokens generated so far
  Reward   : human listening score r ∈ [1,5], normalized to [-1,1]
  Objective: max J(θ) = E[r(X_gen)]
  Update   : ∇θJ(θ) = (1/N) Σᵢ r̂^(i) · ∇θ log pθ(X^(i))   (REINFORCE)
  KL Penalty (prevents reward hacking):
    J'(θ) = E[r] - λ · KL(pθ || pθ₀)
    where pθ₀ is the frozen reference (original Task 3 weights)

"""

import os, math, json, random, numpy as np
import torch, torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pretty_midi
from collections import Counter

# Paths & Config 
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR  = "outputs/task4_rlhf"
CKPT_T3  = "checkpoints/task3_transformer.pt"
CKPT_T4  = "checkpoints/task4_rlhf.pt"
CKPT_RM  = "checkpoints/reward_model.pt"
SURVEY_F = "outputs/task4_rlhf/human_survey.json"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# Transformer hyperparams 
D_MODEL     = 256
N_HEADS     = 8
N_LAYERS    = 4
D_FF        = 1024
DROPOUT     = 0.1
MAX_SEQ_LEN = 512

# RLHF hyperparams 
RL_ITERS      = 20       # REINFORCE iterations
BATCH_SIZE_RL = 8        # samples per iteration (keep low for 4 GB VRAM)
GEN_LEN       = 256      # tokens generated per sample during RL
GEN_LEN_FINAL = 512      # tokens for final 10 compositions
TEMPERATURE   = 1.0
TOP_K         = 50
LR_RL         = 1e-5     # very small LR — standard for RLHF fine-tuning
KL_COEFF      = 0.1      # λ: higher → closer to original policy (less hacking)

# Reward model hyperparams 
RM_EPOCHS  = 30
RM_LR      = 3e-4
FEAT_DIM   = 5

print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU   : {torch.cuda.get_device_name(0)}")


# ════════════════════════════════════════════════════════════════════════════
#  MODULE 1 — Transformer (identical to Task 3)
# ════════════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=2048):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x):
        return self.drop(x + self.pe[:, :x.size(1)])


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff    = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(),
                                   nn.Dropout(dropout), nn.Linear(d_ff, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x, mask):
        n = self.norm1(x)
        a, _ = self.attn(n, n, n, attn_mask=mask)
        x = x + self.drop(a)
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


class MusicTransformer(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, D_MODEL)
        self.pos_enc = PositionalEncoding(D_MODEL, DROPOUT, max_len=MAX_SEQ_LEN + 1)
        self.layers  = nn.ModuleList([
            TransformerBlock(D_MODEL, N_HEADS, D_FF, DROPOUT) for _ in range(N_LAYERS)
        ])
        self.norm = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def causal_mask(self, T, device):
        return torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()

    def forward(self, x):
        T    = x.size(1)
        mask = self.causal_mask(T, x.device)
        h    = self.pos_enc(self.tok_emb(x))
        for layer in self.layers:
            h = layer(h, mask)
        return self.head(self.norm(h))   # (B, T, vocab_size)



# Musical Feature Extraction


def extract_features(midi_path):
    """
    5 scalar features that correlate with musical quality.
    Returns np.array(5,) or None.
    """
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
    except Exception:
        return None
    notes = sorted([n for inst in pm.instruments for n in inst.notes], key=lambda n: n.start)
    if len(notes) < 5:
        return None

    dur      = max(pm.get_end_time(), 1e-6)
    pitches  = [n.pitch for n in notes]
    vels     = [n.velocity for n in notes]

    # 1. note density (normalized; 0.4 ≈ 8 notes/sec is ideal)
    density  = min(len(notes) / dur / 20.0, 1.0)

    # 2. pitch range (88 keys total)
    p_range  = (max(pitches) - min(pitches)) / 87.0

    # 3. rhythm diversity
    d_ms = [round((n.end - n.start) * 1000 / 50) * 50 for n in notes]
    rhy_div = len(set(d_ms)) / len(d_ms)

    # 4. repetition ratio (4-grams)
    ngrams = [tuple(pitches[i:i+4]) for i in range(len(pitches) - 3)]
    counts = Counter(ngrams)
    rep    = sum(1 for c in counts.values() if c > 1) / max(len(ngrams), 1)

    # 5. avg velocity (0-1)
    avg_vel = np.mean(vels) / 127.0

    return np.array([density, p_range, rhy_div, rep, avg_vel], dtype=np.float32)


# Reward Functions


def proxy_reward(midi_path):
    """
    Rule-based reward when no human labels exist.
    Encodes known musical quality heuristics.
    Returns float in [-1, 1].

    """
    f = extract_features(midi_path)
    if f is None:
        return -1.0
    density, p_range, rhy_div, rep, avg_vel = f

    r = 0.0
    # Ideal density: not too sparse, not too dense
    r += 1.0 - abs(density - 0.4) * 2.5
    # Wide pitch range is better (more musical)
    r += min(p_range * 2.0, 1.0)
    # Rhythm variety is better
    r += rhy_div
    # Repetition: penalize extremes
    if   rep < 0.05: r -= 0.5   # too random
    elif rep > 0.80: r -= 1.0   # too repetitive
    else:            r += 0.5   # coherent range

    return float(np.clip(r / 3.0, -1.0, 1.0))


class RewardModel(nn.Module):
    """Small MLP: features(5,) → scalar reward."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FEAT_DIM, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),       nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def get_reward(midi_path, rm=None):
    """Return scalar reward. Prefer trained reward model, fall back to proxy."""
    if rm is not None:
        f = extract_features(midi_path)
        if f is not None:
            rm.eval()
            with torch.no_grad():
                x = torch.tensor(f, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                return rm(x).item()
    return proxy_reward(midi_path)



# MIDI / Token Utilities


def tokens_to_midi(ids, vocab_size, out_path):
    """Converting token list to MIDI, with heuristic fallback."""
    try:
        from miditok import REMI, TokenizerConfig, TokSequence
        cfg = TokenizerConfig(num_velocities=32, use_chords=False, use_programs=False)
        tok = REMI(cfg)
        clipped = [min(int(t), vocab_size - 1) for t in ids]
        seq     = TokSequence(ids=clipped)
        midi    = tok.tokens_to_midi([seq])
        midi.dump(out_path)
    except Exception:
        _fallback_midi(ids, out_path)


def _fallback_midi(ids, out_path, fs=16):
    pm   = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0, name="Piano")
    t    = 0.0
    dt   = 1.0 / fs
    for tok in ids:
        pitch = 21 + (int(tok) % 88)
        inst.notes.append(pretty_midi.Note(velocity=80, pitch=pitch,
                                           start=t, end=t + dt * 2))
        t += dt
    pm.instruments.append(inst)
    pm.write(out_path)


def find_seed_token():
    """Finding the Bar token id from the REMI tokenizer."""
    try:
        from miditok import REMI, TokenizerConfig
        cfg = TokenizerConfig(num_velocities=32, use_chords=False, use_programs=False)
        tok = REMI(cfg)
        for k, v in tok.vocab.items():
            if "Bar" in str(k):
                return v
    except Exception:
        pass
    return 2   # default fallback


#  single sample generation

@torch.no_grad()
def generate_sample(policy, vocab_size, seed_id, gen_len=GEN_LEN):
    """Autoregressive top-k sampling. Returns token id list."""
    policy.eval()
    ids = [seed_id]
    for _ in range(gen_len):
        inp    = torch.tensor([ids[-MAX_SEQ_LEN:]], dtype=torch.long, device=DEVICE)
        logits = policy(inp)[0, -1, :] / TEMPERATURE
        logits[0] = -float("inf")           # suppress padding
        topk_val, _ = torch.topk(logits, TOP_K)
        logits[logits < topk_val[-1]] = -float("inf")
        probs  = torch.softmax(logits, -1)
        nxt    = torch.multinomial(probs, 1).item()
        ids.append(nxt)
    return ids



# REINFORCE Core

def log_prob_of_sequence(policy, token_ids, vocab_size):
    """
    Computes  Σ_t log pθ(x_t | x_{<t})  over the generated sequence.
    This is the quantity we scale by the reward in REINFORCE.
    Requires grad on policy parameters. 

    """
    if len(token_ids) < 2:
        return torch.zeros(1, device=DEVICE, requires_grad=True).sum()

    ids  = [min(int(t), vocab_size - 1) for t in token_ids]
    inp  = torch.tensor([ids[:-1]], dtype=torch.long, device=DEVICE)
    tgt  = torch.tensor([ids[1:]],  dtype=torch.long, device=DEVICE)

    if inp.size(1) > MAX_SEQ_LEN:
        inp = inp[:, -MAX_SEQ_LEN:]
        tgt = tgt[:, -MAX_SEQ_LEN:]

    logits   = policy(inp)                              # (1, T, V)
    log_p    = torch.log_softmax(logits, dim=-1)
    tgt_clip = tgt.clamp(0, logits.size(-1) - 1)
    lp       = log_p.gather(2, tgt_clip.unsqueeze(-1)).squeeze(-1)  # (1, T)
    return lp.sum()                                     # scalar, has grad


def kl_penalty(policy, ref_policy, token_ids, vocab_size):
    """
    Approximate KL(pθ || pθ₀) at the last context position.
    Keeps the fine-tuned policy close to the reference (anti-reward-hacking). 

    """
    if len(token_ids) < 2:
        return torch.zeros(1, device=DEVICE).sum()

    ids = [min(int(t), vocab_size - 1) for t in token_ids[-MAX_SEQ_LEN:]]
    inp = torch.tensor([ids[:-1]], dtype=torch.long, device=DEVICE)
    if inp.size(1) == 0:
        return torch.zeros(1, device=DEVICE).sum()

    with torch.no_grad():
        ref_log_p = torch.log_softmax(ref_policy(inp)[:, -1, :], dim=-1)

    curr_log_p = torch.log_softmax(policy(inp)[:, -1, :], dim=-1)
    curr_p     = curr_log_p.exp()
    kl         = (curr_p * (curr_log_p - ref_log_p)).sum(dim=-1).mean()
    return kl


# Training Loop

def rl_train(policy, ref_policy, rm, vocab_size, seed_id):
    """
    Full REINFORCE training loop.

    Per iteration:
      1. Generate BATCH_SIZE_RL sequences from pθ  (policy.eval during generation)
      2. Save each as temp MIDI, compute reward r^(i)
      3. Normalize rewards: r̂ = (r - μ) / (σ + ε)       ← variance reduction
      4. policy.train(); compute loss:
           L = -(1/N) Σᵢ r̂^(i) · log pθ(X^(i))
             + λ · KL(pθ || pθ₀)
      5. loss.backward(); clip gradients; optimizer.step() 

    """
    optimizer = torch.optim.Adam(policy.parameters(), lr=LR_RL)
    tmp_dir   = os.path.join(OUT_DIR, "_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    all_rewards, all_losses, all_kls = [], [], []
    best_reward = -float("inf")

    print(f"\n── REINFORCE RL Fine-Tuning ──")
    print(f"   Iterations   : {RL_ITERS}")
    print(f"   Batch size   : {BATCH_SIZE_RL}")
    print(f"   Gen length   : {GEN_LEN} tokens")
    print(f"   LR           : {LR_RL}")
    print(f"   KL coeff (λ) : {KL_COEFF}")
    print()

    for it in range(1, RL_ITERS + 1):

        # Generating batch and collect rewards 
        sequences = []
        rewards   = []
        for b in range(BATCH_SIZE_RL):
            ids      = generate_sample(policy, vocab_size, seed_id, GEN_LEN)
            tmp_path = os.path.join(tmp_dir, f"tmp_{b}.mid")
            tokens_to_midi(ids, vocab_size, tmp_path)
            r        = get_reward(tmp_path, rm)
            sequences.append(ids)
            rewards.append(r)

        # Normalizing rewards (variance reduction for REINFORCE stability) 
        r_arr  = np.array(rewards, dtype=np.float32)
        r_norm = (r_arr - r_arr.mean()) / (r_arr.std() + 1e-8)

        # Computing REINFORCE loss + KL penalty
        policy.train()
        optimizer.zero_grad()

        loss_reinforce = torch.zeros(1, device=DEVICE).sum()
        loss_kl        = torch.zeros(1, device=DEVICE).sum()

        for ids, r_hat in zip(sequences, r_norm):
            lp  = log_prob_of_sequence(policy, ids, vocab_size)
            # Negative because we maximize (gradient ASCENT = minimize negative)
            loss_reinforce = loss_reinforce + (-float(r_hat) * lp)

            kl  = kl_penalty(policy, ref_policy, ids, vocab_size)
            loss_kl = loss_kl + kl

        loss_reinforce = loss_reinforce / BATCH_SIZE_RL
        loss_kl        = loss_kl        / BATCH_SIZE_RL
        total_loss     = loss_reinforce + KL_COEFF * loss_kl

        # Gradient update
        total_loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()

        # Logging 
        mean_r  = float(r_arr.mean())
        mean_kl = float(loss_kl.item())
        all_rewards.append(mean_r)
        all_losses.append(float(total_loss.item()))
        all_kls.append(mean_kl)

        print(f"  Iter {it:3d}/{RL_ITERS} | Reward: {mean_r:+.4f} | KL: {mean_kl:.4f} | Loss: {total_loss.item():.4f}")

        if mean_r > best_reward:
            best_reward = mean_r
            torch.save(policy.state_dict(), CKPT_T4)

    print(f"\n  Best mean reward: {best_reward:.4f}")
    print(f"  Best checkpoint: {CKPT_T4}")
    return all_rewards, all_losses, all_kls



# Human Survey

def generate_survey_samples(policy, vocab_size, seed_id, n=20):
    """Generate samples for human rating."""
    survey_dir = os.path.join(OUT_DIR, "survey_samples")
    os.makedirs(survey_dir, exist_ok=True)
    print(f"\n── Generating {n} survey samples ──")
    paths = []
    for i in range(1, n + 1):
        ids  = generate_sample(policy, vocab_size, seed_id)
        path = os.path.join(survey_dir, f"survey_{i:02d}.mid")
        tokens_to_midi(ids, vocab_size, path)
        paths.append(path)
        print(f"  survey_{i:02d}.mid")

    template = {
        "instructions": (
            "Convert each MIDI to audio using MuseScore (File→Open→Export→MP3). "
            "Play each clip to at least 10 listeners. "
            "Ask them to rate 1 (worst) to 5 (best) for musical quality. "
            "Add each rater's score to 'scores_per_rater'. "
            "Do NOT modify the file name or structure."
        ),
        "samples": [
            {
                "file": f"survey_{i:02d}.mid",
                "scores_per_rater": [],
                "mean_score": None
            }
            for i in range(1, n + 1)
        ]
    }
    with open(SURVEY_F, "w") as f:
        json.dump(template, f, indent=2)

    print(f"\n  Survey template written → {SURVEY_F}")
    return paths


def load_survey(path):
    """Return {filename: normalized_reward} from filled survey JSON."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    out = {}
    for s in data["samples"]:
        scores = s.get("scores_per_rater", [])
        if scores:
            out[s["file"]] = (float(np.mean(scores)) - 3.0) / 2.0
    return out if out else None


def train_reward_model(survey_path):
    """Train RM on human-labeled features."""
    print("\n Training Reward Model ")
    rewards_dict = load_survey(survey_path)
    if not rewards_dict:
        print("  Survey not filled. Reward model skipped — using proxy reward.")
        return None

    survey_dir = os.path.join(OUT_DIR, "survey_samples")
    X_list, y_list = [], []
    for fname, r in rewards_dict.items():
        feat = extract_features(os.path.join(survey_dir, fname))
        if feat is not None:
            X_list.append(feat)
            y_list.append(r)

    if len(X_list) < 5:
        print(f"  Only {len(X_list)} labeled samples (need ≥5). Using proxy.")
        return None

    X  = torch.tensor(np.array(X_list), dtype=torch.float32)
    y  = torch.tensor(np.array(y_list), dtype=torch.float32)
    ds = torch.utils.data.TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=8, shuffle=True)

    rm  = RewardModel().to(DEVICE)
    opt = torch.optim.Adam(rm.parameters(), lr=RM_LR, weight_decay=1e-4)
    mse = nn.MSELoss()

    print(f"  Training on {len(X_list)} labeled samples ({RM_EPOCHS} epochs)")
    for ep in range(1, RM_EPOCHS + 1):
        rm.train()
        ep_loss = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = mse(rm(xb), yb)
            loss.backward()
            opt.step()
            ep_loss += loss.item()
        if ep % 10 == 0:
            print(f"  RM epoch {ep:3d} | MSE: {ep_loss/len(dl):.4f}")

    torch.save(rm.state_dict(), CKPT_RM)
    print(f"  Reward model saved → {CKPT_RM}")
    return rm


#  MODULE 9 — Before/After Comparison

def compute_metrics_for_folder(folder):
    import glob
    files = glob.glob(os.path.join(folder, "*.mid"))
    rds, rrs, ncs = [], [], []
    for f in files:
        try:
            pm    = pretty_midi.PrettyMIDI(f)
            notes = sorted([n for inst in pm.instruments for n in inst.notes], key=lambda x: x.start)
            if len(notes) < 5:
                continue
            pitches = [n.pitch for n in notes]
            durs    = [round((n.end - n.start) * 1000 / 50) * 50 for n in notes]
            ngrams  = [tuple(pitches[i:i+4]) for i in range(len(pitches) - 3)]
            counts  = Counter(ngrams)
            rep     = sum(1 for c in counts.values() if c > 1) / max(len(ngrams), 1)
            rd      = len(set(durs)) / max(len(durs), 1)
            rds.append(rd)
            rrs.append(rep)
            ncs.append(len(notes))
        except Exception:
            continue
    return {
        "rhythm_diversity": np.mean(rds) if rds else float("nan"),
        "repetition_ratio": np.mean(rrs) if rrs else float("nan"),
        "avg_note_count":   np.mean(ncs) if ncs else float("nan"),
        "n_files":          len(files),
    }


def print_before_after(t3_dir, t4_dir):
    before = compute_metrics_for_folder(t3_dir)
    after  = compute_metrics_for_folder(t4_dir)
    print("\n" + "═" * 60)
    print("  BEFORE / AFTER RLHF COMPARISON")
    print("═" * 60)
    print(f"  {'Metric':<22} {'Task 3 (Before)':>16} {'Task 4 RLHF':>16}")
    print("  " + "─" * 56)
    for k in ["rhythm_diversity", "repetition_ratio", "avg_note_count"]:
        b = f"{before[k]:.3f}" if not math.isnan(before[k]) else "N/A"
        a = f"{after[k]:.3f}"  if not math.isnan(after[k])  else "N/A"
        print(f"  {k:<22} {b:>16} {a:>16}")
    print("═" * 60)
    print("  Note: add perplexity + human score to this table manually")
    print("        after collecting survey results.\n")


def plot_before_after_comparison(t3_dir, t4_dir):
    """
    Visualize before/after RLHF comparison as grouped bar chart.
    Shows key metrics: rhythm diversity, repetition ratio, avg note count.
    """
    before = compute_metrics_for_folder(t3_dir)
    after  = compute_metrics_for_folder(t4_dir)
    
    metrics = ["rhythm_diversity", "repetition_ratio", "avg_note_count"]
    metric_labels = ["Rhythm Diversity", "Repetition Ratio", "Avg Note Count"]
    
    before_vals = [before.get(k, float('nan')) for k in metrics]
    after_vals = [after.get(k, float('nan')) for k in metrics]
    
    # Filter out NaN values
    valid_metrics = []
    valid_before = []
    valid_after = []
    for i, m in enumerate(metrics):
        if not math.isnan(before_vals[i]) and not math.isnan(after_vals[i]):
            valid_metrics.append(metric_labels[i])
            valid_before.append(before_vals[i])
            valid_after.append(after_vals[i])
    
    if not valid_metrics:
        print("  ⚠  Insufficient data for before/after visualization.")
        return
    
    x = np.arange(len(valid_metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, valid_before, width, label="Task 3 (Before)", 
                   color="#3498db", alpha=0.8, edgecolor="black", linewidth=1.5)
    bars2 = ax.bar(x + width/2, valid_after, width, label="Task 4 RLHF (After)", 
                   color="#2ecc71", alpha=0.8, edgecolor="black", linewidth=1.5)
    
    ax.set_xlabel("Metrics", fontsize=12, fontweight="bold")
    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_title("Task 4: Before vs After RLHF Fine-Tuning", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(valid_metrics)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig("plots/task4_before_after_comparison.png", dpi=150)
    print("  Before/After chart → plots/task4_before_after_comparison.png")
    plt.close()


#  MODULE 10 — RL Plots

def plot_rl_curves(rewards, losses, kls):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(rewards, marker="o", linewidth=2, color="#27ae60")
    axes[0].axhline(0, color="gray", linestyle="--", lw=1)
    axes[0].set_title("Mean Reward per RL Iteration")
    axes[0].set_xlabel("Iteration"); axes[0].set_ylabel("Reward")

    axes[1].plot(losses, marker="o", linewidth=2, color="#c0392b")
    axes[1].set_title("Total REINFORCE Loss")
    axes[1].set_xlabel("Iteration"); axes[1].set_ylabel("Loss")

    axes[2].plot(kls, marker="o", linewidth=2, color="#8e44ad")
    axes[2].set_title("KL Divergence from Reference Policy")
    axes[2].set_xlabel("Iteration"); axes[2].set_ylabel("KL")

    plt.suptitle("Task 4 — REINFORCE RLHF", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("plots/task4_rl_curves.png", dpi=150)
    print("  RL curves → plots/task4_rl_curves.png")


#  MODULE 11 — Final 10-Sample Generation

def generate_final(policy, vocab_size, seed_id, n=10):
    print(f"\n── Generating {n} final RL-tuned compositions ──")
    policy.eval()
    for i in range(1, n + 1):
        ids  = generate_sample(policy, vocab_size, seed_id, gen_len=GEN_LEN_FINAL)
        path = os.path.join(OUT_DIR, f"rlhf_composition_{i:02d}.mid")
        tokens_to_midi(ids, vocab_size, path)
        print(f"  rlhf_composition_{i:02d}.mid")


#  MAIN

if __name__ == "__main__":
    print("=" * 65)
    print("TASK 4: REINFORCE RLHF Fine-Tuning")
    print("=" * 65)

    # Loading vocab size 
    if not os.path.exists("data/processed/vocab_size.npy"):
        print("Error: data/processed/vocab_size.npy not found")
        exit(1)
    vocab_size = int(np.load("data/processed/vocab_size.npy")[0])
    print(f"Vocabulary size: {vocab_size}")

    # Loading Task 3 checkpoint into two model instances ──
    if not os.path.exists(CKPT_T3):
        print(f"Error: {CKPT_T3} not found. Run 05_task3_transformer.py first.")
        exit(1)

    # Fine-tuned policy — parameters will be updated
    policy = MusicTransformer(vocab_size).to(DEVICE)
    policy.load_state_dict(torch.load(CKPT_T3, map_location=DEVICE))

    # Reference policy — frozen; used only for KL penalty computation
    ref_policy = MusicTransformer(vocab_size).to(DEVICE)
    ref_policy.load_state_dict(torch.load(CKPT_T3, map_location=DEVICE))
    for p in ref_policy.parameters():
        p.requires_grad = False
    ref_policy.eval()
    print(f"Task 3 checkpoint loaded: {CKPT_T3}")

    seed_id = find_seed_token()
    print(f"Seed token ID: {seed_id}")

    # Check if RLHF checkpoint exists - if so, skip RL training
    skip_rl_training = os.path.exists(CKPT_T4)
    
    if skip_rl_training:
        print(f"\n✓ Loading pretrained RLHF checkpoint: {CKPT_T4}")
        print("  Skipping RL training phase...")
    else:
        print(f"\n⚠ RLHF checkpoint not found. Running full RL training pipeline...")
        
        # Phase 1: Human Survey 
        if not os.path.exists(SURVEY_F):
            print("\n Phase 1: Generating Survey Samples (first run) ")
            generate_survey_samples(policy, vocab_size, seed_id, n=20)
            print("\nSurvey samples created. Need to fill human_survey.json and re-run.")
            print("   To proceed immediately with proxy reward, type 'y' below.")
            ans = input("   Continue with proxy reward now? [y/n]: ").strip().lower()
            if ans != "y":
                exit(0)
        else:
            print(f"  Existing survey: {SURVEY_F}")

        # Phase 2: Reward Model
        rm = train_reward_model(SURVEY_F)

        # Phase 3: REINFORCE RL Training 
        rewards, losses, kls = rl_train(policy, ref_policy, rm, vocab_size, seed_id)

        # Phase 4: Plots 
        plot_rl_curves(rewards, losses, kls)

    # Phase 5: Load best checkpoint → generate final 10 samples 
    policy.load_state_dict(torch.load(CKPT_T4, map_location=DEVICE))
    if not skip_rl_training:
        generate_final(policy, vocab_size, seed_id, n=10)
    else:
        print("  Loading existing outputs for visualization...")

    # Phase 6: Before/After comparison 
    print_before_after(
        t3_dir="outputs/task3_transformer",
        t4_dir=OUT_DIR
    )
    plot_before_after_comparison(
        t3_dir="outputs/task3_transformer",
        t4_dir=OUT_DIR
    )

    print(" Task 4 complete!")
    print("   Deliverables:")
    print(f"   • Survey template           → {SURVEY_F}")
    print(f"   • Survey MIDI samples       → {OUT_DIR}/survey_samples/")
    print(f"   • Reward model              → {CKPT_RM}")
    print(f"   • RL curves                 → plots/task4_rl_curves.png")
    print(f"   • Before/After chart        → plots/task4_before_after_comparison.png")
    print(f"   • 10 RL-tuned MIDIs         → {OUT_DIR}/rlhf_composition_*.mid")
    print(f"   • Before/After table        → see above")
