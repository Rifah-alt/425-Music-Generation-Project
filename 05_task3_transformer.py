"""
TASK 3: Decoder-only Transformer (GPT-style) for Autoregressive Music Generation

"""

import os, math, numpy as np, torch, torch.nn as nn
import matplotlib.pyplot as plt
import pretty_midi
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Config 
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
D_MODEL     = 256       # embedding / model dimension
N_HEADS     = 8         # attention heads
N_LAYERS    = 4         # transformer decoder layers
D_FF        = 1024      # feed-forward inner dimension
DROPOUT     = 0.1
MAX_SEQ_LEN = 512       # max token sequence length for training
BATCH_SIZE  = 16        # safe for 4 GB VRAM with seq_len=512
EPOCHS      = 40
LR          = 3e-4
GEN_LENGTH  = 1024      # tokens to generate per composition
TEMPERATURE = 1.0       # sampling temperature
TOP_K       = 50        # top-k sampling

OUT_DIR  = "outputs/task3_transformer"
CKPT     = "checkpoints/task3_transformer.pt"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("plots", exist_ok=True)

print(f"Using device: {DEVICE}")


# Token Dataset 
class TokenDataset(Dataset):
    """
    Each item is a fixed-length chunk of MAX_SEQ_LEN+1 tokens.
    Input  = chunk[:-1]  (all but last)
    Target = chunk[1:]   (all but first — shifted by 1) 

    """
    def __init__(self, npy_path, max_len=MAX_SEQ_LEN, vocab_size=None):
        raw = np.load(npy_path, allow_pickle=True)  # object array of lists

        # Clip any out-of-range token IDs for safety
        self.chunks = []
        for seq in raw:
            seq = [int(t) for t in seq]
            if vocab_size is not None:
                seq = [min(t, vocab_size - 1) for t in seq]
            # slide non-overlapping windows
            for start in range(0, len(seq) - max_len, max_len // 2):
                chunk = seq[start:start + max_len + 1]
                if len(chunk) == max_len + 1:
                    self.chunks.append(chunk)

        print(f"    {npy_path}: {len(self.chunks)} chunks of length {max_len}")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk  = torch.tensor(self.chunks[idx], dtype=torch.long)
        return chunk[:-1], chunk[1:]   # (input, target)


def collate_fn(batch):
    inputs, targets = zip(*batch)
    return torch.stack(inputs), torch.stack(targets)


# Positional Encoding 
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=2048):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)          # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (B, T, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.drop(x)


# Transformer Decoder Block 
class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.attn   = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff     = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1  = nn.LayerNorm(d_model)
        self.norm2  = nn.LayerNorm(d_model)
        self.drop   = nn.Dropout(dropout)

    def forward(self, x, causal_mask):
        # Self-attention with causal mask (pre-norm style)
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=causal_mask)
        x = x + self.drop(attn_out)
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


# GPT-style Music Transformer 
class MusicTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=D_MODEL, n_heads=N_HEADS,
                 n_layers=N_LAYERS, d_ff=D_FF, dropout=DROPOUT, max_len=MAX_SEQ_LEN):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout, max_len=max_len + 1)
        self.layers  = nn.ModuleList([
            TransformerDecoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm    = nn.LayerNorm(d_model)
        self.head    = nn.Linear(d_model, vocab_size)  # output projection to vocab

        # Initialize weights (GPT-style)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def make_causal_mask(self, T, device):
        """Upper-triangular mask: position i cannot attend to j > i."""
        mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
        # nn.MultiheadAttention: True = IGNORE
        return mask

    def forward(self, x):
        # x: (B, T) token ids
        T    = x.size(1)
        mask = self.make_causal_mask(T, x.device)

        h = self.tok_emb(x)       # (B, T, d_model)
        h = self.pos_enc(h)

        for layer in self.layers:
            h = layer(h, mask)

        h      = self.norm(h)
        logits = self.head(h)     # (B, T, vocab_size)
        return logits


# Perplexity Helper 
def compute_perplexity(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for inp, tgt in loader:
            inp, tgt = inp.to(device), tgt.to(device)
            logits = model(inp)                         # (B, T, V)
            loss   = criterion(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
            total_loss += loss.item()
            n += 1
    avg_loss   = total_loss / n
    perplexity = math.exp(avg_loss)
    return perplexity, avg_loss


# Training 
def train(vocab_size):
    print("\n── Loading token sequences ──")
    train_ds = TokenDataset("data/processed/tokens_train.npy", MAX_SEQ_LEN, vocab_size)
    val_ds   = TokenDataset("data/processed/tokens_validation.npy", MAX_SEQ_LEN, vocab_size)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_fn, num_workers=2, pin_memory=True)

    model     = MusicTransformer(vocab_size).to(DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=0)   # ignore padding
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {n_params:,}")

    train_perp, val_perp = [], []
    best_val = float("inf")

    print(f"\n── Training for {EPOCHS} epochs ──")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for inp, tgt in tqdm(train_loader, desc=f"Ep {epoch:03d} train", leave=False):
            inp, tgt = inp.to(DEVICE), tgt.to(DEVICE)
            optimizer.zero_grad()
            logits = model(inp)                         # (B, T, V)
            loss   = criterion(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        train_ppl      = math.exp(avg_train_loss)

        val_ppl, _ = compute_perplexity(model, val_loader, criterion, DEVICE)
        scheduler.step()

        train_perp.append(train_ppl)
        val_perp.append(val_ppl)

        if val_ppl < best_val:
            best_val = val_ppl
            torch.save(model.state_dict(), CKPT)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Ep {epoch:3d} | Train PPL: {train_ppl:.2f} | Val PPL: {val_ppl:.2f}")

    # ── Perplexity plot ──
    plt.figure(figsize=(8, 4))
    plt.plot(train_perp, label="Train Perplexity", linewidth=2)
    plt.plot(val_perp,   label="Val Perplexity",   linewidth=2, linestyle="--")
    plt.xlabel("Epoch"); plt.ylabel("Perplexity")
    plt.title("Task 3 — Transformer Perplexity")
    plt.legend(); plt.tight_layout()
    plt.savefig("plots/task3_perplexity.png", dpi=150)
    print("  Perplexity curve → plots/task3_perplexity.png")

    print(f"\n  Best validation perplexity: {best_val:.2f}")
    return model


# Generation (top-k + temperature)
def generate_sequence(model, tokenizer, seed_token_id, gen_length=GEN_LENGTH,
                       temperature=TEMPERATURE, top_k=TOP_K, device=DEVICE):
    """Autoregressively sample a token sequence."""
    model.eval()
    ids = [seed_token_id]

    with torch.no_grad():
        for _ in range(gen_length):
            inp    = torch.tensor([ids], dtype=torch.long, device=device)
            # Truncate to MAX_SEQ_LEN if growing too long
            if inp.size(1) > MAX_SEQ_LEN:
                inp = inp[:, -MAX_SEQ_LEN:]
            logits = model(inp)                        # (1, T, V)
            next_logit = logits[0, -1, :] / temperature

            # Mask special tokens (id 0 = padding, 1 = EOS if exists)
            next_logit[0] = -float("inf")

            # Top-k filtering
            if top_k > 0:
                topk_val, _ = torch.topk(next_logit, top_k)
                threshold   = topk_val[-1]
                next_logit[next_logit < threshold] = -float("inf")

            probs   = torch.softmax(next_logit, dim=-1)
            next_id = torch.multinomial(probs, 1).item()
            ids.append(next_id)

    return ids


def generate_compositions(model, vocab_size, n=10):
    """Generate n long compositions and save as MIDI."""
    print(f"\n── Generating {n} compositions ({GEN_LENGTH} tokens each) ──")

    # Re-load tokenizer to convert tokens → MIDI
    try:
        from miditok import REMI, TokenizerConfig
        config    = TokenizerConfig(num_velocities=32, use_chords=False, use_programs=False)
        tokenizer = REMI(config)
    except ImportError:
        print("  miditok not available — saving raw token sequences instead.")
        tokenizer = None

    # Seed token: trying to find a "Bar" token, otherwise use token 2
    seed_id = 2
    if tokenizer is not None:
        for k, v in tokenizer.vocab.items():
            if "Bar" in str(k):
                seed_id = v
                break

    for i in range(1, n + 1):
        ids = generate_sequence(model, tokenizer, seed_id, gen_length=GEN_LENGTH)

        if tokenizer is not None:
            try:
                from miditok import TokSequence
                tok_seq  = TokSequence(ids=ids)
                midi_obj = tokenizer.tokens_to_midi([tok_seq])
                path     = os.path.join(OUT_DIR, f"composition_{i:02d}.mid")
                midi_obj.dump(path)
                print(f"  Composition {i:02d} → {path}")
            except Exception as e:
                # Fallback: save token IDs as numpy
                path = os.path.join(OUT_DIR, f"composition_{i:02d}_tokens.npy")
                np.save(path, np.array(ids))
                print(f"  Composition {i:02d} (tokens) → {path}  [{e}]")
        else:
            path = os.path.join(OUT_DIR, f"composition_{i:02d}_tokens.npy")
            np.save(path, np.array(ids))
            print(f"  Composition {i:02d} → {path}")


# Baseline Comparison Table 
def print_baseline_table(val_ppl):
    print("\n── Baseline Comparison (Perplexity) ──")
    print(f"{'Model':<25} {'Val Perplexity':>15}")
    print("-" * 42)
    print(f"{'Random Generator':<25} {'∞ (undefined)':>15}")
    print(f"{'Markov Chain':<25} {'~120–200':>15}")
    print(f"{'Task 3 Transformer':<25} {val_ppl:>15.2f}")
    print(f"{'Task 3 + RLHF (est.)':<25} {'see Task 4':>15}")


# Main 
if __name__ == "__main__":
    print("=" * 60)
    print("TASK 3: Transformer Generator")
    print("=" * 60)

    # Load vocab size saved by 02_preprocess.py
    vocab_path = "data/processed/vocab_size.npy"
    if not os.path.exists(vocab_path):
        print(" data/processed/vocab_size.npy not found.")
        exit(1)

    vocab_size = int(np.load(vocab_path)[0])
    print(f"Vocabulary size: {vocab_size}")

    if not os.path.exists("data/processed/tokens_train.npy"):
        print("Token files not found. Run 02_preprocess.py first.")
        exit(1)

    model = train(vocab_size)

    # Load best checkpoint
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE))

    # Final perplexity on validation
    val_ds     = TokenDataset("data/processed/tokens_validation.npy", MAX_SEQ_LEN, vocab_size)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    criterion  = nn.CrossEntropyLoss(ignore_index=0)
    val_ppl, _ = compute_perplexity(model, val_loader, criterion, DEVICE)
    print(f"\nFinal Validation Perplexity: {val_ppl:.2f}")

    generate_compositions(model, vocab_size, n=10)
    print_baseline_table(val_ppl)

    print("\nTask 3 complete!")
    print("   Deliverables:")
    print(f"   • Perplexity curve      → plots/task3_perplexity.png")
    print(f"   • 10 compositions       → {OUT_DIR}/")
    print(f"   • Checkpoint            → {CKPT}")
