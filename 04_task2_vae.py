"""
TASK 2: Variational Autoencoder (VAE) for Multi-Genre Piano-Roll Generation

"""

import os, numpy as np, torch, torch.nn as nn
import matplotlib.pyplot as plt
import pretty_midi
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Config 
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WIN_LEN    = 128
PITCH_DIM  = 88
HIDDEN_DIM = 256
LATENT_DIM = 64
BATCH_SIZE = 32
EPOCHS     = 80
LR         = 1e-3
DROPOUT    = 0.2
FS         = 16
PITCH_LOW  = 21

# KL annealing: β linearly increases from 0 → 1 over WARMUP epochs
WARMUP_EPOCHS = 30
BETA_MAX      = 1.0

OUT_DIR   = "outputs/task2_vae"
CKPT      = "checkpoints/task2_vae.pt"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("plots", exist_ok=True)

print(f"Using device: {DEVICE}")


# Dataset 
class PianoRollDataset(Dataset):
    def __init__(self, npy_path):
        data = np.load(npy_path)
        self.data = torch.from_numpy(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# Focal Loss 
class FocalBCELoss(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=20.0):
        super().__init__()
        self.gamma      = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=torch.tensor(self.pos_weight, device=logits.device),
            reduction="none"
        )
        probs = torch.sigmoid(logits)
        pt    = torch.where(targets == 1, probs, 1 - probs)
        return ((1 - pt) ** self.gamma * bce).mean()


# VAE Encoder 
class VAEEncoder(nn.Module):
    """
    Outputs µ and log σ² (two separate heads on the LSTM final hidden state).
    """
    def __init__(self, input_dim=88, hidden_dim=256, latent_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm    = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                               batch_first=True,
                               dropout=dropout if num_layers > 1 else 0.0)
        self.fc_mu   = nn.Linear(hidden_dim, latent_dim)
        self.fc_logv = nn.Linear(hidden_dim, latent_dim)
        self.drop    = nn.Dropout(dropout)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        h_last = self.drop(h_n[-1])         # (B, hidden_dim)
        mu     = self.fc_mu(h_last)         # (B, latent_dim)
        log_var= self.fc_logv(h_last)       # (B, latent_dim)
        return mu, log_var


# VAE Decoder 
class VAEDecoder(nn.Module):
    def __init__(self, latent_dim=64, hidden_dim=256, output_dim=88, seq_len=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.seq_len = seq_len
        self.fc_in   = nn.Linear(latent_dim, hidden_dim)
        self.lstm    = nn.LSTM(latent_dim + hidden_dim, hidden_dim, num_layers=num_layers,
                               batch_first=True,
                               dropout=dropout if num_layers > 1 else 0.0)
        self.fc_out  = nn.Linear(hidden_dim, output_dim)
        # NO sigmoid — raw logits

    def forward(self, z):
        B    = z.size(0)
        seed = torch.relu(self.fc_in(z))
        z_rep    = z.unsqueeze(1).expand(-1, self.seq_len, -1)
        seed_rep = seed.unsqueeze(1).expand(-1, self.seq_len, -1)
        dec_in   = torch.cat([z_rep, seed_rep], dim=-1)
        out, _   = self.lstm(dec_in)
        return self.fc_out(out)             # (B, T, 88) raw logits


# VAE 
class MusicVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = VAEEncoder(PITCH_DIM, HIDDEN_DIM, LATENT_DIM, num_layers=2, dropout=DROPOUT)
        self.decoder = VAEDecoder(LATENT_DIM, HIDDEN_DIM, PITCH_DIM, WIN_LEN, num_layers=2, dropout=DROPOUT)

    def reparameterize(self, mu, log_var):
        """z = µ + σ ⊙ ε,  ε ~ N(0, I)"""
        std = torch.exp(0.5 * log_var)      # σ = exp(0.5 * log σ²)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z           = self.reparameterize(mu, log_var)
        logits      = self.decoder(z)
        return logits, mu, log_var

    def kl_divergence(self, mu, log_var):
        """Closed-form KL(q || N(0,I))"""
        # KL = -0.5 * sum(1 + log σ² - µ² - σ²)
        return -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())


def get_beta(epoch, warmup=WARMUP_EPOCHS, beta_max=BETA_MAX):
    """Linearly increase β from 0 to beta_max over warmup epochs."""
    return min(beta_max, beta_max * epoch / warmup)


# Training 
def train():
    print("\n── Loading data ──")
    train_ds = PianoRollDataset("data/processed/train.npy")
    val_ds   = PianoRollDataset("data/processed/validation.npy")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    print(f"  Train: {len(train_ds)}   Val: {len(val_ds)}")

    model     = MusicVAE().to(DEVICE)
    recon_fn  = FocalBCELoss(gamma=2.0, pos_weight=20.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    train_total, train_recon, train_kl = [], [], []
    val_total, val_recon, val_kl       = [], [], []
    betas                              = []
    best_val = float("inf")

    print(f"\n── Training for {EPOCHS} epochs (KL warmup = {WARMUP_EPOCHS}) ──")
    for epoch in range(1, EPOCHS + 1):
        beta = get_beta(epoch)
        betas.append(beta)

        # Train
        model.train()
        ep_total = ep_recon = ep_kl = 0.0
        for batch in tqdm(train_loader, desc=f"Ep {epoch:03d} train", leave=False):
            batch = batch.to(DEVICE)
            optimizer.zero_grad()

            logits, mu, log_var = model(batch)
            recon_loss = recon_fn(logits, batch)
            kl_loss    = model.kl_divergence(mu, log_var)
            loss       = recon_loss + beta * kl_loss

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            ep_total += loss.item()
            ep_recon += recon_loss.item()
            ep_kl    += kl_loss.item()

        n = len(train_loader)
        train_total.append(ep_total / n)
        train_recon.append(ep_recon / n)
        train_kl.append(ep_kl    / n)

        # Validate
        model.eval()
        vt = vr = vk = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                logits, mu, log_var = model(batch)
                vr += recon_fn(logits, batch).item()
                vk += model.kl_divergence(mu, log_var).item()
        m = len(val_loader)
        avg_vr  = vr / m
        avg_vk  = vk / m
        avg_vt  = avg_vr + beta * avg_vk
        val_total.append(avg_vt)
        val_recon.append(avg_vr)
        val_kl.append(avg_vk)

        scheduler.step(avg_vt)
        if avg_vt < best_val:
            best_val = avg_vt
            torch.save(model.state_dict(), CKPT)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Ep {epoch:3d} | β={beta:.2f} | Recon={avg_vr:.4f} | KL={avg_vk:.4f} | Total={avg_vt:.4f}")

    # Plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].plot(train_total, label="Train Total", linewidth=2)
    axes[0].plot(val_total,   label="Val Total",   linewidth=2, linestyle="--")
    axes[0].set_title("VAE — Total Loss"); axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(train_recon, label="Train Recon", linewidth=2)
    axes[1].plot(train_kl,    label="Train KL",    linewidth=2)
    axes[1].plot(val_recon,   label="Val Recon",   linewidth=2, linestyle="--")
    axes[1].plot(val_kl,      label="Val KL",      linewidth=2, linestyle="--")
    axes[1].set_title("VAE — Recon vs KL Loss"); axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("plots/task2_loss_curves.png", dpi=150)
    print("  Loss curves saved → plots/task2_loss_curves.png")
    return model


# MIDI Export (shared helper) 
def piano_roll_to_midi(binary_roll, out_path, fs=FS, velocity=80, program=0):
    pm   = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=program, name="Piano")
    dt   = 1.0 / fs
    T, P = binary_roll.shape
    for p in range(P):
        col   = binary_roll[:, p]
        onset = None
        for t in range(T):
            if col[t] == 1 and onset is None:
                onset = t * dt
            elif col[t] == 0 and onset is not None:
                offset = t * dt
                if offset - onset >= dt:
                    inst.notes.append(pretty_midi.Note(velocity=velocity, pitch=PITCH_LOW + p,
                                                        start=onset, end=offset))
                onset = None
        if onset is not None:
            offset = T * dt
            inst.notes.append(pretty_midi.Note(velocity=velocity, pitch=PITCH_LOW + p,
                                                start=onset, end=offset))
    pm.instruments.append(inst)
    pm.write(out_path)


# Multi-Genre Generation (8 samples from N(0,I)) 
def generate_samples(model, n_samples=8, threshold=0.3):
    model.eval()
    print(f"\n── Generating {n_samples} multi-genre samples ──")
    with torch.no_grad():
        z      = torch.randn(n_samples, LATENT_DIM, device=DEVICE)
        logits = model.decoder(z)
        probs  = torch.sigmoid(logits).cpu().numpy()

    for i, prob in enumerate(probs):
        binary = (prob > threshold).astype(np.uint8)
        if binary.sum() == 0:
            binary = (prob > 0.1).astype(np.uint8)
        path = os.path.join(OUT_DIR, f"multigenre_{i+1:02d}.mid")
        piano_roll_to_midi(binary, path)
        print(f"  Saved {path}")


# Latent Interpolation (8 steps between 2 encoded pieces)
def latent_interpolation(model, train_npy, n_steps=8, threshold=0.3):
    """
    Encoding two training windows → µ1, µ2.
    Generating 8 intermediate points zα = (1-α)µ1 + α µ2. 

    """
    print(f"\n── Latent Interpolation ({n_steps} steps) ──")
    data   = torch.from_numpy(np.load(train_npy))
    idx1, idx2 = 0, min(500, len(data) - 1)   # picking two different pieces
    x1 = data[idx1].unsqueeze(0).to(DEVICE)
    x2 = data[idx2].unsqueeze(0).to(DEVICE)

    model.eval()
    with torch.no_grad():
        mu1, _ = model.encoder(x1)
        mu2, _ = model.encoder(x2)

    alphas = np.linspace(0, 1, n_steps)
    for i, alpha in enumerate(alphas):
        z_alpha = (1 - alpha) * mu1 + alpha * mu2    # (1, latent_dim)
        with torch.no_grad():
            logits = model.decoder(z_alpha)
            probs  = torch.sigmoid(logits).cpu().numpy()[0]
        binary = (probs > threshold).astype(np.uint8)
        if binary.sum() == 0:
            binary = (probs > 0.1).astype(np.uint8)
        path = os.path.join(OUT_DIR, f"interpolation_{i+1:02d}_alpha{alpha:.2f}.mid")
        piano_roll_to_midi(binary, path)
        print(f"  α={alpha:.2f} → {path}")


def latent_interpolation_heatmap(model, train_npy, n_steps=8, threshold=0.3):
    """
    Visualize latent interpolation as a heatmap of piano-rolls.
    Shows 8 interpolated outputs side-by-side for qualitative assessment.
    """
    import matplotlib.pyplot as plt
    
    print(f"\n── Latent Interpolation Heatmap ──")
    data   = torch.from_numpy(np.load(train_npy))
    idx1, idx2 = 0, min(500, len(data) - 1)
    x1 = data[idx1].unsqueeze(0).to(DEVICE)
    x2 = data[idx2].unsqueeze(0).to(DEVICE)

    model.eval()
    with torch.no_grad():
        mu1, _ = model.encoder(x1)
        mu2, _ = model.encoder(x2)

    alphas = np.linspace(0, 1, n_steps)
    heatmaps = []
    
    for alpha in alphas:
        z_alpha = (1 - alpha) * mu1 + alpha * mu2
        with torch.no_grad():
            logits = model.decoder(z_alpha)
            probs  = torch.sigmoid(logits).cpu().numpy()[0]
        binary = (probs > threshold).astype(np.uint8)
        if binary.sum() == 0:
            binary = (probs > 0.1).astype(np.uint8)
        heatmaps.append(binary)

    # Create heatmap visualization
    fig, axes = plt.subplots(1, n_steps, figsize=(16, 3))
    for i, (ax, alpha, hm) in enumerate(zip(axes, alphas, heatmaps)):
        ax.imshow(hm, aspect='auto', cmap='Blues', origin='lower')
        ax.set_title(f'α={alpha:.2f}', fontsize=10)
        ax.set_xlabel('Pitch')
        if i == 0:
            ax.set_ylabel('Time')
        else:
            ax.set_yticks([])

    plt.suptitle('Task 2: Latent Space Interpolation (Piano-Roll Heatmaps)', 
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "interpolation_heatmap.png"), dpi=150, bbox_inches='tight')
    print(f"  Interpolation heatmap → {OUT_DIR}/interpolation_heatmap.png")


# Main
if __name__ == "__main__":
    print("=" * 60)
    print("TASK 2: Variational Autoencoder")
    print("=" * 60)

    if not os.path.exists("data/processed/train.npy"):
        print("Error: Training data not found.")
        exit(1)

    # Load or train model
    model = MusicVAE().to(DEVICE)
    if os.path.exists(CKPT):
        print(f"\n✓ Loading pretrained checkpoint: {CKPT}")
        model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    else:
        print(f"\n✗ Checkpoint not found. Training from scratch...")
        model = train()
        model.load_state_dict(torch.load(CKPT, map_location=DEVICE))

    generate_samples(model, n_samples=8, threshold=0.3)
    latent_interpolation(model, "data/processed/train.npy", n_steps=8)
    latent_interpolation_heatmap(model, "data/processed/train.npy", n_steps=8)

    print("\nTask 2 complete!")
    print("   Deliverables:")
    print(f"   • Loss curves                    → plots/task2_loss_curves.png")
    print(f"   • 8 multi-genre MIDIs           → {OUT_DIR}/multigenre_*.mid")
    print(f"   • 8 interpolation MIDIs          → {OUT_DIR}/interpolation_*.mid")
    print(f"   • Interpolation heatmap          → {OUT_DIR}/interpolation_heatmap.png")
    print(f"   • Checkpoint                     → {CKPT}")

