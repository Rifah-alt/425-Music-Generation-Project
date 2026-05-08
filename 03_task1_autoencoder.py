"""
TASK 1: LSTM Autoencoder for Piano-Roll Music Generation

(batch=32, hidden=256, latent=64 for GTX 1650)

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
EPOCHS     = 60
LR         = 1e-3
DROPOUT    = 0.2
FS         = 16      # frames per second (for MIDI export)
PITCH_LOW  = 21      # A0

OUT_DIR    = "outputs/task1_ae"
CKPT       = "checkpoints/task1_ae.pt"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("plots", exist_ok=True)

print(f"Using device: {DEVICE}")


# Dataset 
class PianoRollDataset(Dataset):
    def __init__(self, npy_path):
        data = np.load(npy_path)          # (N, 128, 88) float32
        self.data = torch.from_numpy(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]             # (128, 88)


# Focal Loss (for binary classification with class imbalance) (handles 97-98% sparsity) 
class FocalBCELoss(nn.Module):

    def __init__(self, gamma=2.0, pos_weight=20.0):
        super().__init__()
        self.gamma      = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        # logits: raw (no sigmoid), targets: binary float
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=torch.tensor(self.pos_weight, device=logits.device),
            reduction="none"
        )
        probs  = torch.sigmoid(logits)
        pt     = torch.where(targets == 1, probs, 1 - probs)
        focal  = ((1 - pt) ** self.gamma) * bce
        return focal.mean()


# Encoder 
class Encoder(nn.Module):
    def __init__(self, input_dim=88, hidden_dim=256, latent_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_dim, latent_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, 88)
        _, (h_n, _) = self.lstm(x)
        # h_n: (num_layers, B, hidden_dim) — take last layer
        h_last = self.drop(h_n[-1])       # (B, hidden_dim)
        z      = self.fc(h_last)          # (B, latent_dim)
        return z


# Decoder 
class Decoder(nn.Module):
    def __init__(self, latent_dim=64, hidden_dim=256, output_dim=88, seq_len=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.seq_len = seq_len
        self.fc_in   = nn.Linear(latent_dim, hidden_dim)
        self.lstm    = nn.LSTM(
            latent_dim + hidden_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc_out  = nn.Linear(hidden_dim, output_dim)
        #NO sigmoid here — raw logits for BCEWithLogitsLoss / FocalLoss

    def forward(self, z):
        # z: (B, latent_dim)
        B = z.size(0)
        # Repeating z across time and concatenate with projected z as seed input
        seed  = torch.relu(self.fc_in(z))           # (B, hidden_dim)
        z_rep = z.unsqueeze(1).expand(-1, self.seq_len, -1)   # (B, T, latent_dim)
        seed_rep = seed.unsqueeze(1).expand(-1, self.seq_len, -1)  # (B, T, hidden_dim)
        dec_in = torch.cat([z_rep, seed_rep], dim=-1)  # (B, T, latent+hidden)
        out, _ = self.lstm(dec_in)                  # (B, T, hidden_dim)
        logits = self.fc_out(out)                   # (B, T, 88) — raw logits
        return logits


# Autoencoder 
class LSTMAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(PITCH_DIM, HIDDEN_DIM, LATENT_DIM, num_layers=2, dropout=DROPOUT)
        self.decoder = Decoder(LATENT_DIM, HIDDEN_DIM, PITCH_DIM, WIN_LEN, num_layers=2, dropout=DROPOUT)

    def forward(self, x):
        z      = self.encoder(x)
        logits = self.decoder(z)
        return logits, z


# Training 
def train():
    print("\n── Loading data ──")
    train_ds = PianoRollDataset("data/processed/train.npy")
    val_ds   = PianoRollDataset("data/processed/validation.npy")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    print(f"  Train windows: {len(train_ds)}")
    print(f"  Val   windows: {len(val_ds)}")

    model     = LSTMAutoencoder().to(DEVICE)
    criterion = FocalBCELoss(gamma=2.0, pos_weight=20.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    train_losses, val_losses = [], []
    best_val = float("inf")

    print(f"\n Training for {EPOCHS} epochs ")
    for epoch in range(1, EPOCHS + 1):
        # Train 
        model.train()
        epoch_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Ep {epoch:03d} train", leave=False):
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            logits, _ = model(batch)
            loss      = criterion(logits, batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # gradient clipping
            optimizer.step()
            epoch_loss += loss.item()
        avg_train = epoch_loss / len(train_loader)

        # Validate 
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch  = batch.to(DEVICE)
                logits, _ = model(batch)
                val_loss += criterion(logits, batch).item()
        avg_val = val_loss / len(val_loader)

        scheduler.step(avg_val)
        train_losses.append(avg_train)
        val_losses.append(avg_val)

        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), CKPT)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | Train: {avg_train:.4f} | Val: {avg_val:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

    # Plot 
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train Loss", linewidth=2)
    plt.plot(val_losses,   label="Val Loss",   linewidth=2, linestyle="--")
    plt.xlabel("Epoch"); plt.ylabel("Focal BCE Loss")
    plt.title("Task 1 — LSTM Autoencoder Training")
    plt.legend(); plt.tight_layout()
    plt.savefig("plots/task1_loss_curve.png", dpi=150)
    print("  Loss curve saved → plots/task1_loss_curve.png")
    return model


# Generation 
"""
    Sample z ~ N(0, I), decode, binarized with threshold < 0.5
    (compensates for model underestimating note probabilities on sparse data).

"""
def generate_samples(model, n_samples=5, threshold=0.3):
    
    model.eval()
    print(f"\n── Generating {n_samples} MIDI samples (threshold={threshold}) ──")

    with torch.no_grad():
        z      = torch.randn(n_samples, LATENT_DIM, device=DEVICE)
        logits = model.decoder(z)               # (N, T, 88)
        probs  = torch.sigmoid(logits).cpu().numpy()

    for i, prob_matrix in enumerate(probs):
        binary = (prob_matrix > threshold).astype(np.uint8)  # (128, 88)
        note_count = binary.sum()
        print(f"  Sample {i+1}: {note_count} active note cells")

        if note_count == 0:
            print(f" All cells silent — lowering threshold to 0.1")
            binary = (prob_matrix > 0.1).astype(np.uint8)

        midi_path = os.path.join(OUT_DIR, f"generated_{i+1:02d}.mid")
        piano_roll_to_midi(binary, midi_path)
        print(f"    Saved → {midi_path}")


def piano_roll_to_midi(binary_roll, out_path, fs=FS, velocity=80, program=0):
    
    # Pitch index p → MIDI note PITCH_LOW + p
    
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
                if offset - onset >= dt:   # at least one frame duration
                    inst.notes.append(
                        pretty_midi.Note(velocity=velocity, pitch=PITCH_LOW + p,
                                         start=onset, end=offset)
                    )
                onset = None
        if onset is not None:
            offset = T * dt
            inst.notes.append(
                pretty_midi.Note(velocity=velocity, pitch=PITCH_LOW + p,
                                 start=onset, end=offset)
            )

    pm.instruments.append(inst)
    pm.write(out_path)


# Main 
if __name__ == "__main__":
    print("=" * 60)
    print("TASK 1: LSTM Autoencoder")
    print("=" * 60)

    # Check data exists
    if not os.path.exists("data/processed/train.npy"):
        print("data/processed/train.npy not found.")
        exit(1)

    model = train()

    # Load best checkpoint for generation
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    generate_samples(model, n_samples=5, threshold=0.3)

    print("\n Task 1 complete!")
    print("   Deliverables:")
    print(f"   • Loss curve  → plots/task1_loss_curve.png")
    print(f"   • 5 MIDI files → {OUT_DIR}/")
    print(f"   • Checkpoint  → {CKPT}")

