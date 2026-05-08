"""
Preprocessing : Building Piano-Roll Windows (Tasks 1 & 2) + Building Token Sequences (Task 3 & 4)

"""

import os, glob, numpy as np, pandas as pd, pretty_midi
from tqdm import tqdm

# Config 
FS          = 16          # frames per second  (128 frames = 8 seconds)
WIN_LEN     = 128         # piano-roll window length
MIN_ACTIVE  = 0.02        # discard windows with < 2% active cells
PITCH_LOW   = 21          # A0 (lowest piano key)
PITCH_HIGH  = 108         # C8 (highest piano key)
PITCH_RANGE = PITCH_HIGH - PITCH_LOW + 1  # 88


def find_maestro_root(base="data/maestro"):
    """Walking data/maestro to find the directory containing the CSV."""
    for root, dirs, files in os.walk(base):
        for f in files:
            if f.endswith(".csv"):
                return root
    raise FileNotFoundError("Cannot find maestro CSV under data/maestro/. Run 01_download_data.py first.")


def load_metadata(maestro_root):
    csv_path = glob.glob(os.path.join(maestro_root, "*.csv"))[0]
    df = pd.read_csv(csv_path)
    # Building absolute MIDI paths
    df["abs_midi_path"] = df["midi_filename"].apply(
        lambda rel: os.path.join(maestro_root, rel)
    )
    return df


def midi_to_windows(midi_path, fs=FS, win_len=WIN_LEN, min_active=MIN_ACTIVE):
    """
    Loading one MIDI file to return a list of binary piano-roll windows.
    Each window has shape (win_len, 88).

    """
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
    except Exception as e:
        return []   # skip malformed files

    # Piano-roll: shape (128, T) — rows = all MIDI pitches
    roll = pm.get_piano_roll(fs=fs)  # float, velocity values

    # Slice to 88-key piano range and binarize
    roll = roll[PITCH_LOW:PITCH_HIGH + 1, :]  # (88, T)
    roll = (roll > 0).astype(np.float32)       # binary

    # Transpose → (T, 88)
    roll = roll.T

    T = roll.shape[0]
    windows = []
    for start in range(0, T - win_len + 1, win_len):
        w = roll[start:start + win_len]        # (128, 88)
        if w.mean() >= min_active:             # keep if active enough
            windows.append(w)
    return windows


def build_split(df, split_name, out_dir, limit=None):
    """Process all files for one split and save as a single .npy array."""
    subset = df[df["split"] == split_name]
    if limit:
        subset = subset.head(limit)

    all_windows = []
    skipped = 0
    for _, row in tqdm(subset.iterrows(), total=len(subset), desc=f"  {split_name}"):
        path = row["abs_midi_path"]
        if not os.path.exists(path):
            skipped += 1
            continue
        wins = midi_to_windows(path)
        all_windows.extend(wins)

    if not all_windows:
        print(f"  ⚠ No windows collected for {split_name}. Check your data path.")
        return

    arr = np.stack(all_windows, axis=0)    # (N, 128, 88)
    out_path = os.path.join(out_dir, f"{split_name}.npy")
    np.save(out_path, arr)
    print(f" {split_name}: {arr.shape[0]} windows → {out_path}  (skipped {skipped} files)")
    return arr


# Token sequences : Tokenizing MIDI files using miditok REMI and save per-split

def build_token_sequences(df, out_dir, limit_per_split=None):

    try:
        from miditok import REMI, TokenizerConfig
    except ImportError:
        print("miditok not installed.")
        print("need to run pip install miditok symusic")
        return

    config = TokenizerConfig(num_velocities=32, use_chords=False, use_programs=False)
    tokenizer = REMI(config)

    for split_name in ["train", "validation", "test"]:
        subset = df[df["split"] == split_name]
        if limit_per_split:
            subset = subset.head(limit_per_split)

        sequences = []
        for _, row in tqdm(subset.iterrows(), total=len(subset), desc=f"  tokens/{split_name}"):
            path = row["abs_midi_path"]
            if not os.path.exists(path):
                continue
            try:
                tok = tokenizer(path)
                # tok is a TokSequence; access .ids for integer list
                ids = tok[0].ids if isinstance(tok, list) else tok.ids
                if len(ids) > 10:
                    sequences.append(ids)
            except Exception:
                continue

        out_path = os.path.join(out_dir, f"tokens_{split_name}.npy")
        # Save as object array (variable-length sequences)
        np.save(out_path, np.array(sequences, dtype=object))
        print(f"tokens/{split_name}: {len(sequences)} sequences → {out_path}")

    # Save vocab size
    vocab_size = tokenizer.vocab_size
    np.save(os.path.join(out_dir, "vocab_size.npy"), np.array([vocab_size]))
    print(f"  Vocabulary size: {vocab_size}")


# Exploratory Data Analysis (EDA)

def compute_duration_histogram(df, n_bins=50):
    """
    Extract piece durations from MAESTRO training set.
    Returns duration distribution for histogram visualization.
    """
    durations = []
    for _, row in tqdm(df[df["split"] == "train"].iterrows(), desc="  Computing durations"):
        path = row["abs_midi_path"]
        if not os.path.exists(path):
            continue
        try:
            pm = pretty_midi.PrettyMIDI(path)
            dur = pm.get_end_time()
            if dur > 0:
                durations.append(dur)
        except Exception:
            continue
    return np.array(durations)


def compute_pitch_distribution(df):
    """
    Extract pitch class distribution (12 pitch classes: C, C#, D, ..., B)
    from MAESTRO training set.
    """
    pitch_counts = np.zeros(12)
    for _, row in tqdm(df[df["split"] == "train"].iterrows(), desc="  Computing pitch distribution"):
        path = row["abs_midi_path"]
        if not os.path.exists(path):
            continue
        try:
            pm = pretty_midi.PrettyMIDI(path)
            for inst in pm.instruments:
                for note in inst.notes:
                    pitch_class = note.pitch % 12
                    pitch_counts[pitch_class] += 1
        except Exception:
            continue
    return pitch_counts


def plot_eda(durations, pitch_counts):
    """
    Generate and save EDA visualizations: duration histogram + pitch distribution.
    """
    import matplotlib.pyplot as plt
    
    os.makedirs("plots", exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Duration histogram
    axes[0].hist(durations, bins=50, color="#3498db", edgecolor="black", alpha=0.7)
    axes[0].axvline(np.mean(durations), color="red", linestyle="--", linewidth=2, label=f"Mean: {np.mean(durations):.1f}s")
    axes[0].set_xlabel("Piece Duration (seconds)", fontsize=11)
    axes[0].set_ylabel("Frequency", fontsize=11)
    axes[0].set_title("MAESTRO Training Set: Duration Distribution", fontsize=12, fontweight="bold")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Pitch distribution (12 pitch classes)
    pitch_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    axes[1].bar(pitch_names, pitch_counts, color="#2ecc71", edgecolor="black", alpha=0.7)
    axes[1].set_xlabel("Pitch Class", fontsize=11)
    axes[1].set_ylabel("Frequency", fontsize=11)
    axes[1].set_title("MAESTRO Training Set: Pitch Class Distribution", fontsize=12, fontweight="bold")
    axes[1].grid(alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.savefig("plots/eda_duration_and_pitch.png", dpi=150)
    print("  EDA plots → plots/eda_duration_and_pitch.png")


# Main 
if __name__ == "__main__":
    print("=" * 60)
    print("MAESTRO Preprocessing")
    print("=" * 60)

    maestro_root = find_maestro_root()
    print(f"Found MAESTRO at: {maestro_root}")
    df = load_metadata(maestro_root)
    print(f"Total recordings: {len(df)}")
    print(f"Splits: {df['split'].value_counts().to_dict()}\n")

    PIANO_ROLL_DIR = "data/processed"
    TOKEN_DIR      = "data/processed"
    os.makedirs(PIANO_ROLL_DIR, exist_ok=True)

    print("── Piano-Roll Windows (Tasks 1 & 2) ──")
    # Using full training set,limiting val/test for speed on GTX 1650
    build_split(df, "train",      PIANO_ROLL_DIR)
    build_split(df, "validation", PIANO_ROLL_DIR, limit=50)
    build_split(df, "test",       PIANO_ROLL_DIR, limit=50)

    print("\n── Token Sequences (Task 3 & 4) ──")
    build_token_sequences(df, TOKEN_DIR, limit_per_split=None)

    print("\n── Exploratory Data Analysis (EDA) ──")
    durations = compute_duration_histogram(df)
    pitch_counts = compute_pitch_distribution(df)
    plot_eda(durations, pitch_counts)

    print("\n Preprocessing complete.")

