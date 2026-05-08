"""
EVALUATION METRICS — Fixed version

"""

import os, glob, numpy as np, pretty_midi, random, math, json
from collections import Counter

PITCH_LOW = 21   # A0

# Metric helpers

def pitch_histogram(midi_path):
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
    except Exception:
        return None
    hist = np.zeros(12)
    for inst in pm.instruments:
        for note in inst.notes:
            hist[note.pitch % 12] += 1
    total = hist.sum()
    return (hist / total) if total > 0 else None


def pitch_histogram_similarity(gen_path, ref_path):
    q = pitch_histogram(gen_path)
    p = pitch_histogram(ref_path)
    if q is None or p is None:
        return None
    return float(np.sum(np.abs(p - q)))


def rhythm_diversity(midi_path, quantize_ms=50):
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
    except Exception:
        return None
    durations = []
    for inst in pm.instruments:
        for note in inst.notes:
            dur_ms = (note.end - note.start) * 1000
            q = round(dur_ms / quantize_ms) * quantize_ms
            if q > 0:
                durations.append(q)
    if not durations:
        return None
    return len(set(durations)) / len(durations)


def repetition_ratio(midi_path, n=4):
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
    except Exception:
        return None
    notes   = sorted([note for inst in pm.instruments for note in inst.notes],
                     key=lambda x: x.start)
    pitches = [note.pitch for note in notes]
    if len(pitches) < n + 1:
        return None
    ngrams  = [tuple(pitches[i:i+n]) for i in range(len(pitches) - n + 1)]
    counts  = Counter(ngrams)
    repeated = sum(1 for c in counts.values() if c > 1)
    return repeated / len(ngrams)


def verify_midi(midi_path, min_notes=50, min_dur=5.0):
    try:
        pm    = pretty_midi.PrettyMIDI(midi_path)
        notes = [n for inst in pm.instruments for n in inst.notes]
        return len(notes) >= min_notes and pm.get_end_time() >= min_dur
    except Exception:
        return False


# Safe metrics for one folder

EMPTY_METRICS = {
    "Rhythm Diversity": float("nan"),
    "Repetition Ratio": float("nan"),
    "Pitch Hist Sim":   float("nan"),
    "n_files": 0,
}


def metrics_for_folder(folder, ref_midis=None, label=""):
    midi_files = sorted(
        glob.glob(os.path.join(folder, "*.mid")) +
        glob.glob(os.path.join(folder, "*.midi"))
    )
    if not midi_files:
        print(f"  ⚠  No MIDI files found in: {folder}")
        return dict(EMPTY_METRICS)   # return dict with NaN values, never empty {}

    rds, rrs, phss = [], [], []
    for path in midi_files:
        rd = rhythm_diversity(path)
        rr = repetition_ratio(path)
        if rd is not None: rds.append(rd)
        if rr is not None: rrs.append(rr)
        if ref_midis:
            ref = random.choice(ref_midis)
            phs = pitch_histogram_similarity(path, ref)
            if phs is not None: phss.append(phs)

    return {
        "Rhythm Diversity": np.mean(rds)  if rds  else float("nan"),
        "Repetition Ratio": np.mean(rrs)  if rrs  else float("nan"),
        "Pitch Hist Sim":   np.mean(phss) if phss else float("nan"),
        "n_files": len(midi_files),
    }


# Recovering Task 3 MIDIs from saved .npy token files

def tokens_npy_to_midi(npy_path, out_dir, vocab_size, n=10):
    """
    Converting composition_XX_tokens.npy files (saved when miditok failed)
    into MIDI files so the metrics script can process them.

    """
    npy_files = sorted(glob.glob(os.path.join(out_dir, "*_tokens.npy")))
    if not npy_files:
        return 0

    print(f"  Converting {len(npy_files)} token .npy files → MIDI …")
    converted = 0
    for npy_file in npy_files:
        try:
            ids      = np.load(npy_file, allow_pickle=True).tolist()
            base     = os.path.basename(npy_file).replace("_tokens.npy", ".mid")
            out_path = os.path.join(out_dir, base)
            if os.path.exists(out_path):
                continue
            _heuristic_tokens_to_midi(ids, vocab_size, out_path)
            converted += 1
        except Exception as e:
            print(f"    Could not convert {npy_file}: {e}")
    print(f"  Converted {converted} files → {out_dir}")
    return converted


def _heuristic_tokens_to_midi(ids, vocab_size, out_path, fs=16):

    """
    Fallback when miditok is unavailable: map each token to a piano note
    using modular arithmetic so the output covers the full piano range.

    """
    pm   = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0, name="Piano")
    t    = 0.0
    dt   = 1.0 / fs

    # Group tokens into note-like events (pitch + duration clusters)
    # Simple approach: every 3rd token triggers a note event
    for i in range(0, len(ids) - 2, 3):
        pitch    = PITCH_LOW + (int(ids[i])     % 88)
        velocity = 40 + (int(ids[i+1]) % 88)
        dur      = dt * (1 + int(ids[i+2]) % 8)
        inst.notes.append(
            pretty_midi.Note(velocity=min(velocity, 127), pitch=pitch,
                             start=t, end=t + dur)
        )
        t += dt * 0.5   # some notes overlap slightly (more musical)

    if not inst.notes:
        return

    pm.instruments.append(inst)
    pm.write(out_path)


# Baselines 

def generate_random_baseline(out_dir="outputs/baselines/random", n=5):
    os.makedirs(out_dir, exist_ok=True)
    durations_pool = [0.125, 0.25, 0.5, 1.0]
    for i in range(n):
        pm   = pretty_midi.PrettyMIDI()
        inst = pretty_midi.Instrument(program=0)
        t    = 0.0
        for _ in range(200):
            pitch = random.randint(21, 108)
            dur   = random.choice(durations_pool)
            inst.notes.append(pretty_midi.Note(velocity=80, pitch=pitch,
                                               start=t, end=t + dur))
            t += dur + random.uniform(0, 0.1)
        pm.instruments.append(inst)
        pm.write(os.path.join(out_dir, f"random_{i+1:02d}.mid"))
    print(f"  Random baseline: {n} files → {out_dir}")
    return out_dir


def build_markov_model(maestro_search_paths, n_files=100):
    midi_files = []
    for pattern in maestro_search_paths:
        midi_files.extend(glob.glob(pattern, recursive=True))
    midi_files = midi_files[:n_files]

    if not midi_files:
        print("  No reference MIDI files found for Markov model. Skipping.")
        return None, []

    trans    = np.zeros((128, 128), dtype=np.float64)
    dur_pool = []

    for path in midi_files:
        try:
            pm    = pretty_midi.PrettyMIDI(path)
            notes = sorted([n for inst in pm.instruments for n in inst.notes],
                           key=lambda x: x.start)
            pitches = [n.pitch for n in notes]
            durs    = [n.end - n.start for n in notes if n.end - n.start > 0]
            dur_pool.extend(durs)
            for a, b in zip(pitches[:-1], pitches[1:]):
                trans[a, b] += 1
        except Exception:
            continue

    row_sums = trans.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    trans = trans / row_sums
    return trans, dur_pool


def generate_markov(trans, dur_pool, out_dir="outputs/baselines/markov", n=5, seq_len=300):
    os.makedirs(out_dir, exist_ok=True)
    if trans is None:
        print("  No training data found for Markov model. Skipping.")
        return out_dir

    pitches_range = list(range(21, 109))
    for i in range(n):
        pm   = pretty_midi.PrettyMIDI()
        inst = pretty_midi.Instrument(program=0)
        cur  = random.choice(pitches_range)
        t    = 0.0
        for _ in range(seq_len):
            row = trans[cur]
            cur = int(np.random.choice(128, p=row)) if row.sum() > 0 else random.choice(pitches_range)
            if not (21 <= cur <= 108):
                cur = random.choice(pitches_range)
            dur = float(random.choice(dur_pool)) if dur_pool else 0.25
            inst.notes.append(pretty_midi.Note(velocity=80, pitch=cur,
                                               start=t, end=t + dur))
            t += dur + 0.05
        pm.instruments.append(inst)
        pm.write(os.path.join(out_dir, f"markov_{i+1:02d}.mid"))
    print(f"  Markov baseline: {n} files → {out_dir}")
    return out_dir


# Table printer

def fmt(v):
    """Format a metric value safely."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "N/A"
    return f"{v:.3f}"


def print_comparison_table(results: dict):
    print("\n" + "=" * 78)
    print(f"  {'Model':<26} {'Files':>5}  {'Rhythm Div':>11}  {'Repet.':>8}  {'Pitch Sim':>10}")
    print("=" * 78)
    for name, r in results.items():
        n   = r.get("n_files", 0)
        rd  = fmt(r.get("Rhythm Diversity"))
        rr  = fmt(r.get("Repetition Ratio"))
        phs = fmt(r.get("Pitch Hist Sim"))
        print(f"  {name:<26} {n:>5}  {rd:>11}  {rr:>8}  {phs:>10}")
    print("=" * 78)
    print("  Rhythm Div : higher = more variety")
    print("  Repet.     : 0.1–0.5 = coherent; ~0 or ~1 = degenerate")
    print("  Pitch Sim  : lower = closer to real music (0 = identical)")


# Human Survey Analysis

def compute_human_scores_from_survey():
    """
    Read human_survey.json and compute mean human scores per model category.
    Models are identified by file prefixes: survey_XX = RLHF samples, others are baselines.
    """
    survey_file = "outputs/task4_rlhf/human_survey.json"
    if not os.path.exists(survey_file):
        print(f"  ⚠  {survey_file} not found. Skipping human scores.")
        return None
    
    with open(survey_file, 'r') as f:
        data = json.load(f)
    
    if "samples" not in data or not data["samples"]:
        print("  ⚠  No survey samples found. Skipping human scores.")
        return None
    
    human_scores = {}
    for sample in data["samples"]:
        mean_score = sample.get("mean_score", 0)
        # All survey samples are from RLHF (Task 4)
        if "Task 4: RLHF" not in human_scores:
            human_scores["Task 4: RLHF"] = []
        human_scores["Task 4: RLHF"].append(mean_score)
    
    # Compute aggregates
    final_scores = {}
    for model, scores in human_scores.items():
        if scores:
            final_scores[model] = {
                "mean": np.mean(scores),
                "std": np.std(scores),
                "n": len(scores)
            }
    
    return final_scores


def plot_human_scores(human_scores):
    """
    Create bar chart showing mean human rating per model with error bars.
    """
    import matplotlib.pyplot as plt
    
    if not human_scores:
        return
    
    models = list(human_scores.keys())
    means = [human_scores[m]["mean"] for m in models]
    stds = [human_scores[m]["std"] for m in models]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#e74c3c"]  # Red for RLHF
    bars = ax.bar(models, means, yerr=stds, capsize=5, color=colors, alpha=0.7, 
                   edgecolor="black", linewidth=1.5)
    
    ax.set_ylabel("Mean Human Rating (1-5)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Model", fontsize=12, fontweight="bold")
    ax.set_title("Task 4: Human Rating Evaluation (RLHF)", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 5.5)
    ax.grid(alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.1,
                f'{mean:.2f}±{std:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("plots/human_scores_bar_chart.png", dpi=150)
    print("  Human scores → plots/human_scores_bar_chart.png")
    plt.close()


# Enhanced Comparison Table with 5 Metrics

def print_enhanced_comparison_table(results):
    """
    Enhanced table with all 6 models × 5 metrics, saved as both table and image.
    Metrics: Rhythm Diversity, Repetition Ratio, Pitch Similarity, N Files, and a summary note.
    """
    import matplotlib.pyplot as plt
    
    # Prepare table data
    model_names = list(results.keys())
    metrics = ["Rhythm Div", "Repet. Ratio", "Pitch Sim", "Files"]
    
    table_data = []
    for model in model_names:
        r = results[model]
        rd  = f"{r.get('Rhythm Diversity', float('nan')):.3f}" if not math.isnan(r.get('Rhythm Diversity', float('nan'))) else "N/A"
        rr  = f"{r.get('Repetition Ratio', float('nan')):.3f}" if not math.isnan(r.get('Repetition Ratio', float('nan'))) else "N/A"
        phs = f"{r.get('Pitch Hist Sim', float('nan')):.3f}" if not math.isnan(r.get('Pitch Hist Sim', float('nan'))) else "N/A"
        nf  = str(r.get("n_files", 0))
        table_data.append([model, rd, rr, phs, nf])
    
    # Print to console
    print("\n" + "=" * 100)
    print(f"  {'Model':<26} {'Rhythm Div':>12}  {'Repet. Ratio':>13}  {'Pitch Sim':>11}  {'Files':>6}")
    print("=" * 100)
    for row in table_data:
        print(f"  {row[0]:<26} {row[1]:>12}  {row[2]:>13}  {row[3]:>11}  {row[4]:>6}")
    print("=" * 100)
    print("  Rhythm Div   : higher = more variety")
    print("  Repet. Ratio : 0.1–0.5 = coherent; ~0 or ~1 = degenerate")
    print("  Pitch Sim    : lower = closer to real music (0 = identical)")
    
    # Create visualization of table
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Format data for matplotlib table
    cell_text = []
    for row in table_data:
        cell_text.append(row)
    
    # Create table
    table = ax.table(cellText=cell_text, 
                     colLabels=["Model", "Rhythm Div", "Repet. Ratio", "Pitch Sim", "Files"],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.3, 0.15, 0.15, 0.15, 0.1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(5):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(5):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')
            else:
                table[(i, j)].set_facecolor('#ffffff')
    
    plt.title('Comparison: Automatic Metrics (6 Models × 4 Metrics)', 
              fontsize=13, fontweight='bold', pad=20)
    plt.savefig("plots/comparison_table.png", dpi=150, bbox_inches='tight')
    print("\n  Comparison table image → plots/comparison_table.png")
    plt.close()


# Main 

if __name__ == "__main__":
    print("=" * 60)
    print("EVALUATION METRICS")
    print("=" * 60)

    # reference MIDIs (real MAESTRO files for Pitch Hist Sim) ──
    maestro_patterns = [
        "data/maestro/**/*.mid",
        "data/maestro/**/*.midi",
        "data/maestro-v3.0.0/**/*.mid",
        "data/maestro-v3.0.0/**/*.midi",
    ]
    ref_midis = []
    for pat in maestro_patterns:
        ref_midis.extend(glob.glob(pat, recursive=True))
    ref_midis = ref_midis[:50]
    if ref_midis:
        print(f"Reference MIDIs found: {len(ref_midis)}")
    else:
        print("⚠  No reference MIDIs found — Pitch Hist Sim will show N/A")
        print("   (This is fine; the other two metrics still work.)")

    # Loading vocab size ──
    vocab_size = 32000   # safe default
    vpath = "data/processed/vocab_size.npy"
    if os.path.exists(vpath):
        vocab_size = int(np.load(vpath)[0])
        print(f"Vocab size: {vocab_size}")

    # convert token .npy files → MIDI if needed ──
    t3_dir = "outputs/task3_transformer"
    if os.path.exists(t3_dir):
        existing_mid = glob.glob(os.path.join(t3_dir, "*.mid"))
        npy_tokens   = glob.glob(os.path.join(t3_dir, "*_tokens.npy"))
        if not existing_mid and npy_tokens:
            print(f"\n  Task 3 has {len(npy_tokens)} token files but no MIDIs.")
            print("  Converting tokens → MIDI using heuristic method …")
            tokens_npy_to_midi(None, t3_dir, vocab_size)

    results = {}

    # Baselines
    print("\n── Building Baselines ──")
    rand_dir = generate_random_baseline()

    print("  Building Markov model …")
    trans, dur_pool = build_markov_model(maestro_patterns, n_files=100)
    markov_dir = generate_markov(trans, dur_pool)

    results["Random Generator"]  = metrics_for_folder(rand_dir,   ref_midis)
    results["Markov Chain"]      = metrics_for_folder(markov_dir, ref_midis)

    # Task outputs
    print("\n── Computing Task Metrics ──")
    task_dirs = {
        "Task 1: LSTM AE":          "outputs/task1_ae",
        "Task 2: VAE":              "outputs/task2_vae",
        "Task 3: Transformer":      "outputs/task3_transformer",
        "Task 4: RLHF":             "outputs/task4_rlhf",
    }
    for label, folder in task_dirs.items():
        if os.path.exists(folder):
            results[label] = metrics_for_folder(folder, ref_midis, label)
        else:
            print(f"  Skipping {label} — folder not found: {folder}")

    # Print table
    print("\n── Automatic Metrics ──")
    print_comparison_table(results)
    print_enhanced_comparison_table(results)

    # Human survey analysis
    print("\n── Human Evaluation (Survey) ──")
    human_scores = compute_human_scores_from_survey()
    if human_scores:
        print(f"  Task 4 RLHF: {human_scores['Task 4: RLHF']['mean']:.2f}±{human_scores['Task 4: RLHF']['std']:.2f} ({human_scores['Task 4: RLHF']['n']} samples)")
        plot_human_scores(human_scores)

    # MIDI validity check 
    print("\n── MIDI Validity Check (≥50 notes, ≥5 sec) ──")
    all_midis = (
        glob.glob("outputs/**/*.mid",  recursive=True) +
        glob.glob("outputs/**/*.midi", recursive=True)
    )
    valid   = sum(1 for m in all_midis if verify_midi(m))
    invalid = len(all_midis) - valid
    print(f"  Valid  : {valid}/{len(all_midis)}")
    if invalid > 0:
        print(f"  Invalid: {invalid} (too short or too few notes — regenerate if needed)")

    print("\n Evaluation complete.")
