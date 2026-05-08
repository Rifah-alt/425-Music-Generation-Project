"""
Downloading the MAESTRO Dataset (MIDI-only, ~57 MB compressed)

"""

import urllib.request, zipfile, os, shutil

URL = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip"
DEST_ZIP = "data/maestro-v3.0.0-midi.zip"
DEST_DIR = "data/maestro"

def download_with_progress(url, dest):
    print(f"Downloading MAESTRO MIDI dataset from:\n  {url}")
    def progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        pct = min(100, downloaded * 100 // total_size)
        bar = "█" * (pct // 2) + "░" * (50 - pct // 2)
        print(f"\r  [{bar}] {pct}%  ({downloaded//1024//1024} MB / {total_size//1024//1024} MB)", end="", flush=True)
    urllib.request.urlretrieve(url, dest, reporthook=progress)
    print("\nDownload complete.")

def extract(zip_path, dest_dir):
    print(f"Extracting to {dest_dir}/ ...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest_dir)
    print("Extraction complete.")

def verify():
    import pandas as pd
    csv_candidates = []
    for root, dirs, files in os.walk(DEST_DIR):
        for f in files:
            if f.endswith(".csv"):
                csv_candidates.append(os.path.join(root, f))
    if not csv_candidates:
        print("Could not find maestro CSV. Check extraction path.")
        return
    csv_path = csv_candidates[0]
    df = pd.read_csv(csv_path)
    print(f"\nMetadata loaded: {len(df)} recordings")
    print(f"   Splits: {df['split'].value_counts().to_dict()}")
    print(f"   CSV path: {csv_path}")
    # Count MIDI files
    midi_count = sum(1 for root, _, files in os.walk(DEST_DIR) for f in files if f.endswith(".midi") or f.endswith(".mid"))
    print(f"   MIDI files found: {midi_count}")
    

if not os.path.exists(DEST_DIR + "/maestro-v3.0.0"):
    if not os.path.exists(DEST_ZIP):
        download_with_progress(URL, DEST_ZIP)
    extract(DEST_ZIP, DEST_DIR)
else:
    print("Dataset folder already exists, skipping download.")

verify()
