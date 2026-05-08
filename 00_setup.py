"""
Environment Setup and package installation script for the Music Generation Project.
 
"""

import subprocess, sys, os

packages = [
    "pretty_midi",
    "miditok",
    "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",
    "numpy pandas matplotlib",
    "music21",
    "tqdm",
]

print("Installing packages...")
for pkg in packages:
    subprocess.run([sys.executable, "-m", "pip", "install"] + pkg.split(), check=False)

# Creating project directory structure
dirs = [
    "data/maestro",
    "data/processed/train",
    "data/processed/val",
    "data/processed/test",
    "outputs/task1_ae",
    "outputs/task2_vae",
    "outputs/task3_transformer",
    "checkpoints",
    "plots",
]
for d in dirs:
    os.makedirs(d, exist_ok=True)

print("\nAll directories created.")

