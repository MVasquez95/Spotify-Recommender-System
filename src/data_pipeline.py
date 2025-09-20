# src/data_pipeline.py

import os
import zipfile
import json
from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")
SUBSET_DIR = Path("data/subset")
ZIP_PATH = RAW_DIR / "spotify_million_playlist_dataset.zip"
SUBSET_FILE_JSON = SUBSET_DIR / "spotify_million_playlist_dataset.json"
SUBSET_FILE_PARQUET = SUBSET_DIR / "spotify_million_playlist_dataset.parquet"

PLAYLISTS_PER_SLICE = 1000
N_SLICES = 150

def extract_subset(zip_path=ZIP_PATH, n_slices=N_SLICES, playlists_per_slice=PLAYLISTS_PER_SLICE):
    if not zip_path.exists():
        raise FileNotFoundError(
            f"⚠️ MPD zip not found at {zip_path}. "
            "Please download it manually from AIcrowd and place it here."
        )

    SUBSET_DIR.mkdir(parents=True, exist_ok=True)
    subset_playlists = []

    print(f"Extracting first {n_slices} slices (~{n_slices*playlists_per_slice} playlists)...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        slice_files = sorted([f for f in z.namelist() if f.endswith('.json')])
        for i, slice_file in enumerate(slice_files[:n_slices]):
            with z.open(slice_file) as f:
                data = json.load(f)
                subset_playlists.extend(data["playlists"])
            print(f"Slice {i+1}/{n_slices} extracted.")

    # Save JSON
    with open(SUBSET_FILE_JSON, 'w', encoding='utf-8') as f:
        json.dump(subset_playlists, f)
    print(f"Subset JSON saved to {SUBSET_FILE_JSON} ({len(subset_playlists)} playlists).")

    # Save Parquet
    df = pd.json_normalize(subset_playlists)
    df.to_parquet(SUBSET_FILE_PARQUET, index=False)
    print(f"Subset Parquet saved to {SUBSET_FILE_PARQUET} ({df.shape[0]} playlists).")

    return subset_playlists

# Load the 150k playlists subset. If not present, extract it.
def load_subset(force_extract=False):
    if SUBSET_FILE_PARQUET.exists() and not force_extract:
        print(f"Loading subset from {SUBSET_FILE_PARQUET}...")
        return pd.read_parquet(SUBSET_FILE_PARQUET)
    else:
        return pd.DataFrame(extract_subset())

# Quick test
if __name__ == "__main__":
    df = load_subset()
    print(df.head())
    print(f"Total playlists: {df.shape[0]}")
