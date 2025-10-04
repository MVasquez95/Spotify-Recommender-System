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

# Original parameters
PLAYLISTS_PER_SLICE = 1000
# 500k playlists
N_SLICES = 500

# Reduced subset for modeling (50k playlists)
REDUCED_N_SLICES = 50

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


def load_subset(force_extract=False, reduced=False):
    if reduced:
        n_slices = REDUCED_N_SLICES
        subset_file = SUBSET_DIR / "spotify_million_playlist_dataset_50k.parquet"
    else:
        n_slices = N_SLICES
        subset_file = SUBSET_FILE_PARQUET

    if subset_file.exists() and not force_extract:
        print(f"Loading subset from {subset_file}...")
        return pd.read_parquet(subset_file)
    else:
        playlists = extract_subset(n_slices=n_slices)
        df = pd.DataFrame(playlists)
        # Save reduced subset if applicable
        if reduced:
            df.to_parquet(subset_file, index=False)
        return df


# Quick test
if __name__ == "__main__":
    # Load reduced subset for modeling
    df = load_subset(reduced=True)
    print(df.head())
    print(f"Total playlists (reduced subset): {df.shape[0]}")