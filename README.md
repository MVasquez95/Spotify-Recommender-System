# Spotify Recommender System 🎶

## Aim
Build a song/playlist recommendation system using Spotify data.  
**Baseline (popularity)**, **collaborative filtering (ALS, LightFM)** and **hybrid** models are implemented.

## Structure
- `notebooks/`: analysis and experiments.
- `src/`: reusable functions (pipeline, models, metrics).
- `data/`: datasets (raw and processed).
   -`raw/`: original downloaded zips
   .`subset/`: subset of 150k playlists (JSON + Parquet)

## Bookstores
See `requirements.txt`.

## Dataset
Can be used:
- Spotify Million Playlist Dataset (MPD)
- Spotify Challenge Dataset (10k playlists)
- Spotify API (optional, for additional experiments)
- Subsets created for prototypes

## Dataset setup
1. **Download the MPD dataset (training set, ~6GB) from AIcrowd:**
   https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge/dataset_files
   and place it in `data/raw/spotify_million_playlist_dataset.zip`.

2. **Optionally, download the challenge/test set (~10k playlists):**
   `spotify_million_playlist_dataset_challenge.zip` for evaluation/demo purposes.

3. **Generate the subset (150k playlists)**
- Run `src/data_pipeline.py`
- Or load it directly in your notebook:
   ```Python
   from src.data_pipeline import load_subset
   df = load_subset()
   ```
   to generate the subset of 150k playlists.

## Notes
- **Do NOT commit raw or subset data**. The repository `.gitignore` already ignores:
   ```Console
   data/raw/*
   data/subset/*
   ```
- Pipeline creates **JSON** and **Parquet** formats for fast loading.
- Use the subset for **EDA, model training, and Kaggle deployment**.
- Challenge/test set can be used for **evaluation or demo purposes.**

## Planned Notebooks
1. **01_EDA.ipynb** → Explore the subset, generate statistics and wow-factor visualizations.
2. **02_Baseline_Models.ipynb** → Popularity-based recommendations.
3. **03_Collaborative_Models.ipynb** → ALS / LightFM models.
4. **04_Hybrid_Models.ipynb** → Combine collaborative + content features.
5. **05_Evaluation.ipynb** → Metrics: Precision@K, Recall@K, NDCG@K, Hit Rate, Diversity, Novelty.

## Spotify Recommender System – Data & Notebook Flow

```
┌─────────────────────────────────────────┐
│ Spotify Million Playlist Dataset (1M)   │
│      mpd.zip in data/raw/               │
└─────────────────────────────────────────┘
                   │
                   ▼
        ┌───────────────────────┐
        │  src/data_pipeline.py │
        │  - Extract subset     │
        │  - Generate 150k      │
        └───────────────────────┘
                   │
                   ▼
  ┌─────────────────────────────┐
  │ Subset: 150k playlists      │
  │  JSON + Parquet             │
  └─────────────────────────────┘
        │           │           │
        ▼           ▼           ▼
┌─────────────┐ ┌─────────────┐ ┌──────────────┐
│ 01_EDA      │ │ 02_Baseline │ │ 03_Collab    │
│ - Explore   │ │ - Popularity│ │ - ALS/LightFM│
│ - Stats     │ │             │ │ Models       │
└─────────────┘ └─────────────┘ └──────────────┘
        │
        ▼
┌─────────────┐
│ 04_Hybrid   │
│ - Combine CF│
│   + Content │
└─────────────┘
        │
        ▼
┌──────────────┐
│ 05_Evaluation│
│ - Precision@K│
│ - Recall@K   │
│ - NDCG@K     │
│ - HitRate    │
│ - Diversity  │
│ - Optional:  │
│   Challenge  │
│   Test set   │
└──────────────┘
```