# Spotify Recommender System 🎶

## Aim
Build a song/playlist recommendation system using Spotify data.  
We implemented and compared **baseline (popularity)** and **collaborative filtering (ALS)** models, evaluating them with standard ranking metrics.

## Structure
- `notebooks/`: analysis and experiments (EDA, baseline, ALS, evaluation).
- `src/`: reusable functions (pipeline, models, metrics).
- `data/`: datasets (raw, subset, processed).
  - `raw/`: original downloaded zips.
  - `subset/`: subset of 150k playlists (JSON + Parquet).
  - `processed/`: intermediate CSR matrices and mappings.
- `models/`: trained model files (e.g., `als_optuna.pkl`).

## Requirements
See `requirements.txt`.

## Dataset
We used:
- **Spotify Million Playlist Dataset (MPD)** (~6GB).
- **Spotify Challenge Dataset** (10k playlists) for evaluation/demo.
- **Subset (150k playlists)** created for faster prototyping.


## Dataset setup
1. **Download MPD dataset (training set, ~6GB) from AIcrowd:**
   https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge/dataset_files  
   Place it in `data/raw/spotify_million_playlist_dataset.zip`.

2. **Optionally, download the challenge/test set (~10k playlists):**
   `spotify_million_playlist_dataset_challenge.zip` → place in `data/raw/`.

3. **Generate subset (500k playlists)**:
   - Run `src/data_pipeline.py`
   - Or load it directly in your notebook:
     ```python
     from src.data_pipeline import load_subset
     df = load_subset()

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
1. 01_EDA.ipynb → Explore subset, generate statistics & visualizations.
2. 02_baseline_popularity.ipynb → Popularity-based recommendations.
3. 03_data_prep.ipynb → ALS with hyperparameter tuning via Optuna.
4. 04_ALS_training.ipynb → ALS with hyperparameter tuning via Optuna & Compare ALS vs. Popularity using Precision@K, Recall@K, NDCG@K, and Hit Rate.

## Results
- ALS significantly outperforms Popularity across all metrics (Precision, Recall, HitRate, NDCG).
- Improvements are especially large at smaller cutoffs (K=10), which are more relevant in practice.
- Full results in als_optuna_eval.csv and popularity_model_eval.csv.

## Future Work
- Incorporate side information (user/item metadata).
- Explore sequence-aware or neural recommenders (LightFM, Transformers, GNNs).
- Perform online evaluation (A/B testing).
- Investigate hybrid models (collaborative + content-based).

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
        │  - Generate 500k      │
        └───────────────────────┘
                   │
                   ▼
  ┌─────────────────────────────┐
  │ Subset: 500k playlists      │
  │  JSON + Parquet             │
  └─────────────────────────────┘
        │           │           │
        ▼           ▼           ▼
┌─────────────┐ ┌─────────────┐ ┌──────────────────┐
│ 01_EDA      │ │ 02_Baseline │ │ 03_data_prep     │
│ - Explore   │ │ - Popularity│ │ - DATA PROCESS   │
│ - Stats     │ │             │ │ FOR USE IN MODELS│
└─────────────┘ └─────────────┘ └──────────────────┘
                                       │
                                       ▼
                               ┌───────────────────┐
                               │ 04_ALS_training   │
                               │ - Train ALS       │
                               │ - Compare results │
                               │ - Precision@K     │
                               │ - Recall@K        │
                               │ - NDCG@K          │
                               │ - HitRate         │
                               └───────────────────┘
```