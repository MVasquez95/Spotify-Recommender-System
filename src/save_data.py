import numpy as np
import pandas as pd
import pickle, os
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix, save_npz

def process_user_safe(u, items_u, rng, test_ratio, freq_threshold, item_counts):
    train_rows, train_cols, test_rows, test_cols = [], [], [], []
    n_items = len(items_u)

    if n_items == 1:
        # Just one item: all by train
        train_rows.append(u)
        train_cols.append(items_u[0])

    else:
        # Filter valid candidates for test (frequent)
        candidates = [it for it in items_u if item_counts.get(it, 0) >= freq_threshold]

        if len(candidates) == 0:
            # No candidates: all to train
            train_rows.extend([u]*n_items)
            train_cols.extend(items_u)
        else:
            n_test = min(max(1, int(n_items * test_ratio)), len(candidates))
            test_items = rng.choice(candidates, size=n_test, replace=False)

            for it in items_u:
                if it in test_items:
                    test_rows.append(u)
                    test_cols.append(it)
                else:
                    train_rows.append(u)
                    train_cols.append(it)

    return train_rows, train_cols, test_rows, test_cols


def build_and_save_split_safe(
    input_parquet="data/subset/tracks_subset.parquet",
    output_dir="data/processed",
    seed=42,
    n_jobs=-1,
    test_ratio=0.2,
    freq_threshold=5,
    chunk_size=20000
):
    # Load data
    df = pd.read_parquet(input_parquet)
    df["item_key"] = df["track_uri"].astype(str)

    # Create indexes
    user2idx = {pid: i for i, pid in enumerate(df["playlist_id"].unique())}
    item2idx = {it: j for j, it in enumerate(df["item_key"].unique())}

    df["user_idx"] = df["playlist_id"].map(user2idx).astype(np.int32)
    df["item_idx"] = df["item_key"].map(item2idx).astype(np.int32)

    n_users, n_items = len(user2idx), len(item2idx)
    print(f"Users={n_users}, Items={n_items}")

    # Count item frequency
    item_counts = df["item_idx"].value_counts().to_dict()

    # Group items by user
    grouped = df.groupby("user_idx", sort=False)["item_idx"].agg(list)
    rng = np.random.default_rng(seed)

    # Chunk processing
    all_train_rows, all_train_cols, all_test_rows, all_test_cols = [], [], [], []
    user_ids = grouped.index.to_numpy()
    n_chunks = (len(user_ids) + chunk_size - 1) // chunk_size

    for i in tqdm(range(n_chunks), desc="Processing chunks"):
        chunk_users = user_ids[i * chunk_size : (i + 1) * chunk_size]
        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(process_user_safe)(u, grouped[u], rng, test_ratio, freq_threshold, item_counts)
            for u in chunk_users
        )
        for tr_r, tr_c, te_r, te_c in results:
            all_train_rows.extend(tr_r)
            all_train_cols.extend(tr_c)
            all_test_rows.extend(te_r)
            all_test_cols.extend(te_c)

    # Create CSR matrices
    train_interactions = csr_matrix(
        (np.ones(len(all_train_rows), dtype=np.int8), (all_train_rows, all_train_cols)),
        shape=(n_users, n_items)
    )
    test_interactions = csr_matrix(
        (np.ones(len(all_test_rows), dtype=np.int8), (all_test_rows, all_test_cols)),
        shape=(n_users, n_items)
    )
    interactions = train_interactions + test_interactions

    # Save
    os.makedirs(output_dir, exist_ok=True)
    save_npz(f"{output_dir}/interactions.npz", interactions)
    save_npz(f"{output_dir}/train_interactions.npz", train_interactions)
    save_npz(f"{output_dir}/test_interactions.npz", test_interactions)

    with open(f"{output_dir}/user2idx.pkl", "wb") as f:
        pickle.dump(user2idx, f)
    with open(f"{output_dir}/item2idx.pkl", "wb") as f:
        pickle.dump(item2idx, f)

    print("Split saved in", output_dir)
    print(f"Train nnz: {train_interactions.nnz}, Test nnz: {test_interactions.nnz}")