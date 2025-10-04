import pickle
from pathlib import Path
from scipy.sparse import load_npz
from implicit.nearest_neighbours import bm25_weight, tfidf_weight

def load_data(data_dir: Path, weighting: str = "none"):
    train_csr = load_npz(data_dir / "train_interactions.npz")
    test_csr = load_npz(data_dir / "test_interactions.npz")
    interactions = load_npz(data_dir / "interactions.npz")

    # apply weighting to item-user
    item_user = train_csr.T.tocsr()
    if weighting == "bm25":
        wmat = bm25_weight(item_user).tocsr()
    elif weighting == "tfidf":
        wmat = tfidf_weight(item_user).tocsr()
    else:
        wmat = item_user

    return train_csr, test_csr, interactions, wmat