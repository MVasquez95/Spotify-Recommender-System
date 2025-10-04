import numpy as np
import pandas as pd

def precision_at_k(recs, ground_truth, k=10):
    precisions = []
    for u, rec_list in recs.items():
        gt = ground_truth.get(u, set())
        if not gt:
            continue
        hits = sum(1 for item in rec_list[:k] if item in gt)
        precisions.append(hits / k)
    return np.mean(precisions)

def recall_at_k(recs, ground_truth, k=10):
    recalls = []
    for u, rec_list in recs.items():
        gt = ground_truth.get(u, set())
        if not gt:
            continue
        hits = sum(1 for item in rec_list[:k] if item in gt)
        recalls.append(hits / len(gt))
    return np.mean(recalls)

def ndcg_at_k(recs, ground_truth, k=10):
    ndcgs = []
    for u, rec_list in recs.items():
        gt = ground_truth.get(u, set())
        if not gt:
            continue
        dcg = sum(1 / np.log2(i + 2) for i, item in enumerate(rec_list[:k]) if item in gt)
        ideal_hits = min(len(gt), k)
        idcg = sum(1 / np.log2(i + 2) for i in range(ideal_hits))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
    return np.mean(ndcgs)

def hit_rate_at_k(recs, ground_truth, k=10):
    hits = 0
    total = 0
    for u, rec_list in recs.items():
        gt = ground_truth.get(u, set())
        if not gt:
            continue
        total += 1
        if set(rec_list[:k]) & gt:
            hits += 1
    return hits / total if total > 0 else 0.0

def normalize_eval_df(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    out = {}
    for _, row in df.iterrows():
        if "K" in row:
            k = int(row["K"])
            out[f"precision@{k}"] = row.get("Precision@K", row.get(f"precision@{k}"))
            out[f"recall@{k}"]    = row.get("Recall@K", row.get(f"recall@{k}"))
            out[f"hitrate@{k}"]   = row.get("HitRate@K", row.get(f"hitrate@{k}"))
            out[f"ndcg@{k}"]      = row.get("NDCG@K", row.get(f"ndcg@{k}"))
        else:
            for col in df.columns:
                out[col] = row[col]
    return pd.DataFrame([out], index=[model_name])