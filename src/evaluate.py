import numpy as np

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