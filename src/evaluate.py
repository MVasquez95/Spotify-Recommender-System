import numpy as np
from .metrics import precision_at_k, recall_at_k, ndcg_at_k, hit_rate_at_k
from .models import generate_recommendations

def evaluate(model, train_csr, test_csr, user_limit=1000):
    n_users = train_csr.shape[0]
    user_ids = np.arange(n_users)
    if user_limit and user_limit < n_users:
        user_ids = np.random.choice(user_ids, size=user_limit, replace=False)

    recs = generate_recommendations(model, train_csr, user_ids, N=100)
    ground_truth = {u: set(test_csr[u].indices) for u in user_ids}

    k_values = [10, 50, 100]
    metrics = {}
    for k in k_values:
        metrics[f"precision@{k}"] = precision_at_k(recs, ground_truth, k)
        metrics[f"recall@{k}"]   = recall_at_k(recs, ground_truth, k)
        metrics[f"ndcg@{k}"]     = ndcg_at_k(recs, ground_truth, k)
        metrics[f"hitrate@{k}"]  = hit_rate_at_k(recs, ground_truth, k)

    return metrics