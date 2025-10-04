import numpy as np
import gc
import multiprocessing
from implicit.als import AlternatingLeastSquares

def train_als(wmat, factors=64, regularization=0.05, iterations=15, num_threads=None):
    if num_threads is None:
        num_threads = multiprocessing.cpu_count()

    model = AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        iterations=iterations,
        num_threads=num_threads
    )

    model.fit(wmat.T)

    gc.collect()
    return model


def generate_recommendations(model, train_csr, user_ids, N=100):
    recs = {}
    for uid in user_ids:
        user_vec = model.user_factors[uid]
        scores = model.item_factors @ user_vec

        seen = train_csr[uid].indices
        scores[seen] = -np.inf

        topN = np.argpartition(scores, -N)[-N:]
        ranked = topN[np.argsort(scores[topN])[::-1]]
        recs[uid] = ranked.tolist()

    return recs
