"""
metrics.py
----------
Standard IR evaluation metrics: NDCG@K, MRR, Precision@K, Recall@K.
Used to compare retrieval strategies in MLflow experiments.
"""

import numpy as np


def dcg_at_k(rels: list, k: int) -> float:
    rels = rels[:k]
    if not rels:
        return 0.0
    g = np.array(rels, dtype=float)
    d = np.log2(np.arange(2, len(g) + 2))
    return float(np.sum(g / d))


def ndcg_at_k(rels: list, k: int) -> float:
    dcg  = dcg_at_k(rels, k)
    idcg = dcg_at_k(sorted(rels, reverse=True), k)
    return dcg / idcg if idcg > 0 else 0.0


def reciprocal_rank(rels: list, threshold: float = 0.1) -> float:
    for i, r in enumerate(rels):
        if r >= threshold:
            return 1.0 / (i + 1)
    return 0.0


def precision_at_k(rels: list, k: int, threshold: float = 0.1) -> float:
    top = rels[:k]
    return sum(1 for r in top if r >= threshold) / k if top else 0.0


def recall_at_k(rels: list, total_rel: int, k: int, threshold: float = 0.1) -> float:
    if total_rel == 0:
        return 0.0
    found = sum(1 for r in rels[:k] if r >= threshold)
    return found / total_rel


def evaluate_ranking(query_results: list, k_values: list = None) -> dict:
    """
    Aggregate metrics across all evaluated queries.
    Each item in query_results: {"query", "relevances", "total_relevant"}
    """
    if k_values is None:
        k_values = [5, 10, 20]

    ndcg  = {k: [] for k in k_values}
    prec  = {k: [] for k in k_values}
    rec   = {k: [] for k in k_values}
    mrr_list = []

    for qr in query_results:
        rels     = qr["relevances"]
        tot_rel  = qr.get("total_relevant",
                           sum(1 for r in rels if r > 0.1))
        mrr_list.append(reciprocal_rank(rels))
        for k in k_values:
            ndcg[k].append(ndcg_at_k(rels, k))
            prec[k].append(precision_at_k(rels, k))
            rec[k].append(recall_at_k(rels, tot_rel, k))

    out = {"MRR": float(np.mean(mrr_list)), "num_queries": len(query_results)}
    for k in k_values:
        out[f"NDCG@{k}"]   = float(np.mean(ndcg[k]))
        out[f"P@{k}"]      = float(np.mean(prec[k]))
        out[f"Recall@{k}"] = float(np.mean(rec[k]))
    return out


def print_metrics(metrics: dict) -> None:
    print("\n" + "=" * 52)
    print("  RANKING EVALUATION")
    print("=" * 52)
    print(f"  Queries : {metrics.get('num_queries','?')}")
    print(f"  MRR     : {metrics.get('MRR', 0):.4f}")
    for k in [5, 10, 20]:
        n = metrics.get(f"NDCG@{k}")
        p = metrics.get(f"P@{k}")
        r = metrics.get(f"Recall@{k}")
        if n is not None:
            print(f"  NDCG@{k:<2}  : {n:.4f}   P@{k}: {p:.4f}   R@{k}: {r:.4f}")
    print("=" * 52 + "\n")
