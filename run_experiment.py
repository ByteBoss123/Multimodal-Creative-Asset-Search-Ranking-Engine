"""
run_experiment.py
-----------------
MLflow experiment: 3 retrieval strategies on real MS COCO val2014 data.

  baseline  — FAISS ANN only (raw cosine ordering)
  pixelseek — FAISS + composite reranking (tag bonus + category prior)
  diverse   — pixelseek + MMR diversity reranking

Logs NDCG@5/10/20, MRR, P@10, Recall@10, P99 latency per strategy.
"""

import sys, time, logging
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlflow
from src.embedder import LSAEmbedder
from src.indexer import FAISSIndexer
from src.reranker import Reranker
from src.corpus import load_asset_lookup, EVAL_QUERIES
from src.metrics import evaluate_ranking, print_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CORPUS_PATH   = "data/corpus.jsonl"
INDEX_PATH    = "models/faiss.index"
EMBEDDER_PATH = "models/embedder.pkl"
RECALL_K = 50
TOP_K    = 10


def get_rels(result_ids, assets, query):
    return [assets.get(aid, {}).get("relevance_labels", {}).get(query, 0.0)
            for aid in result_ids]


def count_total_relevant(assets, query, threshold=0.05):
    return sum(1 for a in assets.values()
               if a.get("relevance_labels", {}).get(query, 0.0) > threshold)


def run_strategy(name, embedder, indexer, reranker, assets, diversity=False):
    latencies, query_results = [], []

    for query in EVAL_QUERIES:
        t0 = time.perf_counter()
        qvec  = embedder.embed_text(query)
        cands = indexer.search(qvec, top_k=RECALL_K)

        if name == "baseline":
            result_ids = [aid for aid, _ in cands[:TOP_K]]
        else:
            ranked = reranker.rerank(query, cands, assets, top_k=TOP_K)
            if diversity:
                ranked = reranker.diversity_rerank(ranked)
            result_ids = [r.asset_id for r in ranked]

        latencies.append((time.perf_counter() - t0) * 1000)
        rels = get_rels(result_ids, assets, query)
        query_results.append({
            "query": query,
            "relevances": rels,
            "total_relevant": count_total_relevant(assets, query),
        })

    metrics = evaluate_ranking(query_results, k_values=[5, 10, 20])
    metrics["p99_latency_ms"]  = float(np.percentile(latencies, 99))
    metrics["mean_latency_ms"] = float(np.mean(latencies))
    return metrics


def main():
    for p in [CORPUS_PATH, INDEX_PATH, EMBEDDER_PATH]:
        if not Path(p).exists():
            print(f"Missing: {p}. Run scripts/ingest_corpus.py and build_index.py first.")
            return

    logger.info("Loading assets + index + embedder...")
    assets   = load_asset_lookup(CORPUS_PATH)
    embedder = LSAEmbedder.load(EMBEDDER_PATH)
    indexer  = FAISSIndexer.load(INDEX_PATH)
    reranker = Reranker()

    mlflow.set_experiment("pixelseek-coco-retrieval")

    for strat, diversity in [("baseline", False), ("pixelseek", False), ("diverse", True)]:
        logger.info(f"\n── {strat} ──")
        with mlflow.start_run(run_name=strat):
            mlflow.log_params({
                "strategy":    strat,
                "recall_k":    RECALL_K,
                "top_k":       TOP_K,
                "diversity":   diversity,
                "num_queries": len(EVAL_QUERIES),
                "corpus_size": len(assets),
                "dataset":     "MS COCO val2014",
                "embedder":    "TF-IDF/LSA-512",
                "index":       "FAISS IVFFlat nlist=201 nprobe=32",
            })

            metrics = run_strategy(strat, embedder, indexer, reranker, assets, diversity)

            for k, v in metrics.items():
                mlflow.log_metric(k.replace("@", "_at_"), v)

            print(f"\nStrategy: {strat.upper()}")
            print_metrics(metrics)

    logger.info("\nDone. View results: mlflow ui --port 5000")


if __name__ == "__main__":
    main()
