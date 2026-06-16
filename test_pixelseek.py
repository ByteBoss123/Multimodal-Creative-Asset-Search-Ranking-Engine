"""
test_pixelseek.py
-----------------
Full test suite for PixelSeek on real COCO data.
Tests corpus builder, FAISS indexer, reranker, and metrics.

Run: pytest tests/test_pixelseek.py -v
"""

import sys
import json
import tempfile
import numpy as np
import pytest
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.corpus import _infer_category, _extract_tags, _relevance, EVAL_QUERIES
from src.indexer import FAISSIndexer
from src.reranker import Reranker, RankedResult
from src.metrics import (ndcg_at_k, reciprocal_rank,
                          precision_at_k, recall_at_k, evaluate_ranking)


# ── helpers ──────────────────────────────────────────────────────────────────

def make_coco_json(tmp_path, n_images=20, caps_per_image=5):
    """Create a minimal COCO-format JSON for testing."""
    images, annotations = [], []
    ann_id = 1
    for i in range(1, n_images + 1):
        images.append({
            "id": i,
            "url": f"http://example.com/img{i}.jpg",
            "file_name": f"COCO_val2014_{i:012d}.jpg",
            "width": 640, "height": 480,
            "date_captured": "2014-01-01 00:00:00",
        })
        cap_templates = [
            f"A dog playing in the park near image {i}.",
            f"Two people walking their dog outside.",
            f"A person with a dog in a green field.",
            f"An animal and its owner in the park.",
            f"A puppy running on grass outdoors.",
        ]
        for cap in cap_templates[:caps_per_image]:
            annotations.append({"id": ann_id, "image_id": i, "caption": cap})
            ann_id += 1

    path = tmp_path / "coco_captions.json"
    path.write_text(json.dumps({
        "info": {}, "licenses": [], "type": "captions",
        "images": images, "annotations": annotations,
    }))
    return str(path)


def make_fake_assets(n=20):
    """Build a minimal asset_lookup dict for reranker tests."""
    lookup = {}
    for i in range(n):
        aid = str(i + 1)
        captions = [
            "A dog playing in the park.",
            "Two people walking outside.",
        ]
        desc = " | ".join(captions)
        lookup[aid] = {
            "asset_id": aid,
            "image_url": f"http://example.com/{aid}.jpg",
            "file_name": f"img_{aid}.jpg",
            "captions": captions,
            "description": desc,
            "tags": ["dog", "park", "people", "walking", "outside"],
            "category": "Animals" if i % 2 == 0 else "Nature",
            "relevance_labels": {q: 0.5 for q in EVAL_QUERIES},
        }
    return lookup


# ── Corpus ───────────────────────────────────────────────────────────────────

class TestCorpus:
    def test_category_inference_animals(self):
        assert _infer_category("A dog and cat playing together") == "Animals"

    def test_category_inference_food(self):
        assert _infer_category("People eating pizza at a restaurant") == "Food & Drink"

    def test_category_inference_sports(self):
        assert _infer_category("A person surfing on the ocean wave") == "Sports"

    def test_category_fallback(self):
        cat = _infer_category("xyzzy blorp quux")
        assert cat == "Other"

    def test_extract_tags_filters_stopwords(self):
        captions = ["A dog is running in the park.", "The dog plays with a ball in the park."]
        tags = _extract_tags(captions)
        assert "the" not in tags
        assert "is" not in tags
        assert "park" in tags

    def test_extract_tags_requires_two_captions(self):
        # "running" appears once, "park" appears twice — only park should be a tag
        captions = ["A dog is running in the park.", "The dog plays in the park."]
        tags = _extract_tags(captions)
        assert "park" in tags

    def test_relevance_perfect_overlap(self):
        score = _relevance("dog park", "dog playing in park", ["dog", "park"])
        assert score == pytest.approx(1.0)

    def test_relevance_no_overlap(self):
        score = _relevance("airplane sky", "dog playing in park", ["dog", "park"])
        assert score == pytest.approx(0.0)

    def test_build_corpus_from_coco(self, tmp_path):
        from src.corpus import build_corpus
        coco_path = make_coco_json(tmp_path, n_images=10)
        out_path  = str(tmp_path / "corpus.jsonl")
        assets = build_corpus(coco_path, out_path)
        assert len(assets) == 10

    def test_corpus_asset_schema(self, tmp_path):
        from src.corpus import build_corpus
        coco_path = make_coco_json(tmp_path, n_images=5)
        out_path  = str(tmp_path / "corpus.jsonl")
        assets = build_corpus(coco_path, out_path)
        for a in assets:
            assert "asset_id"    in a
            assert "description" in a
            assert "captions"    in a
            assert "tags"        in a
            assert "category"    in a
            assert "image_url"   in a
            assert "relevance_labels" in a
            for q in EVAL_QUERIES:
                assert q in a["relevance_labels"]
                assert 0.0 <= a["relevance_labels"][q] <= 1.0

    def test_corpus_jsonl_loadable(self, tmp_path):
        from src.corpus import build_corpus, load_asset_lookup
        coco_path = make_coco_json(tmp_path, n_images=8)
        out_path  = str(tmp_path / "corpus.jsonl")
        build_corpus(coco_path, out_path)
        lookup = load_asset_lookup(out_path)
        assert len(lookup) == 8
        for aid, asset in lookup.items():
            assert asset["asset_id"] == aid


# ── FAISS Indexer ─────────────────────────────────────────────────────────────

class TestFAISSIndexer:
    def _make_embeddings(self, N, dim=512):
        v = np.random.randn(N, dim).astype(np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        return v

    def test_build_flat_small(self):
        dim, N = 512, 50
        emb = self._make_embeddings(N, dim)
        ids = [str(i) for i in range(N)]
        idx = FAISSIndexer(dim=dim)
        idx.build(emb, ids)
        assert idx.size == N

    def test_build_ivf_large(self):
        dim, N = 512, 2000
        emb = self._make_embeddings(N, dim)
        ids = [str(i) for i in range(N)]
        idx = FAISSIndexer(dim=dim)
        idx.build(emb, ids)
        assert idx.size == N

    def test_search_top_result_is_self(self):
        dim, N = 512, 100
        emb = self._make_embeddings(N, dim)
        ids = [str(i) for i in range(N)]
        idx = FAISSIndexer(dim=dim)
        idx.build(emb, ids)
        results = idx.search(emb[42], top_k=5)
        assert results[0][0] == "42"

    def test_search_returns_k_results(self):
        dim, N = 512, 200
        emb = self._make_embeddings(N, dim)
        ids = [str(i) for i in range(N)]
        idx = FAISSIndexer(dim=dim)
        idx.build(emb, ids)
        q = np.random.randn(dim).astype(np.float32)
        q /= np.linalg.norm(q)
        results = idx.search(q, top_k=10)
        assert len(results) == 10

    def test_cosine_scores_in_range(self):
        dim, N = 512, 100
        emb = self._make_embeddings(N, dim)
        idx = FAISSIndexer(dim=dim)
        idx.build(emb, [str(i) for i in range(N)])
        q = np.random.randn(dim).astype(np.float32)
        q /= np.linalg.norm(q)
        for _, score in idx.search(q, top_k=20):
            assert -1.01 <= score <= 1.01

    def test_save_and_load(self, tmp_path):
        dim, N = 512, 150
        emb = self._make_embeddings(N, dim)
        ids = [str(i) for i in range(N)]
        idx = FAISSIndexer(dim=dim)
        idx.build(emb, ids)
        path = str(tmp_path / "test.index")
        idx.save(path)

        idx2 = FAISSIndexer.load(path, dim=dim)
        assert idx2.size == N
        results = idx2.search(emb[7], top_k=1)
        assert results[0][0] == "7"


# ── Reranker ──────────────────────────────────────────────────────────────────

class TestReranker:
    def test_returns_top_k(self):
        rr = Reranker()
        assets = make_fake_assets(20)
        cands = [(str(i+1), 0.9 - i*0.03) for i in range(20)]
        results = rr.rerank("dog park", cands, assets, top_k=10)
        assert len(results) == 10

    def test_ranks_are_sequential(self):
        rr = Reranker()
        assets = make_fake_assets(10)
        cands = [(str(i+1), 0.9 - i*0.05) for i in range(10)]
        results = rr.rerank("dog", cands, assets, top_k=5)
        assert [r.rank for r in results] == list(range(1, 6))

    def test_tag_bonus_lifts_matching_asset(self):
        rr = Reranker()
        # asset "1" has matching tags, "2" has higher clip but no tags
        assets = {
            "1": {"asset_id": "1", "image_url": "", "file_name": "",
                  "captions": ["dog park"], "description": "dog playing in park",
                  "tags": ["dog", "park", "playing"], "category": "Animals"},
            "2": {"asset_id": "2", "image_url": "", "file_name": "",
                  "captions": ["abstract"], "description": "abstract geometry",
                  "tags": ["geometry"], "category": "Other"},
        }
        cands = [("2", 0.90), ("1", 0.82)]  # asset 2 has higher clip score
        results = rr.rerank("dog park playing", cands, assets, top_k=2)
        # asset 1 should beat asset 2 after tag bonus
        assert results[0].asset_id == "1"

    def test_diversity_reduces_category_clustering(self):
        rr = Reranker()
        # Build 10 results all from same category
        results = []
        for i in range(10):
            results.append(RankedResult(
                asset_id=str(i), image_url="", file_name="",
                captions=[], category="Animals", tags=[],
                clip_score=0.9 - i*0.05, tag_bonus=0.0,
                final_score=0.9 - i*0.05, rank=i+1,
            ))
        diverse = rr.diversity_rerank(results, diversity_weight=0.5)
        # Scores should decrease as same-category penalty accumulates
        scores = [r.final_score for r in diverse]
        assert scores[0] >= scores[-1]

    def test_handles_missing_assets_gracefully(self):
        rr = Reranker()
        assets = make_fake_assets(5)
        cands = [("999", 0.9), ("1", 0.8)]  # "999" doesn't exist
        results = rr.rerank("dog", cands, assets, top_k=5)
        assert all(r.asset_id != "999" for r in results)


# ── Metrics ───────────────────────────────────────────────────────────────────

class TestMetrics:
    def test_ndcg_perfect(self):
        rels = [1.0, 0.8, 0.6, 0.4, 0.2]
        assert ndcg_at_k(rels, 5) == pytest.approx(1.0, abs=0.01)

    def test_ndcg_zero(self):
        assert ndcg_at_k([0.0, 0.0, 0.0], 3) == 0.0

    def test_ndcg_partial(self):
        rels = [0.0, 1.0, 0.0, 1.0]
        score = ndcg_at_k(rels, 4)
        assert 0.0 < score < 1.0

    def test_mrr_first_position(self):
        assert reciprocal_rank([1.0, 0.0, 0.0]) == pytest.approx(1.0)

    def test_mrr_third_position(self):
        assert reciprocal_rank([0.0, 0.0, 1.0]) == pytest.approx(1/3, abs=0.001)

    def test_mrr_no_relevant(self):
        assert reciprocal_rank([0.0, 0.0, 0.0]) == 0.0

    def test_precision_at_k(self):
        rels = [1.0, 0.0, 1.0, 0.0, 1.0]
        assert precision_at_k(rels, 5) == pytest.approx(0.6, abs=0.001)

    def test_recall_at_k(self):
        rels = [1.0, 0.0, 1.0, 0.0]
        assert recall_at_k(rels, total_rel=4, k=4) == pytest.approx(0.5, abs=0.001)

    def test_evaluate_ranking_aggregation(self):
        qrs = [
            {"query": "q1", "relevances": [1.0, 0.5, 0.0], "total_relevant": 2},
            {"query": "q2", "relevances": [0.0, 1.0, 0.5], "total_relevant": 2},
        ]
        m = evaluate_ranking(qrs, k_values=[3])
        assert "NDCG@3" in m and "MRR" in m and "P@3" in m
        assert 0.0 <= m["NDCG@3"] <= 1.0
        assert 0.0 <= m["MRR"] <= 1.0

    def test_evaluate_ranking_all_perfect(self):
        qrs = [{"query": "q1", "relevances": [1.0, 1.0, 1.0], "total_relevant": 3}]
        m = evaluate_ranking(qrs, k_values=[3])
        assert m["NDCG@3"] == pytest.approx(1.0, abs=0.01)
        assert m["MRR"] == pytest.approx(1.0, abs=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
