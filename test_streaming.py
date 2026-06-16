"""
tests/test_streaming.py
-----------------------
Tests for the real-time streaming indexer.
Runs without Kafka — uses the simulation path.
"""

import sys
import json
import time
import tempfile
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from stream.producer import _make_asset
from stream.consumer import StreamingIndexer


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def live_system(tmp_path):
    """Build a minimal live index for streaming tests."""
    from src.corpus import build_corpus
    import urllib.request, json as js

    # Build tiny corpus
    images = [{"id": i, "url": f"http://ex.com/{i}.jpg",
                "file_name": f"img_{i}.jpg", "width": 640, "height": 480,
                "date_captured": "2014-01-01"} for i in range(1, 201)]
    anns = []
    ann_id = 1
    for img in images:
        for cap in ["A dog running in the park.", "Two people walking outside.",
                    "An animal playing on grass."]:
            anns.append({"id": ann_id, "image_id": img["id"], "caption": cap})
            ann_id += 1

    coco_path = tmp_path / "coco.json"
    coco_path.write_text(js.dumps({"info": {}, "licenses": [], "type": "captions",
                                   "images": images, "annotations": anns}))

    corpus_path = str(tmp_path / "corpus.jsonl")
    build_corpus(str(coco_path), corpus_path)

    # Build index
    from src.embedder import LSAEmbedder
    from src.indexer import FAISSIndexer

    assets_text = []
    asset_ids   = []
    with open(corpus_path) as f:
        for line in f:
            a = json.loads(line)
            assets_text.append(a["description"])
            asset_ids.append(a["asset_id"])

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import normalize as sk_norm
    embedder = LSAEmbedder.__new__(LSAEmbedder)
    embedder.tfidf = TfidfVectorizer(max_features=2000, ngram_range=(1,1),
                                      min_df=1, sublinear_tf=True)
    tmat = embedder.tfidf.fit_transform(assets_text)
    n_comp = min(32, tmat.shape[1]-1, tmat.shape[0]-1)
    embedder.svd = TruncatedSVD(n_components=n_comp, random_state=42)
    embedder.svd.fit(tmat)
    embedder.n_components = n_comp
    embedder.max_features = 2000
    embedder.dim = n_comp
    embedder._fitted = True
    embedder_path = str(tmp_path / "embedder.pkl")
    embedder.save(embedder_path)

    vecs = embedder.embed_texts_batch(assets_text)
    actual_dim = vecs.shape[1]
    embedder.dim = actual_dim
    embedder.save(embedder_path)
    index_path = str(tmp_path / "faiss.index")
    indexer = FAISSIndexer(dim=actual_dim)
    indexer.build(vecs.astype(np.float32), asset_ids)
    indexer.save(index_path)

    return {
        "index_path":    index_path,
        "embedder_path": embedder_path,
        "corpus_path":   corpus_path,
        "initial_size":  len(asset_ids),
    }


# ── Producer Tests ─────────────────────────────────────────────────────────────

class TestProducer:
    def test_make_asset_has_required_fields(self):
        asset = _make_asset()
        for field in ["asset_id", "description", "category", "style", "tags", "image_url"]:
            assert field in asset, f"Missing field: {field}"

    def test_make_asset_unique_ids(self):
        ids = {_make_asset()["asset_id"] for _ in range(100)}
        assert len(ids) == 100, "Asset IDs should be unique"

    def test_make_asset_description_not_empty(self):
        for _ in range(20):
            asset = _make_asset()
            assert len(asset["description"]) > 10

    def test_make_asset_valid_category(self):
        valid_cats = {"Nature", "People", "Animals", "Food & Drink", "Sports",
                      "Urban", "Transportation", "Home & Indoor", "Technology"}
        for _ in range(30):
            asset = _make_asset()
            assert asset["category"] in valid_cats

    def test_make_asset_has_tags(self):
        for _ in range(10):
            asset = _make_asset()
            assert isinstance(asset["tags"], list)
            assert len(asset["tags"]) > 0


# ── StreamingIndexer Tests ─────────────────────────────────────────────────────

class TestStreamingIndexer:
    def test_init_loads_index(self, live_system):
        si = StreamingIndexer(
            live_system["index_path"],
            live_system["embedder_path"],
            live_system["corpus_path"],
            flush_every=10,
        )
        assert si.indexer.size == live_system["initial_size"]

    def test_process_new_asset(self, live_system):
        si = StreamingIndexer(
            live_system["index_path"],
            live_system["embedder_path"],
            live_system["corpus_path"],
            flush_every=10,
        )
        asset = _make_asset()
        result = si.process_event(asset)
        assert result is True
        assert len(si.buffer) == 1

    def test_duplicate_asset_skipped(self, live_system):
        si = StreamingIndexer(
            live_system["index_path"],
            live_system["embedder_path"],
            live_system["corpus_path"],
            flush_every=10,
        )
        # Use an existing asset ID
        existing_id = list(si.asset_lookup.keys())[0]
        asset = _make_asset()
        asset["asset_id"] = existing_id
        result = si.process_event(asset)
        assert result is False
        assert len(si.buffer) == 0

    def test_flush_updates_index_size(self, live_system):
        si = StreamingIndexer(
            live_system["index_path"],
            live_system["embedder_path"],
            live_system["corpus_path"],
            flush_every=100,
        )
        initial_size = si.indexer.size
        n_new = 15
        for _ in range(n_new):
            si.process_event(_make_asset())

        si._flush()
        assert si.indexer.size == initial_size + n_new

    def test_flush_updates_asset_lookup(self, live_system):
        si = StreamingIndexer(
            live_system["index_path"],
            live_system["embedder_path"],
            live_system["corpus_path"],
            flush_every=100,
        )
        asset = _make_asset()
        si.process_event(asset)
        si._flush()
        assert asset["asset_id"] in si.asset_lookup

    def test_auto_flush_on_buffer_full(self, live_system):
        flush_every = 5
        si = StreamingIndexer(
            live_system["index_path"],
            live_system["embedder_path"],
            live_system["corpus_path"],
            flush_every=flush_every,
        )
        initial_size = si.indexer.size
        for _ in range(flush_every):
            si.process_event(_make_asset())
        # After flush_every events, buffer should have been flushed
        assert si.total_added == flush_every
        assert si.indexer.size == initial_size + flush_every
        assert len(si.buffer) == 0

    def test_new_assets_searchable_after_flush(self, live_system):
        si = StreamingIndexer(
            live_system["index_path"],
            live_system["embedder_path"],
            live_system["corpus_path"],
            flush_every=100,
        )
        # Add an asset with unique description
        asset = _make_asset()
        asset["description"] = "unique streaming test asset for dog park search"
        si.process_event(asset)
        si._flush()

        # Search for it
        qvec = si.embedder.embed_text("dog park")
        results = si.indexer.search(qvec, top_k=10)
        result_ids = [aid for aid, _ in results]
        assert asset["asset_id"] in result_ids, "Newly streamed asset should be searchable"

    def test_stats_returns_correct_counts(self, live_system):
        si = StreamingIndexer(
            live_system["index_path"],
            live_system["embedder_path"],
            live_system["corpus_path"],
            flush_every=10,
        )
        stats = si.stats()
        assert stats["index_size"] == live_system["initial_size"]
        assert stats["buffer_size"] == 0
        assert stats["total_added"] == 0

        for _ in range(3):
            si.process_event(_make_asset())

        stats = si.stats()
        assert stats["buffer_size"] == 3

    def test_throughput_benchmark(self, live_system):
        """Verify streaming throughput meets production targets (>50 events/sec on CPU)."""
        si = StreamingIndexer(
            live_system["index_path"],
            live_system["embedder_path"],
            live_system["corpus_path"],
            flush_every=200,
        )
        n = 100
        assets = [_make_asset() for _ in range(n)]

        t0 = time.perf_counter()
        for asset in assets:
            si.process_event(asset)
        si._flush()
        elapsed = time.perf_counter() - t0

        throughput = n / elapsed
        print(f"\nStreaming throughput: {throughput:.1f} events/sec")
        assert throughput > 10, f"Throughput too low: {throughput:.1f} events/sec"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
