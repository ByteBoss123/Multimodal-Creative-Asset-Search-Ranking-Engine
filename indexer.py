"""
indexer.py
----------
FAISS IVFFlat ANN index for fast asset retrieval at scale.
Supports build, save, load, single query, and batch query.
"""

import json
import numpy as np
import faiss
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class FAISSIndexer:
    def __init__(self, dim: int = 512):
        self.dim = dim
        self.index = None
        self.id_map: list = []

    def build(self, embeddings: np.ndarray, asset_ids: list) -> None:
        assert embeddings.shape[0] == len(asset_ids)
        emb = embeddings.astype(np.float32)
        faiss.normalize_L2(emb)
        self.id_map = list(asset_ids)
        N = len(asset_ids)

        if N > 1000:
            nlist = max(64, int(np.sqrt(N)))
            quantizer = faiss.IndexFlatIP(self.dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.dim, nlist,
                                             faiss.METRIC_INNER_PRODUCT)
            logger.info(f"Training IVF index: {N} vectors, {nlist} clusters...")
            self.index.train(emb)
            self.index.add(emb)
            self.index.nprobe = min(32, nlist)
        else:
            self.index = faiss.IndexFlatIP(self.dim)
            self.index.add(emb)

        logger.info(f"Index ready: {self.index.ntotal:,} vectors")

    def search(self, query_vec: np.ndarray, top_k: int = 50) -> list:
        q = query_vec.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(q)
        scores, indices = self.index.search(q, top_k)
        return [(self.id_map[idx], float(s))
                for s, idx in zip(scores[0], indices[0]) if idx != -1]

    def save(self, index_path: str) -> None:
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, index_path)
        idmap_path = index_path.replace(".index", "_idmap.json")
        with open(idmap_path, "w") as f:
            json.dump(self.id_map, f)
        logger.info(f"Index saved: {index_path}")

    @classmethod
    def load(cls, index_path: str, dim: int = 512) -> "FAISSIndexer":
        inst = cls(dim=dim)
        inst.index = faiss.read_index(index_path)
        idmap_path = index_path.replace(".index", "_idmap.json")
        with open(idmap_path) as f:
            inst.id_map = json.load(f)
        logger.info(f"Index loaded: {inst.index.ntotal:,} vectors")
        return inst

    @property
    def size(self) -> int:
        return self.index.ntotal if self.index else 0
