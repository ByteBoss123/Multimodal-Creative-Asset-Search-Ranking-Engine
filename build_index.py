"""
build_index.py
--------------
Fit LSA embedder on real COCO corpus, encode all asset descriptions,
and build FAISS IVFFlat index for sub-100ms ANN search.

Pipeline:
  1. Load corpus JSONL
  2. Fit TF-IDF + SVD (LSA) on all descriptions
  3. Batch-encode to 512-dim vectors
  4. Build FAISS IVFFlat index
  5. Save index + embedder to models/

Run:
    python scripts/build_index.py
"""

import sys
import json
import argparse
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.embedder import LSAEmbedder
from src.indexer import FAISSIndexer

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def build(corpus_path: str, index_path: str, embedder_path: str,
          batch_size: int = 2048):

    logger.info(f"Loading corpus: {corpus_path}")
    asset_ids, texts = [], []
    with open(corpus_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            a = json.loads(line)
            asset_ids.append(a["asset_id"])
            # Use all captions joined — richer signal than just one
            texts.append(a["description"][:1000])

    logger.info(f"Corpus: {len(asset_ids):,} assets")

    # Fit LSA embedder on full corpus
    embedder = LSAEmbedder(n_components=512, max_features=50000)
    embedder.fit(texts)
    embedder.save(embedder_path)

    # Batch encode
    logger.info("Encoding all descriptions...")
    embeddings = embedder.embed_texts_batch(texts, batch_size=batch_size)
    logger.info(f"Embeddings: {embeddings.shape}, dtype={embeddings.dtype}")

    # Build FAISS index
    logger.info("Building FAISS IVFFlat index...")
    indexer = FAISSIndexer(dim=embeddings.shape[1])
    indexer.build(embeddings, asset_ids)
    indexer.save(index_path)
    logger.info(f"Done. {indexer.size:,} vectors at {index_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus",   default="data/corpus.jsonl")
    parser.add_argument("--output",   default="models/faiss.index")
    parser.add_argument("--embedder", default="models/embedder.pkl")
    args = parser.parse_args()
    build(args.corpus, args.output, args.embedder)
