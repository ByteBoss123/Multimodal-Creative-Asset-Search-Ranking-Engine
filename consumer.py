"""
stream/consumer.py
------------------
Real-time streaming indexer for PixelSeek.

Consumes asset upload events from Kafka, embeds each new asset
using the LSA embedder, and adds it to the live FAISS index —
no full index rebuild required.

This mirrors how Adobe Stock indexes billions of assets:
new uploads flow through a streaming pipeline and become
searchable within seconds, not hours.

Architecture:
    Kafka topic: asset-uploads
         │
         ▼
    StreamingIndexer (consumer)
         │
         ├── embed_text(description)     → LSA 512-dim vector
         ├── faiss_index.add(vector)     → in-place index update
         ├── asset_lookup[id] = metadata → in-memory store update
         └── flush to disk every N assets
         │
         ▼
    Updated FAISS index (live, searchable immediately)

Usage (requires Kafka running locally):
    python stream/consumer.py
    python stream/consumer.py --bootstrap localhost:9092 --flush-every 100
"""

import json
import time
import logging
import argparse
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

TOPIC         = "asset-uploads"
INDEX_PATH    = "models/faiss.index"
EMBEDDER_PATH = "models/embedder.pkl"
CORPUS_PATH   = "data/corpus.jsonl"


class StreamingIndexer:
    """
    Real-time FAISS index updater.
    Consumes Kafka messages and adds new asset embeddings
    to the live index without full rebuild.
    """

    def __init__(self, index_path: str, embedder_path: str, corpus_path: str,
                 flush_every: int = 100):
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from src.embedder import LSAEmbedder
        from src.indexer import FAISSIndexer
        from src.corpus import load_asset_lookup

        logger.info("Loading embedder and index...")
        self.embedder     = LSAEmbedder.load(embedder_path)
        self.indexer      = FAISSIndexer.load(index_path)
        self.asset_lookup = load_asset_lookup(corpus_path)
        self.corpus_path  = corpus_path
        self.index_path   = index_path
        self.flush_every  = flush_every
        self.buffer       = []   # pending (embedding, asset) pairs
        self.total_added  = 0

        logger.info(f"StreamingIndexer ready. Index size: {self.indexer.size:,}")

    def process_event(self, event: dict) -> bool:
        """
        Process a single asset upload event.
        Returns True if successfully indexed.
        """
        asset_id = event.get("asset_id")
        if not asset_id:
            return False

        # Skip duplicates
        if asset_id in self.asset_lookup:
            logger.debug(f"Duplicate asset {asset_id} — skipping")
            return False

        description = event.get("description", "")
        if not description:
            return False

        # Embed the new asset description
        vec = self.embedder.embed_text(description).astype(np.float32)

        # Store in buffer
        self.buffer.append((vec, event))

        # Flush when buffer is full
        if len(self.buffer) >= self.flush_every:
            self._flush()

        return True

    def _flush(self):
        """Batch-add buffered embeddings to FAISS and persist to disk."""
        if not self.buffer:
            return

        vecs      = np.vstack([v for v, _ in self.buffer]).astype(np.float32)
        asset_ids = [a["asset_id"] for _, a in self.buffer]

        # Add to FAISS index in-place
        import faiss
        faiss.normalize_L2(vecs)
        self.indexer.index.add(vecs)
        self.indexer.id_map.extend(asset_ids)

        # Update in-memory lookup
        for _, asset in self.buffer:
            self.asset_lookup[asset["asset_id"]] = asset

        # Append to corpus JSONL
        with open(self.corpus_path, "a") as f:
            for _, asset in self.buffer:
                f.write(json.dumps(asset) + "\n")

        # Persist updated index to disk
        self.indexer.save(self.index_path)

        self.total_added += len(self.buffer)
        logger.info(f"Flushed {len(self.buffer)} assets. "
                    f"Index size: {self.indexer.size:,} | "
                    f"Total added this session: {self.total_added}")
        self.buffer.clear()

    def stats(self) -> dict:
        return {
            "index_size":    self.indexer.size,
            "assets_loaded": len(self.asset_lookup),
            "buffer_size":   len(self.buffer),
            "total_added":   self.total_added,
        }


def run_consumer(bootstrap: str = "localhost:9092", flush_every: int = 100,
                 group_id: str = "pixelseek-indexer"):
    """Main consumer loop — connects to Kafka and processes events."""
    try:
        from kafka import KafkaConsumer
        consumer = KafkaConsumer(
            TOPIC,
            bootstrap_servers=bootstrap,
            group_id=group_id,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            auto_offset_reset="earliest",
            enable_auto_commit=True,
            consumer_timeout_ms=30000,   # exit after 30s of no messages
        )
    except Exception as e:
        logger.error(f"Kafka not available ({e}). Running in simulation mode.")
        _simulate(flush_every)
        return

    indexer = StreamingIndexer(INDEX_PATH, EMBEDDER_PATH, CORPUS_PATH, flush_every)

    logger.info(f"Consuming from topic '{TOPIC}'...")
    processed = 0
    t0 = time.time()

    try:
        for msg in consumer:
            event = msg.value
            if indexer.process_event(event):
                processed += 1
                if processed % 50 == 0:
                    elapsed = time.time() - t0
                    rate = processed / elapsed
                    logger.info(f"Processed {processed} events | "
                                f"Rate: {rate:.1f} events/sec | "
                                f"Stats: {indexer.stats()}")
    except KeyboardInterrupt:
        logger.info("Consumer stopped by user.")
    finally:
        indexer._flush()   # flush any remaining buffer
        consumer.close()
        logger.info(f"Consumer done. Total processed: {processed}")


def _simulate(flush_every: int):
    """
    Simulation mode (no Kafka required).
    Generates synthetic events and indexes them in real-time,
    demonstrating the full streaming pipeline end-to-end.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from stream.producer import _make_asset

    logger.info("SIMULATION MODE — running streaming pipeline without Kafka")
    indexer = StreamingIndexer(INDEX_PATH, EMBEDDER_PATH, CORPUS_PATH, flush_every)

    n_events = 200
    t0 = time.time()

    for i in range(n_events):
        asset = _make_asset()
        indexer.process_event(asset)

        # Print throughput every 50 events
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            logger.info(f"Streamed {i+1}/{n_events} events | "
                        f"Throughput: {rate:.1f} events/sec | "
                        f"Index: {indexer.indexer.size:,} vectors")

    # Final flush
    indexer._flush()
    elapsed = time.time() - t0
    logger.info(f"\nSimulation complete.")
    logger.info(f"  Events processed : {n_events}")
    logger.info(f"  Total time       : {elapsed:.2f}s")
    logger.info(f"  Throughput       : {n_events/elapsed:.1f} events/sec")
    logger.info(f"  Final index size : {indexer.indexer.size:,} vectors")
    logger.info(f"  New assets added : {indexer.total_added}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap",   default="localhost:9092")
    parser.add_argument("--flush-every", type=int, default=100)
    parser.add_argument("--group-id",    default="pixelseek-indexer")
    parser.add_argument("--simulate",    action="store_true",
                        help="Run without Kafka (simulation mode)")
    args = parser.parse_args()

    if args.simulate:
        _simulate(args.flush_every)
    else:
        run_consumer(args.bootstrap, args.flush_every, args.group_id)
