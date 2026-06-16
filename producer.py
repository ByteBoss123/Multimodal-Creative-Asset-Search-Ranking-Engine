"""
stream/producer.py
------------------
Simulates a creative asset upload stream — the same pattern used by
Adobe Stock, Shutterstock, and Getty when new images are submitted.

Each asset upload event is published to a Kafka topic as a JSON message.
The streaming indexer (consumer.py) picks these up and indexes them
into FAISS in near-real-time without rebuilding the full index.

Usage (requires Kafka running locally):
    python stream/producer.py --rate 10   # 10 assets/sec
    python stream/producer.py --batch data/corpus.jsonl  # replay corpus
"""

import json
import time
import uuid
import random
import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

TOPIC = "asset-uploads"

CATEGORIES = ["Nature", "People", "Animals", "Food & Drink", "Sports",
               "Urban", "Transportation", "Home & Indoor", "Technology"]
STYLES = ["photorealistic", "illustration", "vector", "minimalist", "cinematic"]

def _make_asset():
    cat = random.choice(CATEGORIES)
    style = random.choice(STYLES)
    subjects = {
        "Nature": ["sunset", "mountain", "ocean", "forest"],
        "People": ["portrait", "crowd", "athlete", "family"],
        "Animals": ["dog", "cat", "bird", "lion"],
        "Food & Drink": ["pizza", "coffee", "sushi", "cocktail"],
        "Sports": ["basketball", "surfing", "cycling", "tennis"],
        "Urban": ["skyline", "bridge", "street", "architecture"],
        "Transportation": ["car", "airplane", "bicycle", "train"],
        "Home & Indoor": ["kitchen", "bedroom", "living room", "garden"],
        "Technology": ["laptop", "robot", "drone", "server"],
    }
    subject = random.choice(subjects.get(cat, ["scene"]))
    description = f"A {style} {cat.lower()} image featuring {subject} with vibrant tones"
    return {
        "asset_id":    str(uuid.uuid4()),
        "event":       "asset.uploaded",
        "timestamp":   time.time(),
        "category":    cat,
        "style":       style,
        "description": description,
        "tags":        [subject, cat.lower(), style],
        "image_url":   f"https://example.com/assets/{uuid.uuid4()}.jpg",
        "file_name":   f"asset_{uuid.uuid4().hex[:8]}.jpg",
    }


def produce_stream(rate_per_sec: int = 5, max_events: int = 1000,
                   bootstrap: str = "localhost:9092"):
    """Publish synthetic asset upload events to Kafka."""
    try:
        from kafka import KafkaProducer
        producer = KafkaProducer(
            bootstrap_servers=bootstrap,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            acks="all",
            retries=3,
        )
    except Exception as e:
        logger.error(f"Kafka not available ({e}). Running in dry-run mode.")
        _dry_run(rate_per_sec, max_events)
        return

    interval = 1.0 / rate_per_sec
    sent = 0
    logger.info(f"Producing to topic '{TOPIC}' at {rate_per_sec} events/sec...")

    try:
        while sent < max_events:
            asset = _make_asset()
            future = producer.send(TOPIC, value=asset)
            future.get(timeout=10)
            sent += 1
            if sent % 50 == 0:
                logger.info(f"Published {sent}/{max_events} asset events")
            time.sleep(interval)
    except KeyboardInterrupt:
        logger.info("Producer stopped.")
    finally:
        producer.flush()
        producer.close()
        logger.info(f"Done. Total published: {sent}")


def produce_from_corpus(corpus_path: str, bootstrap: str = "localhost:9092",
                        rate_per_sec: int = 100):
    """Replay existing corpus as a Kafka stream (backfill scenario)."""
    try:
        from kafka import KafkaProducer
        producer = KafkaProducer(
            bootstrap_servers=bootstrap,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            acks=1,
        )
    except Exception as e:
        logger.error(f"Kafka not available: {e}")
        return

    interval = 1.0 / rate_per_sec
    sent = 0
    logger.info(f"Replaying corpus: {corpus_path}")

    with open(corpus_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            asset = json.loads(line)
            asset["event"] = "asset.uploaded"
            asset["timestamp"] = time.time()
            producer.send(TOPIC, value=asset)
            sent += 1
            if sent % 500 == 0:
                logger.info(f"Replayed {sent} assets...")
            time.sleep(interval)

    producer.flush()
    producer.close()
    logger.info(f"Corpus replay complete: {sent} events published")


def _dry_run(rate_per_sec: int, max_events: int):
    """Dry run without Kafka — prints events to stdout."""
    logger.info(f"DRY RUN: generating {max_events} asset events at {rate_per_sec}/sec")
    interval = 1.0 / rate_per_sec
    for i in range(max_events):
        asset = _make_asset()
        print(json.dumps(asset))
        time.sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rate",      type=int, default=10, help="Events per second")
    parser.add_argument("--max",       type=int, default=500, help="Max events")
    parser.add_argument("--bootstrap", default="localhost:9092")
    parser.add_argument("--batch",     default=None, help="Corpus JSONL path for replay")
    args = parser.parse_args()

    if args.batch:
        produce_from_corpus(args.batch, args.bootstrap, args.rate)
    else:
        produce_stream(args.rate, args.max, args.bootstrap)
