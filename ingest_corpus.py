"""
ingest_corpus.py
----------------
Build the searchable corpus from the real MS COCO val2014 captions JSON.

Downloads raw COCO captions if not already present, then processes
into asset JSONL format with tags, categories, and relevance labels.

Run:
    python scripts/ingest_corpus.py
    python scripts/ingest_corpus.py --limit 5000   # smaller subset
"""

import sys
import argparse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.corpus import build_corpus

COCO_URL    = ("https://raw.githubusercontent.com/tylin/coco-caption"
               "/master/annotations/captions_val2014.json")
COCO_PATH   = "data/coco_captions.json"
CORPUS_PATH = "data/corpus.jsonl"


def download_coco_if_needed():
    if Path(COCO_PATH).exists():
        print(f"COCO captions already at {COCO_PATH}")
        return
    Path(COCO_PATH).parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading MS COCO val2014 captions from GitHub...")
    req = urllib.request.Request(COCO_URL, headers={"User-Agent": "pixelseek/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read()
    with open(COCO_PATH, "wb") as f:
        f.write(data)
    print(f"Saved {len(data)/1e6:.1f} MB → {COCO_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Max images to include (default: all ~40K)")
    parser.add_argument("--coco",   default=COCO_PATH)
    parser.add_argument("--output", default=CORPUS_PATH)
    args = parser.parse_args()

    download_coco_if_needed()
    assets = build_corpus(args.coco, args.output, limit=args.limit)
    print(f"\nCorpus ready: {len(assets):,} assets at {args.output}")
