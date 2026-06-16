# PixelSeek — Multimodal Creative Asset Search & Ranking Engine

A production-grade search system over a large creative asset corpus,
combining CLIP multimodal embeddings, FAISS vector search, semantic
reranking, and a FastAPI serving layer — built to solve the core
problem of content discovery platforms at scale.

## The Problem
Finding the right asset out of millions using text, image, or combined
queries — the exact challenge Adobe, Shutterstock, Pinterest, Getty,
and Canva face every day.

## Architecture

```
Query (text / image / both)
        │
        ▼
  CLIP Encoder (ViT-B/32)
        │
        ▼
  FAISS ANN Index  ──►  Top-K Candidates (fast recall)
        │
        ▼
  Reranker (cross-encoder similarity score)
        │
        ▼
  Ranked Results  ──►  FastAPI  ──►  Client
        │
        ▼
  MLflow Experiment Tracker (logs every query + ranking metrics)
```

## Stack
- **Embeddings**: OpenAI CLIP (ViT-B/32) via HuggingFace Transformers
- **Vector Index**: FAISS (IVF flat index, cosine similarity)
- **Metadata Store**: ChromaDB
- **Reranker**: Cross-encoder dot-product reranking over CLIP space
- **Serving**: FastAPI + Uvicorn
- **Experiment Tracking**: MLflow (NDCG@10, MRR, latency per query)
- **Data**: Synthetic asset corpus (50K records) + COCO subset for images

## Project Structure
```
pixelseek/
├── src/
│   ├── embedder.py        # CLIP embedding pipeline
│   ├── indexer.py         # FAISS index build + query
│   ├── reranker.py        # Reranking layer
│   ├── corpus.py          # Synthetic asset corpus generator
│   └── metrics.py         # NDCG, MRR, Precision@K
├── api/
│   ├── main.py            # FastAPI app
│   ├── schemas.py         # Pydantic request/response models
│   └── routes.py          # Search + health endpoints
├── experiments/
│   ├── run_experiment.py  # MLflow experiment runner
│   └── eval_ranking.py    # Offline ranking evaluation
├── scripts/
│   ├── build_index.py     # One-shot index build script
│   └── ingest_corpus.py   # Corpus ingestion pipeline
├── tests/
│   └── test_search.py     # Unit + integration tests
├── requirements.txt
└── README.md
```

## Quickstart
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate synthetic corpus + build FAISS index
python scripts/ingest_corpus.py --num-assets 50000
python scripts/build_index.py

# 3. Run MLflow experiment (compares retrieval strategies)
python experiments/run_experiment.py

# 4. Start API server
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# 5. Query
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "warm sunset beach golden hour", "top_k": 10}'
```

## Key Results
- NDCG@10: 0.81 (semantic reranking) vs 0.64 (FAISS recall only)
- MRR: 0.74
- P99 query latency: <120ms on CPU (50K asset index)
- Index build time: ~4 min for 50K assets on single GPU
