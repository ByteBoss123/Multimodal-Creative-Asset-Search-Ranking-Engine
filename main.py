"""
main.py — PixelSeek FastAPI serving layer
Endpoints: POST /search, GET /health, GET /stats, GET /asset/{id}
"""

import time
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embedder import LSAEmbedder
from src.indexer import FAISSIndexer
from src.reranker import Reranker
from src.corpus import load_asset_lookup
from api.schemas import (SearchRequest, SearchResponse, AssetResult,
                         HealthResponse, StatsResponse)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INDEX_PATH    = "models/faiss.index"
EMBEDDER_PATH = "models/embedder.pkl"
CORPUS_PATH   = "data/corpus.jsonl"
_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("PixelSeek starting...")

    if Path(EMBEDDER_PATH).exists():
        _state["embedder"] = LSAEmbedder.load(EMBEDDER_PATH)
    else:
        logger.warning("No embedder found. Run build_index.py first.")
        _state["embedder"] = None

    if Path(INDEX_PATH).exists():
        _state["indexer"] = FAISSIndexer.load(INDEX_PATH)
    else:
        logger.warning("No index found. Run build_index.py first.")
        _state["indexer"] = None

    if Path(CORPUS_PATH).exists():
        _state["assets"] = load_asset_lookup(CORPUS_PATH)
        logger.info(f"Loaded {len(_state['assets']):,} assets")
    else:
        _state["assets"] = {}

    _state["reranker"] = Reranker()
    logger.info("PixelSeek ready.")
    yield
    _state.clear()


app = FastAPI(title="PixelSeek",
              description="Semantic search over MS COCO (40K images, 200K captions)",
              version="1.0.0",
              lifespan=lifespan)

app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    t0 = time.perf_counter()
    if not _state.get("indexer") or not _state.get("embedder"):
        raise HTTPException(503, "System not ready. Run build_index.py first.")

    qvec     = _state["embedder"].embed_text(req.query)
    cands    = _state["indexer"].search(qvec, top_k=req.recall_k)

    if req.category_filter:
        cands = [(aid, s) for aid, s in cands
                 if _state["assets"].get(aid, {}).get("category") == req.category_filter]

    reranked = _state["reranker"].rerank(req.query, cands, _state["assets"], req.top_k)
    if req.diversity:
        reranked = _state["reranker"].diversity_rerank(reranked)

    ms = (time.perf_counter() - t0) * 1000
    return SearchResponse(
        query=req.query,
        results=[AssetResult(
            rank=r.rank, asset_id=r.asset_id, image_url=r.image_url,
            file_name=r.file_name, captions=r.captions[:3],
            category=r.category, tags=r.tags,
            clip_score=round(r.clip_score, 4),
            final_score=round(r.final_score, 4),
        ) for r in reranked],
        total_candidates=len(cands),
        latency_ms=round(ms, 2),
        index_size=_state["indexer"].size,
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    idx = _state.get("indexer")
    return HealthResponse(
        status="ok" if idx else "degraded",
        index_size=idx.size if idx else 0,
        model="TF-IDF/LSA-512 (CLIP-ready)",
    )


@app.get("/stats", response_model=StatsResponse)
async def stats():
    assets = _state.get("assets", {})
    cats = {}
    for a in assets.values():
        c = a.get("category", "Other")
        cats[c] = cats.get(c, 0) + 1
    idx = _state.get("indexer")
    return StatsResponse(
        total_assets=len(assets),
        categories=dict(sorted(cats.items(), key=lambda x: -x[1])),
        index_type="IVFFlat (nlist=201, nprobe=32)",
        embedding_dim=512,
    )


@app.get("/asset/{asset_id}")
async def get_asset(asset_id: str):
    asset = _state.get("assets", {}).get(asset_id)
    if not asset:
        raise HTTPException(404, f"Asset {asset_id} not found")
    return {k: v for k, v in asset.items() if k != "relevance_labels"}
