"""
reranker.py
-----------
Two-stage reranker for creative asset search.

Stage 1: FAISS returns top-K candidates (fast recall)
Stage 2: Composite reranking using:
  - CLIP cosine similarity  (primary signal)
  - Tag overlap bonus       (query-tag lexical match)
  - Category prior          (learned engagement weights)
  - Optional MMR diversity  (penalise same-category clustering)
"""

import re
import numpy as np
from dataclasses import dataclass

CATEGORY_PRIORS = {
    "People":         0.13,
    "Animals":        0.12,
    "Nature":         0.11,
    "Food & Drink":   0.10,
    "Sports":         0.10,
    "Urban":          0.09,
    "Transportation": 0.09,
    "Home & Indoor":  0.08,
    "Technology":     0.07,
    "Other":          0.05,
}


@dataclass
class RankedResult:
    asset_id:    str
    image_url:   str
    file_name:   str
    captions:    list
    category:    str
    tags:        list
    clip_score:  float
    tag_bonus:   float
    final_score: float
    rank:        int = 0


class Reranker:
    def __init__(self, tag_weight: float = 0.15, prior_weight: float = 0.05):
        self.tag_weight   = tag_weight
        self.prior_weight = prior_weight

    def rerank(self, query: str, candidates: list,
               asset_lookup: dict, top_k: int = 10) -> list:
        q_tokens = set(re.findall(r"[a-z]+", query.lower()))
        scored = []

        for asset_id, clip_score in candidates:
            asset = asset_lookup.get(asset_id)
            if not asset:
                continue

            # Tag overlap signal
            tag_tokens  = set(t.lower() for t in asset.get("tags", []))
            desc_tokens = set(re.findall(r"[a-z]+",
                                         asset.get("description", "").lower()))
            overlap = len(q_tokens & (tag_tokens | desc_tokens))
            s_tag = min(1.0, overlap / max(len(q_tokens), 1))

            # Category prior
            s_prior = CATEGORY_PRIORS.get(asset.get("category", "Other"), 0.05)

            final = (clip_score
                     + self.tag_weight * s_tag
                     + self.prior_weight * s_prior)

            scored.append(RankedResult(
                asset_id=asset_id,
                image_url=asset.get("image_url", ""),
                file_name=asset.get("file_name", ""),
                captions=asset.get("captions", []),
                category=asset.get("category", "Other"),
                tags=asset.get("tags", []),
                clip_score=float(clip_score),
                tag_bonus=float(s_tag),
                final_score=float(final),
            ))

        scored.sort(key=lambda r: r.final_score, reverse=True)
        results = scored[:top_k]
        for i, r in enumerate(results):
            r.rank = i + 1
        return results

    def diversity_rerank(self, results: list,
                         diversity_weight: float = 0.3) -> list:
        """MMR-style: penalise repeated categories."""
        seen: dict = {}
        for r in results:
            count = seen.get(r.category, 0)
            r.final_score -= diversity_weight * count * 0.05
            seen[r.category] = count + 1
        results.sort(key=lambda r: r.final_score, reverse=True)
        for i, r in enumerate(results):
            r.rank = i + 1
        return results
