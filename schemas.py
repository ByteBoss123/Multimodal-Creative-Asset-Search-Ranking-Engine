from pydantic import BaseModel, Field
from typing import Optional


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(default=10, ge=1, le=100)
    category_filter: Optional[str] = None
    diversity: bool = False
    recall_k: int = Field(default=50, ge=10, le=200)

    model_config = {"json_schema_extra": {"example": {
        "query": "a dog playing in the park",
        "top_k": 10, "diversity": True
    }}}


class AssetResult(BaseModel):
    rank:        int
    asset_id:    str
    image_url:   str
    file_name:   str
    captions:    list[str]
    category:    str
    tags:        list[str]
    clip_score:  float
    final_score: float


class SearchResponse(BaseModel):
    query:            str
    results:          list[AssetResult]
    total_candidates: int
    latency_ms:       float
    index_size:       int


class HealthResponse(BaseModel):
    status:     str
    index_size: int
    model:      str
    version:    str = "1.0.0"


class StatsResponse(BaseModel):
    total_assets: int
    categories:   dict
    index_type:   str
    embedding_dim: int
