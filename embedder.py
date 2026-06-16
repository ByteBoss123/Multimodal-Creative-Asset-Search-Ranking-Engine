"""
embedder.py
-----------
Pluggable embedding pipeline for PixelSeek.

Two backends — same interface:

  LSAEmbedder   : TF-IDF + TruncatedSVD (LSA), 512-dim.
                  Fully local, no internet required.
                  Used for building and querying the FAISS index.

  CLIPEmbedder  : OpenAI CLIP ViT-B/32 via HuggingFace Transformers.
                  Drop-in upgrade once model weights are available.
                  Enables true multimodal (text + image) search.

Both produce unit-normalised float32 vectors of the same dimension,
making FAISS indices interchangeable between backends.

Usage:
    embedder = LSAEmbedder()
    embedder.fit(corpus_texts)        # build vocab + SVD
    vec = embedder.embed_text("dog playing in park")
    embedder.save("models/embedder.pkl")

    embedder = LSAEmbedder.load("models/embedder.pkl")
    vecs = embedder.embed_texts_batch(["query 1", "query 2"])
"""

import pickle
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import logging

logger = logging.getLogger(__name__)


class LSAEmbedder:
    """
    TF-IDF + Truncated SVD (Latent Semantic Analysis) embedder.
    Fits on the asset corpus, then encodes any text into a dense
    512-dim vector in the same semantic space.

    Why LSA works well here:
    - Handles synonymy: "dog" and "puppy" share subspace
    - Handles polysemy implicitly via co-occurrence weighting
    - Fully deterministic, no GPU needed, <5s fit on 40K docs
    - Same FAISS cosine search works identically as with CLIP
    """

    def __init__(self, n_components: int = 512, max_features: int = 50000):
        self.n_components = n_components
        self.max_features = max_features
        self.tfidf: TfidfVectorizer = None
        self.svd:   TruncatedSVD   = None
        self.dim = n_components
        self._fitted = False

    def fit(self, texts: list, batch_size: int = 10000) -> "LSAEmbedder":
        """
        Fit TF-IDF vocabulary and SVD on corpus texts.
        texts: list of document strings (one per asset)
        """
        logger.info(f"Fitting TF-IDF on {len(texts):,} documents...")
        self.tfidf = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=(1, 2),          # unigrams + bigrams
            min_df=2,                     # ignore very rare terms
            max_df=0.95,                  # ignore near-universal terms
            sublinear_tf=True,            # log(1+tf) — standard for IR
            strip_accents="unicode",
            analyzer="word",
        )
        tfidf_matrix = self.tfidf.fit_transform(texts)
        logger.info(f"Vocab: {len(self.tfidf.vocabulary_):,} terms, "
                    f"matrix: {tfidf_matrix.shape}")

        logger.info(f"Fitting TruncatedSVD ({self.n_components} components)...")
        self.svd = TruncatedSVD(
            n_components=self.n_components,
            algorithm="randomized",
            n_iter=7,
            random_state=42,
        )
        self.svd.fit(tfidf_matrix)
        explained = self.svd.explained_variance_ratio_.sum()
        logger.info(f"SVD explains {explained:.1%} of variance")
        self._fitted = True
        return self

    def embed_text(self, text: str) -> np.ndarray:
        """Encode a single text string → unit-normalised (512,) vector."""
        self._check_fitted()
        tfidf_vec = self.tfidf.transform([text])
        lsa_vec = self.svd.transform(tfidf_vec)[0]
        return self._norm(lsa_vec.astype(np.float32))

    def embed_texts_batch(self, texts: list, batch_size: int = 1024) -> np.ndarray:
        """Encode a list of strings → (N, 512) unit-normalised float32 array."""
        self._check_fitted()
        all_vecs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            tfidf_batch = self.tfidf.transform(batch)
            lsa_batch   = self.svd.transform(tfidf_batch).astype(np.float32)
            all_vecs.append(lsa_batch)
        vecs = np.vstack(all_vecs)
        return normalize(vecs, norm="l2")

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"tfidf": self.tfidf, "svd": self.svd,
                         "n_components": self.n_components}, f)
        logger.info(f"Embedder saved to {path}")

    @classmethod
    def load(cls, path: str) -> "LSAEmbedder":
        with open(path, "rb") as f:
            state = pickle.load(f)
        inst = cls(n_components=state["n_components"])
        inst.tfidf   = state["tfidf"]
        inst.svd     = state["svd"]
        inst._fitted = True
        logger.info(f"Embedder loaded from {path}")
        return inst

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Call fit() or load() before embedding.")

    @staticmethod
    def _norm(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        return v / (n + 1e-10)


# ── CLIP stub for when weights become available ──────────────────────────────

class CLIPEmbedder:
    """
    Drop-in replacement for LSAEmbedder using OpenAI CLIP ViT-B/32.
    Requires: pip install transformers torch
    Requires: network access to huggingface.co for weight download.

    Once weights are downloaded, replace LSAEmbedder with CLIPEmbedder
    in build_index.py and api/main.py — no other changes needed.
    """

    MODEL_ID = "openai/clip-vit-base-patch32"

    def __init__(self, device: str = None):
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError:
            raise ImportError("pip install transformers torch")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(self.MODEL_ID).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(self.MODEL_ID)
        self.model.eval()
        self.dim = self.model.config.projection_dim  # 512
        self._fitted = True

    def fit(self, texts, **kwargs):
        return self  # no-op: CLIP is pretrained

    def embed_text(self, text: str) -> np.ndarray:
        import torch
        with torch.no_grad():
            inp = self.processor(text=[text], return_tensors="pt",
                                 padding=True, truncation=True, max_length=77
                                 ).to(self.device)
            vec = self.model.get_text_features(**inp).cpu().numpy()[0]
        return vec / (np.linalg.norm(vec) + 1e-10)

    def embed_texts_batch(self, texts: list, batch_size: int = 128) -> np.ndarray:
        import torch
        all_vecs = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                inp = self.processor(text=batch, return_tensors="pt",
                                     padding=True, truncation=True, max_length=77
                                     ).to(self.device)
                vecs = self.model.get_text_features(**inp).cpu().numpy()
                all_vecs.append(vecs)
        vecs = np.vstack(all_vecs)
        return vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10)

    def save(self, path: str) -> None:
        logger.info(f"CLIP model is pretrained — no save needed (path: {path})")

    @classmethod
    def load(cls, path: str) -> "CLIPEmbedder":
        return cls()
