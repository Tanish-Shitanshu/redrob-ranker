"""
Semantic similarity layer. Default: TF-IDF + cosine similarity — zero external
downloads, deterministic, fast enough for 100K candidates well within the
5-minute compute budget, and fully inspectable/defensible in a live interview.

Optional upgrade: if a local sentence-transformers model directory exists at
models/bge-small (see scripts/precompute_embeddings.py, run once locally with
network), semantic_similarity() blends in dense embedding cosine similarity.
This never touches the network at rank time either way.
"""
from __future__ import annotations
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import jd_config as J

EMBED_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "bge-small")


def tfidf_similarity(jd_text: str, candidate_texts: list[str]) -> np.ndarray:
    corpus = [jd_text] + candidate_texts
    vec = TfidfVectorizer(
        stop_words="english",
        max_features=20000,
        ngram_range=(1, 2),
        min_df=2,
    )
    tfidf = vec.fit_transform(corpus)
    jd_vec = tfidf[0:1]
    cand_vecs = tfidf[1:]
    sims = cosine_similarity(jd_vec, cand_vecs)[0]
    return sims


def embedding_similarity_available() -> bool:
    return os.path.isdir(EMBED_MODEL_DIR)


def embedding_similarity(jd_text: str, candidate_texts: list[str]) -> np.ndarray:
    """Only called if a local model directory is present. Loads from local path
    only — never hits the network, satisfying the no-network-at-rank-time rule."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBED_MODEL_DIR)
    jd_vec = model.encode([jd_text], normalize_embeddings=True)
    cand_vecs = model.encode(candidate_texts, normalize_embeddings=True, batch_size=64, show_progress_bar=True)
    sims = cosine_similarity(jd_vec, cand_vecs)[0]
    return sims


def compute_semantic_scores(candidate_texts: list[str]) -> np.ndarray:
    tfidf_sims = tfidf_similarity(J.JD_FULL_TEXT, candidate_texts)
    # normalize tfidf sims to 0-1 range via min-max (cosine sims on tfidf tend to be small/skewed)
    lo, hi = tfidf_sims.min(), tfidf_sims.max()
    if hi > lo:
        tfidf_norm = (tfidf_sims - lo) / (hi - lo)
    else:
        tfidf_norm = tfidf_sims

    if embedding_similarity_available():
        emb_sims = embedding_similarity(J.JD_FULL_TEXT, candidate_texts)
        elo, ehi = emb_sims.min(), emb_sims.max()
        emb_norm = (emb_sims - elo) / (ehi - elo) if ehi > elo else emb_sims
        # blend: embeddings capture meaning better, tfidf keeps exact-term precision
        return 0.6 * emb_norm + 0.4 * tfidf_norm

    return tfidf_norm
