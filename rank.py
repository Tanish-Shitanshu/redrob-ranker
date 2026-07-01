#!/usr/bin/env python3
"""
Main entrypoint. Reads candidates.jsonl, produces the top-100 ranked
submission.csv per submission_spec.md Sections 2-3.

Usage:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Compute: designed to run within 5 min / 16GB RAM / CPU-only / no network on
the full 100,000-candidate pool. No hosted LLM calls anywhere in this file.

Two-pass streaming design (memory efficiency):
  Pass 1: stream every candidate once, extract lightweight features (scalars,
          short strings, hit counts, full_text for semantic scoring) — the
          raw parsed JSON and heavy nested fields (career_history, skills,
          education) are dropped immediately after use, never held for all
          100K candidates at once. Score everyone, keep only the top 100 ids.
  Pass 2: stream the file again, and only for the ~100 winning candidate_ids,
          re-parse with full (non-light) features to build grounded reasoning
          text. This keeps peak memory to "100K lightweight feature dicts"
          instead of "100K full nested candidate objects".

Semantic similarity defaults to a local TF-IDF layer (zero downloads); if
models/bge-small/ exists locally (see scripts/precompute_embeddings.py), a
local sentence-transformers model is blended in automatically — still no
network access at rank time either way.
"""
from __future__ import annotations
import argparse
import csv
import gc
import gzip
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.features import extract_features
from src.honeypot import check_honeypot
from src.semantic import compute_semantic_scores
from src.scorer import score_candidate
from src.reasoning import generate_reasoning

TOP_N = 100


def _opener(path: str):
    return gzip.open if str(path).endswith(".gz") else open


def iter_candidates(path: str):
    with _opener(path)(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t0 = time.time()

    # ---------- Pass 1: lightweight feature extraction + scoring for all ----------
    light_feats = []
    texts = []
    severities = {}
    n = 0
    for cand in iter_candidates(args.candidates):
        f = extract_features(cand, light=True)
        violations, severity = check_honeypot(cand)
        light_feats.append(f)
        texts.append(f["full_text"])
        severities[f["candidate_id"]] = severity
        n += 1
    print(f"[pass1] loaded+featurized {n} candidates in {time.time()-t0:.1f}s", file=sys.stderr)
    gc.collect()

    t1 = time.time()
    sem_scores = compute_semantic_scores(texts)
    print(f"[pass1] semantic scoring in {time.time()-t1:.1f}s", file=sys.stderr)
    del texts
    gc.collect()

    t2 = time.time()
    results = []
    for f, sem in zip(light_feats, sem_scores):
        sev = severities[f["candidate_id"]]
        r = score_candidate(f, float(sem), sev)
        r["_light_feats"] = f
        results.append(r)
    print(f"[pass1] scored all candidates in {time.time()-t2:.1f}s", file=sys.stderr)

    results.sort(key=lambda r: (-r["score"], r["candidate_id"]))
    top = results[:TOP_N]
    top_ids = {r["candidate_id"] for r in top}

    del light_feats, results, sem_scores
    gc.collect()

    # ---------- Pass 2: re-extract full features only for top 100, for reasoning ----------
    t3 = time.time()
    full_feats_by_id = {}
    for cand in iter_candidates(args.candidates):
        cid = cand.get("candidate_id")
        if cid in top_ids:
            full_feats_by_id[cid] = extract_features(cand, light=False)
            if len(full_feats_by_id) == len(top_ids):
                break
    print(f"[pass2] re-extracted full features for top {len(full_feats_by_id)} in {time.time()-t3:.1f}s", file=sys.stderr)

    # ---------- Build output rows ----------
    rows = []
    prev_score = None
    for i, r in enumerate(top):
        rank = i + 1
        score = r["score"]
        if prev_score is not None and score > prev_score:
            score = prev_score  # enforce non-increasing (safety net)
        prev_score = score

        cid = r["candidate_id"]
        full_f = full_feats_by_id.get(cid, r["_light_feats"])
        reasoning = generate_reasoning(full_f, r, rank)
        rows.append({
            "candidate_id": cid,
            "rank": rank,
            "score": round(score, 6),
            "reasoning": reasoning,
        })

    out_path = Path(args.out)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        writer.writerows(rows)

    honeypots_in_top100 = sum(1 for r in top if severities.get(r["candidate_id"], 0) > 0)
    print(f"Wrote {len(rows)} rows to {out_path}", file=sys.stderr)
    print(f"Honeypot-flagged in top {TOP_N}: {honeypots_in_top100} ({honeypots_in_top100/TOP_N:.1%})", file=sys.stderr)
    print(f"Total runtime: {time.time()-t0:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
