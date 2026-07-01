"""
HuggingFace Spaces sandbox app. Accepts a small candidate sample (<=100
candidates) as a .jsonl upload, runs the exact same ranking pipeline as
rank.py, and returns a ranked CSV — satisfying submission_spec.md Section
10.5 (sandbox must accept a small sample and produce a ranked CSV end-to-end,
within the compute budget).

This does NOT re-implement scoring logic — it imports directly from src/, so
there is zero drift between what the sandbox demonstrates and what actually
produced the real submission.
"""
import json
import tempfile
import gradio as gr
import pandas as pd

from src.features import extract_features
from src.honeypot import check_honeypot
from src.semantic import compute_semantic_scores
from src.scorer import score_candidate
from src.reasoning import generate_reasoning

SAMPLE_PATH = "data/sample_candidates.jsonl"  # bundled small sample, ships with the Space


def run_ranker(uploaded_file):
    if uploaded_file is not None:
        path = uploaded_file.name
    else:
        path = SAMPLE_PATH  # fall back to the bundled sample if nothing uploaded

    candidates = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))

    if len(candidates) > 100:
        candidates = candidates[:100]

    feats_list = [extract_features(c, light=False) for c in candidates]
    texts = [f["full_text"] for f in feats_list]
    sem_scores = compute_semantic_scores(texts)

    results = []
    for c, f, sem in zip(candidates, feats_list, sem_scores):
        violations, severity = check_honeypot(c)
        r = score_candidate(f, float(sem), severity)
        r["_feats"] = f
        results.append(r)

    results.sort(key=lambda r: (-r["score"], r["candidate_id"]))

    rows = []
    for i, r in enumerate(results):
        rank = i + 1
        reasoning = generate_reasoning(r["_feats"], r, rank)
        rows.append({
            "candidate_id": r["candidate_id"],
            "rank": rank,
            "score": round(r["score"], 6),
            "reasoning": reasoning,
        })

    df = pd.DataFrame(rows)

    out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".csv").name
    df.to_csv(out_path, index=False)

    return df, out_path


with gr.Blocks(title="Redrob Ranker — ThunderCode") as demo:
    gr.Markdown(
        "# Redrob Candidate Ranker — Team ThunderCode\n"
        "Upload a small `.jsonl` candidate sample (≤100 candidates) to see the "
        "ranking pipeline run end-to-end. Leave empty to use the bundled sample. "
        "This runs the exact same code (`src/`) as the full 100K-candidate "
        "submission — CPU-only, no network calls, no GPU."
    )
    with gr.Row():
        upload = gr.File(label="Upload candidates.jsonl (optional)", file_types=[".jsonl"])
    run_btn = gr.Button("Run ranker", variant="primary")
    output_table = gr.Dataframe(label="Ranked results")
    output_file = gr.File(label="Download ranked CSV")

    run_btn.click(fn=run_ranker, inputs=[upload], outputs=[output_table, output_file])

if __name__ == "__main__":
    demo.launch()
