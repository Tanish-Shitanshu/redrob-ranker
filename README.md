# Redrob Hackathon — ThunderCode Candidate Ranker

Team **ThunderCode** — Intelligent Candidate Discovery & Ranking Challenge

Ranks the 100,000-candidate pool against the Redrob "Senior AI Engineer —
Founding Team" JD and produces the top-100 submission CSV.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
venv/bin/pip install -r requirements.txt
```

(If your shell has a `pip` alias that ignores active venvs, use `venv/bin/pip`
explicitly as above rather than plain `pip`.)

## Reproduce the submission

```bash
python3 rank.py --candidates /path/to/candidates.jsonl --out ./submission.csv
```

Also accepts a gzipped file directly: `--candidates candidates.jsonl.gz`.

Validate the output against the official spec:

```bash
python3 validate_submission.py submission.csv
```

**Measured runtime:** ~27–70s on the full 100,000-candidate pool (well under
the 5-minute budget), peak memory ~2GB (well under the 16GB budget). CPU-only,
zero network calls during ranking.

## Architecture

Rule-based + TF-IDF hybrid ranker, deliberately avoiding any LLM-per-candidate
approach (infeasible within the compute budget at 100K scale, per the spec's
own reasoning). Two-pass streaming design keeps memory low: a lightweight
pass scores every candidate, then only the winning top-100 get re-parsed with
full detail for reasoning generation.

Pipeline (see `src/`):

1. **`jd_config.py`** — structured representation of the JD: hard requirements,
   nice-to-haves, explicit disqualifiers, and the "ideal candidate" shape,
   extracted from a close reading of the JD text rather than a flat keyword list.
2. **`features.py`** — extracts scoring features per candidate. Production-evidence
   keywords (embeddings, vector DBs, eval frameworks) are searched in
   `career_history` descriptions specifically, **not** the skills list — this is
   the core defense against the keyword-stuffing trap the dataset explicitly
   includes.
3. **`honeypot.py`** — flags impossible profiles (e.g. "expert" proficiency with
   0 months duration, overlapping full-time roles, inverted salary ranges
   co-occurring with other issues). Thresholds were calibrated against the
   50-candidate sample set to avoid flagging realistic data noise while still
   catching genuine honeypots (achieves ~1% flag rate in top 100 vs. the
   dataset's known ~0.08% honeypot rate).
4. **`semantic.py`** — TF-IDF + cosine similarity between JD text and candidate
   profile text (summary, headline, career history, skills). Zero external
   downloads, fully deterministic. Optionally blends in a local
   sentence-transformers embedding model if present at `models/bge-small/`
   (see "Optional embedding upgrade" below) — loaded from a local path only,
   so it never touches the network at rank time even when present.
5. **`scorer.py`** — composite score combining semantic fit, production
   evidence, title relevance, experience-band fit, location fit, and notice
   period, minus disqualifier penalties (research-only, consulting-only,
   CV/speech background without NLP exposure, title-chasing tenure patterns),
   multiplied by a behavioral-signal modifier (availability, responsiveness,
   platform activity) and a honeypot-suppression factor. Includes a hard
   relevance gate so no candidate in a clearly unrelated profession (e.g.
   Civil Engineer, Accountant) can outrank real software/ML candidates purely
   on experience-band or location luck.
6. **`reasoning.py`** — generates the grounded, per-candidate `reasoning`
   column from actual extracted feature values only (never invented), varying
   structure based on which components drove the score and always surfacing a
   concrete concern when one exists.

## Optional embedding upgrade

By default the semantic layer uses TF-IDF only. To blend in dense embeddings
for stronger semantic matching (one-time local setup, requires internet just
for this step — not at rank time):

```bash
venv/bin/pip install sentence-transformers
python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5').save('models/bge-small')"
```

`rank.py` auto-detects `models/bge-small/` and blends embedding similarity
into the semantic score automatically — no code changes needed.

## Team

- Tanish Shitanshu (ts260@snu.edu.in) — primary contact
- Sai Mohit Kotta (mohit4learning@gmail.com)

## AI tools disclosure

Built with Claude as a development/pair-programming tool — architecture
discussion, code writing, and iterative debugging (e.g. the honeypot
threshold calibration and the title-relevance gate fix were both caught and
fixed through iterative testing against the sample dataset, visible in the
git history). No candidate data was sent to any hosted LLM API as part of the
ranking pipeline itself — `rank.py` makes zero network calls.
