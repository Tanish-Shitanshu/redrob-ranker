"""
Generates the `reasoning` column. Per submission_spec Section 3, this is
manually reviewed at Stage 4 against: specific facts, JD connection, honest
concerns, no hallucination, cross-candidate variation, and rank-consistent tone.

Design: build reasoning from actual extracted feature values (never invent
anything), vary sentence structure based on which components drove the score,
and always name at least one concrete gap/concern if one exists rather than
only praising — even for top-ranked candidates, if a real concern exists
(e.g. long notice period) it gets a clause, matching the doc's own example
row 3 ("some concern on notice period... but otherwise strong fit").
"""
from __future__ import annotations
import random


def _fmt_hits(hits: list[str], max_n=2) -> str:
    return ", ".join(hits[:max_n])


def generate_reasoning(feats: dict, result: dict, rank: int) -> str:
    rng = random.Random(feats["candidate_id"])  # deterministic per-candidate variation
    yoe = feats["years_of_experience"]
    title = feats["current_title"] or "their current role"
    company = feats["current_company"] or "their current company"
    comps = result["components"]

    clauses = []

    # Opening: years + title + strongest concrete evidence
    if feats["embed_hits"] or feats["vdb_hits"]:
        evidence = _fmt_hits(feats["embed_hits"] + feats["vdb_hits"])
        clauses.append(f"{yoe} years of experience, currently {title} at {company}, with career history showing hands-on production work involving {evidence}")
    elif feats["eval_hits"]:
        clauses.append(f"{yoe} years of experience as {title} at {company}, with direct experience in ranking/evaluation work ({_fmt_hits(feats['eval_hits'])})")
    elif comps["title_relevance"] >= 0.5:
        clauses.append(f"{yoe} years of experience in a software/ML-adjacent role ({title} at {company}), though career history doesn't show explicit embeddings or vector-search production work")
    else:
        clauses.append(f"{yoe} years of experience as {title} at {company}")

    # JD connection: title/role fit
    if comps["title_relevance"] >= 0.9:
        clauses.append("role trajectory lines up closely with the AI/ML engineering scope the JD describes")
    elif comps["title_relevance"] >= 0.5:
        clauses.append("background is software/data engineering adjacent to the role rather than a direct AI/ML-engineering match")
    elif comps["title_relevance"] <= 0.05:
        clauses.append("title and career history are not in a software/ML engineering track, which is a fundamental mismatch for this JD regardless of other signals")

    # Location
    if feats["is_preferred_location"]:
        clauses.append(f"based in {feats['location']}, matching the JD's Pune/Noida preference")
    elif feats["is_acceptable_india_location"]:
        clauses.append(f"based in {feats['location']}, one of the JD's accepted India locations")
    elif feats["is_india"] and feats["willing_to_relocate"]:
        clauses.append(f"based in {feats['location']} but marked willing to relocate")
    elif not feats["is_india"]:
        clauses.append(f"based outside India ({feats['location']}), and the JD does not sponsor visas, so location is a real constraint")

    # Honest concerns (always include if any exist, even for top candidates)
    concerns = []
    if feats["notice_period_days"] > 60:
        concerns.append(f"a {feats['notice_period_days']}-day notice period, longer than the JD's stated preference")
    if feats["days_inactive"] > 90:
        concerns.append(f"no platform activity in {feats['days_inactive']} days, raising availability concerns")
    if feats["recruiter_response_rate"] < 0.2:
        concerns.append(f"a low recruiter response rate ({feats['recruiter_response_rate']:.0%})")
    if result["disqualifier_reasons"]:
        concerns.append(result["disqualifier_reasons"][0])
    if comps["experience_fit"] < 0.6:
        concerns.append(f"{yoe} years of experience is outside the JD's stated 5-9 year band")

    if concerns:
        clauses.append("concern: " + concerns[0])

    reasoning = "; ".join(clauses) + "."
    reasoning = reasoning[0].upper() + reasoning[1:]

    # Trim to keep it to roughly 1-2 sentences as spec requests
    if len(reasoning) > 320:
        reasoning = reasoning[:317].rsplit(";", 1)[0] + "."

    return reasoning
