"""
Composite scorer. Structure:

  base_score = weighted sum of:
      semantic_fit          (TF-IDF cosine similarity, JD vs candidate full text)
      production_evidence   (career_history evidence of embeddings/vector-db/eval work)
      title_relevance       (does the actual job title/role match, not just skills tags)
      experience_fit         (soft band around 5-9 yrs)
      location_fit

  base_score -= disqualifier_penalty   (research-only, consulting-only, etc.)

  final_score = base_score * behavioral_multiplier * honeypot_suppression

This ordering matters: disqualifiers and honeypot suppression are applied as
penalties/multipliers on top of a fit score, not folded into a single flat sum,
so a candidate can't "buy back" a real disqualifier with, say, a great location.
"""
from __future__ import annotations
import math
from typing import Any

from . import jd_config as J


def title_relevance(feats: dict) -> float:
    """0-1. Looks at current_title AND all past titles, not just current, since
    someone titled 'Backend Engineer' who previously built ranking systems still
    counts. Explicitly penalizes non-technical and non-software-engineering
    titles regardless of skills list content — this is the JD's stated
    anti-keyword-stuffing check (a 'Marketing Manager' with AI skills tags is
    not a fit)."""
    titles = [feats["current_title"]] + [j.get("title", "") for j in feats["career_history"]]
    titles_text = " ".join(t.lower() for t in titles if t)

    strong_terms = ["ai engineer", "ml engineer", "machine learning engineer", "search engineer",
                     "ranking", "retrieval", "recommendation", "applied scientist", "nlp engineer",
                     "data scientist", "research engineer"]
    software_adjacent_terms = ["backend engineer", "software engineer", "data engineer",
                                "platform engineer", "full stack", "founding engineer",
                                "infrastructure engineer", "devops", "site reliability",
                                "mobile developer", "frontend engineer", "java developer",
                                "python developer", "cloud engineer", "systems engineer",
                                "database", "qa engineer", "test engineer", "analytics engineer"]
    clearly_unrelated_terms = [
        "civil engineer", "mechanical engineer", "electrical engineer", "chemical engineer",
        "structural engineer", "accountant", "accounting", "hr ", "human resources", "recruiter",
        "sales", "marketing", "business development", "account manager", "content writer",
        "customer success", "customer support", "teacher", "doctor", "nurse", "lawyer",
        "legal counsel", "operations manager", "logistics", "supply chain", "graphic designer",
        "financial analyst", "auditor",
    ]

    if any(t in titles_text for t in clearly_unrelated_terms) and not any(t in titles_text for t in strong_terms):
        return 0.03

    if any(t in titles_text for t in strong_terms):
        return 1.0
    if any(t in titles_text for t in software_adjacent_terms):
        return 0.55
    # generic/unclear title with no unrelated OR software signal at all — low, not moderate
    return 0.1


def production_evidence(feats: dict) -> float:
    """0-1. Weighted toward the JD's explicit hard requirements: embeddings +
    vector-db/hybrid-search + eval-framework experience, evidenced in career text."""
    embed = min(feats["embed_hit_count"], 3) / 3
    vdb = min(feats["vdb_hit_count"], 3) / 3
    eval_ = min(feats["eval_hit_count"], 2) / 2
    prod = min(feats["prod_hit_count"], 3) / 3
    nice = min(feats["nice_hit_count"], 4) / 4

    # hard requirements weighted heavily; "nice to have" is a small bonus only
    score = 0.35 * embed + 0.30 * vdb + 0.20 * eval_ + 0.10 * prod + 0.05 * nice
    return min(1.0, score)


def experience_fit(feats: dict) -> float:
    lo, hi = J.EXPERIENCE_BAND
    yoe = feats["years_of_experience"]
    if lo <= yoe <= hi:
        return 1.0
    dist = (lo - yoe) if yoe < lo else (yoe - hi)
    # soft decay: ~0.5 at 3 years off-band, floor at 0.15 so it's never a hard wall
    return max(0.15, 1.0 - dist / 6.0)


def location_fit(feats: dict) -> float:
    if feats["is_preferred_location"]:
        return 1.0
    if feats["is_acceptable_india_location"]:
        return 0.85
    if feats["is_india"] and feats["willing_to_relocate"]:
        return 0.7
    if feats["is_india"]:
        return 0.5
    # outside India: JD says case-by-case, no visa sponsorship
    if feats["willing_to_relocate"]:
        return 0.3
    return 0.15


def notice_period_fit(feats: dict) -> float:
    days = feats["notice_period_days"]
    if days <= J.NOTICE_GOOD:
        return 1.0
    if days <= J.NOTICE_GOOD + J.NOTICE_BUYOUT_LIMIT:  # buyout covers up to 30 extra days
        return 0.85
    if days <= 90:
        return 0.6
    return 0.4


def disqualifier_penalty(feats: dict) -> tuple[float, list[str]]:
    """Returns (penalty in [0, ~0.9], list of human-readable reasons)."""
    penalty = 0.0
    reasons = []

    prod_evidence = feats["embed_hit_count"] + feats["vdb_hit_count"] + feats["prod_hit_count"]

    # Pure research/academic with zero production evidence
    if feats["research_only_hits"] > 0 and prod_evidence == 0:
        penalty += 0.55
        reasons.append("research/academic background with no production deployment evidence")

    # Recent LangChain-only AI experience, no pre-LLM production ML evidence
    if feats["langchain_only_hits"] > 0 and feats["eval_hit_count"] == 0 and feats["vdb_hit_count"] == 0 and prod_evidence <= 1:
        penalty += 0.35
        reasons.append("AI experience appears limited to LLM-API wrapper work, no deeper production ML evidence")

    # Architect/tech-lead titles with no recent hands-on code signal (approximated:
    # architect-type titles dominate AND no production evidence at all)
    if feats["architect_hits"] > 0 and prod_evidence == 0 and feats["eval_hit_count"] == 0:
        penalty += 0.25
        reasons.append("architecture/leadership-heavy title history with limited hands-on production evidence")

    # Consulting-only entire career
    if feats["consulting_only"]:
        penalty += 0.4
        reasons.append("entire career at consulting/IT-services firms, no product-company experience")

    # CV/speech/robotics primary without NLP/IR exposure
    if feats["cv_hit_count"] >= 2 and feats["nlp_hit_count"] == 0:
        penalty += 0.35
        reasons.append("background concentrated in computer vision/speech/robotics with no NLP/IR exposure")

    # Title-chasing pattern (churny short tenures)
    if feats["tenure_churn_score"] < 0.6:
        penalty += (0.6 - feats["tenure_churn_score"]) * 0.5
        reasons.append("career pattern shows several short (<18mo) stints")

    return min(0.9, penalty), reasons


def behavioral_multiplier(feats: dict) -> float:
    """Multiplicative modifier in roughly [0.45, 1.15]. A great candidate who's
    unreachable/inactive is down-weighted, not zeroed — per the JD's explicit
    instruction to weigh availability."""
    m = 1.0

    if not feats["open_to_work_flag"]:
        m *= 0.85

    days_inactive = feats["days_inactive"]
    if days_inactive > 180:
        m *= 0.55
    elif days_inactive > 90:
        m *= 0.75
    elif days_inactive > 30:
        m *= 0.92

    rr = feats["recruiter_response_rate"]
    if rr < 0.1:
        m *= 0.7
    elif rr < 0.3:
        m *= 0.88

    if feats["interview_completion_rate"] < 0.3 and feats["interview_completion_rate"] >= 0:
        m *= 0.85

    completeness = feats["profile_completeness_score"]
    if completeness < 40:
        m *= 0.8
    elif completeness > 85:
        m *= 1.05

    if feats["verified_email"] and feats["verified_phone"]:
        m *= 1.03

    if feats["saved_by_recruiters_30d"] >= 5 or feats["search_appearance_30d"] >= 100:
        m *= 1.05  # external validation: recruiters already interested

    oar = feats["offer_acceptance_rate"]
    if oar != -1 and oar < 0.2:
        m *= 0.9  # history of rejecting/not-accepting offers

    return max(0.4, min(1.2, m))


def score_candidate(feats: dict, semantic_sim: float, honeypot_severity: float) -> dict[str, Any]:
    sem = semantic_sim
    prod = production_evidence(feats)
    title = title_relevance(feats)
    exp = experience_fit(feats)
    loc = location_fit(feats)
    notice = notice_period_fit(feats)

    base = (
        0.28 * sem +
        0.30 * prod +
        0.20 * title +
        0.12 * exp +
        0.07 * loc +
        0.03 * notice
    )

    penalty, dq_reasons = disqualifier_penalty(feats)
    base = max(0.0, base - penalty)

    # Hard relevance gate: a clearly-unrelated profession (civil engineer,
    # accountant, HR, etc.) must never outrank real software/ML candidates
    # purely on experience-band + location luck. title_relevance == 0.03 is the
    # sentinel for "clearly unrelated" from title_relevance() above.
    if title <= 0.03 and prod < 0.15:
        base *= 0.15
        dq_reasons.append("title/career history indicates a non-software profession, not a fit regardless of other signals")

    mult = behavioral_multiplier(feats)
    honeypot_factor = 1.0 - min(0.95, honeypot_severity)  # near-zero out strong honeypots, never fully to 0

    final = base * mult * honeypot_factor

    return {
        "candidate_id": feats["candidate_id"],
        "score": final,
        "components": {
            "semantic_fit": sem,
            "production_evidence": prod,
            "title_relevance": title,
            "experience_fit": exp,
            "location_fit": loc,
            "notice_fit": notice,
            "disqualifier_penalty": penalty,
            "behavioral_multiplier": mult,
            "honeypot_severity": honeypot_severity,
        },
        "disqualifier_reasons": dq_reasons,
    }
