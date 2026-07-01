"""
Turn a raw candidate record (per candidate_schema.json) into a flat feature dict
used by the scorer. Kept separate from scoring logic so features are easy to
inspect/debug independently (dumped to data/features_sample.json for review).
"""
from __future__ import annotations
from datetime import date, datetime
from typing import Any

from . import jd_config as J

TODAY = date(2026, 7, 1)  # dataset is synthetic/fixed; using a stable "as of" date


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError:
        return None


def _career_text(candidate: dict) -> str:
    """All career_history descriptions + titles concatenated — this is where we
    hunt for real production evidence, deliberately separate from the skills list."""
    parts = []
    for job in candidate.get("career_history", []):
        parts.append(job.get("title", ""))
        parts.append(job.get("description", ""))
    return " \n ".join(parts).lower()


def _full_text(candidate: dict) -> str:
    """Everything textual about the candidate — used for semantic similarity."""
    profile = candidate.get("profile", {})
    parts = [profile.get("headline", ""), profile.get("summary", "")]
    parts.append(_career_text(candidate))
    skill_names = " ".join(s.get("name", "") for s in candidate.get("skills", []))
    parts.append(skill_names)
    return " \n ".join(parts)


def _term_hits(text: str, terms: list[str]) -> list[str]:
    return [t for t in terms if t in text]


def _tenure_churn_score(career_history: list[dict]) -> float:
    """Detects title-chasing pattern: many short (<18mo) stints. Returns 0-1,
    1 = no churn concern, 0 = strong title-chaser pattern."""
    if len(career_history) < 3:
        return 1.0
    durations = [j.get("duration_months", 0) or 0 for j in career_history]
    short_stints = sum(1 for d in durations if 0 < d < 18)
    ratio = short_stints / len(durations)
    return max(0.0, 1.0 - ratio * 1.3)


def _consulting_only(career_history: list[dict]) -> bool:
    companies = [ (j.get("company") or "").lower() for j in career_history ]
    if not companies:
        return False
    all_consulting = all(any(cf in c for cf in J.CONSULTING_FIRMS) for c in companies)
    return all_consulting


def _days_since(d: date | None) -> int | None:
    if d is None:
        return None
    return (TODAY - d).days


def extract_features(candidate: dict) -> dict[str, Any]:
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])

    career_text = _career_text(candidate)
    full_text = _full_text(candidate)

    embed_hits = _term_hits(career_text, J.EMBEDDING_RETRIEVAL_TERMS)
    vdb_hits = _term_hits(career_text, J.VECTOR_DB_TERMS)
    eval_hits = _term_hits(career_text, J.EVAL_FRAMEWORK_TERMS)
    prod_hits = _term_hits(career_text, J.PRODUCTION_SCALE_TERMS)
    nice_hits = _term_hits(career_text, J.NICE_TO_HAVE_TERMS)
    nlp_hits = _term_hits(career_text, J.NLP_IR_TERMS)
    cv_hits = _term_hits(career_text, J.CV_SPEECH_ROBOTICS_TERMS)
    research_hits = _term_hits(career_text, J.RESEARCH_ONLY_TERMS)
    langchain_only_hits = _term_hits(career_text, J.RECENT_LANGCHAIN_ONLY_TERMS)
    architect_hits = _term_hits(career_text, J.ARCHITECT_NO_CODE_TERMS)

    location = (profile.get("location") or "").lower()
    country = (profile.get("country") or "").lower()

    last_active = _parse_date(signals.get("last_active_date"))
    days_inactive = _days_since(last_active)

    feats = {
        "candidate_id": candidate.get("candidate_id"),
        "years_of_experience": profile.get("years_of_experience", 0) or 0,
        "current_title": profile.get("current_title", ""),
        "current_company": profile.get("current_company", ""),
        "location": profile.get("location", ""),
        "country": profile.get("country", ""),

        # production-evidence signal counts (career_history only, not skills)
        "embed_hit_count": len(embed_hits),
        "vdb_hit_count": len(vdb_hits),
        "eval_hit_count": len(eval_hits),
        "prod_hit_count": len(prod_hits),
        "nice_hit_count": len(nice_hits),
        "embed_hits": embed_hits,
        "vdb_hits": vdb_hits,
        "eval_hits": eval_hits,

        # NLP/IR vs CV-speech-robotics balance
        "nlp_hit_count": len(nlp_hits),
        "cv_hit_count": len(cv_hits),

        # disqualifier raw signals
        "research_only_hits": len(research_hits),
        "langchain_only_hits": len(langchain_only_hits),
        "architect_hits": len(architect_hits),
        "consulting_only": _consulting_only(career),
        "tenure_churn_score": _tenure_churn_score(career),
        "num_jobs": len(career),

        # location fit
        "is_preferred_location": any(p in location for p in J.PREFERRED_LOCATIONS),
        "is_acceptable_india_location": (country == "india") and any(p in location for p in J.ACCEPTABLE_INDIA_LOCATIONS),
        "is_india": country == "india",
        "willing_to_relocate": bool(signals.get("willing_to_relocate", False)),

        # notice period / logistics
        "notice_period_days": signals.get("notice_period_days", 999),

        # behavioral signals (raw, for the multiplier)
        "open_to_work_flag": bool(signals.get("open_to_work_flag", False)),
        "days_inactive": days_inactive if days_inactive is not None else 9999,
        "recruiter_response_rate": signals.get("recruiter_response_rate", 0) or 0,
        "interview_completion_rate": signals.get("interview_completion_rate", 0) or 0,
        "profile_completeness_score": signals.get("profile_completeness_score", 0) or 0,
        "verified_email": bool(signals.get("verified_email", False)),
        "verified_phone": bool(signals.get("verified_phone", False)),
        "search_appearance_30d": signals.get("search_appearance_30d", 0) or 0,
        "saved_by_recruiters_30d": signals.get("saved_by_recruiters_30d", 0) or 0,
        "offer_acceptance_rate": signals.get("offer_acceptance_rate", -1),
        "github_activity_score": signals.get("github_activity_score", -1),

        # for semantic similarity + reasoning generation
        "full_text": full_text,
        "summary": profile.get("summary", ""),
        "headline": profile.get("headline", ""),
        "career_history": career,
        "skills": skills,
        "education": candidate.get("education", []),
        "raw_signals": signals,
    }
    return feats
