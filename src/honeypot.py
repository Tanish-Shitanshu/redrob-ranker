"""
Detects the ~80 honeypot candidates with subtly impossible profiles, per
README/submission_spec: e.g. "expert" proficiency in a skill with 0 years used,
or career timelines that don't add up. These are generic, schema-level
consistency checks — NOT JD-specific — matching the spec's framing that a good
system should "naturally avoid them" through careful profile inspection, not
by special-casing known honeypot IDs.

Returns a list of violation strings (empty list = clean) and a severity score.
Used by the scorer as a multiplicative suppression factor, not a hard drop —
so a borderline data-quality quirk doesn't necessarily nuke a real candidate,
but a candidate with multiple violations gets pushed far down.
"""
from __future__ import annotations
from datetime import datetime


def _parse(d):
    if not d:
        return None
    try:
        return datetime.strptime(d, "%Y-%m-%d").date()
    except ValueError:
        return None


def check_honeypot(candidate: dict) -> tuple[list[str], float]:
    violations = []
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    education = candidate.get("education", [])
    yoe = profile.get("years_of_experience", 0) or 0

    # NOTE on calibration: this dataset has a lot of realistic *noise* (e.g. skill
    # duration_months that don't perfectly reconcile with years_of_experience, or
    # inverted salary ranges from messy self-reported data). Only ~80/100,000
    # candidates are true honeypots, so these checks are intentionally strict —
    # tuned against the sample set to avoid flagging normal noisy-but-real profiles.

    # 1. "Expert" proficiency with near-zero duration_months — the doc's own example
    zero_duration_experts = [
        s for s in skills
        if s.get("proficiency") == "expert" and s.get("duration_months") == 0
    ]
    if len(zero_duration_experts) >= 1:
        names = ", ".join(s.get("name", "?") for s in zero_duration_experts[:3])
        violations.append(f"{len(zero_duration_experts)} skill(s) marked 'expert' with 0 months duration ({names})")

    # 2. Too many "expert" skills relative to total experience (breadth-depth mismatch)
    expert_count = sum(1 for s in skills if s.get("proficiency") == "expert")
    if expert_count >= 10:
        violations.append(f"{expert_count} skills marked 'expert' — implausible breadth")

    # 3. Skill duration grossly exceeds total career experience (>2x + large absolute margin,
    # to avoid flagging normal data noise)
    total_months_possible = yoe * 12
    for s in skills:
        dur = s.get("duration_months")
        if dur is not None and total_months_possible > 0 and dur > total_months_possible * 2 and dur - total_months_possible > 24:
            violations.append(f"skill '{s.get('name')}' duration {dur}mo grossly exceeds total experience ({yoe}yr)")

    # 5. Overlapping full-time roles (more than one is_current, or overlapping date ranges)
    current_count = sum(1 for j in career if j.get("is_current"))
    if current_count > 1:
        violations.append(f"{current_count} roles marked is_current simultaneously")

    parsed_ranges = []
    for j in career:
        sd = _parse(j.get("start_date"))
        ed = _parse(j.get("end_date")) if j.get("end_date") else None
        if sd:
            parsed_ranges.append((sd, ed))
    parsed_ranges.sort()
    for i in range(len(parsed_ranges) - 1):
        _, end1 = parsed_ranges[i]
        start2, _ = parsed_ranges[i + 1]
        if end1 and start2 and start2 < end1:
            overlap_days = (end1 - start2).days
            if overlap_days > 90:  # small overlaps common (notice periods), big ones suspicious
                violations.append(f"overlapping roles by {overlap_days} days")

    # 6. Education timeline inconsistency
    for e in education:
        sy, ey = e.get("start_year"), e.get("end_year")
        if sy and ey and ey < sy:
            violations.append(f"education end_year {ey} before start_year {sy}")
        if sy and ey and (ey - sy) > 10:
            violations.append(f"education span {sy}-{ey} implausibly long")

    # 7. Experience years grossly inconsistent with earliest education start_year
    # (e.g. 12+ years of experience but started undergrad 9 years ago). Wide slack
    # (8 years) to absorb people who started working before/during studies, career
    # switchers, or missing/partial education records.
    if education:
        earliest_edu_start = min((e.get("start_year") or 9999) for e in education)
        years_since_edu_start = 2026 - earliest_edu_start
        if yoe > years_since_edu_start + 8:
            violations.append(f"{yoe}yr experience implausible given earliest education start {earliest_edu_start}")

    # 8. Salary range inverted — treated as a soft data-quality flag, not counted
    # toward severity on its own (observed too frequently in real-looking profiles
    # in this dataset to be a reliable honeypot signal by itself; only escalates
    # severity when it co-occurs with another violation).
    signals = candidate.get("redrob_signals", {})
    sal = signals.get("expected_salary_range_inr_lpa", {})
    salary_inverted = bool(sal) and sal.get("min") is not None and sal.get("max") is not None and sal["min"] > sal["max"]
    if salary_inverted and violations:
        violations.append("expected salary min > max (co-occurring with other issues)")

    severity = min(1.0, len(violations) * 0.4)
    return violations, severity
