"""
features.py
Pure functions that turn one raw candidate record (as parsed from
candidates.jsonl) into a flat dict of interpretable features.

Design principle: every feature here is something a human reviewer could
recompute by hand from the candidate's profile in under a minute. No
black-box embeddings live in this file — that keeps Stage-5 ("defend your
design choices") tractable, and keeps the reasoning column honest (we only
ever write down things we actually computed).
"""

from datetime import date

import config as cfg


def _to_date(s):
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except (ValueError, TypeError):
        return None


def _text_blob(candidate):
    """Concatenate all free-text fields into one lowercase blob for keyword
    and TF-IDF matching."""
    parts = []
    profile = candidate.get("profile", {})
    parts.append(profile.get("headline", ""))
    parts.append(profile.get("summary", ""))
    parts.append(profile.get("current_title", ""))
    for ch in candidate.get("career_history", []):
        parts.append(ch.get("title", ""))
        parts.append(ch.get("description", ""))
    for s in candidate.get("skills", []):
        parts.append(s.get("name", ""))
    return " | ".join(p for p in parts if p).lower()


def _count_hits(text, terms):
    return sum(1 for t in terms if t in text)


def _skill_lookup(candidate):
    return {s.get("name", "").lower(): s for s in candidate.get("skills", [])}


def core_skill_features(candidate, text):
    """Mandatory-skill evidence, trust-adjusted against keyword stuffing.

    For each of the four mandatory pillars (embeddings, vector DB/hybrid
    search, ranking & eval, LLM/RAG) we check both (a) whether the term
    appears anywhere in free text / skills, and (b) whether that evidence is
    *corroborated* — appears in career-history narrative (not just a bare
    skill tag), and the skill entry (if any) has nonzero duration_months or
    endorsements. A "skill" with proficiency=expert, 0 duration, 0
    endorsements, never mentioned in any job description, is exactly the
    keyword-stuffing trap the JD calls out — it gets ~0 trust weight.
    """
    skills = _skill_lookup(candidate)
    career_text = " | ".join(
        ch.get("description", "") for ch in candidate.get("career_history", [])
    ).lower()

    def pillar_score(term_list):
        hits_anywhere = _count_hits(text, term_list)
        if hits_anywhere == 0:
            return 0.0
        hits_in_career_narrative = _count_hits(career_text, term_list)
        corroboration = min(1.0, hits_in_career_narrative / max(1, hits_anywhere))
        base = min(1.0, hits_anywhere / 2.0)
        return base * (0.35 + 0.65 * corroboration)

    embeddings = pillar_score(cfg.EMBEDDING_TERMS)
    vectordb = pillar_score(cfg.VECTOR_DB_TERMS)
    ranking_eval = pillar_score(cfg.RANKING_EVAL_TERMS)
    llm = pillar_score(cfg.LLM_TERMS)
    python_ev = 1.0 if _count_hits(text, cfg.PYTHON_TERMS) > 0 else 0.0

    high_prof_skills = [
        s for s in skills.values() if s.get("proficiency") in ("expert", "advanced")
    ]
    if high_prof_skills:
        corroborated = 0
        for s in high_prof_skills:
            mentioned = s.get("name", "").lower() in career_text
            has_track_record = (s.get("duration_months") or 0) > 0
            has_social_proof = (s.get("endorsements") or 0) > 0
            if has_track_record and (mentioned or has_social_proof):
                corroborated += 1
        skill_trust_ratio = corroborated / len(high_prof_skills)
    else:
        skill_trust_ratio = 0.5

    pillars = [embeddings, vectordb, ranking_eval, max(llm, 0.3 * embeddings)]
    soft_min = (min(pillars) * 0.5) + (sum(pillars) / len(pillars) * 0.5)

    core_match = soft_min * (0.5 + 0.5 * python_ev) * (0.4 + 0.6 * skill_trust_ratio)

    return {
        "embeddings_evidence": embeddings,
        "vectordb_evidence": vectordb,
        "ranking_eval_evidence": ranking_eval,
        "llm_evidence": llm,
        "python_evidence": python_ev,
        "skill_trust_ratio": skill_trust_ratio,
        "core_skill_match": min(1.0, core_match),
    }


def credibility_features(candidate, text):
    """Production-language vs. tutorial-language, plus open validation
    (papers/talks/OSS) for the "closed-source-only 5+ years" soft penalty."""
    prod_hits = _count_hits(text, cfg.PRODUCTION_TERMS)
    tutorial_hits = _count_hits(text, cfg.TUTORIAL_ONLY_TERMS)
    open_validation_hits = _count_hits(text, cfg.OPEN_VALIDATION_TERMS)

    if prod_hits + tutorial_hits == 0:
        production_ratio = 0.5
    else:
        production_ratio = prod_hits / (prod_hits + tutorial_hits)

    yoe = candidate.get("profile", {}).get("years_of_experience", 0) or 0
    closed_source_penalty = 0.0
    if yoe >= 5 and open_validation_hits == 0 and production_ratio < 0.6:
        closed_source_penalty = 0.12

    credibility = max(0.0, min(1.0, production_ratio) - closed_source_penalty)
    return {
        "production_language_ratio": production_ratio,
        "open_validation_hits": open_validation_hits,
        "credibility_modifier": credibility,
    }


def experience_seniority_features(candidate, text):
    """Years-of-experience band fit (soft Gaussian around 7y) plus the
    "title-chaser" trajectory check and the JD's hard/soft disqualifiers
    (research-only, architecture-only-recently)."""
    yoe = candidate.get("profile", {}).get("years_of_experience", 0) or 0
    band_fit = pow(2.0, -((yoe - cfg.IDEAL_YOE_CENTER) ** 2) / (2 * cfg.IDEAL_YOE_WIDTH ** 2))
    band_fit = max(0.15, band_fit)

    history = candidate.get("career_history", [])
    recent = [h for h in history if (h.get("duration_months") or 0) > 0]
    short_stints = sum(1 for h in recent if (h.get("duration_months") or 0) < 18)
    title_escalation_words = ["senior", "staff", "principal", "lead", "head"]
    escalating = 0
    titles = [h.get("title", "").lower() for h in recent]
    for i in range(len(titles) - 1):
        a, b = titles[i], titles[i + 1]
        a_rank = next((i2 for i2, w in enumerate(title_escalation_words) if w in a), -1)
        b_rank = next((i2 for i2, w in enumerate(title_escalation_words) if w in b), -1)
        if a_rank > b_rank >= 0:
            escalating += 1

    title_chaser_flag = len(recent) >= 3 and short_stints >= len(recent) - 1 and escalating >= 1
    title_chaser_penalty = 0.25 if title_chaser_flag else 0.0

    research_only = bool(history) and all(
        _count_hits(h.get("description", "").lower(), cfg.RESEARCH_ONLY_TERMS) > 0
        or _count_hits(h.get("industry", "").lower(), ["research", "academia"]) > 0
        for h in history
    )

    current_jobs = [h for h in history if h.get("is_current")]
    architecture_only_recent = False
    if current_jobs:
        cur = current_jobs[0]
        is_architect_title = any(t in cur.get("title", "").lower() for t in cfg.ARCHITECTURE_ONLY_TITLES)
        long_enough = (cur.get("duration_months") or 0) >= 18
        no_code_language = _count_hits(
            cur.get("description", "").lower(),
            ["wrote code", "coded", "implemented", "built", "shipped code"],
        ) == 0
        architecture_only_recent = is_architect_title and long_enough and no_code_language

    seniority_score = band_fit - title_chaser_penalty
    if research_only:
        seniority_score *= 0.15
    if architecture_only_recent:
        seniority_score *= 0.55

    return {
        "yoe_band_fit": band_fit,
        "title_chaser_flag": title_chaser_flag,
        "research_only_flag": research_only,
        "architecture_only_recent_flag": architecture_only_recent,
        "experience_seniority_fit": max(0.0, min(1.0, seniority_score)),
    }


def domain_negative_list_features(candidate, text):
    """JD section "Things we explicitly do NOT want" — applied as a
    multiplicative penalty mass, not a hard zero, since the JD itself gives
    escape clauses (e.g. consulting-only is fine IF prior product co exp)."""
    history = candidate.get("career_history", [])
    companies = [h.get("company", "").lower() for h in history]
    industries = [h.get("industry", "").lower() for h in history]

    all_consulting = bool(history) and all(
        any(cf in c for cf in cfg.CONSULTING_FIRMS) for c in companies
    )
    has_product_experience = any(
        "it services" not in ind and not any(cf in c for cf in cfg.CONSULTING_FIRMS)
        for c, ind in zip(companies, industries)
    )
    consulting_only_penalty = 0.30 if (all_consulting and not has_product_experience) else 0.0

    cv_hits = _count_hits(text, cfg.CV_SPEECH_ROBOTICS_TERMS)
    nlp_hits = _count_hits(text, cfg.NLP_IR_TERMS)
    cv_only_penalty = 0.25 if (cv_hits >= 2 and nlp_hits == 0) else 0.0

    cur = next((h for h in history if h.get("is_current")), None)
    recent_llm_only_penalty = 0.0
    if cur and (cur.get("duration_months") or 0) < 12:
        cur_text = cur.get("description", "").lower()
        is_langchain_wrapper = ("langchain" in cur_text or "openai api" in cur_text) and \
            _count_hits(cur_text, cfg.RANKING_EVAL_TERMS + cfg.VECTOR_DB_TERMS) == 0
        has_pre_llm_history = len(history) > 1 and any(
            (h.get("duration_months") or 0) >= 24 for h in history[1:]
        )
        if is_langchain_wrapper and not has_pre_llm_history:
            recent_llm_only_penalty = 0.30

    total_penalty = min(0.7, consulting_only_penalty + cv_only_penalty + recent_llm_only_penalty)
    score = 1.0 - total_penalty
    return {
        "consulting_only_flag": all_consulting and not has_product_experience,
        "cv_speech_robotics_only_flag": cv_hits >= 2 and nlp_hits == 0,
        "recent_llm_wrapper_only_flag": recent_llm_only_penalty > 0,
        "domain_negative_list": max(0.0, score),
    }


def location_logistics_features(candidate):
    """City preference, relocation willingness (no visa sponsorship per JD),
    and notice period."""
    profile = candidate.get("profile", {})
    location = (profile.get("location") or "").lower()
    country = (profile.get("country") or "").lower()
    signals = candidate.get("redrob_signals", {})
    willing_to_relocate = signals.get("willing_to_relocate", False)

    if any(c in location for c in cfg.PREFERRED_CITIES):
        city_score = 1.0
    elif any(c in location for c in cfg.TIER1_WELCOME_CITIES):
        city_score = 0.8
    elif country == "india":
        city_score = 0.55 if willing_to_relocate else 0.35
    else:
        city_score = 0.25 if willing_to_relocate else 0.08

    notice_days = signals.get("notice_period_days", 90)
    if notice_days is None:
        notice_days = 90
    if notice_days <= 30:
        notice_score = 1.0
    elif notice_days <= 60:
        notice_score = 0.7
    elif notice_days <= 90:
        notice_score = 0.45
    else:
        notice_score = 0.25

    score = 0.65 * city_score + 0.35 * notice_score
    return {
        "city_score": city_score,
        "notice_score": notice_score,
        "location_logistics_fit": score,
    }


def behavioral_availability_features(candidate, today=None):
    """Redrob platform signals: is this candidate actually reachable/hireable
    right now, independent of how good they look on paper."""
    today = today or date.today()
    sig = candidate.get("redrob_signals", {})

    last_active = _to_date(sig.get("last_active_date"))
    if last_active:
        days_inactive = max(0, (today - last_active).days)
        recency_score = max(0.1, 1.0 - days_inactive / 180.0)
    else:
        recency_score = 0.3

    open_to_work = 1.0 if sig.get("open_to_work_flag") else 0.5
    response_rate = sig.get("recruiter_response_rate", 0.5) or 0.0
    interview_completion = sig.get("interview_completion_rate", 0.7) or 0.0

    oar = sig.get("offer_acceptance_rate", -1)
    if oar is None:
        oar = -1
    offer_accept_score = 0.75 if oar == -1 else max(0.0, min(1.0, oar))

    verified = (1.0 if sig.get("verified_email") else 0.7) * \
               (1.0 if sig.get("verified_phone") else 0.85)
    completeness = (sig.get("profile_completeness_score", 50) or 0) / 100.0

    score = (
        0.28 * recency_score +
        0.16 * open_to_work +
        0.20 * response_rate +
        0.14 * interview_completion +
        0.10 * offer_accept_score +
        0.07 * verified +
        0.05 * completeness
    )
    return {
        "recency_score": recency_score,
        "response_rate": response_rate,
        "interview_completion": interview_completion,
        "behavioral_availability": max(0.0, min(1.0, score)),
    }


def honeypot_flag(candidate):
    """Best-effort implausibility screen. We don't need to catch every
    honeypot (the relevance model independently down-weights mismatched
    profiles) — this is a backstop for profiles that are *numerically*
    impossible regardless of topical relevance."""
    skills = candidate.get("skills", [])
    history = candidate.get("career_history", [])
    yoe = candidate.get("profile", {}).get("years_of_experience", 0) or 0

    expert_zero = sum(
        1 for s in skills
        if s.get("proficiency") in ("expert", "advanced") and (s.get("duration_months") or 0) == 0
    )
    if expert_zero >= cfg.HONEYPOT_EXPERT_ZERO_DURATION_THRESHOLD:
        return True

    tot_months = sum(h.get("duration_months", 0) or 0 for h in history)
    if yoe > 0 and tot_months / 12.0 > yoe * cfg.HONEYPOT_CAREER_VS_YOE_RATIO:
        return True

    parsed = []
    for h in history:
        s = _to_date(h.get("start_date"))
        e = _to_date(h.get("end_date")) if h.get("end_date") else date.today()
        if s:
            parsed.append((s, e))
    parsed.sort()
    for i in range(len(parsed) - 1):
        if (parsed[i][1] - parsed[i + 1][0]).days > cfg.HONEYPOT_OVERLAP_DAYS:
            return True

    for e in candidate.get("education", []):
        if (e.get("end_year") or 9999) < (e.get("start_year") or 0):
            return True

    return False


def extract_all_features(candidate):
    text = _text_blob(candidate)
    feats = {"candidate_id": candidate["candidate_id"], "text_blob": text}
    feats.update(core_skill_features(candidate, text))
    feats.update(credibility_features(candidate, text))
    feats.update(experience_seniority_features(candidate, text))
    feats.update(domain_negative_list_features(candidate, text))
    feats.update(location_logistics_features(candidate))
    feats.update(behavioral_availability_features(candidate))
    feats["is_honeypot"] = honeypot_flag(candidate)

    feats["profile_headline"] = candidate.get("profile", {}).get("headline", "")
    feats["current_title"] = candidate.get("profile", {}).get("current_title", "")
    feats["current_company"] = candidate.get("profile", {}).get("current_company", "")
    feats["location"] = candidate.get("profile", {}).get("location", "")
    feats["country"] = candidate.get("profile", {}).get("country", "")
    feats["years_of_experience"] = candidate.get("profile", {}).get("years_of_experience", 0)
    feats["notice_period_days"] = candidate.get("redrob_signals", {}).get("notice_period_days")
    feats["open_to_work_flag"] = candidate.get("redrob_signals", {}).get("open_to_work_flag")
    feats["last_active_date"] = candidate.get("redrob_signals", {}).get("last_active_date")
    return feats
