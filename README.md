# Redrob Candidate Ranker — Intelligent Candidate Discovery & Ranking Challenge

A feature-engineered, fully-explainable candidate ranker for Redrob's
"Senior AI Engineer — Founding Team" job description. No hosted LLM calls,
no GPU, CPU-only, ~70 seconds end-to-end on the full 100,000-candidate pool.

## Why feature engineering, not an LLM judge

The compute budget (5 min, CPU-only, no network, 16GB RAM) rules out
per-candidate LLM scoring at this scale — see Section 3 of
`submission_spec.docx`. Instead, every signal block below is a closed-form
function of the candidate JSON that a reviewer can recompute by hand. That
also makes the `reasoning` column honest by construction: it's generated
from the exact fields used to score, so it can't hallucinate a skill the
candidate doesn't have or contradict the rank.

## How scoring works

```
final_score =
    0.34 * core_skill_match          (embeddings / vector-DB / ranking-eval / LLM evidence,
                                       trust-weighted against keyword stuffing, blended
                                       30% with TF-IDF similarity to the JD text)
  + 0.16 * experience_seniority_fit  (5-9y band fit + title-chaser / research-only /
                                       architecture-only detectors)
  + 0.16 * domain_negative_list      (JD's explicit "do not want" list, with its own
                                       escape clauses encoded)
  + 0.12 * location_logistics_fit    (city tier, relocation, notice period — no visa
                                       sponsorship per JD)
  + 0.12 * behavioral_availability   (Redrob signals: recency, response rate, interview
                                       completion, offer acceptance, verification)
  + 0.10 * credibility_modifier      (production language vs. tutorial language)
```

Honeypots (numerically impossible profiles — e.g. several "expert" skills
with zero months used, or total career-history duration exceeding stated
years of experience) are screened out before ranking and forced to the
bottom. On the released pool this flags 45 candidates; 0 of them land in
the final top 100.

Full rationale for every weight and keyword list lives in `config.py` —
each list is commented with the JD clause it implements.

## Repo layout

```
config.py     — JD-derived keyword taxonomies and scoring weights
features.py   — pure feature-extraction functions, one per signal block
rank.py       — main pipeline: load → extract → TF-IDF → score → rank → write CSV/XLSX
requirements.txt
submission_metadata.yaml
data/candidates.jsonl   — (not committed; place the released file here)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduce the submission

```bash
python rank.py --candidates ./data/candidates.jsonl --out ./submission.csv --xlsx ./submission_review.xlsx
```

- `submission.csv` — the strict 4-column format required by
  `submission_spec.docx` (`candidate_id,rank,score,reasoning`), validated
  against the organizers' `validate_submission.py` with zero errors.
- `submission_review.xlsx` — the same top-100 with every intermediate
  signal column exposed, for human review / Stage-5 prep.

Measured runtime on the full 100K pool: **~67 seconds**, single process,
CPU-only, no network calls — see `submission_metadata.yaml`.

## Validating the output

```bash
python validate_submission.py submission.csv   # organizer-provided validator
```

## Design notes for Stage 5 (defend your work)

- **Anti-keyword-stuffing**: a skill tagged "expert" with 0 endorsements,
  0 duration, and never mentioned in any job description gets near-zero
  trust weight regardless of how many times the term appears. See
  `skill_trust_ratio` in `features.py::core_skill_features`.
- **Negative-list clauses encoded with their escape hatches**: e.g.
  consulting-only career is only penalized if the candidate has *no*
  product-company stint anywhere in their history, matching the JD's own
  "if you're currently at one of these but have prior product-company
  experience, that's fine."
- **Behavioral signals modify, don't replace, topical fit**: a highly
  engaged irrelevant candidate cannot outrank a relevant disengaged one —
  `behavioral_availability` is capped at 12% of the composite.
- **Known limitations**: the keyword taxonomies in `config.py` are
  English-term lists; a candidate describing identical work with different
  terminology relies on the 30%-weighted TF-IDF blend to be caught, which
  is a softer signal. The honeypot screen is best-effort and intentionally
  conservative (favors false negatives over flagging legitimate candidates)
  — it does not aim to catch all ~80 honeypots, only to keep the top-100
  honeypot rate near zero, which it does on the released pool.
