#!/usr/bin/env python3
"""
rank.py — Redrob Hackathon: Intelligent Candidate Discovery & Ranking

Usage:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Design goals (see methodology_summary in submission_metadata.yaml for the
full writeup):
  1. No hosted LLM calls, no GPU, CPU-only, runs in well under 5 minutes
     on 100K candidates / 16GB RAM — this is a feature-engineered scorer
     over precomputed signals, not a per-candidate LLM judge.
  2. Explicitly anti-keyword-stuffing: skill claims are trust-weighted by
     corroboration (endorsements, duration, career-history mentions), not
     taken at face value.
  3. Explicitly encodes the JD's negative list (consulting-only, CV/speech/
     robotics-only, research-only, recent-LangChain-only) as penalties.
  4. Behavioral availability (Redrob signals) modifies but never replaces
     topical fit — a highly engaged irrelevant candidate should not outrank
     a relevant disengaged one, and vice versa.
  5. A best-effort honeypot screen removes numerically-impossible profiles
     before ranking.
  6. Every reasoning string is generated from the same computed fields used
     for scoring — nothing is invented, so nothing in the reasoning column
     can contradict the rank or hallucinate a skill the candidate doesn't
     have.
"""

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import config as cfg
import features as feat

JD_QUERY_TEXT = """
Senior AI Engineer founding team. Own the intelligence layer: ranking,
retrieval, and matching systems. Production experience with embeddings
based retrieval systems sentence-transformers OpenAI embeddings BGE E5
deployed to real users, handled embedding drift index refresh retrieval
quality regression in production. Production experience with vector
databases or hybrid search infrastructure Pinecone Weaviate Qdrant Milvus
OpenSearch Elasticsearch FAISS. Strong Python production code quality.
Hands on experience designing evaluation frameworks for ranking systems
NDCG MRR MAP offline to online correlation A/B test interpretation.
Ship a v2 ranking system embeddings hybrid retrieval LLM based re-ranking.
Set up evaluation infrastructure offline benchmarks online A/B testing.
Shipped at least one end to end ranking search or recommendation system to
real users at meaningful scale. Strong opinions about retrieval hybrid vs
dense, evaluation offline vs online, LLM integration fine tune vs prompt.
"""


def load_candidates(path):
    opener = open
    if str(path).endswith(".gz"):
        import gzip
        opener = gzip.open
    # Try a set of common encodings, then fall back to a safe binary-read
    encodings = ("utf-8", "utf-8-sig", "cp1252", "latin-1")
    for enc in encodings:
        try:
            with opener(path, "rt", encoding=enc, errors="strict") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        # skip malformed/non-JSON lines
                        continue
            return
        except UnicodeDecodeError:
            # try the next encoding
            continue
        except FileNotFoundError:
            # propagate missing file errors immediately
            raise

    # Final fallback: read raw bytes and decode lines with replacement
    mode = "rb"
    with opener(path, mode) as f:
        for raw in f:
            try:
                if isinstance(raw, bytes):
                    line = raw.decode("utf-8", errors="replace").strip()
                else:
                    # in case opener returned text lines unexpectedly
                    line = str(raw).strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        # skip malformed lines
                        continue
            except Exception:
                continue


def build_reasoning(row):
    bits = []
    title = row["current_title"] or "Unspecified title"
    company = row["current_company"] or "unspecified company"
    yoe = row["years_of_experience"]
    bits.append(f"{title} at {company}, {yoe:.1f}y experience")

    pillars = []
    if row["embeddings_evidence"] > 0.3:
        pillars.append("embeddings/retrieval")
    if row["vectordb_evidence"] > 0.3:
        pillars.append("vector DB/hybrid search")
    if row["ranking_eval_evidence"] > 0.3:
        pillars.append("ranking eval (NDCG/MAP/MRR)")
    if row["llm_evidence"] > 0.3:
        pillars.append("LLM/RAG")
    if pillars:
        bits.append("evidence of " + ", ".join(pillars))
    else:
        bits.append("limited direct evidence of the core retrieval/ranking stack")

    if row["skill_trust_ratio"] < 0.4 and row["core_skill_match"] > 0.2:
        bits.append("some listed skills lack duration/endorsement corroboration")

    if row["consulting_only_flag"]:
        bits.append("career entirely at IT-services firms, no product-co stint")
    if row["title_chaser_flag"]:
        bits.append("short-tenure title-escalation pattern")
    if row["research_only_flag"]:
        bits.append("research-only background, no production deployment")
    if row["cv_speech_robotics_only_flag"]:
        bits.append("CV/speech/robotics background without NLP/IR exposure")

    loc = row["location"] or "location unknown"
    bits.append(f"based in {loc}")
    if row["notice_period_days"] is not None:
        bits.append(f"{int(row['notice_period_days'])}d notice")

    if row["behavioral_availability"] < 0.4:
        bits.append("low recent platform engagement")
    elif row["behavioral_availability"] > 0.75:
        bits.append("strong recent platform engagement")

    text = "; ".join(bits) + "."
    return text[:400]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--xlsx", default=None, help="optional .xlsx mirror of the CSV")
    ap.add_argument("--top-n", type=int, default=100)
    args = ap.parse_args()

    t0 = time.time()
    print(f"[1/5] Loading & extracting features from {args.candidates} ...", file=sys.stderr)

    records = []
    n = 0
    for cand in load_candidates(args.candidates):
        records.append(feat.extract_all_features(cand))
        n += 1
        if n % 20000 == 0:
            print(f"  ...{n} candidates processed ({time.time()-t0:.1f}s)", file=sys.stderr)
    if n == 0:
        print(f"Error: no valid JSON candidates found in {args.candidates}.\n"
              "The file may be binary or not formatted as JSONL.", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(records)
    print(f"  loaded {len(df)} candidates in {time.time()-t0:.1f}s", file=sys.stderr)

    print("[2/5] Fitting TF-IDF JD-similarity model ...", file=sys.stderr)
    vectorizer = TfidfVectorizer(
        max_features=20000, stop_words="english", ngram_range=(1, 2), min_df=2,
    )
    corpus = df["text_blob"].tolist() + [JD_QUERY_TEXT.lower()]
    tfidf = vectorizer.fit_transform(corpus)
    jd_vec = tfidf[-1]
    cand_vecs = tfidf[:-1]
    sim = cosine_similarity(cand_vecs, jd_vec).ravel()
    # rescale to 0-1 using a robust max so the metric isn't squashed near 0
    sim_scaled = sim / (np.percentile(sim, 99.5) + 1e-9)
    df["jd_text_similarity"] = np.clip(sim_scaled, 0, 1)
    print(f"  done in {time.time()-t0:.1f}s", file=sys.stderr)

    print("[3/5] Combining signal blocks into final composite score ...", file=sys.stderr)
    # core_skill_match blends the rule-based pillar evidence with the
    # broader TF-IDF topical-similarity signal (catches genuine relevance
    # the keyword lists miss, e.g. synonyms/phrasing we didn't enumerate).
    df["core_skill_match_blended"] = 0.7 * df["core_skill_match"] + 0.3 * df["jd_text_similarity"]

    w = cfg.WEIGHTS
    df["final_score"] = (
        w["core_skill_match"] * df["core_skill_match_blended"] +
        w["experience_seniority_fit"] * df["experience_seniority_fit"] +
        w["domain_negative_list"] * df["domain_negative_list"] +
        w["location_logistics_fit"] * df["location_logistics_fit"] +
        w["behavioral_availability"] * df["behavioral_availability"] +
        w["credibility_modifier"] * df["credibility_modifier"]
    )
    # honeypots: force to the bottom regardless of any other signal
    df.loc[df["is_honeypot"], "final_score"] = -1.0

    n_honeypots = int(df["is_honeypot"].sum())
    print(f"  flagged {n_honeypots} implausible/honeypot profiles "
          f"({100*n_honeypots/len(df):.2f}% of pool)", file=sys.stderr)

    print("[4/5] Ranking top candidates ...", file=sys.stderr)
    # Round first, then sort on the rounded score — the submitted CSV's
    # score column is what the tie-break rule (score desc, candidate_id
    # asc) must be consistent with, not the unrounded internal score.
    df["score_rounded"] = df["final_score"].round(4)
    df_sorted = df.sort_values(
        ["score_rounded", "candidate_id"], ascending=[False, True]
    ).reset_index(drop=True)
    top = df_sorted.head(args.top_n).copy()
    top["rank"] = np.arange(1, len(top) + 1)
    top["score"] = top["score_rounded"]
    top["reasoning"] = top.apply(build_reasoning, axis=1)

    honeypots_in_top = int(top["is_honeypot"].sum())
    print(f"  honeypots in top {args.top_n}: {honeypots_in_top} "
          f"({100*honeypots_in_top/args.top_n:.1f}%)", file=sys.stderr)

    out_cols = ["candidate_id", "rank", "score", "reasoning"]
    submission = top[out_cols]
    submission.to_csv(args.out, index=False)
    print(f"[5/5] Wrote {args.out}", file=sys.stderr)

    if args.xlsx:
        # richer sheet for human review, alongside the strict-format CSV
        extra_cols = [
            "candidate_id", "rank", "score", "reasoning", "current_title",
            "current_company", "location", "country", "years_of_experience",
            "notice_period_days", "core_skill_match", "experience_seniority_fit",
            "domain_negative_list", "location_logistics_fit",
            "behavioral_availability", "credibility_modifier",
        ]
        top[extra_cols].to_excel(args.xlsx, index=False, sheet_name="top_100_ranked")
        print(f"[5/5] Wrote {args.xlsx}", file=sys.stderr)

    print(f"Total runtime: {time.time()-t0:.1f}s for {len(df)} candidates", file=sys.stderr)


if __name__ == "__main__":
    main()
