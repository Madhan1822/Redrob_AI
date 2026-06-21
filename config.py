"""
config.py
Job-description-derived signal vocabulary and scoring weights for the
Redrob "Senior AI Engineer — Founding Team" ranking challenge.

Every list here is traceable to a specific clause in job_description.docx.
Keeping this in one file makes the scoring auditable and easy to defend
in the Stage-5 interview: every weight has a one-line justification.
"""

# ---------------------------------------------------------------------------
# Core technical evidence the JD says is mandatory ("things you absolutely
# need"). Matched against career-history text, skills, and headline/summary.
# ---------------------------------------------------------------------------
EMBEDDING_TERMS = [
    "sentence-transformers", "sentence transformers", "openai embeddings",
    "bge", "e5 embedding", "text embedding", "embedding model", "embeddings",
    "dense retrieval", "bi-encoder", "cross-encoder",
]

VECTOR_DB_TERMS = [
    "pinecone", "weaviate", "qdrant", "milvus", "opensearch",
    "elasticsearch", "faiss", "vector database", "vector db", "vector store",
    "hybrid search", "ann index", "approximate nearest neighbor",
]

RANKING_EVAL_TERMS = [
    "ndcg", "mrr", "map@", "mean average precision", "learning to rank",
    "ltr", "precision@", "recall@", "a/b test", "ab test", "offline eval",
    "online eval", "click-through", "ctr ", "ranking model", "re-rank",
    "reranking", "search relevance",
]

LLM_TERMS = [
    "llm", "large language model", "gpt-", "fine-tun", "lora", "qlora",
    "peft", "prompt engineering", "rag", "retrieval augmented",
    "retrieval-augmented",
]

PYTHON_TERMS = ["python"]

# ---------------------------------------------------------------------------
# Credibility signals: distinguishes "shipped it" from "read about it".
# This is the anti-keyword-stuffing layer.
# ---------------------------------------------------------------------------
PRODUCTION_TERMS = [
    "production", "deployed", "shipped", "real users", "at scale",
    "live traffic", "rolled out", "in prod", "serving traffic",
    "millions of", "thousands of users", "p99", "latency budget",
    "on-call", "incident",
]

TUTORIAL_ONLY_TERMS = [
    "tutorial", "side project", "toy project", "personal project",
    "kaggle competition", "course project", "bootcamp project",
    "followed a tutorial", "udemy", "coursera certificate",
]

# ---------------------------------------------------------------------------
# Domain-fit / explicit negative-list signals from the JD.
# ---------------------------------------------------------------------------
CV_SPEECH_ROBOTICS_TERMS = [
    "computer vision", "image classification", "object detection",
    "image segmentation", "speech recognition", "robotics", "autonomous",
    "lidar", "slam", "asr ", "text-to-speech", "tts ", "self-driving",
]

NLP_IR_TERMS = [
    "nlp", "natural language processing", "information retrieval",
    "search engine", "retrieval", "ranking", "text classification",
    "named entity", "question answering", "semantic search",
    "document retrieval", "query understanding",
]

CONSULTING_FIRMS = [
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini",
]

RESEARCH_ONLY_TERMS = [
    "research scientist", "research engineer", "postdoc", "phd candidate",
    "academic researcher", "university lab", "research lab",
]

ARCHITECTURE_ONLY_TITLES = [
    "architect", "tech lead", "technical lead", "engineering manager",
    "director of engineering", "head of engineering", "vp engineering",
]

OPEN_VALIDATION_TERMS = [
    "open source", "open-source", "github.com", "published a paper",
    "conference talk", "blog post", "wrote about", "spoke at",
    "open sourced",
]

# ---------------------------------------------------------------------------
# Geography (JD: "Pune/Noida preferred...flexible...Hyderabad, Pune, Mumbai,
# Delhi NCR welcome...outside India: case-by-case, no visa sponsorship")
# ---------------------------------------------------------------------------
PREFERRED_CITIES = ["pune", "noida"]
TIER1_WELCOME_CITIES = [
    "hyderabad", "mumbai", "delhi", "gurugram", "gurgaon", "bengaluru",
    "bangalore", "ncr",
]

# ---------------------------------------------------------------------------
# Scoring weights. These sum (within each block) to a normalized 0-1 score;
# blocks are then combined per the weighting below. Documented in the PPT
# and methodology summary so they're not "magic numbers".
# ---------------------------------------------------------------------------
WEIGHTS = {
    "core_skill_match": 0.34,      # embeddings/vectorDB/ranking/LLM/python evidence, trust-adjusted
    "experience_seniority_fit": 0.16,  # years-of-experience band + career trajectory shape
    "domain_negative_list": 0.16,  # JD's explicit "do not want" list, applied as penalty mass
    "location_logistics_fit": 0.12,  # city, relocation, notice period
    "behavioral_availability": 0.12,  # Redrob signals: is this person actually reachable/hireable
    "credibility_modifier": 0.10,  # production-vs-tutorial language, skill corroboration
}

IDEAL_YOE_CENTER = 7.0   # midpoint of the 5-9y band
IDEAL_YOE_WIDTH = 4.5    # soft Gaussian width; JD explicitly says band is flexible

HONEYPOT_EXPERT_ZERO_DURATION_THRESHOLD = 3   # >=3 expert/advanced skills with 0 months used
HONEYPOT_CAREER_VS_YOE_RATIO = 1.3            # career_history total months / (yoe*12)
HONEYPOT_OVERLAP_DAYS = 45                     # concurrent non-current roles overlapping > this
