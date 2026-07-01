"""
Structured representation of the Redrob JD (Senior AI Engineer — Founding Team).

This is deliberately NOT a bag of keywords. The JD itself says the "right answer"
requires reasoning about the gap between what it says and what it means. So this
config encodes: hard requirements, soft/nice-to-haves, explicit disqualifiers,
and the "ideal candidate" shape described in the JD's own "how to read between
the lines" section.

All of this is used downstream by src/scorer.py. Keyword lists here are used for
two purposes: (1) production-evidence detection in career_history descriptions
(NOT the skills list — that's the anti-keyword-stuffing design choice), and
(2) building the JD text block used for TF-IDF / embedding semantic similarity.
"""

JD_TITLE = "Senior AI Engineer — Founding Team"
JD_COMPANY = "Redrob AI"

# Full JD text (condensed to the substantive content) — used for semantic similarity
# against candidate summary + career_history descriptions.
JD_FULL_TEXT = """
Senior AI Engineer, founding AI engineering team at a Series A AI-native talent
intelligence platform. Own the intelligence layer: ranking, retrieval, and
matching systems deciding what recruiters see when searching candidates.
Audit existing BM25 and rule-based scoring. Ship a v2 ranking system using
embeddings, hybrid retrieval, and LLM-based re-ranking. Build evaluation
infrastructure: offline benchmarks, online A/B testing, recruiter feedback loops.
Own long-term architecture for candidate-JD matching at scale. Mentor engineers
as team grows from 4 to 12. Production experience with embeddings-based
retrieval systems such as sentence-transformers, OpenAI embeddings, BGE, E5,
deployed to real users, handling embedding drift, index refresh, retrieval
quality regression in production. Production experience with vector databases
or hybrid search infrastructure: Pinecone, Weaviate, Qdrant, Milvus, OpenSearch,
Elasticsearch, or FAISS. Strong Python and code quality. Hands-on experience
designing evaluation frameworks for ranking systems: NDCG, MRR, MAP,
offline-to-online correlation, A/B test interpretation. Nice to have: LLM
fine-tuning with LoRA QLoRA PEFT, learning-to-rank models, XGBoost-based or
neural ranking, HR-tech or recruiting tech or marketplace product experience,
distributed systems, large-scale inference optimization, open source
contributions. Scrappy product-engineering attitude, willing to ship a working
ranker in a week even if suboptimal. Shipped at least one end-to-end ranking,
search, or recommendation system to real users at meaningful scale.
""".strip()

# --- Hard requirements: production evidence keyword groups -----------------
# These are searched primarily in career_history[].description text, not the
# flat skills list, because the JD's stated trap is candidates who list AI
# keywords as skills without ever having shipped anything with them.
EMBEDDING_RETRIEVAL_TERMS = [
    "embedding", "embeddings", "sentence-transformers", "sentence transformers",
    "openai embeddings", "bge", "e5 embedding", "dense retrieval", "dense vector",
    "semantic search", "vector search", "retrieval-augmented", "rag pipeline",
    "nearest neighbor search", "ann search", "cosine similarity search",
]

VECTOR_DB_TERMS = [
    "pinecone", "weaviate", "qdrant", "milvus", "opensearch", "elasticsearch",
    "faiss", "vector database", "vector db", "hybrid search", "hybrid retrieval",
    "bm25",
]

EVAL_FRAMEWORK_TERMS = [
    "ndcg", "mrr", "map@", "mean average precision", "a/b test", "ab test",
    "offline evaluation", "online evaluation", "recall@", "precision@",
    "ranking evaluation", "evaluation framework", "click-through rate", "ctr",
]

PRODUCTION_SCALE_TERMS = [
    "production", "deployed", "real users", "at scale", "millions of",
    "shipped", "in production", "serving", "latency", "throughput",
]

# --- Nice-to-have terms ------------------------------------------------------
NICE_TO_HAVE_TERMS = [
    "lora", "qlora", "peft", "fine-tuning", "fine tuned", "learning to rank",
    "learning-to-rank", "xgboost", "recommendation system", "recommender system",
    "hr-tech", "hrtech", "recruiting tech", "ats", "marketplace", "distributed systems",
    "kubernetes", "kafka", "spark", "open source", "open-source", "github stars",
]

# --- Disqualifier signals ----------------------------------------------------
CONSULTING_FIRMS = [
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant", "capgemini",
]

FRAMEWORK_TUTORIAL_TERMS = [
    "langchain tutorial", "how i used", "hackathon demo", "toy project",
    "learning project", "followed a tutorial", "built a demo",
]

CV_SPEECH_ROBOTICS_TERMS = [
    "computer vision", "image classification", "object detection", "speech recognition",
    "robotics", "autonomous", "slam", "lidar", "audio processing",
]
NLP_IR_TERMS = [
    "nlp", "natural language", "text classification", "information retrieval",
    "search ranking", "language model", "llm", "transformer", "bert", "named entity",
]

RESEARCH_ONLY_TERMS = ["research scientist", "research lab", "phd research", "academic", "postdoc", "publication"]

RECENT_LANGCHAIN_ONLY_TERMS = ["langchain", "openai api wrapper", "prompt engineering only"]

ARCHITECT_NO_CODE_TERMS = ["tech lead", "engineering manager", "architect", "head of engineering", "director of engineering"]

# --- Location / logistics ----------------------------------------------------
PREFERRED_LOCATIONS = ["pune", "noida", "delhi", "gurgaon", "gurugram", "new delhi", "delhi ncr"]
ACCEPTABLE_INDIA_LOCATIONS = PREFERRED_LOCATIONS + ["hyderabad", "mumbai", "bengaluru", "bangalore"]

# --- Experience band ----------------------------------------------------------
EXPERIENCE_BAND = (5, 9)   # soft band, not a hard cutoff
IDEAL_APPLIED_ML_YEARS = (4, 5)  # years specifically in applied ML/AI, from the "ideal candidate" section

# Notice period thresholds (days)
NOTICE_GOOD = 30
NOTICE_BUYOUT_LIMIT = 30  # can buy out up to 30 days on top of stated notice
