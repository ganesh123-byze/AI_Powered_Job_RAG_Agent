"""
skill_matcher.py  –  Robust Skill Matching & Domain Detection
==============================================================
Improvements:
  1. Expanded SKILL_ALIASES (300+ mappings including Indian tech stack)
  2. skills_match() uses 4 strategies: exact → alias → substring → token overlap
  3. detect_domain() scores all domains, picks highest with tie-breaking
  4. get_search_queries() returns role + skill combos, not just role strings
  5. should_filter_job() is now less aggressive (returns False by default)
"""

import re
from typing import List, Tuple, Optional

# ---------------------------------------------------------------------------
# Comprehensive alias map (normalises any spelling to canonical lowercase form)
# ---------------------------------------------------------------------------

SKILL_ALIASES: dict[str, str] = {
    # ── Python ecosystem ────────────────────────────────────────────────────
    "python": "python",
    "py": "python",
    "python3": "python",
    "python 3": "python",
    "scikit-learn": "scikit learn",
    "scikit learn": "scikit learn",
    "sklearn": "scikit learn",
    "pandas": "pandas",
    "numpy": "numpy",
    "np": "numpy",
    "scipy": "scipy",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "plotly": "plotly",
    "jupyter": "jupyter notebook",
    "jupyter notebook": "jupyter notebook",
    "ipython": "jupyter notebook",
    "pydantic": "pydantic",
    "fastapi": "fastapi",
    "flask": "flask",
    "django": "django",
    "celery": "celery",
    "sqlalchemy": "sqlalchemy",
    "alembic": "alembic",
    "poetry": "poetry",
    "pip": "pip",
    "conda": "conda",

    # ── ML / AI ─────────────────────────────────────────────────────────────
    "ml": "machine learning",
    "machine learning": "machine learning",
    "ai": "artificial intelligence",
    "artificial intelligence": "artificial intelligence",
    "deep learning": "deep learning",
    "dl": "deep learning",
    "neural network": "neural networks",
    "neural networks": "neural networks",
    "ann": "neural networks",
    "cnn": "convolutional neural networks",
    "convolutional neural networks": "convolutional neural networks",
    "rnn": "recurrent neural networks",
    "recurrent neural networks": "recurrent neural networks",
    "lstm": "lstm",
    "gru": "gru",
    "attention": "attention mechanism",
    "attention mechanism": "attention mechanism",
    "supervised learning": "supervised learning",
    "supervised": "supervised learning",
    "unsupervised learning": "unsupervised learning",
    "unsupervised": "unsupervised learning",
    "reinforcement learning": "reinforcement learning",
    "rl": "reinforcement learning",
    "regression": "regression",
    "classification": "classification",
    "clustering": "clustering",
    "dimensionality reduction": "dimensionality reduction",
    "pca": "pca",
    "feature engineering": "feature engineering",
    "feature selection": "feature selection",
    "ensemble methods": "ensemble methods",
    "ensemble": "ensemble methods",
    "random forest": "random forests",
    "random forests": "random forests",
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
    "lgbm": "lightgbm",
    "catboost": "catboost",
    "gradient boosting": "gradient boosting",
    "gbm": "gradient boosting",
    "svm": "svm",
    "support vector machine": "svm",
    "naive bayes": "naive bayes",
    "knn": "knn",
    "k-nearest neighbors": "knn",
    "decision tree": "decision trees",
    "decision trees": "decision trees",
    "cross-validation": "cross validation",
    "hyperparameter tuning": "hyperparameter tuning",
    "grid search": "grid search",
    "model evaluation": "model evaluation",

    # ── Deep Learning frameworks ─────────────────────────────────────────────
    "tensorflow": "tensorflow",
    "tf": "tensorflow",
    "pytorch": "pytorch",
    "torch": "pytorch",
    "keras": "keras",
    "jax": "jax",
    "huggingface": "hugging face",
    "hugging face": "hugging face",
    "transformers": "transformers",
    "diffusers": "diffusers",
    "timm": "timm",

    # ── Computer Vision ──────────────────────────────────────────────────────
    "computer vision": "computer vision",
    "cv": "computer vision",
    "opencv": "opencv",
    "cv2": "opencv",
    "image processing": "image processing",
    "image classification": "image classification",
    "object detection": "object detection",
    "yolo": "yolo",
    "yolov5": "yolo",
    "yolov8": "yolo",
    "semantic segmentation": "semantic segmentation",
    "instance segmentation": "instance segmentation",
    "pillow": "pillow",
    "pil": "pillow",
    "torchvision": "torchvision",

    # ── NLP ─────────────────────────────────────────────────────────────────
    "nlp": "natural language processing",
    "natural language processing": "natural language processing",
    "tfidf": "tfidf",
    "tf-idf": "tfidf",
    "tf idf": "tfidf",
    "word2vec": "word2vec",
    "glove": "glove",
    "fasttext": "fasttext",
    "bert": "bert",
    "gpt": "gpt",
    "gpt-3": "gpt",
    "gpt-4": "gpt",
    "t5": "t5",
    "roberta": "roberta",
    "xlnet": "xlnet",
    "seq2seq": "seq2seq",
    "ner": "named entity recognition",
    "named entity recognition": "named entity recognition",
    "sentiment analysis": "sentiment analysis",
    "text classification": "text classification",
    "tokenization": "tokenization",
    "lemmatization": "lemmatization",
    "stemming": "stemming",
    "spacy": "spacy",
    "nltk": "nltk",
    "gensim": "gensim",
    "stanza": "stanza",

    # ── GenAI / LLM ──────────────────────────────────────────────────────────
    "llm": "large language models",
    "llms": "large language models",
    "large language models": "large language models",
    "large language model": "large language models",
    "genai": "generative ai",
    "generative ai": "generative ai",
    "generative": "generative ai",
    "langchain": "langchain",
    "llamaindex": "llamaindex",
    "llama index": "llamaindex",
    "rag": "retrieval augmented generation",
    "retrieval augmented generation": "retrieval augmented generation",
    "prompt engineering": "prompt engineering",
    "prompt eng": "prompt engineering",
    "openai": "openai",
    "openai api": "openai api",
    "chatgpt": "chatgpt",
    "ollama": "ollama",
    "anthropic": "anthropic",
    "claude": "claude",
    "gemini": "gemini",
    "mistral": "mistral",
    "llama": "llama",
    "crewai": "crewai",
    "autogen": "autogen",
    "agentic": "agentic ai",
    "agentic ai": "agentic ai",
    "vector database": "vector database",
    "chromadb": "chromadb",
    "pinecone": "pinecone",
    "weaviate": "weaviate",
    "qdrant": "qdrant",
    "faiss": "faiss",
    "milvus": "milvus",

    # ── Data Science / Analytics ─────────────────────────────────────────────
    "data science": "data science",
    "data analysis": "data analysis",
    "eda": "exploratory data analysis",
    "exploratory data analysis": "exploratory data analysis",
    "statistics": "statistics",
    "statistical": "statistics",
    "probability": "probability",
    "hypothesis testing": "hypothesis testing",
    "a/b testing": "ab testing",
    "ab testing": "ab testing",
    "data visualization": "data visualization",
    "data viz": "data visualization",
    "power bi": "power bi",
    "powerbi": "power bi",
    "tableau": "tableau",
    "looker": "looker",
    "excel": "excel",
    "google sheets": "google sheets",

    # ── SQL / Databases ──────────────────────────────────────────────────────
    "sql": "sql",
    "mysql": "mysql",
    "postgresql": "postgresql",
    "postgres": "postgresql",
    "sqlite": "sqlite",
    "oracle": "oracle",
    "mssql": "mssql",
    "sql server": "mssql",
    "nosql": "nosql",
    "mongodb": "mongodb",
    "mongo": "mongodb",
    "cassandra": "cassandra",
    "redis": "redis",
    "elasticsearch": "elasticsearch",
    "neo4j": "neo4j",
    "dynamodb": "dynamodb",
    "firebase": "firebase",
    "supabase": "supabase",
    "snowflake": "snowflake",
    "bigquery": "bigquery",
    "redshift": "redshift",

    # ── Data Engineering ─────────────────────────────────────────────────────
    "apache spark": "spark",
    "spark": "spark",
    "pyspark": "pyspark",
    "hadoop": "hadoop",
    "hive": "hive",
    "kafka": "kafka",
    "apache kafka": "kafka",
    "airflow": "airflow",
    "apache airflow": "airflow",
    "flink": "flink",
    "dbt": "dbt",
    "etl": "etl",
    "data pipeline": "data pipeline",
    "databricks": "databricks",

    # ── Cloud ────────────────────────────────────────────────────────────────
    "aws": "aws",
    "amazon web services": "aws",
    "gcp": "google cloud",
    "google cloud": "google cloud",
    "google cloud platform": "google cloud",
    "azure": "azure",
    "microsoft azure": "azure",
    "sagemaker": "sagemaker",
    "vertex ai": "vertex ai",
    "lambda": "aws lambda",
    "aws lambda": "aws lambda",
    "s3": "aws s3",
    "ec2": "aws ec2",
    "cloud": "cloud platforms",

    # ── DevOps / MLOps ───────────────────────────────────────────────────────
    "docker": "docker",
    "kubernetes": "kubernetes",
    "k8s": "kubernetes",
    "helm": "helm",
    "terraform": "terraform",
    "ansible": "ansible",
    "jenkins": "jenkins",
    "github actions": "github actions",
    "gitlab ci": "gitlab ci",
    "circleci": "circleci",
    "ci/cd": "ci/cd",
    "ci cd": "ci/cd",
    "mlops": "mlops",
    "mlflow": "mlflow",
    "wandb": "wandb",
    "dvc": "dvc",
    "bentoml": "bentoml",
    "triton": "triton",
    "prometheus": "prometheus",
    "grafana": "grafana",
    "datadog": "datadog",
    "nginx": "nginx",
    "linux": "linux",
    "unix": "unix",
    "bash": "bash",
    "shell": "bash",

    # ── Web / Backend ────────────────────────────────────────────────────────
    "nodejs": "nodejs",
    "node.js": "nodejs",
    "node js": "nodejs",
    "express": "express",
    "react": "react",
    "reactjs": "react",
    "react.js": "react",
    "angular": "angular",
    "vue": "vue",
    "next.js": "next.js",
    "nextjs": "next.js",
    "html": "html",
    "css": "css",
    "javascript": "javascript",
    "js": "javascript",
    "typescript": "typescript",
    "ts": "typescript",
    "rest api": "rest apis",
    "rest apis": "rest apis",
    "restful": "rest apis",
    "graphql": "graphql",
    "grpc": "grpc",
    "microservices": "microservices",
    "websocket": "websocket",
    "spring": "spring",
    "spring boot": "spring boot",
    "java": "java",

    # ── Tools ────────────────────────────────────────────────────────────────
    "git": "git",
    "github": "github",
    "gitlab": "gitlab",
    "vs code": "vs code",
    "vscode": "vs code",
    "jira": "jira",
    "confluence": "confluence",
    "agile": "agile",
    "scrum": "scrum",
    "postman": "postman",
    "streamlit": "streamlit",
    "gradio": "gradio",
    "google colab": "google colab",
    "colab": "google colab",
}

# ---------------------------------------------------------------------------
# Domain keyword config  (keywords → canonical domain)
# ---------------------------------------------------------------------------

DOMAIN_KEYWORDS: dict[str, dict] = {
    "machine_learning": {
        "keywords": [
            "machine learning", "ml engineer", "deep learning",
            "regression", "classification", "supervised", "unsupervised",
            "neural network", "scikit learn", "sklearn",
            "tensorflow", "pytorch", "keras", "xgboost",
            "random forest", "gradient boosting",
        ],
        "search_queries": [
            "Machine Learning Engineer",
            "ML Engineer",
            "Deep Learning Engineer",
            "AI Engineer",
            "Data Scientist",
        ],
        "weight": 1.5,
        "exclude_keywords": [],
    },
    "data_science": {
        "keywords": [
            "data science", "data analysis", "analytics", "eda",
            "statistics", "probability", "pandas", "numpy",
            "matplotlib", "seaborn", "power bi", "tableau",
            "data visualization", "hypothesis testing",
        ],
        "search_queries": [
            "Data Scientist",
            "Data Analyst",
            "Analytics Engineer",
            "Business Analyst",
        ],
        "weight": 1.0,
        "exclude_keywords": [],
    },
    "nlp": {
        "keywords": [
            "nlp", "natural language processing", "text classification",
            "sentiment analysis", "ner", "tfidf", "bert", "gpt",
            "transformers", "hugging face", "seq2seq", "spacy", "nltk",
        ],
        "search_queries": [
            "NLP Engineer",
            "NLP Scientist",
            "Language Model Engineer",
            "Text Analytics Engineer",
        ],
        "weight": 1.3,
        "exclude_keywords": [],
    },
    "genai_llm": {
        "keywords": [
            "generative ai", "large language models", "llm",
            "gpt", "prompt engineering", "langchain", "llamaindex",
            "retrieval augmented generation", "rag", "chatgpt",
            "agentic ai", "vector database", "chromadb",
        ],
        "search_queries": [
            "Generative AI Engineer",
            "LLM Engineer",
            "AI Engineer",
            "Prompt Engineer",
            "RAG Engineer",
        ],
        "weight": 1.4,
        "exclude_keywords": [],
    },
    "computer_vision": {
        "keywords": [
            "computer vision", "image processing", "opencv",
            "object detection", "yolo", "semantic segmentation",
            "image classification", "torchvision", "medical imaging",
        ],
        "search_queries": [
            "Computer Vision Engineer",
            "Vision AI Engineer",
            "Image Processing Engineer",
        ],
        "weight": 1.3,
        "exclude_keywords": [],
    },
    "data_engineering": {
        "keywords": [
            "data engineering", "data pipeline", "etl", "spark",
            "pyspark", "kafka", "airflow", "dbt", "hadoop",
            "databricks", "snowflake", "bigquery", "streaming",
        ],
        "search_queries": [
            "Data Engineer",
            "ETL Engineer",
            "Pipeline Engineer",
            "Big Data Engineer",
        ],
        "weight": 1.0,
        "exclude_keywords": [],
    },
    "backend": {
        "keywords": [
            "backend", "flask", "fastapi", "django", "rest apis",
            "microservices", "nodejs", "spring boot", "postgresql",
            "mongodb", "docker", "kubernetes",
        ],
        "search_queries": [
            "Backend Developer",
            "Python Developer",
            "Backend Engineer",
            "Full Stack Developer",
        ],
        "weight": 0.8,
        "exclude_keywords": [],
    },
    "mlops": {
        "keywords": [
            "mlops", "mlflow", "wandb", "dvc", "model deployment",
            "model serving", "triton", "bentoml", "kubeflow",
            "sagemaker", "vertex ai", "ci/cd", "monitoring",
        ],
        "search_queries": [
            "MLOps Engineer",
            "ML Platform Engineer",
            "AI Infrastructure Engineer",
        ],
        "weight": 1.2,
        "exclude_keywords": [],
    },
}


# ---------------------------------------------------------------------------
# Core normalisation
# ---------------------------------------------------------------------------

def normalize_skill(skill: str) -> str:
    """Normalise skill string to canonical lowercase form."""
    s = skill.lower().strip()
    # Remove common punctuation
    s = re.sub(r"[.\-_/]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return SKILL_ALIASES.get(s, s)


def get_skill_tokens(skill: str) -> set[str]:
    """Return meaningful word tokens from a skill."""
    STOP = {"and", "or", "for", "the", "of", "in", "a", "an",
            "with", "using", "via", "to", "from", "by"}
    normalised = normalize_skill(skill)
    return {t for t in normalised.split() if t not in STOP and len(t) >= 2}


# ---------------------------------------------------------------------------
# Four-strategy skill matching
# ---------------------------------------------------------------------------

def skills_match(skill1: str, skill2: str) -> bool:
    """
    True if the two skills refer to the same technology.

    Strategies (in order):
      1. Exact match after normalisation
      2. One is a substring of the other (min 4 chars)
      3. Token subset (all tokens of shorter ⊆ tokens of longer)
      4. Token overlap ≥ 70%
    """
    n1 = normalize_skill(skill1)
    n2 = normalize_skill(skill2)

    # 1. Exact
    if n1 == n2:
        return True

    # 2. Substring (avoids false positives on 2-char strings)
    if len(n1) >= 4 and len(n2) >= 4:
        if n1 in n2 or n2 in n1:
            return True

    # 3 & 4. Token-based
    t1 = get_skill_tokens(skill1)
    t2 = get_skill_tokens(skill2)
    if not t1 or not t2:
        return False
    if t1.issubset(t2) or t2.issubset(t1):
        return True
    overlap = len(t1 & t2) / max(len(t1), len(t2))
    return overlap >= 0.7


# ---------------------------------------------------------------------------
# Batch matching
# ---------------------------------------------------------------------------

def find_matched_skills(
    resume_skills: List[str],
    jd_skills: List[str],
) -> Tuple[List[str], List[str]]:
    """
    Returns:
      matched_resume_skills  – resume skills that appear in JD
      missing_jd_skills      – JD skills not covered by resume
    """
    matched: list[str] = []
    for rs in resume_skills:
        if any(skills_match(rs, js) for js in jd_skills):
            matched.append(rs)

    missing: list[str] = []
    for js in jd_skills:
        if not any(skills_match(rs, js) for rs in resume_skills):
            missing.append(js)

    return matched, missing


# ---------------------------------------------------------------------------
# Domain detection
# ---------------------------------------------------------------------------

def detect_domain(resume_skills: List[str], preferred_roles: List[str]) -> str:
    """
    Score all domains against resume skills + preferred roles.
    Returns the best-matching domain key.
    """
    all_text = (
        " ".join(normalize_skill(s) for s in resume_skills)
        + " "
        + " ".join(r.lower() for r in preferred_roles)
    )

    scores: dict[str, float] = {}
    for domain, info in DOMAIN_KEYWORDS.items():
        score = 0.0
        for kw in info["keywords"]:
            if kw in all_text:
                score += info.get("weight", 1.0)
        scores[domain] = score

    if not scores or max(scores.values()) == 0:
        return "machine_learning"

    return max(scores, key=scores.get)


def get_search_queries(domain: str) -> List[str]:
    """Return search query strings for a domain."""
    return DOMAIN_KEYWORDS.get(domain, DOMAIN_KEYWORDS["machine_learning"])["search_queries"]


def should_filter_job(domain: str, job_title: str, job_desc: str) -> bool:
    """
    Returns True ONLY when we're very confident the job is in a completely
    different domain. Conservative – prefers keeping jobs over filtering.
    """
    # Only filter if no domain keywords appear anywhere in title+description
    if domain not in DOMAIN_KEYWORDS:
        return False
    job_text = (job_title + " " + job_desc[:300]).lower()
    domain_kws = DOMAIN_KEYWORDS[domain]["keywords"]
    if not any(kw in job_text for kw in domain_kws[:5]):
        # Before filtering, check if ANY domain keyword appears
        # (user might have a broad skill set)
        for d_info in DOMAIN_KEYWORDS.values():
            if any(kw in job_text for kw in d_info["keywords"][:3]):
                return False   # some domain matches → keep
        return True   # nothing matches → safe to filter
    return False


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        ("Machine Learning", "machine learning engineer"),
        ("Scikit-learn", "sklearn"),
        ("TF-IDF", "tfidf"),
        ("Python", "python 3"),
        ("LLM", "large language models"),
        ("RAG", "retrieval augmented generation"),
        ("PyTorch", "torch"),
        ("k8s", "kubernetes"),
        ("postgresql", "postgres"),
    ]
    print("Skill match tests:")
    for s1, s2 in tests:
        r = skills_match(s1, s2)
        print(f"  {'✓' if r else '✗'}  {s1!r:30} vs {s2!r}")

    print("\nDomain detection tests:")
    for skills_list in [
        ["Python", "PyTorch", "TensorFlow", "Deep Learning", "NLP"],
        ["Python", "Flask", "REST API", "PostgreSQL", "Docker"],
        ["LangChain", "RAG", "ChromaDB", "LLM", "Prompt Engineering"],
    ]:
        d = detect_domain(skills_list, [])
        print(f"  {skills_list[:3]} → {d}")