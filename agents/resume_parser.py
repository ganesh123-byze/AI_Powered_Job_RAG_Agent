"""
resume_parser.py  –  Robust Resume Parser
==========================================
Fixes vs old version:
  1. Multi-strategy text extraction (pypdf + pdfminer fallback)
  2. Two-stage LLM parsing: broad extraction → skill enrichment pass
  3. Heuristic skill extractor as hard fallback (regex + keyword bank)
  4. Normalises every skills list before returning
  5. Prefers high token budget (1500) to avoid truncated JSON
  6. Validates and repairs JSON before constructing ResumeProfile
"""

import json
import re
import httpx
import logging
from typing import Optional

from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_LLM_MODEL,
    OPENROUTER_BASE_URL,
    OPENROUTER_HEADERS,
    OPENROUTER_TIMEOUT,
)
from models.resume import ResumeProfile
from utils.pdf_extractor import extract_text_from_pdf

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Comprehensive technical skill bank for heuristic fallback
# ---------------------------------------------------------------------------
SKILL_BANK = [
    # Languages
    "python", "java", "javascript", "typescript", "c", "c++", "c#", "go",
    "rust", "ruby", "php", "swift", "kotlin", "scala", "r", "matlab",
    "perl", "bash", "shell", "powershell", "vba", "dart", "lua",
    # ML / AI
    "machine learning", "deep learning", "neural networks", "nlp",
    "computer vision", "reinforcement learning", "transformers",
    "bert", "gpt", "llm", "rag", "langchain", "llamaindex", "huggingface",
    "scikit-learn", "sklearn", "tensorflow", "keras", "pytorch", "jax",
    "xgboost", "lightgbm", "catboost", "opencv", "yolo",
    "stable diffusion", "diffusion models", "vae", "gan",
    # Data
    "pandas", "numpy", "scipy", "matplotlib", "seaborn", "plotly",
    "tableau", "power bi", "looker", "dbt", "airflow", "spark",
    "pyspark", "hadoop", "hive", "kafka", "flink", "databricks",
    "snowflake", "bigquery", "redshift", "dbt", "etl",
    # Databases
    "sql", "mysql", "postgresql", "sqlite", "oracle", "mssql",
    "mongodb", "cassandra", "redis", "elasticsearch", "neo4j",
    "dynamodb", "firebase", "supabase", "cockroachdb",
    # Cloud / DevOps
    "aws", "azure", "gcp", "google cloud", "docker", "kubernetes",
    "terraform", "ansible", "jenkins", "github actions", "gitlab ci",
    "circleci", "helm", "prometheus", "grafana", "datadog",
    "linux", "unix", "nginx", "apache",
    # Web / Backend
    "fastapi", "flask", "django", "express", "node.js", "nodejs",
    "spring boot", "spring", "rails", "laravel", "asp.net",
    "graphql", "rest", "grpc", "microservices", "websocket",
    # Frontend
    "react", "angular", "vue", "svelte", "next.js", "nuxt",
    "html", "css", "sass", "tailwind", "bootstrap", "webpack",
    # Vector / RAG
    "chromadb", "pinecone", "weaviate", "qdrant", "faiss", "milvus",
    "vector database", "embeddings", "semantic search",
    # MLOps
    "mlflow", "wandb", "dvc", "bentoml", "triton", "torchserve",
    "sagemaker", "vertex ai", "kubeflow",
    # Other
    "git", "github", "gitlab", "jira", "confluence", "agile", "scrum",
    "linux", "regex", "api", "oauth", "jwt", "celery", "rabbitmq",
]

# ---------------------------------------------------------------------------
# Primary LLM prompt  –  increased token budget and strict JSON schema
# ---------------------------------------------------------------------------
EXTRACTION_PROMPT = """\
You are an expert resume parser. Extract ALL information from the resume below.

RULES:
- Return ONLY valid JSON – no markdown fences, no explanation
- Escape special characters properly inside strings
- skills: include EVERY technical tool, language, framework, library mentioned
  anywhere in the resume (projects, experience, skills section, certifications)
- preferred_roles: infer 2-4 roles the candidate is targeting based on their
  skills and experience (e.g. "Machine Learning Engineer", "Data Analyst")
- experience_years: total professional experience in years; 0.0 if fresher/student

JSON schema (return exactly these keys):
{{
  "full_name": string,
  "email": string | null,
  "phone": string | null,
  "skills": [list of strings – EVERY technology mentioned],
  "experience_years": float,
  "current_role": string | null,
  "education": [
    {{"degree": str, "field_of_study": str, "institution": str, "year_of_graduation": int | null}}
  ],
  "projects": [
    {{"name": str, "description": str, "technologies": [list of strings]}}
  ],
  "preferred_roles": [list of strings],
  "summary": string | null
}}

RESUME TEXT:
{resume_text}

Return ONLY the JSON object.
"""

# Second-pass prompt to enrich skills from a previously parsed result
SKILL_ENRICH_PROMPT = """\
A resume parser extracted these skills from a resume: {existing_skills}

Here is the full resume text:
{resume_text}

Task: Find ADDITIONAL technical skills, tools, frameworks, libraries, or
technologies that are mentioned ANYWHERE in the resume (project descriptions,
experience bullets, certifications, coursework) but are NOT in the list above.

Return ONLY a JSON array of the NEW skill strings. Example:
["PyTorch", "FastAPI", "PostgreSQL"]

Return [] if no additional skills found.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _call_openrouter(prompt: str, max_tokens: int = 1500) -> str:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        **OPENROUTER_HEADERS,
    }
    payload = {
        "model": OPENROUTER_LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,           # low temp → deterministic JSON
        "max_tokens": max_tokens,
    }
    with httpx.Client(timeout=OPENROUTER_TIMEOUT) as client:
        resp = client.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            json=payload,
            headers=headers,
        )
    data = resp.json()
    if resp.status_code != 200:
        raise RuntimeError(f"OpenRouter {resp.status_code}: {data}")
    return data["choices"][0]["message"]["content"]


def _strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _find_json_object(text: str) -> Optional[str]:
    """Return the first balanced JSON object substring."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if esc:
            esc = False
            continue
        if ch == "\\" and in_str:
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _repair_json(text: str) -> str:
    """Close unclosed strings and braces (best-effort for truncated LLM output)."""
    start = text.find("{")
    if start == -1:
        return text
    depth = 0
    in_str = False
    esc = False
    for ch in text[start:]:
        if esc:
            esc = False
            continue
        if ch == "\\" and in_str:
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
    suffix = ""
    if in_str:
        suffix += '"'
    if depth > 0:
        suffix += "}" * depth
    return text + suffix


def _safe_parse_json(raw: str) -> dict:
    """Try multiple strategies to parse JSON from an LLM response."""
    text = _strip_fences(raw)

    # 1. Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Extract first balanced object
    candidate = _find_json_object(text)
    if candidate:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # 3. Repair and parse
    repaired = _repair_json(text)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    # 4. Give up
    raise ValueError(f"Cannot parse JSON from LLM response:\n{raw[:400]}")


# ---------------------------------------------------------------------------
# Heuristic skill extraction (used as hard fallback and enrichment)
# ---------------------------------------------------------------------------

def extract_skills_heuristically(text: str) -> list[str]:
    """
    Extract technical skills from raw text using a keyword bank + patterns.
    Returns deduplicated, title-cased skill list.
    """
    text_lower = text.lower()
    found: set[str] = set()

    # 1. Match against known skill bank
    for skill in SKILL_BANK:
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, text_lower):
            found.add(skill)

    # 2. Extract items after section headings (Skills, Technologies, Tools…)
    section_pattern = re.compile(
        r"(?:technical skills?|skills?|tools?|technologies?|tech stack|"
        r"competencies|expertise|proficiency)[:\-\s]*([^\n]{5,200})",
        re.IGNORECASE,
    )
    for m in section_pattern.finditer(text):
        chunk = m.group(1)
        # split on common delimiters
        parts = re.split(r"[,|•·/;]\s*", chunk)
        for part in parts:
            p = part.strip().strip("●■▪-•").strip()
            if 1 < len(p) <= 30 and not p.isdigit():
                found.add(p.lower())

    # 3. Extract parenthesised tech lists e.g. (Python, SQL, Docker)
    for m in re.finditer(r"\(([^)]{5,150})\)", text):
        parts = re.split(r"[,;]\s*", m.group(1))
        for p in parts:
            p = p.strip()
            if 1 < len(p) <= 25 and re.search(r"[a-zA-Z]", p):
                found.add(p.lower())

    # 4. Extract from project technology lines
    tech_line_pattern = re.compile(
        r"(?:built with|developed using|technologies?|stack|implemented with)"
        r"\s*[:\-]?\s*([^\n\.]{5,200})",
        re.IGNORECASE,
    )
    for m in tech_line_pattern.finditer(text):
        parts = re.split(r"[,|•·/;]\s*", m.group(1))
        for part in parts:
            p = part.strip()
            if 1 < len(p) <= 30:
                found.add(p.lower())

    # Normalise: remove pure numbers / single chars / stopwords
    stopwords = {"and", "or", "the", "for", "with", "using", "etc", "other"}
    cleaned = []
    seen_lower: set[str] = set()
    for s in sorted(found):
        s = s.strip()
        if not s or s in stopwords or len(s) < 2:
            continue
        key = s.lower()
        if key in seen_lower:
            continue
        seen_lower.add(key)
        # Preferred capitalisation
        upper_map = {
            "sql", "aws", "gcp", "api", "nlp", "ml", "ai", "ci", "cd",
            "html", "css", "php", "vba", "oop", "mvc", "gpu", "cpu",
        }
        if key in upper_map:
            cleaned.append(key.upper())
        elif key in {"python", "java", "rust", "scala", "go", "dart", "lua"}:
            cleaned.append(key.title())
        else:
            cleaned.append(s.title())

    return cleaned


# ---------------------------------------------------------------------------
# Education / projects normalisation helpers
# ---------------------------------------------------------------------------

def _normalise_education(edu_list) -> list[dict]:
    out = []
    for item in (edu_list or []):
        if isinstance(item, str):
            out.append({
                "degree": item, "field_of_study": "",
                "institution": "", "year_of_graduation": None,
            })
        elif isinstance(item, dict):
            y = item.get("year_of_graduation")
            try:
                year = int(y) if y is not None and str(y).strip() else None
            except Exception:
                year = None
            out.append({
                "degree": item.get("degree") or "",
                "field_of_study": item.get("field_of_study") or "",
                "institution": item.get("institution") or "",
                "year_of_graduation": year,
            })
    return out


def _normalise_projects(proj_list) -> list[dict]:
    out = []
    for item in (proj_list or []):
        if isinstance(item, str):
            out.append({"name": item, "description": "", "technologies": []})
        elif isinstance(item, dict):
            techs = item.get("technologies") or []
            if isinstance(techs, str):
                techs = [t.strip() for t in techs.split(",") if t.strip()]
            out.append({
                "name": item.get("name") or "",
                "description": item.get("description") or "",
                "technologies": techs,
            })
    return out


def _normalise_skills(raw_skills, resume_text: str) -> list[str]:
    """
    Merge LLM-extracted skills with heuristic extraction.
    Deduplicates, normalises, and returns a clean list.
    """
    if isinstance(raw_skills, str):
        raw_skills = [s.strip() for s in raw_skills.split(",") if s.strip()]
    if not isinstance(raw_skills, list):
        raw_skills = []

    # Heuristic extraction always runs as safety net
    heuristic = extract_skills_heuristically(resume_text)

    # Merge
    combined: list[str] = list(raw_skills) + heuristic

    # Deduplicate (case-insensitive), preserve first occurrence casing
    seen: set[str] = set()
    out: list[str] = []
    for s in combined:
        if not isinstance(s, str) or not s.strip():
            continue
        key = s.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s.strip())

    return out


# ---------------------------------------------------------------------------
# Two-pass LLM skill enrichment
# ---------------------------------------------------------------------------

def _enrich_skills_second_pass(
    existing_skills: list[str],
    resume_text: str,
) -> list[str]:
    """Run a second LLM pass focused solely on finding missing skills."""
    if not existing_skills:
        return []
    prompt = SKILL_ENRICH_PROMPT.format(
        existing_skills=json.dumps(existing_skills[:60]),
        resume_text=resume_text[:3000],
    )
    try:
        raw = _call_openrouter(prompt, max_tokens=400)
        text = _strip_fences(raw)
        arr = json.loads(text)
        if isinstance(arr, list):
            return [s for s in arr if isinstance(s, str) and s.strip()]
    except Exception as e:
        logger.debug("Skill enrichment pass failed: %s", e)
    return []


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def parse_resume(pdf_path: str) -> ResumeProfile:
    """
    Parse resume PDF → ResumeProfile.

    Strategy:
      1. Extract raw text (pypdf → pdfminer fallback)
      2. LLM pass 1: full structured extraction
      3. LLM pass 2: skill enrichment
      4. Heuristic skill extraction (always merged in)
      5. Build and return ResumeProfile
    """
    print("  Extracting text from PDF...")
    raw_text = extract_text_from_pdf(pdf_path)
    if not raw_text or len(raw_text.strip()) < 50:
        raise RuntimeError("Could not extract readable text from the PDF.")

    print(f"  Extracted {len(raw_text)} chars. Sending to LLM (pass 1)...")

    # ----- Pass 1: Full extraction -----
    prompt = EXTRACTION_PROMPT.format(resume_text=raw_text[:4000])
    raw_response = _call_openrouter(prompt, max_tokens=1800)
    logger.debug("LLM raw response (first 800):\n%s", raw_response[:800])

    try:
        data = _safe_parse_json(raw_response)
    except ValueError as exc:
        logger.error("Primary parse failed: %s", exc)
        # Minimal skeleton so pipeline can limp forward
        data = {
            "full_name": "Unknown",
            "email": None, "phone": None,
            "skills": [], "experience_years": 0.0,
            "current_role": None, "education": [],
            "projects": [], "preferred_roles": [],
            "summary": None,
        }

    if not isinstance(data, dict):
        data = {"full_name": "Unknown", "skills": [], "experience_years": 0.0,
                "education": [], "projects": [], "preferred_roles": []}

    # ----- Normalise sub-structures -----
    data["education"] = _normalise_education(data.get("education"))
    data["projects"] = _normalise_projects(data.get("projects"))

    # ----- Skills: merge LLM + heuristic -----
    llm_skills: list[str] = data.get("skills") or []
    merged = _normalise_skills(llm_skills, raw_text)

    # ----- Pass 2: LLM skill enrichment (only if we have few skills) -----
    if len(merged) < 15:
        print("  Running skill enrichment pass 2...")
        extra = _enrich_skills_second_pass(merged, raw_text)
        if extra:
            existing_lower = {s.lower() for s in merged}
            for s in extra:
                if s.strip().lower() not in existing_lower:
                    merged.append(s.strip())
                    existing_lower.add(s.strip().lower())

    data["skills"] = merged

    # ----- Preferred roles: infer if empty -----
    if not data.get("preferred_roles"):
        data["preferred_roles"] = _infer_roles(merged, data.get("current_role"))

    print(f"  Skills extracted: {len(data['skills'])}")
    if data["skills"]:
        print(f"  Sample: {data['skills'][:10]}")
    print(f"  Preferred roles: {data.get('preferred_roles')}")

    # ----- Construct Pydantic model -----
    try:
        return ResumeProfile(**data)
    except Exception as exc:
        raise ValueError(
            f"ResumeProfile construction failed: {exc}\n"
            f"Data:\n{json.dumps(data, indent=2, default=str)}"
        ) from exc


def _infer_roles(skills: list[str], current_role: Optional[str]) -> list[str]:
    """Infer likely job roles from skills when LLM didn't return them."""
    skills_lower = {s.lower() for s in skills}
    roles = []

    ml_signals = {"pytorch", "tensorflow", "sklearn", "scikit-learn",
                  "machine learning", "deep learning", "nlp", "computer vision",
                  "transformers", "huggingface", "keras", "xgboost"}
    data_signals = {"pandas", "numpy", "sql", "tableau", "power bi",
                    "matplotlib", "seaborn", "spark", "bigquery", "etl",
                    "databricks", "snowflake"}
    backend_signals = {"fastapi", "django", "flask", "nodejs", "spring",
                       "microservices", "docker", "kubernetes", "rest", "grpc"}
    frontend_signals = {"react", "angular", "vue", "next.js", "html",
                        "css", "javascript", "typescript"}
    devops_signals = {"aws", "azure", "gcp", "terraform", "ansible",
                      "jenkins", "github actions", "helm", "kubernetes"}

    if skills_lower & ml_signals:
        roles.append("Machine Learning Engineer")
    if skills_lower & data_signals:
        roles.append("Data Analyst")
    if len(skills_lower & ml_signals) >= 3 and len(skills_lower & data_signals) >= 2:
        roles.append("Data Scientist")
    if skills_lower & backend_signals:
        roles.append("Backend Developer")
    if skills_lower & frontend_signals:
        roles.append("Frontend Developer")
    if skills_lower & devops_signals:
        roles.append("DevOps Engineer")

    if current_role and not roles:
        roles.append(current_role)

    return roles[:4] or ["Software Engineer"]