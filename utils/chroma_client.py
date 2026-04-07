import logging
import chromadb
import os
from chromadb.config import Settings
from config import CHROMA_DB_PATH, COLLECTION_RESUME, COLLECTION_JOBS

os.environ["ANONYMIZED_TELEMETRY"] = "False"
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
logging.getLogger("posthog").setLevel(logging.CRITICAL)


_client = None


def get_client() -> chromadb.Client:
    """Get or create ChromaDB client."""
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path=CHROMA_DB_PATH, settings=Settings(anonymized_telemetry=False)
        )
    return _client


def get_resume_collection():
    """Get or create resume profiles collection."""
    client = get_client()
    return client.get_or_create_collection(
        name=COLLECTION_RESUME, metadata={"hnsw:space": "cosine"}
    )


def get_job_collection():
    """Get or create job listings collection."""
    client = get_client()
    return client.get_or_create_collection(
        name=COLLECTION_JOBS, metadata={"hnsw:space": "cosine"}
    )


def get_chroma_collection(name: str):
    """Get or create collection by name."""
    client = get_client()
    return client.get_or_create_collection(
        name=name, metadata={"hnsw:space": "cosine"}
    )


def upsert_resume(session_id: str, profile_json: str, embedding: list):
    """Upsert resume to ChromaDB."""
    collection = get_resume_collection()
    collection.upsert(
        ids=[session_id],
        documents=[profile_json],
        embeddings=[embedding],
        metadatas=[{"session_id": session_id}],
    )


def upsert_job(job_id: str, description: str, embedding: list, metadata: dict):
    """Upsert job to ChromaDB."""
    collection = get_job_collection()

    # ✅ Ensure metadata is never empty
    if not metadata:
        metadata = {"job_id": job_id, "source": "adzuna"}
    elif "job_id" not in metadata:
        metadata["job_id"] = job_id

    # Sanitize metadata values: Chroma requires primitive values (no None)
    sanitized = {}
    for k, v in metadata.items():
        if v is None:
            sanitized[k] = ""
        elif isinstance(v, (str, int, float, bool)):
            sanitized[k] = v
        else:
            # Fallback to string representation for non-primitive values
            try:
                sanitized[k] = str(v)
            except Exception:
                sanitized[k] = ""

    collection.upsert(
        ids=[job_id],
        documents=[description],
        embeddings=[embedding],
        metadatas=[sanitized],
    )


def query_jobs(query_embedding: list, n_results: int = 20) -> dict:
    """Query jobs by embedding."""
    collection = get_job_collection()
    count = collection.count()
    if count == 0:
        return {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }
    n_results = min(n_results, count)
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )


def clear_job_collection():
    """Clear all jobs from ChromaDB."""
    client = get_client()
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_JOBS in existing:
        client.delete_collection(COLLECTION_JOBS)


def get_collection_stats() -> dict:
    """Get stats about collections."""
    return {
        "resume_count": get_resume_collection().count(),
        "job_count": get_job_collection().count(),
    }
