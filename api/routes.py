import os
import shutil
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List

from agents.orchestrator import run_pipeline
import traceback
import logging

router = APIRouter()

UPLOAD_DIR = "data/sample_resumes"
# Ensure upload directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)


class SearchRequest(BaseModel):
    session_id: str
    experience_level: str
    freshness_days: int = 30
    locations: List[str]


class JobResultResponse(BaseModel):
    title: str
    company: str
    location: str
    posted_date: str
    apply_url: str
    freshness_bucket: str
    overall_score: int
    skill_score: int
    exp_score: int
    location_score: int
    freshness_score: int
    matched_skills: List[str]
    gaps: List[dict]
    rewrites: List[dict]


@router.post("/upload-resume")
async def upload_resume(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400, detail="Only PDF files are accepted"
        )

    session_id = str(uuid.uuid4())[:8]
    filename = f"{session_id}.pdf"
    filepath = os.path.join(UPLOAD_DIR, filename)

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        shutil.copyfileobj(file.file, f)

    return {
        "session_id": session_id,
        "filename": filename,
        "message": "Resume uploaded successfully",
    }


@router.post("/search-jobs")
async def search_jobs(request: SearchRequest):
    filename = f"{request.session_id}.pdf"
    filepath = os.path.join(UPLOAD_DIR, filename)

    if not os.path.exists(filepath):
        raise HTTPException(
            status_code=404, detail="Resume not found - upload first"
        )

    try:
        profile, results = run_pipeline(
            pdf_path=filepath,
            experience_level=request.experience_level,
            freshness_days=request.freshness_days,
            location_priority=request.locations,
        )

        jobs_out = []
        for r in results:
            jobs_out.append(
                {
                    "title": r.job.title,
                    "company": r.job.company,
                    "location": r.job.location,
                    "posted_date": r.job.posted_date,
                    "apply_url": r.job.apply_url,
                    "freshness_bucket": r.job.freshness_bucket,
                    "overall_score": round(r.score.overall * 100),
                    "skill_score": round(r.score.skill_overlap * 100),
                    "exp_score": round(r.score.experience_alignment * 100),
                    "location_score": round(r.score.location_match * 100),
                    "freshness_score": round(r.score.freshness * 100),
                    "matched_skills": r.matched_skills,
                    "gaps": [
                        {
                            "skill": g.missing_skill,
                            "resource": g.resource,
                            "project": g.micro_project,
                        }
                        for g in r.missing_skills
                    ],
                    # -- NEW: resume rewrites --------------------------
                    "rewrites": [
                        {
                            "original": rw.original,
                            "rewritten": rw.rewritten,
                            "skill_targeted": rw.skill_targeted,
                        }
                        for rw in r.rewrites
                    ],
                }
            )

        return JSONResponse(
            content={
                "candidate": profile.full_name,
                "total_jobs_found": len(results),
                "results": jobs_out,
            }
        )

    except Exception as e:
        # Log full traceback for debugging
        logging.exception("Error in /search-jobs: %s", e)
        traceback.print_exc()
        # Return the error detail (short) while keeping full trace in server logs
        raise HTTPException(
            status_code=500,
            detail="Internal server error; check server logs for details",
        )
