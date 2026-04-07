"""
models/job.py  –  Job-related Pydantic models (unchanged from your version)
No changes needed – models are already robust.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class JobListing(BaseModel):
    job_id: str
    title: str
    company: str
    location: str
    description: str
    posted_date: str
    apply_url: str
    salary_min: Optional[float] = None
    salary_max: Optional[float] = None
    freshness_bucket: str = "older"


class ScoreBreakdown(BaseModel):
    skill_overlap: float = 0.0
    experience_alignment: float = 0.0
    location_match: float = 0.0
    freshness: float = 0.0
    overall: float = 0.0


class SkillGap(BaseModel):
    missing_skill: str
    resource: str
    micro_project: str


class RewrittenBullet(BaseModel):
    original: str
    rewritten: str
    skill_targeted: str = ""


class MatchResult(BaseModel):
    job: JobListing
    score: ScoreBreakdown
    matched_skills: List[str] = Field(default_factory=list)
    missing_skills: List[SkillGap] = Field(default_factory=list)
    rewrites: List[RewrittenBullet] = Field(default_factory=list)
    explanation: str = ""