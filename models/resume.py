from pydantic import BaseModel, Field
from typing import List, Optional


class Education(BaseModel):
    degree: str
    field_of_study: Optional[str] = None
    institution: str
    year_of_graduation: Optional[int] = None


class Project(BaseModel):
    name: str
    description: str
    technologies: List[str] = []


class UserPreferences(BaseModel):
    experience_level: str
    freshness_days: int = 30
    location_priority: List[str]


class ResumeProfile(BaseModel):
    full_name: str = Field(default="Unknown")
    email: Optional[str] = None
    phone: Optional[str] = None
    skills: List[str] = Field(default_factory=list)
    experience_years: float = Field(default=0.0)
    current_role: Optional[str] = None
    education: List[Education] = Field(default_factory=list)
    projects: List[Project] = Field(default_factory=list)
    preferred_roles: List[str] = Field(default_factory=list)
    summary: Optional[str] = None
