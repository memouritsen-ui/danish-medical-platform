from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

class EvidenceLevel(str, Enum):
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    VERY_LOW = "very_low"

class VerificationStatus(str, Enum):
    VERIFIED = "verified"
    CONTRADICTED = "contradicted"
    UNCERTAIN = "uncertain"
    PENDING = "pending"

class PICO(BaseModel):
    population: str = Field(description="Patient population or problem")
    intervention: str = Field(description="Intervention or exposure")
    comparison: Optional[str] = Field(None, description="Comparison intervention")
    outcome: str = Field(description="Outcome of interest")

class SourceMetadata(BaseModel):
    url: str
    title: str
    domain: str
    date_accessed: datetime = Field(default_factory=datetime.now)
    author: Optional[str] = None
    publication_date: Optional[str] = None
    credibility_score: float = Field(0.0, ge=0.0, le=1.0)
    is_paywalled: bool = False

class CochraneReport(BaseModel):
    pico: PICO
    rob_score: str = Field(description="Risk of Bias 2.0 score")
    grade_level: str = Field(description="GRADE certainty level")
    summary: str
    contradictions: List[str] = []
    key_findings: List[str] = []

class ResearchTask(BaseModel):
    task_id: str
    topic: str
    status: str = "pending" # pending, running, completed, failed
    created_at: datetime = Field(default_factory=datetime.now)
    logs: List[str] = []
    result: Optional[Dict[str, Any]] = None

class LogMessage(BaseModel):
    task_id: str
    timestamp: datetime
    message: str
    level: str = "INFO"

