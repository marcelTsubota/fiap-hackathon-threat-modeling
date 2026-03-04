from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator


# -----------------------------
# Enums (fixed / canonical)
# -----------------------------


class ComponentClass(str, Enum):
    # DO NOT CHANGE (fixed classes)
    USER = "User"
    EXTERNAL_SYSTEM = "External System"
    API_GATEWAY = "API Gateway"
    LOAD_BALANCER = "Load Balancer"
    WEB_APPLICATION = "Web Application"
    MOBILE_APPLICATION = "Mobile Application"
    APPLICATION_SERVER = "Application Server"
    SERVERLESS_FUNCTION = "Serverless Function"
    DATABASE = "Database"
    CACHE = "Cache"
    MESSAGE_QUEUE = "Message Queue"
    FILE_STORAGE = "File Storage"
    AUTHENTICATION_SERVICE = "Authentication Service"
    SECURITY_SERVICE = "Security Service (WAF/Firewall/Shield)"


class StrideCategory(str, Enum):
    SPOOFING = "Spoofing"
    TAMPERING = "Tampering"
    REPUDIATION = "Repudiation"
    INFORMATION_DISCLOSURE = "Information Disclosure"
    DENIAL_OF_SERVICE = "Denial of Service"
    ELEVATION_OF_PRIVILEGE = "Elevation of Privilege"


class Severity(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


# -----------------------------
# Common primitives
# -----------------------------


class BBox(BaseModel):
    """
    Bounding box in absolute pixel coordinates.
    (x1, y1) is top-left; (x2, y2) is bottom-right.
    """
    model_config = ConfigDict(extra="forbid")

    x1: float = Field(..., ge=0)
    y1: float = Field(..., ge=0)
    x2: float = Field(..., ge=0)
    y2: float = Field(..., ge=0)

    @model_validator(mode="after")
    def _validate_bbox(self) -> "BBox":
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("Invalid bbox: x2 must be > x1 and y2 must be > y1.")
        return self

    def as_xyxy(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


class ImageMeta(BaseModel):
    """Metadata for the processed image (used to contextualize detections)."""
    model_config = ConfigDict(extra="forbid")

    filename: Optional[str] = None
    width: int = Field(..., ge=1)
    height: int = Field(..., ge=1)
    content_type: Optional[str] = None  # e.g. image/png, image/jpeg


# -----------------------------
# YOLO detection output
# -----------------------------


class Detection(BaseModel):
    """Single detection result from YOLO."""
    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., description="Stable identifier for this detection/component.")
    label: ComponentClass
    confidence: float = Field(..., ge=0, le=1)
    bbox: BBox

    @field_validator("id")
    @classmethod
    def _id_non_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Detection id must be non-empty.")
        return v


class DiagramDetections(BaseModel):
    """
    Structured detection output: image metadata + list of detections.
    This is the canonical JSON between detection and STRIDE.
    """
    model_config = ConfigDict(extra="forbid")

    image: ImageMeta
    detections: List[Detection] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_unique_ids(self) -> "DiagramDetections":
        ids = [d.id for d in self.detections]
        if len(ids) != len(set(ids)):
            raise ValueError("detections must have unique ids.")
        return self


# -----------------------------
# STRIDE results
# -----------------------------


class Threat(BaseModel):
    """A single STRIDE threat entry for a given detected component."""
    model_config = ConfigDict(extra="forbid")

    threat_id: str = Field(..., description="Unique threat id (stable within the report).")
    component_id: str = Field(..., description="References Detection.id.")
    component_label: ComponentClass
    category: StrideCategory
    title: str = Field(..., min_length=3)
    description: str = Field(..., min_length=10)
    impact: str = Field(..., min_length=5)
    mitigations: List[str] = Field(default_factory=list, description="Concrete mitigation steps.")
    severity: Severity
    confidence: float = Field(0.75, ge=0, le=1, description="Model confidence for this threat entry.")

    @field_validator("threat_id", "component_id", "title")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Field must be non-empty.")
        return v

    @field_validator("mitigations")
    @classmethod
    def _mitigations_non_empty(cls, v: List[str]) -> List[str]:
        cleaned = []
        for m in v:
            m2 = (m or "").strip()
            if m2:
                cleaned.append(m2)
        return cleaned


class StrideResult(BaseModel):
    """All threats generated for the diagram."""
    model_config = ConfigDict(extra="forbid")

    threats: List[Threat] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_unique_threat_ids(self) -> "StrideResult":
        ids = [t.threat_id for t in self.threats]
        if len(ids) != len(set(ids)):
            raise ValueError("threats must have unique threat_id values.")
        return self


# -----------------------------
# Enrichment (LLM + RAG) and KB
# -----------------------------


class KBHit(BaseModel):
    """
    Represents a knowledge base retrieval result (local KB JSON + embeddings).
    Keep it generic to support different KB formats.
    """
    model_config = ConfigDict(extra="forbid")

    source: str = Field(..., description="KB source identifier, e.g., 'local_kb.json'")
    key: str = Field(..., description="Entry key or identifier inside the KB.")
    score: float = Field(..., ge=0, le=1)
    snippet: Optional[str] = Field(None, description="Short excerpt used for grounding.")


class LLMUsage(BaseModel):
    """Captures minimal LLM usage telemetry for debugging/traceability."""
    model_config = ConfigDict(extra="forbid")

    model: str = Field(..., description="e.g., gpt-4.1-mini")
    prompt_tokens: Optional[int] = Field(None, ge=0)
    completion_tokens: Optional[int] = Field(None, ge=0)
    total_tokens: Optional[int] = Field(None, ge=0)


class EnrichmentResult(BaseModel):
    """
    Enriched data linked back to threats.
    This can include grounded KB hits and any extra structured fields you want to show in the report.
    """
    model_config = ConfigDict(extra="forbid")

    threat_id: str
    kb_hits: List[KBHit] = Field(default_factory=list)
    notes: Optional[str] = Field(
        None,
        description="Optional LLM commentary/extra context (kept short in PDF)."
    )
    llm_usage: Optional[LLMUsage] = None


class EnrichmentBundle(BaseModel):
    """All enrichments generated for a diagram."""
    model_config = ConfigDict(extra="forbid")

    enrichments: List[EnrichmentResult] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_unique_enrichment_threat_ids(self) -> "EnrichmentBundle":
        ids = [e.threat_id for e in self.enrichments]
        if len(ids) != len(set(ids)):
            raise ValueError("enrichments must have unique threat_id values.")
        return self


# -----------------------------
# Consolidated report model (for PDF rendering)
# -----------------------------


class ReportMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field("Threat Modeling Report (STRIDE)", min_length=3)
    system_name: Optional[str] = Field(None, description="Optional system name provided by the user.")
    generated_at_iso: str = Field(..., description="ISO timestamp of generation (UTC or local).")
    version: str = Field("0.1.0", description="Report schema version for traceability.")


class ThreatModelingReport(BaseModel):
    """
    Single object that contains everything needed for PDF rendering and API response.
    """
    model_config = ConfigDict(extra="forbid")

    meta: ReportMeta
    diagram: DiagramDetections
    stride: StrideResult
    enrichment: Optional[EnrichmentBundle] = None

    @model_validator(mode="after")
    def _validate_cross_refs(self) -> "ThreatModelingReport":
        detection_ids = {d.id for d in self.diagram.detections}
        for t in self.stride.threats:
            if t.component_id not in detection_ids:
                raise ValueError(f"Threat {t.threat_id} references unknown component_id={t.component_id}.")
        if self.enrichment:
            threat_ids = {t.threat_id for t in self.stride.threats}
            for e in self.enrichment.enrichments:
                if e.threat_id not in threat_ids:
                    raise ValueError(f"Enrichment references unknown threat_id={e.threat_id}.")
        return self


# -----------------------------
# API request/response schemas
# -----------------------------


class AnalyzeRequest(BaseModel):
    """
    Optional parameters for analysis (kept minimal).
    Upload is handled by FastAPI via multipart; this schema can be used for JSON-only flows.
    """
    model_config = ConfigDict(extra="forbid")

    system_name: Optional[str] = Field(None, description="Displayed in the report header.")
    # Future-proof: allow client to pass optional hints (without changing core architecture).
    hints: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional hints for the pipeline (non-critical)."
    )


class AnalyzeResponse(BaseModel):
    """Primary response: structured report object + optional paths/ids."""
    model_config = ConfigDict(extra="forbid")

    report: ThreatModelingReport
    pdf_filename: Optional[str] = Field(None, description="Generated PDF file name (if applicable).")
    pdf_path: Optional[str] = Field(None, description="Server-side path (do not expose publicly if unsafe).")


class ErrorResponse(BaseModel):
    """Standardized error response for the API."""
    model_config = ConfigDict(extra="forbid")

    error: str
    detail: Optional[str] = None
    code: Optional[Literal["VALIDATION_ERROR", "PROCESSING_ERROR", "LLM_REQUIRED", "INTERNAL_ERROR"]] = None
