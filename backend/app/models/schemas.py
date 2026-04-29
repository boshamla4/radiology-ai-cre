from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
from enum import Enum


class ModalityEnum(str, Enum):
    CT = "CT"
    MR = "MR"
    OTHER = "OTHER"


class ReportStatusEnum(str, Enum):
    AI_DRAFT = "AI_DRAFT"
    DOCTOR_REVIEWED = "DOCTOR_REVIEWED"
    VALIDATED = "VALIDATED"


# ── DICOM Metadata (extracted automatically) ──────────────────────────────────
class DicomMetadata(BaseModel):
    patient_id: str
    patient_name: str
    birth_date: str
    age: str
    sex: str
    modality: ModalityEnum
    study_date: str
    study_description: str
    series_description: str
    protocol_name: str
    body_part: Optional[str] = None
    institution: str
    accession_number: str
    slice_count: int


# ── Upload ────────────────────────────────────────────────────────────────────
class UploadResponse(BaseModel):
    study_id: int
    metadata: DicomMetadata
    message: str


# ── Report creation (manual fields added by user) ─────────────────────────────
class ReportCreateRequest(BaseModel):
    study_id: int
    indication: str
    prescribing_doctor: str


class FindingItem(BaseModel):
    structure: str
    location: Optional[str] = None
    size_mm: Optional[float] = None
    description: str
    is_pathological: bool
    confidence: float


class SegmentationResult(BaseModel):
    findings: List[FindingItem]
    overlay_image_base64: str
    model_version: str
    dice_score: Optional[float] = None


class ReportDraftResponse(BaseModel):
    report_id: int
    status: ReportStatusEnum
    exam_type: str
    technique: str
    ai_findings: str
    ai_conclusion: str
    segmentation: Optional[SegmentationResult]
    similar_cases_count: int
    ai_confidence: float
    created_at: datetime


# ── Report validation (doctor confirms/edits) ─────────────────────────────────
class ReportValidateRequest(BaseModel):
    report_id: int
    radiologist: str
    final_findings: str
    final_conclusion: str


class ReportValidateResponse(BaseModel):
    report_id: int
    pdf_path: str
    message: str


# ── Statistics ────────────────────────────────────────────────────────────────
class DashboardStats(BaseModel):
    total_studies: int
    ct_studies: int
    mr_studies: int
    validated_reports: int
    ai_drafts_pending: int
    avg_ai_confidence: Optional[float]
    pathologies_found: int
    normal_exams: int
