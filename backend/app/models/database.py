from sqlalchemy import Column, String, Integer, Float, DateTime, Text, Boolean, ForeignKey, Enum
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import DeclarativeBase, relationship
from sqlalchemy.sql import func
import enum

from app.config import settings


engine = create_async_engine(settings.DATABASE_URL, echo=False)


class Base(DeclarativeBase):
    pass


class ModalityEnum(str, enum.Enum):
    CT = "CT"
    MR = "MR"
    OTHER = "OTHER"


class ReportStatusEnum(str, enum.Enum):
    AI_DRAFT = "AI_DRAFT"
    DOCTOR_REVIEWED = "DOCTOR_REVIEWED"
    VALIDATED = "VALIDATED"


class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String, unique=True, index=True)  # P600.XXXXXX from DICOM
    name = Column(String)
    birth_date = Column(String)
    age = Column(String)
    sex = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    studies = relationship("Study", back_populates="patient")


class Study(Base):
    __tablename__ = "studies"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"))
    accession_number = Column(String, index=True)
    modality = Column(Enum(ModalityEnum))
    study_date = Column(String)
    study_description = Column(String)
    series_description = Column(String)
    protocol_name = Column(String)
    body_part = Column(String)
    institution = Column(String)
    dicom_zip_path = Column(String)   # path to stored zip
    slice_count = Column(Integer)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    patient = relationship("Patient", back_populates="studies")
    reports = relationship("Report", back_populates="study")
    segmentation = relationship("Segmentation", back_populates="study", uselist=False)


class Segmentation(Base):
    __tablename__ = "segmentations"

    id = Column(Integer, primary_key=True, index=True)
    study_id = Column(Integer, ForeignKey("studies.id"), unique=True)
    overlay_image_path = Column(String)    # PNG overlay stored on disk
    findings_json = Column(Text)           # JSON: detected structures + measurements
    iou_score = Column(Float, nullable=True)
    dice_score = Column(Float, nullable=True)
    model_version = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    study = relationship("Study", back_populates="segmentation")


class Report(Base):
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, index=True)
    study_id = Column(Integer, ForeignKey("studies.id"))
    status = Column(Enum(ReportStatusEnum), default=ReportStatusEnum.AI_DRAFT)

    # Fields filled at upload (manual)
    indication = Column(Text)
    prescribing_doctor = Column(String)
    radiologist = Column(String, nullable=True)   # filled when doctor validates

    # AI-generated content
    exam_type = Column(String)       # e.g. "TDM CEREBRALE"
    technique = Column(String)
    ai_findings = Column(Text)       # raw AI output
    ai_conclusion = Column(Text)

    # Doctor-edited content
    final_findings = Column(Text, nullable=True)
    final_conclusion = Column(Text, nullable=True)

    # Metadata
    retrieved_similar_cases = Column(Text)   # JSON list of similar case IDs used for RAG
    ai_confidence = Column(Float, nullable=True)
    pdf_path = Column(String, nullable=True)
    is_training_sample = Column(Boolean, default=False)  # flagged after doctor validates
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    validated_at = Column(DateTime(timezone=True), nullable=True)

    study = relationship("Study", back_populates="reports")


class TrainingCompteRendu(Base):
    """Indexed compte rendus from training dataset for RAG retrieval."""
    __tablename__ = "training_compte_rendus"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String)
    modality = Column(String)
    body_part = Column(String)
    exam_type = Column(String)
    indication = Column(Text)
    technique = Column(String)
    findings = Column(Text)
    conclusion = Column(Text)
    raw_text = Column(Text)
    pdf_path = Column(String)
    chroma_id = Column(String, unique=True)  # ID in ChromaDB vector store
    indexed_at = Column(DateTime(timezone=True), server_default=func.now())


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db():
    async with AsyncSession(engine) as session:
        yield session
