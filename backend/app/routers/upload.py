import os
import uuid
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.database import get_db, Patient, Study, ModalityEnum
from app.models.schemas import UploadResponse
from app.services.dicom_processor import extract_metadata_from_zip
from app.config import settings

router = APIRouter(prefix="/upload", tags=["upload"])

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@router.post("/dicom", response_model=UploadResponse)
async def upload_dicom(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    if not file.filename.endswith(".zip"):
        raise HTTPException(400, "Only .zip DICOM archives are accepted.")

    zip_bytes = await file.read()
    if len(zip_bytes) > 500 * 1024 * 1024:  # 500 MB limit
        raise HTTPException(413, "File too large (max 500 MB).")

    # Extract DICOM metadata
    try:
        metadata, datasets = extract_metadata_from_zip(zip_bytes)
    except ValueError as e:
        raise HTTPException(400, str(e))

    # Save zip to disk
    save_name = f"{uuid.uuid4().hex}_{file.filename}"
    save_path = UPLOAD_DIR / save_name
    save_path.write_bytes(zip_bytes)

    # Upsert patient
    result = await db.execute(
        select(Patient).where(Patient.patient_id == metadata.patient_id)
    )
    patient = result.scalar_one_or_none()
    if patient is None:
        patient = Patient(
            patient_id=metadata.patient_id,
            name=metadata.patient_name,
            birth_date=metadata.birth_date,
            age=metadata.age,
            sex=metadata.sex,
        )
        db.add(patient)
        await db.flush()

    # Create study
    modality = ModalityEnum(metadata.modality.value)
    study = Study(
        patient_id=patient.id,
        accession_number=metadata.accession_number,
        modality=modality,
        study_date=metadata.study_date,
        study_description=metadata.study_description,
        series_description=metadata.series_description,
        protocol_name=metadata.protocol_name,
        body_part=metadata.body_part,
        institution=metadata.institution,
        dicom_zip_path=str(save_path),
        slice_count=metadata.slice_count,
    )
    db.add(study)
    await db.commit()
    await db.refresh(study)

    return UploadResponse(
        study_id=study.id,
        metadata=metadata,
        message=f"DICOM uploaded successfully. {metadata.slice_count} slices detected.",
    )
