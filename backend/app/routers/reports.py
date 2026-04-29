import os
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.models.database import get_db, Report, Study, Patient, ReportStatusEnum
from app.models.schemas import ReportValidateRequest, ReportValidateResponse, DashboardStats
from app.services.pdf_generator import generate_compte_rendu

router = APIRouter(prefix="/reports", tags=["reports"])

PDF_DIR = Path("pdfs")
PDF_DIR.mkdir(exist_ok=True)


@router.post("/validate", response_model=ReportValidateResponse)
async def validate_report(
    req: ReportValidateRequest,
    db: AsyncSession = Depends(get_db),
):
    """Doctor reviews the AI draft, edits findings/conclusion, and validates."""
    result = await db.execute(select(Report).where(Report.id == req.report_id))
    report = result.scalar_one_or_none()
    if not report:
        raise HTTPException(404, "Report not found.")

    # Load related study and patient
    study_result = await db.execute(select(Study).where(Study.id == report.study_id))
    study = study_result.scalar_one_or_none()
    patient_result = await db.execute(select(Patient).where(Patient.id == study.patient_id))
    patient = patient_result.scalar_one_or_none()

    # Update report with doctor edits
    report.radiologist = req.radiologist
    report.final_findings = req.final_findings
    report.final_conclusion = req.final_conclusion
    report.status = ReportStatusEnum.VALIDATED
    report.validated_at = datetime.utcnow()
    report.is_training_sample = True  # feed back into training

    # Generate final validated PDF
    pdf_bytes = generate_compte_rendu(
        patient_name=patient.name,
        birth_date=patient.birth_date,
        age=patient.age,
        prescribing_doctor=report.prescribing_doctor,
        radiologist=req.radiologist,
        patient_id=patient.patient_id,
        study_date=study.study_date,
        exam_type=report.exam_type,
        indication=report.indication,
        technique=report.technique,
        findings=req.final_findings,
        conclusion=req.final_conclusion,
        is_ai_draft=False,
    )

    pdf_path = PDF_DIR / f"report_{report.id}_validated.pdf"
    pdf_path.write_bytes(pdf_bytes)
    report.pdf_path = str(pdf_path)

    await db.commit()

    return ReportValidateResponse(
        report_id=report.id,
        pdf_path=str(pdf_path),
        message="Report validated and PDF generated.",
    )


@router.get("/draft-pdf/{report_id}")
async def get_draft_pdf(report_id: int, db: AsyncSession = Depends(get_db)):
    """Generate and return an AI draft PDF for preview."""
    result = await db.execute(select(Report).where(Report.id == report_id))
    report = result.scalar_one_or_none()
    if not report:
        raise HTTPException(404, "Report not found.")

    study_result = await db.execute(select(Study).where(Study.id == report.study_id))
    study = study_result.scalar_one_or_none()
    patient_result = await db.execute(select(Patient).where(Patient.id == study.patient_id))
    patient = patient_result.scalar_one_or_none()

    pdf_bytes = generate_compte_rendu(
        patient_name=patient.name,
        birth_date=patient.birth_date,
        age=patient.age,
        prescribing_doctor=report.prescribing_doctor,
        radiologist=None,
        patient_id=patient.patient_id,
        study_date=study.study_date,
        exam_type=report.exam_type,
        indication=report.indication,
        technique=report.technique,
        findings=report.ai_findings or "",
        conclusion=report.ai_conclusion or "",
        is_ai_draft=True,
    )

    draft_path = PDF_DIR / f"report_{report_id}_draft.pdf"
    draft_path.write_bytes(pdf_bytes)

    return FileResponse(str(draft_path), media_type="application/pdf",
                        filename=f"brouillon_ia_{report_id}.pdf")


@router.get("/download/{report_id}")
async def download_report(report_id: int, db: AsyncSession = Depends(get_db)):
    """Download the final validated PDF."""
    result = await db.execute(select(Report).where(Report.id == report_id))
    report = result.scalar_one_or_none()
    if not report or not report.pdf_path:
        raise HTTPException(404, "Validated report not found.")

    return FileResponse(report.pdf_path, media_type="application/pdf",
                        filename=f"compte_rendu_{report_id}.pdf")


@router.get("/dashboard", response_model=DashboardStats)
async def get_dashboard(db: AsyncSession = Depends(get_db)):
    from sqlalchemy import func as sqlfunc
    from app.models.database import ModalityEnum as DBModality

    total = (await db.execute(select(func.count(Study.id)))).scalar() or 0
    ct = (await db.execute(select(func.count(Study.id)).where(
        Study.modality == DBModality.CT))).scalar() or 0
    mr = (await db.execute(select(func.count(Study.id)).where(
        Study.modality == DBModality.MR))).scalar() or 0
    validated = (await db.execute(select(func.count(Report.id)).where(
        Report.status == ReportStatusEnum.VALIDATED))).scalar() or 0
    drafts = (await db.execute(select(func.count(Report.id)).where(
        Report.status == ReportStatusEnum.AI_DRAFT))).scalar() or 0
    avg_conf = (await db.execute(select(func.avg(Report.ai_confidence)))).scalar()

    return DashboardStats(
        total_studies=total,
        ct_studies=ct,
        mr_studies=mr,
        validated_reports=validated,
        ai_drafts_pending=drafts,
        avg_ai_confidence=round(avg_conf, 2) if avg_conf else None,
        pathologies_found=0,
        normal_exams=0,
    )
