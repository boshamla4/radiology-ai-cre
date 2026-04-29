import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.database import get_db, Study, Segmentation, Report, ReportStatusEnum
from app.models.schemas import ReportCreateRequest, ReportDraftResponse, SegmentationResult
from app.services.dicom_processor import extract_metadata_from_zip, extract_representative_slices
from ml.unet.inference import get_segmentation_model
from ml.rag.report_generator import get_report_generator
from app.config import settings

router = APIRouter(prefix="/analysis", tags=["analysis"])


@router.post("/generate-draft", response_model=ReportDraftResponse)
async def generate_draft(
    req: ReportCreateRequest,
    db: AsyncSession = Depends(get_db),
):
    # Load study
    result = await db.execute(select(Study).where(Study.id == req.study_id))
    study = result.scalar_one_or_none()
    if not study:
        raise HTTPException(404, "Study not found.")

    # Load DICOM — resolve relative paths from the backend/ directory
    zip_path = Path(study.dicom_zip_path)
    if not zip_path.is_absolute():
        # Try relative to the backend directory (where uvicorn is launched)
        backend_dir = Path(__file__).parent.parent.parent
        zip_path = backend_dir / zip_path
    if not zip_path.exists():
        raise HTTPException(500, f"DICOM file not found on disk: {zip_path}")

    try:
        zip_bytes = zip_path.read_bytes()
        from app.services.dicom_processor import extract_metadata_from_zip
        metadata, datasets = extract_metadata_from_zip(zip_bytes)

        # Extract representative slices (5 slices for segmentation)
        slices = extract_representative_slices(datasets, n=5)

        # Run segmentation
        weights_path = Path(settings.MODELS_PATH) / "unet_weights.pth"
        if not weights_path.is_absolute():
            weights_path = Path(__file__).parent.parent.parent / weights_path
        seg_model = get_segmentation_model(weights_path=str(weights_path))
        seg_result = seg_model.run(slices, study.body_part or "GENERIC")

        # Save segmentation overlay
        import base64, io
        from PIL import Image
        overlay_dir = Path(__file__).parent.parent.parent / "uploads"
        overlay_dir.mkdir(exist_ok=True)
        overlay_path = overlay_dir / f"overlay_{study.id}.png"
        if seg_result.overlay_image_base64:
            img_bytes = base64.b64decode(seg_result.overlay_image_base64)
            Image.open(io.BytesIO(img_bytes)).save(overlay_path)

        seg_db = Segmentation(
            study_id=study.id,
            overlay_image_path=str(overlay_path),
            findings_json=json.dumps([f.model_dump() for f in seg_result.findings]),
            model_version=seg_result.model_version,
        )
        db.add(seg_db)
        await db.flush()

        # Generate report draft
        generator = get_report_generator()
        exam_type, technique, findings_text, conclusion_text, similar_count, confidence = \
            generator.generate(
                metadata=metadata,
                indication=req.indication,
                findings=seg_result.findings,
                overlay_base64=seg_result.overlay_image_base64,
            )

        # Save report draft
        report = Report(
            study_id=study.id,
            status=ReportStatusEnum.AI_DRAFT,
            indication=req.indication,
            prescribing_doctor=req.prescribing_doctor,
            exam_type=exam_type,
            technique=technique,
            ai_findings=findings_text,
            ai_conclusion=conclusion_text,
            retrieved_similar_cases=json.dumps(similar_count),
            ai_confidence=confidence,
        )
        db.add(report)
        await db.commit()
        await db.refresh(report)

        from datetime import datetime
        return ReportDraftResponse(
            report_id=report.id,
            status=report.status,
            exam_type=exam_type,
            technique=technique,
            ai_findings=findings_text,
            ai_conclusion=conclusion_text,
            segmentation=seg_result,
            similar_cases_count=similar_count,
            ai_confidence=confidence,
            created_at=report.created_at or datetime.utcnow(),
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        detail = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        print(f"[generate-draft ERROR]\n{detail}")
        raise HTTPException(status_code=500, detail=str(e))
