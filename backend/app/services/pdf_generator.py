"""
Generates compte rendu PDFs styled exactly like Centre Radiologie Emilie.
AI drafts get a watermark. Validated reports get the radiologist signature.
"""
import io
from datetime import datetime
from pathlib import Path
from typing import Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, HRFlowable,
    Table, TableStyle, Image as RLImage, KeepTogether
)
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


# ── Colours matching the CRE header ──────────────────────────────────────────
CRE_BLUE = colors.HexColor("#1a3c6e")
CRE_RED = colors.HexColor("#c0392b")
LIGHT_GRAY = colors.HexColor("#f5f5f5")
MID_GRAY = colors.HexColor("#666666")
BLACK = colors.black


def _styles():
    s = getSampleStyleSheet()
    N = "Helvetica"
    B = "Helvetica-Bold"
    L = 14

    header_clinic  = ParagraphStyle("header_clinic",  fontName=N, leading=L, fontSize=9,  textColor=CRE_BLUE, spaceAfter=2)
    header_contact = ParagraphStyle("header_contact", fontName=N, leading=L, fontSize=8,  textColor=MID_GRAY, spaceAfter=1)
    patient_label  = ParagraphStyle("patient_label",  fontName=B, leading=L, fontSize=10, textColor=BLACK,    spaceAfter=3)
    patient_value  = ParagraphStyle("patient_value",  fontName=N, leading=L, fontSize=10, textColor=BLACK,    spaceAfter=3)
    exam_title     = ParagraphStyle("exam_title",     fontName=B, leading=L, fontSize=12, textColor=BLACK,
                                    alignment=TA_CENTER, spaceAfter=6, spaceBefore=10)
    section_label  = ParagraphStyle("section_label",  fontName=B, leading=L, fontSize=10, textColor=BLACK,
                                    spaceAfter=2, spaceBefore=6)
    body_text      = ParagraphStyle("body_text",      fontName=N, leading=L, fontSize=10, textColor=BLACK,
                                    spaceAfter=4, alignment=TA_JUSTIFY)
    bullet         = ParagraphStyle("bullet",         fontName=N, leading=L, fontSize=10, textColor=BLACK,
                                    spaceAfter=2, leftIndent=16, bulletIndent=8)
    conclusion_text= ParagraphStyle("conclusion_text",fontName=B, leading=L, fontSize=10, textColor=BLACK,    spaceAfter=4)
    footer_text    = ParagraphStyle("footer_text",    fontName=N, leading=L, fontSize=8,  textColor=MID_GRAY, alignment=TA_CENTER)
    signature      = ParagraphStyle("signature",      fontName=N, leading=L, fontSize=10, textColor=BLACK,
                                    alignment=TA_RIGHT, spaceBefore=20)
    date_style     = ParagraphStyle("date_style",     fontName=N, leading=L, fontSize=10, textColor=BLACK,
                                    spaceAfter=6, spaceBefore=4)
    ai_watermark   = ParagraphStyle("ai_watermark",   fontName=B, leading=L, fontSize=8,  textColor=CRE_RED,
                                    alignment=TA_CENTER, spaceAfter=4)
    link_style     = ParagraphStyle("link_style",     fontName=N, leading=L, fontSize=8,  textColor=colors.blue, spaceAfter=2)

    return {
        "header_clinic": header_clinic,
        "header_contact": header_contact,
        "patient_label": patient_label,
        "patient_value": patient_value,
        "exam_title": exam_title,
        "section_label": section_label,
        "body_text": body_text,
        "bullet": bullet,
        "conclusion_text": conclusion_text,
        "footer_text": footer_text,
        "signature": signature,
        "date_style": date_style,
        "ai_watermark": ai_watermark,
        "link_style": link_style,
    }


def _watermark_canvas(c: canvas.Canvas, doc, is_ai_draft: bool):
    """Draw header, footer and optional AI watermark on every page."""
    c.saveState()
    w, h = A4

    # Footer line
    c.setStrokeColor(MID_GRAY)
    c.setLineWidth(0.5)
    c.line(1.5 * cm, 1.8 * cm, w - 1.5 * cm, 1.8 * cm)

    # Footer text
    c.setFont("Helvetica", 7)
    c.setFillColor(MID_GRAY)
    c.drawString(1.5 * cm, 1.2 * cm,
                 "Site Web: www.radiologie-emilie.com")
    c.drawRightString(w - 1.5 * cm, 1.2 * cm,
                      "Tel: +241 (0) 11 74 20 47 / +241 (0) 77 667 005")

    # AI watermark diagonal
    if is_ai_draft:
        c.setFont("Helvetica-Bold", 48)
        c.setFillColor(colors.Color(1, 0, 0, alpha=0.08))
        c.translate(w / 2, h / 2)
        c.rotate(45)
        c.drawCentredString(0, 0, "BROUILLON IA")

    c.restoreState()


def generate_compte_rendu(
    *,
    patient_name: str,
    birth_date: str,
    age: str,
    prescribing_doctor: str,
    radiologist: Optional[str],
    patient_id: str,
    study_date: str,
    exam_type: str,
    indication: str,
    technique: str,
    findings: str,
    conclusion: str,
    is_ai_draft: bool = True,
    logo_path: Optional[str] = None,
) -> bytes:
    """
    Generate a compte rendu PDF.
    Returns the PDF as bytes.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=1.8 * cm, rightMargin=1.8 * cm,
        topMargin=2 * cm, bottomMargin=2.5 * cm,
    )

    st = _styles()
    story = []

    # ── HEADER ───────────────────────────────────────────────────────────────
    logo_cell = ""
    if logo_path and Path(logo_path).exists():
        logo_cell = RLImage(logo_path, width=1.8 * cm, height=1.8 * cm)

    clinic_info = [
        Paragraph("<b>CENTRE RADIOLOGIE EMILIE RADIOTEAMS AFRIQUE</b>", st["header_clinic"]),
        Paragraph("+241 (0) 11 74 20 47 / +241 (0) 77 667 005", st["header_contact"]),
        Paragraph("e-mail : info@radiologie-emilie.com", st["header_contact"]),
        Paragraph("Face Mbolo, Boulevard Triomphal", st["header_contact"]),
        Paragraph("BP : 13 151", st["header_contact"]),
    ]

    header_table = Table(
        [[logo_cell, clinic_info]],
        colWidths=[2.2 * cm, None],
    )
    header_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(header_table)
    story.append(HRFlowable(width="100%", thickness=1, color=CRE_BLUE, spaceAfter=6))

    # ── AI DRAFT BANNER ───────────────────────────────────────────────────────
    if is_ai_draft:
        story.append(Paragraph(
            "⚠ COMPTE RENDU GÉNÉRÉ PAR INTELLIGENCE ARTIFICIELLE — "
            "EN ATTENTE DE VALIDATION PAR UN RADIOLOGUE",
            st["ai_watermark"]
        ))
        story.append(HRFlowable(width="100%", thickness=0.5, color=CRE_RED, spaceAfter=6))

    # ── PATIENT INFO ─────────────────────────────────────────────────────────
    def info_row(label, value):
        return [
            Paragraph(f"<u><b>{label}</b></u> :", st["patient_label"]),
            Paragraph(value, st["patient_value"]),
        ]

    info_data = [
        info_row("NOM & Prénom du patient", patient_name.upper()),
        info_row("Date de Naissance", birth_date),
        info_row("Age", age),
        info_row("Médecin prescripteur", prescribing_doctor),
    ]
    if radiologist:
        info_data.append(info_row("Radiologue", radiologist))

    # Login link
    pid_clean = patient_id.replace("P", "P")
    dob_url = birth_date  # DD/MM/YYYY
    link = f"https://imagerie-gabon.com/index.php?login={patient_id}&password={dob_url}"
    info_data.append([
        Paragraph("<u><b>Lien vers les résultats</b></u> :", st["patient_label"]),
        Paragraph(f'<link href="{link}">{link}</link>', st["link_style"]),
    ])

    info_table = Table(info_data, colWidths=[5.5 * cm, None])
    info_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 4 * mm))

    # ── DATE ─────────────────────────────────────────────────────────────────
    story.append(Paragraph(study_date, st["date_style"]))

    # ── EXAM TITLE ───────────────────────────────────────────────────────────
    story.append(Paragraph(f"<u><b>{exam_type.upper()}</b></u>", st["exam_title"]))

    # ── INDICATION ───────────────────────────────────────────────────────────
    story.append(Paragraph(
        f"<u><b>Indication</b></u> : {indication}", st["body_text"]
    ))

    # ── TECHNIQUE ────────────────────────────────────────────────────────────
    story.append(Paragraph(
        f"<u><b>Technique</b></u> : {technique}", st["body_text"]
    ))
    story.append(Spacer(1, 4 * mm))

    # ── RÉSULTATS ────────────────────────────────────────────────────────────
    story.append(Paragraph("<u><b>RÉSULTAT :</b></u>", st["section_label"]))
    story.append(Spacer(1, 2 * mm))

    for line in findings.strip().split("\n"):
        line = line.strip()
        if not line:
            story.append(Spacer(1, 2 * mm))
            continue
        if line.startswith("-") or line.startswith("–") or line.startswith("•"):
            story.append(Paragraph(f"&nbsp;&nbsp;&nbsp;&nbsp;– {line.lstrip('-–•').strip()}", st["bullet"]))
        else:
            story.append(Paragraph(line, st["body_text"]))

    story.append(Spacer(1, 6 * mm))

    # ── CONCLUSION ───────────────────────────────────────────────────────────
    story.append(Paragraph("<u><b>CONCLUSION :</b></u>", st["section_label"]))
    story.append(Spacer(1, 2 * mm))
    for line in conclusion.strip().split("\n"):
        line = line.strip()
        if line:
            story.append(Paragraph(f"<b>{line}</b>", st["conclusion_text"]))

    # ── SIGNATURE ────────────────────────────────────────────────────────────
    story.append(Spacer(1, 10 * mm))
    sig_name = radiologist if radiologist else ("IA — Validation radiologiste requise" if is_ai_draft else "")
    if sig_name:
        story.append(Paragraph(sig_name, st["signature"]))

    def make_canvas(c, doc):
        _watermark_canvas(c, doc, is_ai_draft)

    doc.build(story, onFirstPage=make_canvas, onLaterPages=make_canvas)
    return buf.getvalue()
