"""
Generates draft compte rendu text using Ollama (LLaVA multimodal model).
The segmentation findings + similar cases (RAG) guide the generation.
"""
import base64
import json
from typing import Optional

import ollama

from app.config import settings
from app.models.schemas import FindingItem, DicomMetadata
from ml.rag.indexer import get_indexer


SYSTEM_PROMPT = """You are an expert radiology assistant at Centre Radiologie Emilie, Libreville, Gabon.
Write radiology reports in French using professional medical style.
Your role is to ASSIST the radiologist, not replace clinical judgment.
The report you generate is an AI DRAFT to be verified by a radiologist.

Rules:
- Use precise professional French medical terminology
- Structure: RESULTAT (detailed findings) then CONCLUSION (short summary)
- For normal exams: list all evaluated elements as normal
- For anomalies: specify location, size if available, characteristics
- Never state a definitive diagnosis; use phrases like "compatible avec", "evocateur de", "a confirmer"
"""


def _build_prompt(
    metadata: DicomMetadata,
    indication: str,
    findings: list[FindingItem],
    similar_cases: list[dict],
    exam_type: str,
) -> str:
    # Format segmentation findings (ASCII-safe labels)
    findings_lines = ""
    if findings:
        for f in findings:
            flag = " [PATHOLOGIQUE]" if f.is_pathological else ""
            size = f" ({f.size_mm}mm)" if f.size_mm else ""
            findings_lines += f"- {f.structure}{size} region {f.location}: {f.description}{flag}\n"
    else:
        findings_lines = "- Examen en cours d'analyse par le systeme IA.\n"

    # One short RAG example for style guidance (keep prompt short)
    example_text = ""
    if similar_cases:
        example_text = f"\nExample compte rendu (style reference):\n{similar_cases[0]['text'][:400]}\n"

    modality = metadata.modality.value if hasattr(metadata.modality, 'value') else str(metadata.modality)

    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Generate a radiology report in French for this exam.\n\n"
        f"Patient: {metadata.patient_name}, age {metadata.age}, sex {metadata.sex}\n"
        f"Exam: {exam_type} | Modality: {modality} | Region: {metadata.body_part or 'unknown'}\n"
        f"Clinical indication: {indication}\n"
        f"Protocol: {metadata.protocol_name}\n\n"
        f"AI segmentation findings:\n{findings_lines}"
        f"{example_text}\n"
        f"Write ONLY the two sections below in French:\n\n"
        f"RESULTAT :\n"
        f"[Write detailed observations using medical French, one dash per finding]\n\n"
        f"CONCLUSION :\n"
        f"[Write a short 1-2 sentence summary]\n"
    )


def _infer_exam_type(metadata: DicomMetadata) -> str:
    """Determine the exam title from DICOM metadata."""
    modality = metadata.modality.value if hasattr(metadata.modality, 'value') else str(metadata.modality)
    body = (metadata.body_part or "").upper()
    protocol = (metadata.protocol_name or "").upper()
    desc = (metadata.study_description or "").upper()

    prefix = "TDM" if modality == "CT" else "IRM"

    body_names = {
        "HEAD": "CEREBRALE",
        "BRAIN": "CEREBRALE",
        "ABDOMEN": "ABDOMINALE",
        "THORAX": "THORACIQUE",
        "KNEE": "DU GENOU",
        "SPINE": "DU RACHIS",
        "FACE": "MAXILLO-FACIALE",
    }

    for key, name in body_names.items():
        if key in body or key in protocol or key in desc:
            return f"{prefix} {name}"

    return f"{prefix} {body or 'CORPS ENTIER'}"


def _infer_technique(metadata: DicomMetadata) -> str:
    protocol = (metadata.protocol_name or "").upper()
    series = (metadata.series_description or "").upper()
    combined = protocol + " " + series

    if "AVEC" in combined or "APC" in combined or "INJECTION" in combined or "CONTRASTE" in combined:
        return "Avec injection de produit de contraste intraveineux (APC)."
    elif "SANS" in combined or "SPC" in combined or "NATIV" in combined:
        return "Sans injection de produit de contraste (SPC)."
    elif "SPC" in combined and "APC" in combined:
        return "Étude sans puis avec injection de produit de contraste intraveineux (SPC et APC)."
    else:
        return "Acquisition volumique multiplanaire."


def _parse_generated_text(text: str) -> tuple[str, str]:
    """Split generated text into findings and conclusion."""
    findings = ""
    conclusion = ""

    import re
    # Match RESULTAT or RÉSULTAT (LLaVA may or may not add accent)
    m_findings = re.search(
        r"R[EÉ]SULTAT\s*:?\s*\n?(.*?)(?=CONCLUSION|$)", text,
        re.DOTALL | re.IGNORECASE
    )
    m_conclusion = re.search(
        r"CONCLUSION\s*:?\s*\n?(.*?)$", text,
        re.DOTALL | re.IGNORECASE
    )

    if m_findings:
        findings = m_findings.group(1).strip()
    if m_conclusion:
        conclusion = m_conclusion.group(1).strip()

    if not findings:
        findings = text.strip()
    if not conclusion:
        conclusion = "À valider par le radiologue."

    return findings, conclusion


class ReportGenerator:
    def __init__(self):
        self.client = ollama.Client(host=settings.OLLAMA_HOST)
        self.indexer = get_indexer()

    def _check_ollama(self) -> bool:
        try:
            self.client.list()
            return True
        except Exception:
            return False

    def generate(
        self,
        metadata: DicomMetadata,
        indication: str,
        findings: list[FindingItem],
        overlay_base64: Optional[str] = None,
    ) -> tuple[str, str, str, int, float]:
        """
        Generate report text using RAG + Ollama.

        Returns:
            (exam_type, technique, findings_text, conclusion_text,
             similar_count, confidence)
        """
        exam_type = _infer_exam_type(metadata)
        technique = _infer_technique(metadata)

        # RAG: retrieve similar cases
        query = f"{exam_type} {indication} {metadata.body_part or ''}"
        similar = self.indexer.retrieve_similar(
            query, n=5,
            modality_filter=metadata.modality.value if hasattr(metadata.modality, 'value') else None,
        )

        prompt = _build_prompt(metadata, indication, findings, similar, exam_type)

        if not self._check_ollama():
            # Fallback: template-based report when Ollama is not running
            findings_text, conclusion_text = self._template_fallback(findings, metadata)
            return exam_type, technique, findings_text, conclusion_text, len(similar), 0.45

        try:
            response = self.client.generate(
                model=settings.OLLAMA_MODEL,
                prompt=prompt,
                options={"temperature": 0.3, "num_predict": 1200, "num_ctx": 4096},
            )
            generated = response.response
            print(f"[LLaVA raw output ({len(generated)} chars)]: {generated[:300]!r}")

            findings_text, conclusion_text = _parse_generated_text(generated)

            # If LLaVA returned empty content, use template fallback
            if not findings_text.strip():
                print("[LLaVA] Empty response — using template fallback")
                findings_text, conclusion_text = self._template_fallback(findings, metadata)
                return exam_type, technique, findings_text, conclusion_text, len(similar), 0.50

            confidence = min(0.90, 0.55 + len(similar) * 0.07)
            return exam_type, technique, findings_text, conclusion_text, len(similar), confidence

        except Exception as e:
            print(f"Ollama error: {e}")
            findings_text, conclusion_text = self._template_fallback(findings, metadata)
            return exam_type, technique, findings_text, conclusion_text, len(similar), 0.40

    def _template_fallback(self, findings: list[FindingItem],
                           metadata: DicomMetadata) -> tuple[str, str]:
        """Rule-based fallback when Ollama is unavailable."""
        pathological = [f for f in findings if f.is_pathological]
        normal = [f for f in findings if not f.is_pathological]

        lines = []
        for f in normal:
            lines.append(f"- {f.structure} : aspect normal.")
        for f in pathological:
            lines.append(f"- {f.description}")

        if not lines:
            lines.append("- Étude en cours d'analyse. Résultats en attente de validation.")

        findings_text = "\n".join(lines)
        conclusion_text = (
            "Anomalie(s) détectée(s) par le système d'IA, nécessitant validation radiologique."
            if pathological else
            "Examen d'aspect normal selon l'analyse IA préliminaire. À confirmer par le radiologue."
        )
        return findings_text, conclusion_text


_generator: Optional[ReportGenerator] = None


def get_report_generator() -> ReportGenerator:
    global _generator
    if _generator is None:
        _generator = ReportGenerator()
    return _generator
