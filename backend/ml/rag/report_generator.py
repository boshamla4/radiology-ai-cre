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


SYSTEM_PROMPT = """Tu es un radiologue expert assistant travaillant au Centre de Radiologie Emilie, Libreville, Gabon.
Tu génères des comptes rendus radiologiques en français, dans un style médical professionnel et précis.
Ton rôle est d'ASSISTER le radiologue, pas de remplacer son jugement clinique.
Le compte rendu que tu génères est un BROUILLON IA qui doit être vérifié et validé par un radiologue.

Règles de rédaction :
- Utilise le français médical précis et professionnel
- Structure : Résultats (observations détaillées) puis Conclusion (synthèse courte)
- Pour les examens normaux : liste tous les éléments évalués comme normaux
- Pour les anomalies : précise localisation, taille si disponible, caractéristiques
- Ne pose jamais de diagnostic définitif, utilise des formulations comme "compatible avec", "évocateur de", "à confirmer"
- Garde le style de la radiologie francophone africaine"""


def _build_prompt(
    metadata: DicomMetadata,
    indication: str,
    findings: list[FindingItem],
    similar_cases: list[dict],
    exam_type: str,
) -> str:
    # Format segmentation findings
    findings_text = ""
    if findings:
        for f in findings:
            flag = " [PATHOLOGIQUE]" if f.is_pathological else ""
            size = f" ({f.size_mm}mm)" if f.size_mm else ""
            findings_text += f"- {f.structure}{size} en {f.location}: {f.description}{flag}\n"
    else:
        findings_text = "- Analyse de segmentation en cours (modèle en phase d'initialisation)\n"

    # Format similar cases as few-shot examples
    examples_text = ""
    if similar_cases:
        examples_text = "\n\n=== EXEMPLES DE COMPTES RENDUS SIMILAIRES (pour le style et la terminologie) ===\n"
        for i, case in enumerate(similar_cases[:3], 1):
            examples_text += f"\n--- Exemple {i} ---\n{case['text'][:600]}\n"

    return f"""Génère un compte rendu radiologique pour l'examen suivant.

=== INFORMATIONS PATIENT ===
Patient : {metadata.patient_name}, {metadata.age}, {metadata.sex}
Examen : {exam_type}
Modalité : {metadata.modality}
Région : {metadata.body_part or 'Non spécifiée'}
Indication clinique : {indication}
Protocole : {metadata.protocol_name}

=== RÉSULTATS DE SEGMENTATION IA ===
{findings_text}
{examples_text}

=== INSTRUCTIONS ===
Génère UNIQUEMENT les deux sections suivantes (sans répéter les informations patient) :

RÉSULTAT :
[Rédige les observations détaillées en français médical professionnel,
 basé sur les résultats de segmentation et les exemples similaires.
 Utilise des tirets pour chaque observation.]

CONCLUSION :
[Rédige une conclusion courte en une ou deux phrases.]"""


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
    # Try to find RÉSULTAT section
    m_findings = re.search(
        r"RÉSULTAT\s*:?\s*\n?(.*?)(?=CONCLUSION|$)", text,
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
            messages = [{"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}]

            # Add image if available and model supports vision
            if overlay_base64 and settings.OLLAMA_MODEL in ("llava", "llava:13b", "llava:7b"):
                messages[-1]["images"] = [overlay_base64]

            response = self.client.chat(
                model=settings.OLLAMA_MODEL,
                messages=messages,
                options={"temperature": 0.3, "num_predict": 800},
            )
            generated = response.message.content
            findings_text, conclusion_text = _parse_generated_text(generated)

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
