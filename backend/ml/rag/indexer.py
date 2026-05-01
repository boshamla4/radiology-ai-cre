"""
Indexes training compte rendus into ChromaDB for RAG retrieval.
Run this once after adding new training data.
"""
import json
import re
from pathlib import Path
from typing import Optional

import chromadb
import pdfplumber
from sentence_transformers import SentenceTransformer

from app.config import settings


COLLECTION_NAME = "compte_rendus"
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  # supports French


def _extract_pdf_text(pdf_path: str) -> str:
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += (page.extract_text() or "") + "\n"
    return text.strip()


def _parse_compte_rendu(text: str) -> dict:
    """Extract structured fields from compte rendu text."""
    result = {
        "exam_type": "",
        "indication": "",
        "technique": "",
        "findings": "",
        "conclusion": "",
        "modality": "",
        "body_part": "",
    }

    lines = text.split("\n")

    # Detect exam type (usually all-caps line after patient info)
    for line in lines:
        line = line.strip()
        if re.match(r"^(TDM|IRM|SCANNER|TOMODENSITOMETRIE|RADIOGRAPHIE)", line, re.IGNORECASE):
            result["exam_type"] = line
            # Infer modality and body part
            upper = line.upper()
            result["modality"] = "CT" if any(w in upper for w in ["TDM", "SCANNER", "TOMODENSITO"]) else "MR"
            for part, keywords in {
                "HEAD": ["CEREBR", "CRAN", "TETE", "HEAD", "MAXILLO"],
                "ABDOMEN": ["ABDOMEN", "ABDO"],
                "THORAX": ["THORAX", "CHEST", "PULMON"],
                "KNEE": ["GENOU", "KNEE"],
                "SPINE": ["RACHIS", "VERTEBR", "LOMBAIRE", "CERVICAL"],
            }.items():
                if any(k in upper for k in keywords):
                    result["body_part"] = part
                    break
            break

    # Extract sections
    full = text
    for pattern, key in [
        (r"Indication\s*:\s*(.+?)(?=Technique|Résultat|RÉSULTAT|$)", "indication"),
        (r"Technique\s*:\s*(.+?)(?=Résultat|RÉSULTAT|Indication|$)", "technique"),
        (r"RÉSULTAT\s*:\s*(.+?)(?=CONCLUSION|$)", "findings"),
        (r"Résultat\s*s?\s*:\s*(.+?)(?=CONCLUSION|$)", "findings"),
        (r"CONCLUSION\s*:\s*(.+?)$", "conclusion"),
    ]:
        m = re.search(pattern, full, re.DOTALL | re.IGNORECASE)
        if m:
            result[key] = m.group(1).strip()

    return result


class CompteRenduIndexer:
    def __init__(self, chroma_path: Optional[str] = None):
        path = chroma_path or settings.CHROMA_PATH
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self.embedder = SentenceTransformer(EMBED_MODEL)

    def index_pdf(self, pdf_path: str, patient_id: str) -> Optional[str]:
        """Index a single compte rendu PDF. Returns chroma_id or None if failed."""
        try:
            text = _extract_pdf_text(pdf_path)
            if not text:
                return None

            parsed = _parse_compte_rendu(text)

            # Create embedding from the full text
            embedding = self.embedder.encode(text, convert_to_numpy=True)

            doc_id = f"cr_{patient_id}_{Path(pdf_path).stem}"

            self.collection.upsert(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[text],
                metadatas=[{
                    "patient_id": patient_id,
                    "exam_type": parsed["exam_type"],
                    "modality": parsed["modality"],
                    "body_part": parsed["body_part"],
                    "indication": parsed["indication"][:200],
                    "pdf_path": pdf_path,
                }],
            )
            return doc_id

        except Exception as e:
            print(f"Failed to index {pdf_path}: {e}")
            return None

    def index_dataset(self, dataset_path: Optional[str] = None) -> int:
        """Batch-index all compte rendus in the DATASET directory."""
        base = Path(dataset_path or settings.DATASET_PATH)
        count = 0
        for patient_dir in sorted(base.iterdir()):
            if not patient_dir.is_dir():
                continue
            for pdf_file in patient_dir.glob("*.pdf"):
                patient_id = patient_dir.name
                doc_id = self.index_pdf(str(pdf_file), patient_id)
                if doc_id:
                    count += 1
                    print(f"  Indexed {patient_id}: {pdf_file.name}")
        return count

    def retrieve_similar(self, query_text: str, n: int = 5,
                          modality_filter: Optional[str] = None,
                          body_part_filter: Optional[str] = None) -> list[dict]:
        """Retrieve n most similar compte rendus for RAG context."""
        count = self.collection.count()
        if count == 0:
            return []

        embedding = self.embedder.encode(query_text, convert_to_numpy=True)

        where = {}
        if modality_filter:
            where["modality"] = modality_filter
        if body_part_filter:
            where["body_part"] = body_part_filter

        kwargs = dict(query_embeddings=[embedding], n_results=min(n, count))
        if where:
            kwargs["where"] = where

        try:
            results = self.collection.query(**kwargs)
        except Exception:
            results = self.collection.query(
                query_embeddings=[embedding], n_results=min(n, count)
            )

        docs = []
        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i]
            docs.append({"text": doc, "metadata": meta,
                          "distance": results["distances"][0][i]})
        return docs


# Singleton
_indexer: Optional[CompteRenduIndexer] = None


def get_indexer() -> CompteRenduIndexer:
    global _indexer
    if _indexer is None:
        _indexer = CompteRenduIndexer()
    return _indexer


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Index CRE compte rendus into ChromaDB")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Path to DATASET root (default: from config)")
    parser.add_argument("--pdf", type=str, default=None,
                        help="Index a single PDF instead of the whole dataset")
    parser.add_argument("--patient-id", type=str, default=None,
                        help="Patient ID for single PDF indexing")
    args = parser.parse_args()

    indexer = CompteRenduIndexer()
    if args.pdf:
        pid = args.patient_id or Path(args.pdf).parent.name
        doc_id = indexer.index_pdf(args.pdf, pid)
        print(f"Indexed: {doc_id}")
    else:
        count = indexer.index_dataset(args.dataset)
        print(f"\nDone. {count} compte rendus indexed into ChromaDB.")
        print(f"Total in collection: {indexer.collection.count()}")
