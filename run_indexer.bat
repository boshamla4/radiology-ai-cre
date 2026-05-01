@echo off
REM ─────────────────────────────────────────────────────────────
REM  ChromaDB Indexer — Indexes all compte rendu PDFs in DATASET/
REM  for RAG-assisted report generation.
REM
REM  Run this every time you add new patient PDFs to DATASET/.
REM  Uses upsert — safe to re-run, no duplicates.
REM  Requires the backend to be installed: pip install -r backend/requirements.txt
REM ─────────────────────────────────────────────────────────────

cd /d "%~dp0\backend"

echo.
echo  ChromaDB Indexer — Centre Radiologie Emilie
echo  ────────────────────────────────────────────
echo  Dataset : %~dp0DATASET
echo  ChromaDB: %~dp0backend\chroma_db
echo.

python -m ml.rag.indexer

echo.
pause
