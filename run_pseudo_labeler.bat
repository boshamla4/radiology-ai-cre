@echo off
REM ─────────────────────────────────────────────────────────────
REM  MedSAM Pseudo-Labeling — Annotates all patients in DATASET/
REM  Output: DATASET/pseudo_labels/ + manifest.json per patient
REM
REM  Safe to re-run: already-labeled patients are skipped.
REM  First run: pip install transformers accelerate opencv-python
REM  Then:      double-click this .bat or run from terminal
REM ─────────────────────────────────────────────────────────────

cd /d "%~dp0"

echo.
echo  MedSAM Pseudo-Labeler — Centre Radiologie Emilie
echo  ──────────────────────────────────────────────────
echo  Dataset root  : %~dp0DATASET
echo  Output dir    : %~dp0DATASET\pseudo_labels
echo  Slices/patient: 10
echo  Skip existing : YES (re-run safe)
echo.

python -m backend.ml.medsam.pseudo_labeler ^
    --dataset-root DATASET ^
    --output-dir DATASET\pseudo_labels ^
    --slices 10 ^
    --device cpu ^
    --verbose

echo.
echo  Done. Check DATASET\pseudo_labels\manifest.json
echo  Next: run zip_pseudo_labels.bat then upload to Kaggle.
echo.
pause
