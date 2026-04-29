@echo off
REM ─────────────────────────────────────────────────────────────
REM  MedSAM Pseudo-Labeling — One-command annotation of DATASET/
REM  Output: DATASET/pseudo_labels/ + manifest.json per patient
REM
REM  First run: pip install transformers accelerate
REM  Then:      double-click this .bat or run from terminal
REM ─────────────────────────────────────────────────────────────

cd /d "%~dp0"

echo.
echo  MedSAM Pseudo-Labeler — Centre Radiologie Emilie
echo  ──────────────────────────────────────────────────
echo  Dataset root : %~dp0DATASET
echo  Output dir   : %~dp0DATASET\pseudo_labels
echo  Slices/patient: 10
echo.

python -m backend.ml.medsam.pseudo_labeler ^
    --dataset-root DATASET ^
    --output-dir DATASET\pseudo_labels ^
    --slices 10 ^
    --device cpu ^
    --verbose

echo.
echo  Done. Check DATASET\pseudo_labels\manifest.json
echo  Upload the pseudo_labels\ folder to Google Colab for fine-tuning.
echo.
pause
