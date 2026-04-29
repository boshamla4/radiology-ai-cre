"""
MedSAM pseudo-labeling pipeline for CRE clinical DICOM data.

Workflow:
  1. Load DICOM slices from a patient ZIP or folder
  2. Generate automatic bounding-box prompts via intensity thresholding
  3. Run MedSAM (SAM ViT-B fine-tuned on medical images) to produce masks
  4. Save masks as .npy arrays paired with their source slices
  5. Write a dataset manifest JSON for downstream U-Net fine-tuning

MedSAM weights (~375 MB) are downloaded automatically on first run from
the official HuggingFace release: wanglab/medsam-vit-base

Usage (CLI):
    python -m backend.ml.medsam.pseudo_labeler \
        --patient-zip DATASET/patient002/patient002.zip \
        --output-dir DATASET/pseudo_labels/patient002 \
        --slices 10

    python -m backend.ml.medsam.pseudo_labeler \
        --dataset-root DATASET \
        --output-dir DATASET/pseudo_labels \
        --slices 5
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

# ── lazy imports (heavy deps only needed at runtime) ──────────────────────────

def _import_torch():
    import torch
    return torch

def _import_cv2():
    import cv2
    return cv2

def _import_pydicom():
    import pydicom
    try:
        import pyjpegls  # noqa: F401 — registers JPEG-LS handler
    except ImportError:
        pass
    return pydicom

def _import_pil():
    from PIL import Image
    return Image

def _load_medsam(device: str, weights_path: Optional[Path] = None):
    """Load MedSAM model. Downloads weights if not cached."""
    torch = _import_torch()
    try:
        from transformers import SamModel, SamProcessor
        HF_AVAILABLE = True
    except ImportError:
        HF_AVAILABLE = False

    if HF_AVAILABLE:
        log.info("Loading MedSAM via HuggingFace transformers…")
        model = SamModel.from_pretrained("flaviagiammarino/medsam-vit-base").to(device)
        processor = SamProcessor.from_pretrained("flaviagiammarino/medsam-vit-base")
        model.eval()
        return model, processor, "hf"

    # Fallback: plain SAM from segment-anything package
    try:
        from segment_anything import sam_model_registry, SamPredictor
        if weights_path and weights_path.exists():
            sam = sam_model_registry["vit_b"](checkpoint=str(weights_path))
            sam.to(device)
            predictor = SamPredictor(sam)
            log.info("Loaded SAM vit-b from local weights: %s", weights_path)
            return sam, predictor, "sam"
    except ImportError:
        pass

    raise RuntimeError(
        "No SAM/MedSAM backend found.\n"
        "Install one of:\n"
        "  pip install transformers accelerate  (recommended)\n"
        "  pip install segment-anything         (requires manual weight download)"
    )


# ── DICOM loading ─────────────────────────────────────────────────────────────

def load_slices_from_zip(zip_path: Path, max_slices: int = 20) -> list[np.ndarray]:
    """Extract representative 2-D uint8 slices from a patient ZIP file."""
    pydicom = _import_pydicom()

    datasets = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        dcm_names = [n for n in zf.namelist() if not n.endswith("/")]
        for name in dcm_names:
            try:
                raw = zf.read(name)
                ds = pydicom.dcmread(io.BytesIO(raw))
                _ = ds.pixel_array  # force decode — skip non-image files
                datasets.append(ds)
            except Exception:
                continue

    if not datasets:
        raise ValueError(f"No valid DICOM slices found in {zip_path}")

    # Sort by slice position
    def _sort_key(ds):
        loc = getattr(ds, "SliceLocation", None)
        if loc is not None:
            return float(loc)
        inst = getattr(ds, "InstanceNumber", None)
        return int(inst) if inst is not None else 0

    datasets.sort(key=_sort_key)

    # Pick evenly-spaced slices
    if len(datasets) > max_slices:
        idxs = np.linspace(0, len(datasets) - 1, max_slices, dtype=int)
        datasets = [datasets[i] for i in idxs]

    slices = []
    for ds in datasets:
        arr = ds.pixel_array.astype(np.float32)

        # HU windowing for CT; min-max for MRI
        modality = getattr(ds, "Modality", "CT")
        if modality == "CT":
            c = float(getattr(ds, "WindowCenter", 40))
            w = float(getattr(ds, "WindowWidth", 400))
            if isinstance(c, list):
                c = c[0]
            if isinstance(w, list):
                w = w[0]
            lo, hi = c - w / 2, c + w / 2
            arr = np.clip(arr, lo, hi)
            arr = (arr - lo) / (hi - lo) * 255.0
        else:
            mn, mx = arr.min(), arr.max()
            if mx > mn:
                arr = (arr - mn) / (mx - mn) * 255.0

        slices.append(arr.astype(np.uint8))

    return slices


def load_slices_from_folder(folder: Path, max_slices: int = 20) -> list[np.ndarray]:
    """Load slices directly from a folder of DICOM files."""
    zips = list(folder.glob("*.zip"))
    if zips:
        return load_slices_from_zip(zips[0], max_slices)

    pydicom = _import_pydicom()
    datasets = []
    for f in sorted(folder.iterdir()):
        if f.suffix.lower() in {".dcm", ""}:
            try:
                ds = pydicom.dcmread(str(f))
                _ = ds.pixel_array
                datasets.append(ds)
            except Exception:
                continue

    if not datasets:
        raise ValueError(f"No valid DICOM files in {folder}")

    def _sort_key(ds):
        loc = getattr(ds, "SliceLocation", None)
        return float(loc) if loc is not None else int(getattr(ds, "InstanceNumber", 0))

    datasets.sort(key=_sort_key)
    if len(datasets) > max_slices:
        idxs = np.linspace(0, len(datasets) - 1, max_slices, dtype=int)
        datasets = [datasets[i] for i in idxs]

    slices = []
    for ds in datasets:
        arr = ds.pixel_array.astype(np.float32)
        mn, mx = arr.min(), arr.max()
        if mx > mn:
            arr = (arr - mn) / (mx - mn) * 255.0
        slices.append(arr.astype(np.uint8))
    return slices


# ── Prompt generation (automatic bounding boxes) ─────────────────────────────

def _auto_bboxes(gray: np.ndarray, n_boxes: int = 3) -> list[list[int]]:
    """
    Generate bounding-box prompts for SAM from a grayscale image.

    Strategy:
      1. Otsu threshold to find foreground
      2. Morphological close to merge nearby blobs
      3. Find connected-component bounding boxes
      4. Return the n_boxes largest components (likely anatomy, not noise)
    """
    cv2 = _import_cv2()

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)

    boxes = []
    for lbl in range(1, num_labels):  # skip background (0)
        x, y, w, h, area = stats[lbl]
        if area < 500:  # ignore tiny noise
            continue
        boxes.append((area, [x, y, x + w, y + h]))

    boxes.sort(key=lambda t: t[0], reverse=True)
    return [b for _, b in boxes[:n_boxes]]


# ── MedSAM inference ──────────────────────────────────────────────────────────

def _run_hf_medsam(model, processor, image_rgb: np.ndarray, boxes: list[list[int]]) -> np.ndarray:
    """Run HuggingFace MedSAM and return a combined binary mask."""
    torch = _import_torch()
    Image = _import_pil()

    pil_img = Image.fromarray(image_rgb)
    combined = np.zeros(image_rgb.shape[:2], dtype=np.uint8)

    for box in boxes:
        inputs = processor(
            images=pil_img,
            input_boxes=[[[box]]],
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)

        masks = processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )
        # masks[0] shape: (1, num_masks, H, W)
        best_mask = masks[0][0, 0].numpy().astype(np.uint8)
        combined = np.maximum(combined, best_mask)

    return combined


def _run_sam_predictor(predictor, image_rgb: np.ndarray, boxes: list[list[int]]) -> np.ndarray:
    """Run plain SAM predictor and return a combined binary mask."""
    import torch
    predictor.set_image(image_rgb)
    combined = np.zeros(image_rgb.shape[:2], dtype=np.uint8)

    for box in boxes:
        box_arr = np.array(box)
        masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box_arr[None, :],
            multimask_output=False,
        )
        if len(masks):
            combined = np.maximum(combined, masks[0].astype(np.uint8))

    return combined


# ── Main labeling logic ───────────────────────────────────────────────────────

def pseudo_label_slices(
    slices: list[np.ndarray],
    model,
    processor,
    backend: str,
    n_boxes: int = 3,
) -> list[np.ndarray]:
    """Return a list of binary mask arrays (same shape as input slices)."""
    masks = []
    for i, gray in enumerate(slices):
        # SAM expects RGB
        rgb = np.stack([gray, gray, gray], axis=-1)
        boxes = _auto_bboxes(gray, n_boxes=n_boxes)

        if not boxes:
            log.warning("Slice %d: no foreground detected, skipping.", i)
            masks.append(np.zeros_like(gray))
            continue

        if backend == "hf":
            mask = _run_hf_medsam(model, processor, rgb, boxes)
        else:
            mask = _run_sam_predictor(processor, rgb, boxes)  # processor is predictor here

        masks.append(mask)
        log.info("Slice %d/%d — %d box(es) → mask coverage %.1f%%",
                 i + 1, len(slices), len(boxes), mask.mean() * 100)

    return masks


def save_pseudo_labels(
    slices: list[np.ndarray],
    masks: list[np.ndarray],
    output_dir: Path,
    patient_id: str,
) -> list[dict]:
    """
    Save each (slice, mask) pair and return manifest entries.
    Files:
        <output_dir>/slices/patient_id_NNNN.npy   — normalised [0,1] float32
        <output_dir>/masks/patient_id_NNNN.npy    — binary uint8 {0,1}
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    slices_dir = output_dir / "slices"
    masks_dir = output_dir / "masks"
    slices_dir.mkdir(exist_ok=True)
    masks_dir.mkdir(exist_ok=True)

    entries = []
    for i, (sl, mk) in enumerate(zip(slices, masks)):
        tag = f"{patient_id}_{i:04d}"
        slice_path = slices_dir / f"{tag}.npy"
        mask_path = masks_dir / f"{tag}.npy"

        np.save(slice_path, (sl.astype(np.float32) / 255.0))
        np.save(mask_path, mk.astype(np.uint8))

        entries.append({
            "slice": str(slice_path),
            "mask": str(mask_path),
            "patient_id": patient_id,
            "slice_idx": i,
        })

    return entries


def label_patient(
    patient_zip_or_folder: Path,
    output_dir: Path,
    model,
    processor,
    backend: str,
    max_slices: int = 20,
    patient_id: Optional[str] = None,
) -> list[dict]:
    """Full pipeline for one patient. Returns manifest entries."""
    pid = patient_id or patient_zip_or_folder.stem

    if patient_zip_or_folder.is_file() and patient_zip_or_folder.suffix == ".zip":
        slices = load_slices_from_zip(patient_zip_or_folder, max_slices)
    else:
        slices = load_slices_from_folder(patient_zip_or_folder, max_slices)

    log.info("Patient %s: %d slices loaded", pid, len(slices))
    masks = pseudo_label_slices(slices, model, processor, backend)
    entries = save_pseudo_labels(slices, masks, output_dir / pid, pid)
    log.info("Patient %s: saved %d slice/mask pairs to %s", pid, len(entries), output_dir / pid)
    return entries


def label_dataset(
    dataset_root: Path,
    output_dir: Path,
    model,
    processor,
    backend: str,
    max_slices: int = 10,
) -> Path:
    """
    Label all patients in dataset_root.
    Looks for:
      - dataset_root/patientXXX/*.zip
      - dataset_root/patientXXX/  (folder of .dcm files)
    Returns path to written manifest.json.
    """
    manifest = []
    patient_dirs = sorted(
        d for d in dataset_root.iterdir()
        if d.is_dir() and d.name.lower().startswith("patient")
    )

    if not patient_dirs:
        raise ValueError(f"No patient folders found in {dataset_root}")

    for patient_dir in patient_dirs:
        try:
            entries = label_patient(
                patient_dir, output_dir, model, processor, backend,
                max_slices=max_slices, patient_id=patient_dir.name,
            )
            manifest.extend(entries)
        except Exception as exc:
            log.warning("Skipping %s: %s", patient_dir.name, exc)

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log.info("Dataset labeling complete. %d entries → %s", len(manifest), manifest_path)
    return manifest_path


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="MedSAM pseudo-labeling for CRE DICOM data")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--patient-zip", type=Path, help="Single patient ZIP file")
    src.add_argument("--patient-folder", type=Path, help="Single patient DICOM folder")
    src.add_argument("--dataset-root", type=Path, help="Root folder containing patientXXX dirs")
    p.add_argument("--output-dir", type=Path, required=True, help="Where to save masks + manifest")
    p.add_argument("--slices", type=int, default=10, help="Max slices per patient (default 10)")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Inference device")
    p.add_argument("--weights", type=Path, default=None,
                   help="Optional local SAM weights .pth (only for segment-anything backend)")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main():
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    log.info("Loading MedSAM model on %s…", args.device)
    model, processor, backend = _load_medsam(args.device, args.weights)
    log.info("Backend: %s", backend)

    if args.patient_zip:
        entries = label_patient(
            args.patient_zip, args.output_dir, model, processor, backend,
            max_slices=args.slices,
        )
        manifest_path = args.output_dir / args.patient_zip.stem / "manifest.json"
        manifest_path.write_text(json.dumps(entries, indent=2))
        log.info("Done. %d entries → %s", len(entries), manifest_path)

    elif args.patient_folder:
        entries = label_patient(
            args.patient_folder, args.output_dir, model, processor, backend,
            max_slices=args.slices,
        )
        manifest_path = args.output_dir / args.patient_folder.name / "manifest.json"
        manifest_path.write_text(json.dumps(entries, indent=2))
        log.info("Done. %d entries → %s", len(entries), manifest_path)

    else:
        manifest_path = label_dataset(
            args.dataset_root, args.output_dir, model, processor, backend,
            max_slices=args.slices,
        )
        log.info("Full dataset manifest: %s", manifest_path)


if __name__ == "__main__":
    main()
