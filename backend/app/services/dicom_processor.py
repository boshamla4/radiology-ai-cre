import io
import zipfile
from pathlib import Path
from typing import Optional
import numpy as np
import pydicom
from PIL import Image

from app.models.schemas import DicomMetadata, ModalityEnum


def _parse_date(date_str: str) -> str:
    """Convert DICOM date YYYYMMDD → DD/MM/YYYY."""
    if not date_str or len(date_str) < 8:
        return date_str or ""
    try:
        return f"{date_str[6:8]}/{date_str[4:6]}/{date_str[:4]}"
    except Exception:
        return date_str


def _parse_name(name) -> str:
    """Convert DICOM PatientName to readable string."""
    if name is None:
        return ""
    s = str(name).replace("^", " ").strip()
    return s


def _find_representative_slice(datasets: list) -> pydicom.Dataset:
    """Return the middle slice as representative."""
    return datasets[len(datasets) // 2]


def extract_metadata_from_zip(zip_bytes: bytes) -> tuple[DicomMetadata, list[pydicom.Dataset]]:
    """
    Read a DICOM zip file, extract metadata from the first valid slice,
    and return all datasets sorted by slice location.
    """
    datasets = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for name in zf.namelist():
            if name.endswith("/") or name.endswith(".txt") or name.endswith(".pdf"):
                continue
            try:
                with zf.open(name) as f:
                    ds = pydicom.dcmread(f, force=True)
                    if hasattr(ds, "PixelData"):
                        datasets.append(ds)
            except Exception:
                continue

    if not datasets:
        raise ValueError("No valid DICOM files found in the zip archive.")

    # Sort by slice location (InstanceNumber fallback)
    def sort_key(ds):
        if hasattr(ds, "SliceLocation"):
            return float(ds.SliceLocation)
        if hasattr(ds, "InstanceNumber"):
            return int(ds.InstanceNumber)
        return 0

    datasets.sort(key=sort_key)
    ref = datasets[0]

    modality_str = getattr(ref, "Modality", "OTHER")
    modality = ModalityEnum.CT if modality_str == "CT" else (
        ModalityEnum.MR if modality_str == "MR" else ModalityEnum.OTHER
    )

    body_part = getattr(ref, "BodyPartExamined", None)
    if not body_part:
        # Infer from study/protocol description
        desc = (
            str(getattr(ref, "StudyDescription", "")) + " " +
            str(getattr(ref, "SeriesDescription", "")) + " " +
            str(getattr(ref, "ProtocolName", ""))
        ).upper()
        if any(w in desc for w in ["BRAIN", "CEREBR", "TETE", "HEAD", "CRANIO"]):
            body_part = "HEAD"
        elif any(w in desc for w in ["ABDOMEN", "ABDO"]):
            body_part = "ABDOMEN"
        elif any(w in desc for w in ["THORAX", "CHEST", "PULMON"]):
            body_part = "THORAX"
        elif any(w in desc for w in ["GENOU", "KNEE"]):
            body_part = "KNEE"
        elif any(w in desc for w in ["FACE", "MAXILLO", "FACIAL"]):
            body_part = "FACE"
        else:
            body_part = "UNKNOWN"

    metadata = DicomMetadata(
        patient_id=str(getattr(ref, "PatientID", "UNKNOWN")),
        patient_name=_parse_name(getattr(ref, "PatientName", "")),
        birth_date=_parse_date(str(getattr(ref, "PatientBirthDate", ""))),
        age=str(getattr(ref, "PatientAge", "")),
        sex=str(getattr(ref, "PatientSex", "")),
        modality=modality,
        study_date=_parse_date(str(getattr(ref, "StudyDate", ""))),
        study_description=str(getattr(ref, "StudyDescription", "")),
        series_description=str(getattr(ref, "SeriesDescription", "")),
        protocol_name=str(getattr(ref, "ProtocolName", "")),
        body_part=body_part,
        institution=str(getattr(ref, "InstitutionName", "CIM Emilie")),
        accession_number=str(getattr(ref, "AccessionNumber", "")),
        slice_count=len(datasets),
    )

    return metadata, datasets


def dicom_to_png(ds: pydicom.Dataset, window_center: Optional[float] = None,
                 window_width: Optional[float] = None) -> np.ndarray:
    """
    Convert a DICOM slice to a displayable uint8 numpy array.
    Applies windowing (HU for CT, intensity for MR).
    """
    pixel_array = ds.pixel_array.astype(np.float32)

    # Apply rescale
    slope = float(getattr(ds, "RescaleSlope", 1))
    intercept = float(getattr(ds, "RescaleIntercept", 0))
    pixel_array = pixel_array * slope + intercept

    # Windowing
    if window_center is None:
        window_center = float(getattr(ds, "WindowCenter", 40))
        if isinstance(window_center, pydicom.multival.MultiValue):
            window_center = float(window_center[0])
    if window_width is None:
        window_width = float(getattr(ds, "WindowWidth", 400))
        if isinstance(window_width, pydicom.multival.MultiValue):
            window_width = float(window_width[0])

    low = window_center - window_width / 2
    high = window_center + window_width / 2
    pixel_array = np.clip(pixel_array, low, high)
    pixel_array = ((pixel_array - low) / (high - low) * 255).astype(np.uint8)

    return pixel_array


def extract_representative_slices(datasets: list[pydicom.Dataset],
                                   n: int = 5) -> list[np.ndarray]:
    """Return n evenly spaced slices as uint8 numpy arrays."""
    total = len(datasets)
    indices = np.linspace(0, total - 1, min(n, total), dtype=int)
    return [dicom_to_png(datasets[i]) for i in indices]


def array_to_pil(arr: np.ndarray) -> Image.Image:
    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L").convert("RGB")
    return Image.fromarray(arr)
