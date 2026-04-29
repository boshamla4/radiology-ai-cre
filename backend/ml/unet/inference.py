"""
U-Net inference pipeline.
Loads pre-trained weights, runs segmentation on DICOM slices,
returns structured findings + overlay images.
"""
import json
import base64
import io
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ml.unet.model import UNet, get_class_map, OVERLAY_COLORS
from app.models.schemas import FindingItem, SegmentationResult


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_SIZE = 512   # resize all slices to this before inference
NUM_CLASSES = 5


class SegmentationModel:
    def __init__(self, weights_path: Optional[str] = None):
        self.model = UNet(in_channels=1, num_classes=NUM_CLASSES).to(DEVICE)
        self.loaded = False

        if weights_path and Path(weights_path).exists():
            state = torch.load(weights_path, map_location=DEVICE)
            self.model.load_state_dict(state)
            self.model.eval()
            self.loaded = True
        else:
            # No weights yet — use random init as placeholder
            # Will be replaced once training on BraTS/CHAOS is done
            self.model.eval()

    def preprocess(self, img_array: np.ndarray) -> torch.Tensor:
        """Normalise and resize a uint8 2D array → (1,1,H,W) tensor."""
        img = img_array.astype(np.float32) / 255.0
        t = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        t = F.interpolate(t, size=(INPUT_SIZE, INPUT_SIZE), mode="bilinear", align_corners=False)
        return t.to(DEVICE)

    @torch.no_grad()
    def predict_slice(self, img_array: np.ndarray) -> np.ndarray:
        """Return (H,W) class index map at original resolution."""
        h, w = img_array.shape[:2]
        t = self.preprocess(img_array)
        logits = self.model(t)                       # (1, C, H, W)
        pred = logits.argmax(dim=1).squeeze(0)       # (H, W)
        pred_np = pred.cpu().numpy().astype(np.uint8)
        # Resize back to original
        pred_img = Image.fromarray(pred_np).resize((w, h), Image.NEAREST)
        return np.array(pred_img)

    def create_overlay(self, original: np.ndarray, mask: np.ndarray) -> Image.Image:
        """Blend original greyscale with colour-coded segmentation mask."""
        if original.ndim == 2:
            base = Image.fromarray(original).convert("RGBA")
        else:
            base = Image.fromarray(original).convert("RGBA")

        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        px = overlay.load()
        for y in range(base.size[1]):
            for x in range(base.size[0]):
                cls = int(mask[y, x])
                if cls < len(OVERLAY_COLORS) and OVERLAY_COLORS[cls][3] > 0:
                    px[x, y] = OVERLAY_COLORS[cls]

        return Image.alpha_composite(base, overlay).convert("RGB")

    def extract_findings(self, mask: np.ndarray, class_map: dict,
                         pixel_spacing: float = 0.5) -> list[FindingItem]:
        """Convert segmentation mask to structured finding objects."""
        findings = []
        for cls_idx, cls_name in class_map.items():
            if cls_idx == 0:
                continue
            region = (mask == cls_idx)
            pixel_count = region.sum()
            if pixel_count == 0:
                continue

            area_mm2 = pixel_count * (pixel_spacing ** 2)
            diameter_mm = float(np.sqrt(4 * area_mm2 / np.pi))

            # Find bounding box centre
            ys, xs = np.where(region)
            cx = float(xs.mean())
            cy = float(ys.mean())
            h, w = mask.shape
            location = _describe_location(cx / w, cy / h)

            is_pathological = cls_name in ("lesion",)
            confidence = 0.75 if self.loaded else 0.40

            findings.append(FindingItem(
                structure=cls_name.replace("_", " ").title(),
                location=location,
                size_mm=round(diameter_mm, 1) if is_pathological else None,
                description=_finding_description(cls_name, diameter_mm, location),
                is_pathological=is_pathological,
                confidence=confidence,
            ))
        return findings

    def run(self, slices: list[np.ndarray], body_part: str) -> SegmentationResult:
        """Full inference on a list of slices. Returns SegmentationResult."""
        class_map = get_class_map(body_part)
        all_findings = []
        representative_overlay = None

        mid = len(slices) // 2
        for i, sl in enumerate(slices):
            mask = self.predict_slice(sl)
            findings = self.extract_findings(mask, class_map)
            all_findings.extend(findings)

            if i == mid:
                overlay_img = self.create_overlay(sl, mask)
                representative_overlay = overlay_img

        # Deduplicate findings by structure (keep highest confidence)
        seen = {}
        for f in all_findings:
            key = f.structure
            if key not in seen or f.confidence > seen[key].confidence:
                seen[key] = f
        unique_findings = list(seen.values())

        # Encode overlay as base64 PNG
        buf = io.BytesIO()
        if representative_overlay:
            representative_overlay.save(buf, format="PNG")
        overlay_b64 = base64.b64encode(buf.getvalue()).decode()

        avg_confidence = (
            sum(f.confidence for f in unique_findings) / len(unique_findings)
            if unique_findings else 0.5
        )

        return SegmentationResult(
            findings=unique_findings,
            overlay_image_base64=overlay_b64,
            model_version="unet-v1.0",
            dice_score=None,  # filled during evaluation on labelled data
        )


def _describe_location(cx_norm: float, cy_norm: float) -> str:
    h = "gauche" if cx_norm < 0.5 else "droite"
    v = "antérieure" if cy_norm < 0.33 else ("postérieure" if cy_norm > 0.66 else "centrale")
    return f"région {v} {h}"


def _finding_description(cls_name: str, diameter_mm: float, location: str) -> str:
    descriptions = {
        "lesion": f"Lésion focale de {diameter_mm:.1f}mm détectée en {location}.",
        "brain_parenchyma": "Parenchyme cérébral visible.",
        "ventricles": "Système ventriculaire visualisé.",
        "skull": "Structures osseuses crâniennes présentes.",
        "liver": "Foie visualisé.",
        "spleen": "Rate visualisée.",
        "kidneys": "Reins visualisés.",
        "organ": "Organe cible visualisé.",
        "bone": "Structures osseuses présentes.",
        "other": "Structure identifiée.",
    }
    return descriptions.get(cls_name, f"Structure '{cls_name}' identifiée.")


# Singleton — loaded once at app startup
_model_instance: Optional[SegmentationModel] = None


def get_segmentation_model(weights_path: Optional[str] = None) -> SegmentationModel:
    global _model_instance
    if _model_instance is None:
        _model_instance = SegmentationModel(weights_path)
    return _model_instance
