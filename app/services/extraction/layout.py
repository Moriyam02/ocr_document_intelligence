import logging
import re
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class BoundingBox(BaseModel):
    x_min: int
    y_min: int
    x_max: int
    y_max: int


class FieldAnnotation(BaseModel):
    field_name: str
    text: str
    bbox: BoundingBox
    confidence: float


class SpatialLayoutAnalyzer:
    """Handles spatial layout geometry, token proximity matching, and visual annotation rendering."""

    @staticmethod
    def normalize_bbox(
        bbox: List[int], original_dim: Tuple[int, int]
    ) -> BoundingBox:
        """Converts raw [x, y, w, h] to absolute pixel coordinates [x_min, y_min, x_max, y_max]."""
        x, y, w, h = bbox
        return BoundingBox(
            x_min=max(0, x),
            y_min=max(0, y),
            x_max=min(original_dim[0], x + w),
            y_max=min(original_dim[1], y + h),
        )

    @staticmethod
    def find_value_near_labels(
        tokens: List[Dict[str, Any]],
        label_keywords: List[str],
        x_tolerance: float = 150.0,
        y_tolerance: float = 15.0,
    ) -> Optional[str]:
        """
        Finds target candidate token values positioned immediately to the right 
        or near specified header key label tokens.
        """
        for i, token in enumerate(tokens):
            text = token.get("text", "").lower().strip()
            if any(kw in text for kw in label_keywords):
                # 1. Immediate next token fallback
                if i + 1 < len(tokens):
                    next_token = tokens[i + 1]
                    val = next_token.get("text", "").strip()
                    if re.match(r"^[A-Z0-9\-_]{3,20}$", val, re.IGNORECASE):
                        return val

                # 2. Bounding Box Proximity Lookup (To the Right)
                label_bbox = token.get("bbox")
                if label_bbox and len(label_bbox) == 4:
                    lx, ly, lw, lh = label_bbox
                    for candidate in tokens:
                        if candidate == token:
                            continue
                        c_bbox = candidate.get("bbox")
                        if c_bbox and len(c_bbox) == 4:
                            cx, cy, cw, ch = c_bbox
                            # Check if token is horizontally adjacent and vertically aligned
                            if (cx >= lx + lw) and (cx - (lx + lw) <= x_tolerance) and (abs(cy - ly) <= y_tolerance):
                                cand_text = candidate.get("text", "").strip()
                                if re.match(r"^[A-Z0-9\-_]{3,20}$", cand_text, re.IGNORECASE):
                                    return cand_text
        return None

    @staticmethod
    def find_value_below_labels(
        tokens: List[Dict[str, Any]],
        label_keywords: List[str],
        y_tolerance: float = 40.0,
    ) -> Optional[str]:
        """Finds target candidate token values positioned directly below label header tokens."""
        for token in tokens:
            text = token.get("text", "").lower().strip()
            if any(kw in text for kw in label_keywords):
                label_bbox = token.get("bbox")
                if label_bbox and len(label_bbox) == 4:
                    lx, ly, lw, lh = label_bbox
                    for candidate in tokens:
                        if candidate == token:
                            continue
                        c_bbox = candidate.get("bbox")
                        if c_bbox and len(c_bbox) == 4:
                            cx, cy, cw, ch = c_bbox
                            # Check if candidate token is directly below horizontally aligned label
                            if (cy >= ly + lh) and (cy - (ly + lh) <= y_tolerance) and (abs(cx - lx) <= lw):
                                return candidate.get("text", "").strip()
        return None

    @staticmethod
    def render_bounding_boxes(
        image_np: np.ndarray, annotations: List[FieldAnnotation]
    ) -> np.ndarray:
        """Draws spatial bounding boxes and labels onto the processed image for visual auditing."""
        annotated_img = image_np.copy()
        # Convert to BGR for color overlays if image is grayscale
        if len(annotated_img.shape) == 2:
            annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_GRAY2BGR)

        for ann in annotations:
            box = ann.bbox
            cv2.rectangle(
                annotated_img,
                (box.x_min, box.y_min),
                (box.x_max, box.y_max),
                (0, 255, 0),  # Green box
                2,
            )
            label = f"{ann.field_name}: {ann.text}"
            cv2.putText(
                annotated_img,
                label,
                (box.x_min, max(15, box.y_min - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 200, 0),
                1,
                cv2.LINE_AA,
            )

        return annotated_img


# Alias class name to maintain compatibility with existing references
LayoutAnalyzer = SpatialLayoutAnalyzer


# Standalone Test Execution Block
if __name__ == "__main__":
    dummy_canvas = np.zeros((400, 600, 3), dtype=np.uint8)

    sample_annotations = [
        FieldAnnotation(
            field_name="invoice_number",
            text="INV-1024",
            bbox=BoundingBox(x_min=50, y_min=40, x_max=200, y_max=70),
            confidence=0.95,
        ),
        FieldAnnotation(
            field_name="grand_total",
            text="$150.00",
            bbox=BoundingBox(x_min=400, y_min=300, x_max=550, y_max=330),
            confidence=0.98,
        ),
    ]

    analyzer = LayoutAnalyzer()
    annotated = analyzer.render_bounding_boxes(dummy_canvas, sample_annotations)

    mock_tokens = [
        {"text": "Invoice:", "bbox": [50, 40, 60, 20]},
        {"text": "INV-1024", "bbox": [120, 40, 80, 20]},
    ]
    found_inv = analyzer.find_value_near_labels(mock_tokens, ["invoice"])

    print(f"Rendered visual annotations on image. Image shape: {annotated.shape}")
    print(f"Proximity test extracted invoice number: {found_inv}")