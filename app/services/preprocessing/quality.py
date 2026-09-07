import logging
import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class QualityAnalyzer:

    def analyze(self, image) -> dict:
        """Analyzes image quality metrics including resolution, blur, brightness, contrast, skew, and boundaries.
        
        Handles PIL Image instances (including PpmImageFile), OpenCV arrays, and NumPy arrays.
        """
        # Convert PIL Image or PpmImageFile to NumPy array if necessary
        if isinstance(image, Image.Image):
            image_np = np.array(image.convert("RGB"))
        elif isinstance(image, np.ndarray):
            image_np = image
        else:
            raise TypeError(f"Unsupported image type provided to QualityAnalyzer: {type(image)}")

        # Extract dimensions and create grayscale representation for CV2 operations
        if len(image_np.shape) == 2:
            # Grayscale image array
            h, w = image_np.shape
            gray = image_np
        elif len(image_np.shape) == 3:
            # Color image array (RGB or BGR)
            h, w = image_np.shape[:2]
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        else:
            raise ValueError(f"Invalid image array shape: {image_np.shape}")

        # Calculate blur score using variance of Laplacian
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # Calculate brightness (mean pixel intensity) and contrast (std deviation)
        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))

        # Estimate skew/rotation angle
        skew_angle = self._estimate_skew_angle(gray)

        # Detect document boundary presence
        has_boundary = self._detect_document_boundary(gray)

        # Estimated DPI assuming standard 8.5-inch document width
        estimated_dpi = round((w / 8.5), 1)

        # Determine recommended preprocessing profile based on refined quality thresholds
        recommended_profile = "BASIC"
        if abs(skew_angle) > 2.0:
            recommended_profile = "SKEWED"
        elif blur_score < 100.0 or estimated_dpi < 120:
            recommended_profile = "SMALL_TEXT"
        elif brightness < 60.0 or contrast < 25.0:  # Adjusted contrast threshold from 40.0 to 25.0 to prevent false triggers
            recommended_profile = "LOW_LIGHT"

        # Determine overall quality label according to spec
        if blur_score < 50.0 or brightness < 40.0 or contrast < 20.0 or estimated_dpi < 100:
            quality_label = "Needs manual review"
        elif blur_score < 80.0 or brightness < 70.0 or contrast < 35.0 or estimated_dpi < 150:
            quality_label = "Poor"
        elif blur_score < 150.0 or brightness < 80.0:
            quality_label = "Acceptable"
        else:
            quality_label = "Good"

        return {
            "dimensions": {"width": w, "height": h},
            "estimated_dpi": estimated_dpi,
            "blur_score": round(blur_score, 2),
            "brightness": round(brightness, 2),
            "contrast": round(contrast, 2),
            "skew_angle": round(skew_angle, 2),
            "has_boundary": has_boundary,
            "resolution_warning": estimated_dpi < 150,
            "recommended_profile": recommended_profile,
            "quality_label": quality_label
        }

    def _estimate_skew_angle(self, gray: np.ndarray) -> float:
        """Estimates skew angle using OpenCV minAreaRect on edge contours."""
        try:
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            coords = np.column_stack(np.where(edges > 0))
            if len(coords) < 10:
                return 0.0
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            elif angle > 45:
                angle = 90 - angle
            return float(angle)
        except Exception:
            return 0.0

    def _detect_document_boundary(self, gray: np.ndarray) -> bool:
        """Checks if a clear document border contour exists in the image."""
        try:
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edged = cv2.Canny(blurred, 75, 200)
            contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                max_c = max(contours, key=cv2.contourArea)
                area_ratio = cv2.contourArea(max_c) / (gray.shape[0] * gray.shape[1])
                return bool(area_ratio > 0.35)
            return False
        except Exception:
            return False