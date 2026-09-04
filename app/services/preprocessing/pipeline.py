import logging
import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class ImagePreprocessor:

    def preprocess(self, image: Image.Image, profile: str = "BASIC") -> Image.Image:
        """Applies designated preprocessing pipeline based on the recommended profile.
        
        Handles PIL Image instances and NumPy arrays, returning a clean PIL Image for downstream OCR.
        """
        # Convert PIL Image to NumPy array for OpenCV operations
        if isinstance(image, Image.Image):
            img_np = np.array(image.convert("RGB"))
        elif isinstance(image, np.ndarray):
            img_np = image
        else:
            raise TypeError(f"Unsupported image type provided to ImagePreprocessor: {type(image)}")

        profile = (profile or "BASIC").upper()

        try:
            if profile in ("ORIGINAL", "NONE"):
                # Return image as-is without transformations
                processed_np = img_np

            elif profile == "SKEWED":
                # Deskewing and perspective/rotation correction
                processed_np = self._deskew_image(img_np)

            elif profile == "LOW_LIGHT":
                # Enhance contrast using CLAHE + adaptive thresholding
                if len(img_np.shape) == 3:
                    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                else:
                    gray = img_np
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(gray)
                thresh = cv2.adaptiveThreshold(
                    enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
                )
                processed_np = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)

            elif profile == "SMALL_TEXT":
                # Upscale resolution and apply sharpening kernel
                h, w = img_np.shape[:2]
                upscaled = cv2.resize(img_np, (int(w * 1.5), int(h * 1.5)), interpolation=cv2.INTER_CUBIC)
                
                if len(upscaled.shape) == 3:
                    gray = cv2.cvtColor(upscaled, cv2.COLOR_RGB2GRAY)
                else:
                    gray = upscaled

                kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
                sharpened = cv2.filter2D(gray, -1, kernel)
                processed_np = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB)

            elif profile == "NOISY_SCAN":
                # Apply median filtering, bilateral noise reduction, and morphological opening
                if len(img_np.shape) == 3:
                    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                else:
                    gray = img_np
                median = cv2.medianBlur(gray, 3)
                filtered = cv2.bilateralFilter(median, 9, 75, 75)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                opened = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
                processed_np = cv2.cvtColor(opened, cv2.COLOR_GRAY2RGB)

            else:
                # BASIC profile: Grayscale conversion, mild denoising, and contrast normalization
                if len(img_np.shape) == 3:
                    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                else:
                    gray = img_np
                denoised = cv2.fastNlMeansDenoising(gray, h=10)
                norm = cv2.normalize(denoised, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
                processed_np = cv2.cvtColor(norm, cv2.COLOR_GRAY2RGB)

            return Image.fromarray(processed_np)

        except Exception as e:
            logger.error(f"Error executing preprocessing profile {profile}: {e}")
            # Fallback: return original image if preprocessing fails
            return image if isinstance(image, Image.Image) else Image.fromarray(img_np)

    def apply_profile(self, image: Image.Image, profile: str = "BASIC") -> Image.Image:
        """Alias method to match background job call signature."""
        return self.preprocess(image, profile)

    def _deskew_image(self, img_np: np.ndarray) -> np.ndarray:
        """Helper method to calculate skew angle and rotate image straight."""
        if len(img_np.shape) == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_np

        edges = cv2.Canny(gray, 50, 150)
        coords = np.column_stack(np.where(edges > 0))

        if len(coords) > 10:
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            elif angle > 45:
                angle = 90 - angle

            if abs(angle) > 0.5:
                h, w = img_np.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(
                    img_np, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
                )
                return rotated

        return img_np