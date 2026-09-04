import cv2
import numpy as np


class ImagePreprocessor:

    @staticmethod
    def deskew(image: np.ndarray, angle: float) -> np.ndarray:
        """Rotates image to align text horizontally."""
        if abs(angle) == 0.0 or abs(angle) == 90.0:
            return image
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(
            image,
            matrix,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

    @staticmethod
    def apply_clahe(gray_image: np.ndarray) -> np.ndarray:
        """Enhances low light and low contrast images using Adaptive Histogram Equalization."""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray_image)

    def process(
        self, image_np: np.ndarray, profile: str, skew_angle: float = 0.0
    ) -> np.ndarray:
        """Applies targeted preprocessing pipelines based on quality evaluation (Section 4.3)."""
        # Ensure grayscale
        if len(image_np.shape) == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_np.copy()

        # Execute deskew if needed
        if abs(skew_angle) > 2.0 and abs(skew_angle) != 90.0:
            gray = self.deskew(gray, skew_angle)

        # Profile selection pipeline
        if profile == "low_light":
            enhanced = self.apply_clahe(gray)
            return cv2.adaptiveThreshold(
                enhanced,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11,
                2,
            )

        elif profile == "small_text":
            # Scale up 2x and apply sharpening
            scaled = cv2.resize(
                gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC
            )
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            return cv2.filter2D(scaled, -1, kernel)

        elif profile == "skewed":
            return cv2.fastNlMeansDenoising(gray, h=10)

        # Default basic profile
        return gray


# Standalone Test Execution Block
if __name__ == "__main__":
    dummy_img = np.zeros((400, 600), dtype=np.uint8)
    cv2.putText(
        dummy_img,
        "Sample Invoice Text",
        (50, 200),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        255,
        2,
    )

    processor = ImagePreprocessor()
    processed_img = processor.process(
        dummy_img, profile="low_light", skew_angle=0.0
    )

    print(
        f"Successfully processed image. Output shape: {processed_img.shape}, Type: {processed_img.dtype}"
    )