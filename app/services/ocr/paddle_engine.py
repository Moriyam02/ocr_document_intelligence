import os
import logging
import time
from typing import Dict, Any
import cv2
import numpy as np
from PIL import Image

# Disable oneDNN/MKLDNN at environment level before importing PaddleOCR
os.environ["FLAGS_use_onednn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"

# Suppress verbose logger output from paddleocr/ppocr
logging.getLogger("ppocr").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)


class PaddleOCREngine:

    def __init__(self, lang: str = "en"):
        """Initialize PaddleOCR engine instance with CPU math acceleration disabled to avoid oneDNN crashes."""
        try:
            from paddleocr import PaddleOCR

            self.ocr = PaddleOCR(
                use_angle_cls=True,
                lang=lang,
                enable_mkldnn=False,
                show_log=False
            )
            self.available = True
            logger.info("PaddleOCR engine initialized successfully (enable_mkldnn=False).")
        except Exception as e:
            self.available = False
            logger.error(f"Failed to initialize PaddleOCR: {e}")

    def _prepare_image_input(self, image: Any) -> np.ndarray:
        """Ensures the image is converted into a standard 3-channel OpenCV array."""
        if isinstance(image, Image.Image):
            # Convert PIL image (including RGBA/L) to RGB array, then to OpenCV BGR
            rgb_np = np.array(image.convert("RGB"))
            return cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
        
        elif isinstance(image, np.ndarray):
            # Single-channel grayscale
            if len(image.shape) == 2:
                return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            # 4-channel RGBA
            elif len(image.shape) == 3 and image.shape[2] == 4:
                return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            # 3-channel (assume standard matrix)
            return image
        
        elif isinstance(image, str):
            # If path is provided directly
            img_bgr = cv2.imread(image)
            if img_bgr is None:
                raise ValueError(f"Could not load image from path: {image}")
            return img_bgr
            
        else:
            raise TypeError(f"Unsupported image type provided to PaddleOCR: {type(image)}")

    def process_image(self, image: Any) -> Dict[str, Any]:
        """Runs PaddleOCR and returns standard OCR response format across engines."""
        start_time = time.time()
        if not self.available:
            return {
                "engine_name": "PaddleOCR",
                "confidence": 0.0,
                "raw_text": "",
                "words": [],
                "processing_time": 0.0,
                "text_coverage": 0.0,
                "validation_readiness": 0.0,
                "error": "PaddleOCR not initialized",
            }

        try:
            # Prepare image matrix safely
            cv2_img = self._prepare_image_input(image)

            # Run PaddleOCR inference
            results = self.ocr.ocr(cv2_img)

            raw_text_lines = []
            words = []
            confidences = []

            if results and len(results) > 0 and results[0]:
                for line in results[0]:
                    if not line:
                        continue

                    # Defensive parsing for varied line structure outputs
                    if isinstance(line, (list, tuple)) and len(line) == 2:
                        bbox, text_info = line[0], line[1]
                    elif isinstance(line, (list, tuple)) and len(line) == 1:
                        bbox, text_info = None, line[0]
                    else:
                        continue

                    # Extract text string and confidence score
                    if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                        text, confidence = str(text_info[0]), float(text_info[1])
                    elif isinstance(text_info, str):
                        text, confidence = text_info, 1.0
                    else:
                        continue

                    if text.strip():
                        raw_text_lines.append(text.strip())
                        conf_val = float(confidence)
                        confidences.append(conf_val)

                        # Extract bounding box safely if available
                        if bbox and len(bbox) >= 4:
                            x_coords = [p[0] for p in bbox if isinstance(p, (list, tuple))]
                            y_coords = [p[1] for p in bbox if isinstance(p, (list, tuple))]
                            if x_coords and y_coords:
                                x_min, y_min = int(min(x_coords)), int(min(y_coords))
                                width = int(max(x_coords) - x_min)
                                height = int(max(y_coords) - y_min)
                                bbox_coords = [x_min, y_min, width, height]
                            else:
                                bbox_coords = [0, 0, 0, 0]
                        else:
                            bbox_coords = [0, 0, 0, 0]

                        words.append({
                            "text": text.strip(),
                            "confidence": round(conf_val, 2),
                            "bbox": bbox_coords
                        })

            full_text = "\n".join(raw_text_lines)
            avg_conf = (sum(confidences) / len(confidences)) if confidences else 0.0
            duration = round(time.time() - start_time, 3)

            return {
                "engine_name": "PaddleOCR",
                "confidence": round(avg_conf, 2),
                "raw_text": full_text,
                "words": words,
                "processing_time": duration,
                "text_coverage": 0.90 if len(words) > 0 else 0.0,
                "validation_readiness": 0.90 if avg_conf > 0.6 else 0.40,
                "error": None,
            }

        except Exception as e:
            logger.error(f"Error executing PaddleOCR: {e}")
            return {
                "engine_name": "PaddleOCR",
                "confidence": 0.0,
                "raw_text": "",
                "words": [],
                "processing_time": round(time.time() - start_time, 3),
                "text_coverage": 0.0,
                "validation_readiness": 0.0,
                "error": str(e),
            }

    def extract_text(self, image: Any) -> str:
        """Required wrapper by ConsensusEngine. Returns raw extracted text."""
        result = self.process_image(image)
        return result.get("raw_text", "")