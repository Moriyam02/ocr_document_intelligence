import os
import logging
import time
from typing import Dict, Any
import cv2
import numpy as np
from PIL import Image

# Disable oneDNN/MKLDNN & PIR API at environment level before importing PaddleOCR
os.environ["FLAGS_use_onednn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"

# Suppress noisy lower-level C++ logs from Paddle
logging.getLogger("ppocr").setLevel(logging.ERROR)

# Use Uvicorn's error logger to ensure logs visible in Uvicorn terminal
logger = logging.getLogger("uvicorn.error")


class PaddleOCREngine:

    def __init__(self, lang: str = "en"):
        """
        Initialize PaddleOCR engine instance cleanly.
        Catches version incompatibility flags automatically.
        """
        self.available = False
        self.ocr = None

        try:
            from paddleocr import PaddleOCR

            # Initialize with core parameters supported across all versions
            self.ocr = PaddleOCR(use_angle_cls=True, lang=lang)
            self.available = True
            logger.info("PaddleOCR engine initialized successfully.")
            print("[INFO] PaddleOCR engine initialized successfully.")
        except Exception as e:
            self.available = False
            self.ocr = None
            logger.error(f"Failed to initialize PaddleOCR: {e}")
            print(f"[ERROR] Failed to initialize PaddleOCR: {e}")

    def _prepare_image_input(self, image: Any) -> np.ndarray:
        """Ensures the image is converted into a standard 3-channel BGR OpenCV array."""
        if isinstance(image, Image.Image):
            rgb_np = np.array(image.convert("RGB"))
            return cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)

        elif isinstance(image, np.ndarray):
            if len(image.shape) == 2:
                return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif len(image.shape) == 3 and image.shape[2] == 4:
                return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            return image

        elif isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image file does not exist at: {image}")
            img_bgr = cv2.imread(image)
            if img_bgr is None:
                raise ValueError(f"Could not load image from path: {image}")
            return img_bgr

        else:
            raise TypeError(f"Unsupported image type provided to PaddleOCR: {type(image)}")

    def process_image(self, image: Any) -> Dict[str, Any]:
        """Runs PaddleOCR and returns standard OCR response format across engines."""
        start_time = time.time()

        if not self.available or self.ocr is None:
            return {
                "engine_name": "PaddleOCR",
                "confidence": 0.0,
                "raw_text": "",
                "text": "",
                "words": [],
                "processing_time": 0.0,
                "text_coverage": 0.0,
                "validation_readiness": 0.0,
                "error": "PaddleOCR engine is not initialized",
            }

        try:
            cv2_img = self._prepare_image_input(image)
            
            # Run inference safely with angle classification
            results = self.ocr.ocr(cv2_img, cls=True)

            raw_text_lines = []
            words = []
            confidences = []

            # Handle cases where image has no detectable text (results is [None] or None)
            if results and len(results) > 0 and results[0] is not None:
                first_res = results[0]

                # Case A: PP-Structure / Pipeline Dictionary Output
                if isinstance(first_res, dict):
                    rec_texts = first_res.get("rec_text") or first_res.get("rec_texts") or []
                    rec_scores = first_res.get("rec_score") or first_res.get("rec_scores") or []
                    rec_boxes = first_res.get("dt_polys") or first_res.get("boxes") or []

                    for idx, text in enumerate(rec_texts):
                        text_str = str(text).strip()
                        if text_str:
                            raw_text_lines.append(text_str)
                            score = float(rec_scores[idx]) if idx < len(rec_scores) else 1.0
                            confidences.append(score)

                            box = rec_boxes[idx] if idx < len(rec_boxes) else None
                            bbox_coords = [0, 0, 0, 0]
                            if box is not None and len(box) >= 4:
                                x_coords = [p[0] for p in box if isinstance(p, (list, tuple, np.ndarray))]
                                y_coords = [p[1] for p in box if isinstance(p, (list, tuple, np.ndarray))]
                                if x_coords and y_coords:
                                    x_min, y_min = int(min(x_coords)), int(min(y_coords))
                                    width = int(max(x_coords) - x_min)
                                    height = int(max(y_coords) - y_min)
                                    bbox_coords = [x_min, y_min, width, height]

                            words.append({
                                "text": text_str,
                                "confidence": round(score, 2),
                                "bbox": bbox_coords
                            })

                # Case B: Classic PaddleOCR List Output [[ [bbox], (text, score) ], ...]
                elif isinstance(first_res, list):
                    for line in first_res:
                        if not line:
                            continue

                        bbox, text_info = None, None

                        if isinstance(line, (list, tuple)) and len(line) == 2:
                            bbox, text_info = line[0], line[1]
                        elif isinstance(line, (list, tuple)) and len(line) == 1:
                            text_info = line[0]
                        else:
                            continue

                        if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                            text, confidence = str(text_info[0]), float(text_info[1])
                        elif isinstance(text_info, str):
                            text, confidence = text_info, 1.0
                        else:
                            continue

                        text_str = text.strip()
                        if text_str:
                            raw_text_lines.append(text_str)
                            conf_val = float(confidence)
                            confidences.append(conf_val)

                            bbox_coords = [0, 0, 0, 0]
                            if bbox and len(bbox) >= 4:
                                x_coords = [p[0] for p in bbox if isinstance(p, (list, tuple, np.ndarray))]
                                y_coords = [p[1] for p in bbox if isinstance(p, (list, tuple, np.ndarray))]
                                if x_coords and y_coords:
                                    x_min, y_min = int(min(x_coords)), int(min(y_coords))
                                    width = int(max(x_coords) - x_min)
                                    height = int(max(y_coords) - y_min)
                                    bbox_coords = [x_min, y_min, width, height]

                            words.append({
                                "text": text_str,
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
                "text": full_text,
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
                "text": "",
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


# Explicitly instantiate the global instance so it triggers on application import/startup
paddle_engine = PaddleOCREngine()