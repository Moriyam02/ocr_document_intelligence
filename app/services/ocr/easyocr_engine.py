import easyocr
import time
import numpy as np
from typing import Dict, Any
from PIL import Image


class EasyOCREngine:

    def __init__(self):
        # Initialize EasyOCR reader on CPU (or GPU if available)
        self.reader = easyocr.Reader(["en"], gpu=False)

    def process_image(self, image: Any) -> Dict[str, Any]:
        start_time = time.time()
        try:
            # Handle PIL Image input by converting to NumPy array
            if isinstance(image, Image.Image):
                image = np.array(image)

            results = self.reader.readtext(image)

            words = []
            confidences = []
            text_tokens = []

            for bbox, text, conf in results:
                cleaned_text = text.strip()
                if cleaned_text:
                    confidences.append(float(conf))
                    text_tokens.append(cleaned_text)

                    x_coords = [p[0] for p in bbox]
                    y_coords = [p[1] for p in bbox]
                    left, top = min(x_coords), min(y_coords)
                    width = max(x_coords) - left
                    height = max(y_coords) - top

                    words.append(
                        {
                            "text": cleaned_text,
                            "confidence": round(float(conf), 2),
                            "bbox": [int(left), int(top), int(width), int(height)],
                        }
                    )

            raw_text = " ".join(text_tokens)
            avg_conf = (
                (sum(confidences) / len(confidences)) if confidences else 0.0
            )
            duration = round(time.time() - start_time, 3)

            return {
                "engine_name": "EasyOCR",
                "confidence": round(avg_conf, 2),
                "raw_text": raw_text,
                "words": words,
                "processing_time": duration,
                "text_coverage": 0.85 if len(text_tokens) > 0 else 0.0,
                "validation_readiness": 0.85 if avg_conf > 0.6 else 0.40,
                "error": None,
            }
        except Exception as e:
            return {
                "engine_name": "EasyOCR",
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