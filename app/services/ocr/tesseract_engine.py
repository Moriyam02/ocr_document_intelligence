import pytesseract
import time
import numpy as np
import cv2
from typing import Dict, Any
from PIL import Image

# Configure local Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


class TesseractEngine:

    def process_image(self, image: Any) -> Dict[str, Any]:
        start_time = time.time()
        try:
            # Handle PIL Image input by converting to NumPy BGR format for cv2
            if isinstance(image, Image.Image):
                image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            data = pytesseract.image_to_data(
                rgb_image, output_type=pytesseract.Output.DICT
            )

            words = []
            confidences = []
            valid_tokens = 0
            total_tokens = len(data["text"])

            for i in range(total_tokens):
                text = data["text"][i].strip()
                conf = float(data["conf"][i])

                if text:
                    valid_tokens += 1
                    if conf > 0:
                        confidences.append(conf)
                        words.append(
                            {
                                "text": text,
                                "confidence": conf / 100.0,
                                "bbox": [
                                    data["left"][i],
                                    data["top"][i],
                                    data["width"][i],
                                    data["height"][i],
                                ],
                            }
                        )

            raw_text = pytesseract.image_to_string(rgb_image).strip()
            avg_conf = (
                (sum(confidences) / len(confidences) / 100.0)
                if confidences
                else 0.0
            )
            coverage = valid_tokens / total_tokens if total_tokens > 0 else 0.0
            duration = round(time.time() - start_time, 3)

            return {
                "engine_name": "Tesseract",
                "confidence": round(avg_conf, 2),
                "raw_text": raw_text,
                "words": words,
                "processing_time": duration,
                "text_coverage": round(coverage, 2),
                "validation_readiness": 0.90 if avg_conf > 0.6 else 0.50,
                "error": None,
            }
        except Exception as e:
            return {
                "engine_name": "Tesseract",
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