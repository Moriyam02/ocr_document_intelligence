import logging
from typing import Dict, Any, List
from PIL import Image

from app.services.ocr.tesseract_engine import TesseractEngine
from app.services.ocr.easyocr_engine import EasyOCREngine
from app.services.ocr.paddle_engine import PaddleOCREngine

logger = logging.getLogger(__name__)


class OCRConsensusEngine:

    def __init__(self):
        """Initializes all 3 OCR engines required by Section 4.4."""
        self.tesseract = TesseractEngine()
        self.easyocr = EasyOCREngine()
        self.paddleocr = PaddleOCREngine()

    def run_all_engines(self, image: Image.Image) -> Dict[str, Any]:
        """Runs Tesseract, EasyOCR, and PaddleOCR concurrently on an image page."""
        outputs = {}

        # 1. Tesseract
        try:
            outputs["tesseract"] = self.tesseract.extract_text(image)
        except Exception as e:
            logger.error(f"Tesseract execution error: {e}")
            outputs["tesseract"] = {
                "engine": "tesseract",
                "text": "",
                "words": [],
                "status": f"ERROR: {str(e)}",
            }

        # 2. EasyOCR
        try:
            outputs["easyocr"] = self.easyocr.extract_text(image)
        except Exception as e:
            logger.error(f"EasyOCR execution error: {e}")
            outputs["easyocr"] = {
                "engine": "easyocr",
                "text": "",
                "words": [],
                "status": f"ERROR: {str(e)}",
            }

        # 3. PaddleOCR
        try:
            outputs["paddleocr"] = self.paddleocr.extract_text(image)
        except Exception as e:
            logger.error(f"PaddleOCR execution error: {e}")
            outputs["paddleocr"] = {
                "engine": "paddleocr",
                "text": "",
                "words": [],
                "status": f"ERROR: {str(e)}",
            }

        return outputs

    # Aliases to maintain compatibility with existing background job routines
    def process(self, image: Image.Image) -> Dict[str, Any]:
        return self.run_all_engines(image)

    def process_page(self, image: Image.Image) -> Dict[str, Any]:
        return self.run_all_engines(image)

    def compute_field_consensus(
        self, engine_outputs: Dict[str, Any], extracted_candidates: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compares extracted candidate values across Tesseract, EasyOCR, and PaddleOCR."""
        field_consensus = {}

        for field_name, candidates in extracted_candidates.items():
            if not candidates:
                continue

            value_counts = {}
            for engine, val in candidates.items():
                if not val:
                    continue
                clean_val = str(val).strip().lower()
                if clean_val not in value_counts:
                    value_counts[clean_val] = {
                        "original_value": val,
                        "engines": [engine],
                        "count": 1,
                    }
                else:
                    value_counts[clean_val]["engines"].append(engine)
                    value_counts[clean_val]["count"] += 1

            if not value_counts:
                field_consensus[field_name] = {
                    "value": None,
                    "confidence": 0.0,
                    "source": "NONE",
                    "agreement_count": 0,
                    "flagged_for_review": True,
                }
                continue

            best_match = max(value_counts.values(), key=lambda x: x["count"])
            
            # Count active engines considering both 'status == SUCCESS' and 'error == None' schemas
            total_active_engines = len([
                e for e, out in engine_outputs.items()
                if isinstance(out, dict) and (
                    out.get("status") == "SUCCESS" or 
                    (out.get("error") is None and "error" in out)
                )
            ])
            
            # Fallback if engines return raw strings or standard dicts
            if total_active_engines == 0:
                total_active_engines = len(engine_outputs)

            agreement_ratio = (
                best_match["count"] / total_active_engines if total_active_engines > 0 else 0.5
            )
            confidence = round(min(1.0, agreement_ratio * 1.15), 2)
            needs_review = best_match["count"] < 2 or confidence < 0.85

            field_consensus[field_name] = {
                "value": best_match["original_value"],
                "confidence": confidence,
                "source": ", ".join(best_match["engines"]),
                "agreement_count": best_match["count"],
                "flagged_for_review": needs_review,
            }

        return field_consensus