import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Optional
from collections import Counter
from PIL import Image

from app.services.ocr.tesseract_engine import TesseractEngine
from app.services.ocr.easyocr_engine import EasyOCREngine
from app.services.ocr.paddle_engine import PaddleOCREngine

logger = logging.getLogger(__name__)


class OCRConsensusEngine:

    def __init__(self, executor_threads: int = 3):
        """Initializes all 3 OCR engines required by Section 4.4."""
        self.tesseract = TesseractEngine()
        self.easyocr = EasyOCREngine()
        self.paddleocr = PaddleOCREngine()
        self.executor = ThreadPoolExecutor(max_workers=executor_threads)

    # --- Async Parallel Engine Execution ---

    def _run_single_engine(self, engine_instance: Any, engine_name: str, image: Image.Image) -> Dict[str, Any]:
        """Safely executes a single OCR engine wrapper."""
        try:
            # Handle both process_image and extract_text style returns
            if hasattr(engine_instance, "process_image"):
                res = engine_instance.process_image(image)
                if isinstance(res, dict):
                    return res
                return {"engine": engine_name, "raw_text": str(res), "status": "SUCCESS", "error": None}
            elif hasattr(engine_instance, "extract_text"):
                res = engine_instance.extract_text(image)
                return {"engine": engine_name, "raw_text": str(res), "status": "SUCCESS", "error": None}
            else:
                raise AttributeError(f"{engine_name} missing extraction method")
        except Exception as e:
            logger.error(f"{engine_name} execution error: {e}")
            return {
                "engine": engine_name,
                "raw_text": "",
                "words": [],
                "status": "ERROR",
                "error": str(e),
            }

    async def run_all_engines_async(self, image: Image.Image) -> Dict[str, Any]:
        """Runs Tesseract, EasyOCR, and PaddleOCR concurrently in separate threads."""
        loop = asyncio.get_event_loop()
        
        futures = {
            "tesseract": loop.run_in_executor(self.executor, self._run_single_engine, self.tesseract, "tesseract", image),
            "easyocr": loop.run_in_executor(self.executor, self._run_single_engine, self.easyocr, "easyocr", image),
            "paddleocr": loop.run_in_executor(self.executor, self._run_single_engine, self.paddleocr, "paddleocr", image),
        }

        results = await asyncio.gather(*futures.values())
        return dict(zip(futures.keys(), results))

    def run_all_engines(self, image: Image.Image) -> Dict[str, Any]:
        """Synchronous wrapper that executes engines concurrently using threads."""
        try:
            loop = asyncio.get_running_loop()
            return loop.run_until_complete(self.run_all_engines_async(image))
        except RuntimeError:
            # Fallback when no active async loop exists
            return {
                "tesseract": self._run_single_engine(self.tesseract, "tesseract", image),
                "easyocr": self._run_single_engine(self.easyocr, "easyocr", image),
                "paddleocr": self._run_single_engine(self.paddleocr, "paddleocr", image),
            }

    # Aliases to maintain compatibility with existing background job routines
    def process(self, image: Image.Image) -> Dict[str, Any]:
        return self.run_all_engines(image)

    def process_page(self, image: Image.Image) -> Dict[str, Any]:
        return self.run_all_engines(image)

    # --- Smart Normalization & Field Consensus ---

    @staticmethod
    def _normalize_field_value(field_name: str, val: Any) -> Optional[str]:
        """Standardizes values across formatting variations (e.g. currency, hyphens, floats)."""
        if val is None:
            return None

        s_val = str(val).strip().lower()
        if s_val in ("", "none", "null"):
            return None

        # Numbers / Totals (e.g., "$25,186.00" -> "25186.00")
        if any(term in field_name.lower() for term in ["total", "subtotal", "tax", "amount"]):
            try:
                clean_num = s_val.replace("$", "").replace("₹", "").replace(",", "")
                return f"{float(clean_num):.2f}"
            except ValueError:
                return s_val

        # Invoice codes (e.g., "INV-2026-0423" -> "inv20260423")
        if "invoice" in field_name.lower() or "number" in field_name.lower():
            return s_val.replace("-", "").replace(" ", "").replace("#", "")

        return s_val

    def compute_field_consensus(
        self, engine_outputs: Dict[str, Any], extracted_candidates: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compares extracted candidate values across Tesseract, EasyOCR, and PaddleOCR."""
        field_consensus = {}

        for field_name, candidates in extracted_candidates.items():
            if not candidates:
                continue

            value_groups: Dict[str, Dict[str, Any]] = {}
            
            # 1. Normalize and group votes
            for engine, val in candidates.items():
                norm_val = self._normalize_field_value(field_name, val)
                if not norm_val:
                    continue

                if norm_val not in value_groups:
                    value_groups[norm_val] = {
                        "original_value": val,
                        "engines": [engine],
                        "count": 1,
                    }
                else:
                    value_groups[norm_val]["engines"].append(engine)
                    value_groups[norm_val]["count"] += 1

            if not value_groups:
                field_consensus[field_name] = {
                    "value": None,
                    "confidence": 0.0,
                    "source": "NONE",
                    "agreement_count": 0,
                    "flagged_for_review": True,
                }
                continue

            # 2. Find winning value
            best_match = max(value_groups.values(), key=lambda x: x["count"])

            # 3. Calculate Active Engine Coverage
            total_active_engines = len([
                e for e, out in engine_outputs.items()
                if isinstance(out, dict) and out.get("error") is None
            ])
            if total_active_engines == 0:
                total_active_engines = len(engine_outputs)

            # 4. Weighted Confidence Scoring
            valid_agreement_ratio = best_match["count"] / len(candidates)
            coverage_ratio = best_match["count"] / total_active_engines

            confidence = round((0.75 * valid_agreement_ratio) + (0.25 * coverage_ratio), 2)
            
            # Require at least 2 agreeing engines or >= 0.80 confidence to pass without review
            needs_review = best_match["count"] < 2 or confidence < 0.80

            field_consensus[field_name] = {
                "value": best_match["original_value"],
                "confidence": confidence,
                "source": ", ".join(best_match["engines"]),
                "agreement_count": best_match["count"],
                "flagged_for_review": needs_review,
            }

        return field_consensus