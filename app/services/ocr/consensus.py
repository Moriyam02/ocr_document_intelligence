import os
# Disable problematic C++ execution optimizations on Windows CPU
os.environ["FLAGS_use_onednn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"

import logging
from typing import Dict, Any, List, Optional
from PIL import Image

logger = logging.getLogger("uvicorn.error")


class OCRConsensusEngine:
    """
    Coordinates multi-engine OCR execution (Tesseract, EasyOCR, PaddleOCR)
    and computes consensus across both scalar fields and structured line items.
    """

    def __init__(self, engine_weights: Optional[Dict[str, float]] = None):
        # Default engine weighting based on extraction accuracy
        self.engine_weights = engine_weights or {
            "paddleocr": 0.40,
            "tesseract": 0.35,
            "easyocr": 0.25,
        }

    def run_all_engines(self, image: Image.Image) -> Dict[str, Any]:
        """
        Executes OCR processing across available OCR engines on a single PIL image.
        Returns raw text, detected boxes, and an average per-engine confidence
        score (0.0-1.0) for each engine, where the underlying library exposes one.
        """
        engine_outputs: Dict[str, Any] = {}

        # 1. Tesseract OCR
        try:
            import pytesseract
            raw_text = pytesseract.image_to_string(image)

            # FIX/ADD: image_to_string alone gives no confidence figure.
            # A second lightweight call via image_to_data exposes per-token
            # confidences, which we average into a single engine-level score.
            conf_score = 0.0
            try:
                data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
                confs = [
                    float(c) for c in data.get("conf", [])
                    if str(c) not in ("-1",) and float(c) >= 0
                ]
                if confs:
                    conf_score = round((sum(confs) / len(confs)) / 100.0, 2)
            except Exception as conf_err:
                logger.warning(f"[OCR CONSENSUS] Tesseract confidence lookup failed: {conf_err}")

            engine_outputs["tesseract"] = {
                "status": "SUCCESS",
                "text": raw_text,
                "raw_text": raw_text,
                "boxes": [],
                "confidence": conf_score,
                "error": None
            }
        except Exception as e:
            logger.warning(f"[OCR CONSENSUS] Tesseract engine failed: {e}")
            engine_outputs["tesseract"] = {
                "status": "FAILED",
                "text": "",
                "raw_text": "",
                "boxes": [],
                "confidence": 0.0,
                "error": str(e)
            }

        # 2. EasyOCR
        try:
            import easyocr
            import numpy as np
            reader = easyocr.Reader(['en'], gpu=False)
            results = reader.readtext(np.array(image))
            extracted_text = "\n".join([res[1] for res in results])

            # ADD: average the per-detection confidence EasyOCR already returns.
            confs = [float(res[2]) for res in results if len(res) > 2]
            conf_score = round(sum(confs) / len(confs), 2) if confs else 0.0

            engine_outputs["easyocr"] = {
                "status": "SUCCESS",
                "text": extracted_text,
                "raw_text": extracted_text,
                "boxes": [res[0] for res in results],
                "confidence": conf_score,
                "error": None
            }
        except Exception as e:
            logger.warning(f"[OCR CONSENSUS] EasyOCR engine failed: {e}")
            engine_outputs["easyocr"] = {
                "status": "FAILED",
                "text": "",
                "raw_text": "",
                "boxes": [],
                "confidence": 0.0,
                "error": str(e)
            }

        # 3. PaddleOCR
        try:
            from paddleocr import PaddleOCR
            import numpy as np

            # Disable oneDNN via instance parameter if supported by version
            ocr_engine = PaddleOCR(use_angle_cls=True, lang='en', enable_mkldnn=False)
            results = ocr_engine.ocr(np.array(image))

            lines = []
            scores = []
            if results and len(results) > 0 and results[0] is not None:
                first_res = results[0]
                if isinstance(first_res, dict):
                    lines = first_res.get("rec_text") or first_res.get("rec_texts") or []
                    raw_scores = first_res.get("rec_score") or first_res.get("rec_scores") or []
                    scores = [float(s) for s in raw_scores]
                elif isinstance(first_res, list):
                    for line in first_res:
                        if line and len(line) > 1 and isinstance(line[1], (list, tuple)):
                            lines.append(line[1][0])
                            if len(line[1]) > 1:
                                try:
                                    scores.append(float(line[1][1]))
                                except (TypeError, ValueError):
                                    pass

            extracted_text = "\n".join([str(l).strip() for l in lines if str(l).strip()])
            conf_score = round(sum(scores) / len(scores), 2) if scores else 0.0

            engine_outputs["paddleocr"] = {
                "status": "SUCCESS",
                "text": extracted_text,
                "raw_text": extracted_text,
                "boxes": [],
                "confidence": conf_score,
                "error": None
            }
        except Exception as e:
            logger.warning(f"[OCR CONSENSUS] PaddleOCR engine failed: {e}")
            engine_outputs["paddleocr"] = {
                "status": "FAILED",
                "text": "",
                "raw_text": "",
                "boxes": [],
                "confidence": 0.0,
                "error": str(e)
            }

        return engine_outputs

    def compute_field_consensus(
        self,
        engine_outputs: Dict[str, Any],
        extracted_candidates: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Computes field-level consensus matching the dictionary structure consumed by jobs.py.
        """
        consensus_result: Dict[str, Dict[str, Any]] = {}

        for field_name, engine_values in extracted_candidates.items():
            if not engine_values:
                consensus_result[field_name] = {
                    "value": None,
                    "confidence": 0.0,
                    "sources": [],
                }
                continue

            if field_name == "line_items":
                consensus_result[field_name] = self._compute_line_items_consensus(engine_values)
            else:
                consensus_result[field_name] = self._compute_scalar_consensus(engine_values)

        return consensus_result

    def _compute_scalar_consensus(self, engine_values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculates weighted consensus vote for scalar fields (strings, dates, floats).
        """
        weighted_votes: Dict[Any, float] = {}
        value_sources: Dict[Any, List[str]] = {}

        for engine, value in engine_values.items():
            if value is None or value == "":
                continue

            weight = self.engine_weights.get(engine.lower(), 0.3)
            vote_key = str(value).strip().lower() if isinstance(value, str) else value

            weighted_votes[vote_key] = weighted_votes.get(vote_key, 0.0) + weight
            if vote_key not in value_sources:
                value_sources[vote_key] = []
            value_sources[vote_key].append(engine)

        if not weighted_votes:
            return {"value": None, "confidence": 0.0, "sources": []}

        best_vote_key = max(weighted_votes, key=weighted_votes.get)
        total_possible_weight = sum(
            self.engine_weights.get(eng.lower(), 0.3) for eng in engine_values.keys()
        )
        confidence = round(weighted_votes[best_vote_key] / max(total_possible_weight, 1.0), 2)

        winner_source = value_sources[best_vote_key][0]
        original_value = engine_values[winner_source]

        return {
            "value": original_value,
            "confidence": min(confidence, 1.0),
            "sources": value_sources[best_vote_key],
        }

    def _compute_line_items_consensus(self, engine_values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluates extracted line item candidate lists and selects the most accurate and complete item array.
        """
        valid_candidates: List[Dict[str, Any]] = []

        for engine, items in engine_values.items():
            if isinstance(items, list) and len(items) > 0:
                engine_weight = self.engine_weights.get(engine.lower(), 0.3)
                item_count_score = min(len(items) / 5.0, 1.0)
                score = round((engine_weight * 0.7) + (item_count_score * 0.3), 2)

                valid_candidates.append({
                    "engine": engine,
                    "items": items,
                    "score": score,
                    "item_count": len(items)
                })

        if not valid_candidates:
            return {
                "value": [],
                "confidence": 0.0,
                "sources": [],
            }

        best_candidate = max(valid_candidates, key=lambda x: (x["item_count"], x["score"]))

        supporting_sources = [c["engine"] for c in valid_candidates]
        consensus_confidence = round(
            sum(self.engine_weights.get(e.lower(), 0.3) for e in supporting_sources), 2
        )

        return {
            "value": best_candidate["items"],
            "confidence": min(consensus_confidence, 1.0),
            "sources": supporting_sources,
        }