import os
import uuid
import logging
import pymupdf as fitz
import numpy as np
from PIL import Image
from typing import Dict, Any, List, Union, Optional
from fastapi import UploadFile

from app.core.database import SessionLocal
from app.repositories.document_repository import DocumentRepository
from app.services.ocr.consensus import OCRConsensusEngine
from app.services.extraction.parser import FieldExtractor
from app.services.validation.rules import BusinessRulesValidator
from app.services.validation.confidence_router import RoutingEngine
# ADD: wire in the quality/preprocessing services that were previously being
# passed into process_document_background() from main.py but silently
# discarded via **kwargs and never actually used.
from app.services.preprocessing.quality import QualityAnalyzer
from app.services.preprocessing.pipeline import ImagePreprocessor

logger = logging.getLogger(__name__)

# Storage directory for local uploads
UPLOAD_DIR = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


class JobProcessor:
    def __init__(self):
        self.consensus_engine = OCRConsensusEngine()
        self.field_extractor = FieldExtractor()
        self.validator = BusinessRulesValidator()
        self.router = RoutingEngine()
        self.quality_analyzer = QualityAnalyzer()
        self.image_processor = ImagePreprocessor()
        # Memory map store for quick byte access: doc_id -> raw bytes
        self._memory_store: Dict[str, bytes] = {}
        # Path map store: doc_id -> local disk path
        self._path_store: Dict[str, str] = {}

    def store_file(self, doc_id: str, contents: Union[bytes, UploadFile, str], *args, **kwargs) -> str:
        """
        Stores document bytes/files mapped directly by document ID (`doc_id`).
        Supports bytes (from main.py), UploadFile objects, or disk path strings.
        """
        saved_filename = f"{doc_id}.pdf"
        file_path = os.path.join(UPLOAD_DIR, saved_filename)

        if isinstance(contents, bytes):
            self._memory_store[doc_id] = contents
            with open(file_path, "wb") as f:
                f.write(contents)
        elif hasattr(contents, "file"):
            raw_bytes = contents.file.read()
            contents.file.seek(0)
            self._memory_store[doc_id] = raw_bytes
            with open(file_path, "wb") as f:
                f.write(raw_bytes)
        elif isinstance(contents, str) and os.path.exists(contents):
            file_path = contents
            with open(file_path, "rb") as f:
                self._memory_store[doc_id] = f.read()

        self._path_store[doc_id] = file_path
        return file_path

    def get_file(self, doc_id: str) -> Optional[bytes]:
        """
        Retrieves file bytes for visual preview or export endpoints in main.py.
        """
        if doc_id in self._memory_store:
            return self._memory_store[doc_id]

        file_path = self.get_file_path(doc_id)
        if file_path and os.path.exists(file_path):
            with open(file_path, "rb") as f:
                bytes_data = f.read()
                self._memory_store[doc_id] = bytes_data
                return bytes_data

        return None

    def get_file_path(self, doc_id: str) -> Optional[str]:
        """
        Resolves local disk file path for a given doc_id.
        """
        if doc_id in self._path_store and os.path.exists(self._path_store[doc_id]):
            return self._path_store[doc_id]

        # Search upload folder for matching file prefix
        if os.path.exists(UPLOAD_DIR):
            for fname in os.listdir(UPLOAD_DIR):
                if fname.startswith(str(doc_id)):
                    full_p = os.path.join(UPLOAD_DIR, fname)
                    self._path_store[doc_id] = full_p
                    return full_p

        return None

    def _convert_pdf_to_images(self, file_path: str) -> List[Image.Image]:
        images = []
        doc = fitz.open(file_path)
        for page in doc:
            pix = page.get_pixmap(dpi=200)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
        doc.close()
        return images

    @staticmethod
    def _detect_color_mode(img: Image.Image) -> str:
        """Cheap heuristic: compares RGB channels to flag scanned grayscale/B&W
        pages vs true color images, for display on the quality-check page."""
        try:
            arr = np.array(img.convert("RGB")).astype(int)
            channel_diff = np.abs(arr[..., 0] - arr[..., 1]) + np.abs(arr[..., 1] - arr[..., 2])
            return "Grayscale" if channel_diff.mean() < 3.0 else "Color"
        except Exception:
            return "Unknown"

    def process_document_job(self, document_id: str, file_path: Optional[str] = None):
        logger.info(f"[JOB START] Processing Document ID: {document_id}")
        db = SessionLocal()
        repo = DocumentRepository(db)

        try:
            repo.update_status(doc_id=document_id, status="PROCESSING", progress=10.0)

            # Resolve path if not explicitly provided
            if not file_path:
                file_path = self.get_file_path(document_id)

            if not file_path or not os.path.exists(file_path):
                raise FileNotFoundError(f"File path for document {document_id} could not be located on disk.")

            if file_path.lower().endswith(".pdf"):
                images = self._convert_pdf_to_images(file_path)
            else:
                with Image.open(file_path) as img:
                    images = [img.convert("RGB")]

            total_pages = len(images)
            pages_data = []
            consolidated_consensus: Dict[str, Any] = {}

            for idx, img in enumerate(images):
                page_num = idx + 1
                logger.info(f"Processing Page {page_num}/{total_pages} for Document ID: {document_id}")

                # --- STEP 1: Quality analysis (ADD: this was previously never run) ---
                quality_report = self.quality_analyzer.analyze(img)
                quality_report["color_mode"] = self._detect_color_mode(img)

                quality_progress = 10 + int(20 * (page_num / total_pages))
                repo.update_status(doc_id=document_id, status="PROCESSING", progress=float(quality_progress))

                # --- STEP 2: Preprocessing (ADD: previously computed but never applied) ---
                recommended_profile = quality_report.get("recommended_profile", "BASIC")
                try:
                    preprocessed_img = self.image_processor.apply_profile(img, recommended_profile)
                except Exception as prep_err:
                    logger.warning(f"Preprocessing failed for page {page_num}, using original image: {prep_err}")
                    preprocessed_img = img

                # --- STEP 3: Multi-engine OCR ---
                engine_outputs = self.consensus_engine.run_all_engines(preprocessed_img)
                extracted_candidates = self.field_extractor.build_candidate_dict_from_engines(engine_outputs)
                page_consensus = self.consensus_engine.compute_field_consensus(
                    engine_outputs, extracted_candidates
                )

                ocr_progress = 30 + int(55 * (page_num / total_pages))
                repo.update_status(doc_id=document_id, status="PROCESSING", progress=float(ocr_progress))

                for field_name, field_data in page_consensus.items():
                    if field_data.get("value") is not None:
                        consolidated_consensus[field_name] = field_data

                # Format engine outputs to retain raw extracted text + confidence
                formatted_engine_outputs = {}
                for eng, res in engine_outputs.items():
                    if isinstance(res, dict):
                        formatted_engine_outputs[eng] = {
                            "status": res.get("status", "SUCCESS" if not res.get("error") else "FAILED"),
                            "raw_text": res.get("text") or res.get("raw_text") or res.get("full_text") or "",
                            "detected_boxes": res.get("boxes") or res.get("lines") or [],
                            "confidence": res.get("confidence", 0.0),
                            "error": res.get("error")
                        }
                    else:
                        formatted_engine_outputs[eng] = {
                            "status": "SUCCESS" if res else "FAILED",
                            "raw_text": str(res) if res else "",
                            "confidence": 0.0,
                            "error": None
                        }

                pages_data.append({
                    "page_number": page_num,
                    "quality": quality_report,
                    "preprocessing_profile": recommended_profile,
                    "engine_outputs": formatted_engine_outputs,
                    "consensus": page_consensus
                })

            # --- STEP 4: Validation & routing ---
            repo.update_status(doc_id=document_id, status="PROCESSING", progress=90.0)

            validation_issues = self.validator.validate(consolidated_consensus)
            routing_decision = self.router.evaluate(consolidated_consensus, validation_issues)

            if "overall_confidence" not in routing_decision:
                routing_decision["overall_confidence"] = routing_decision.get("confidence_score", 0.0)

            serialized_issues = []
            for issue in validation_issues:
                if hasattr(issue, "model_dump"):
                    serialized_issues.append(issue.model_dump())
                elif hasattr(issue, "dict"):
                    serialized_issues.append(issue.dict())
                else:
                    serialized_issues.append(issue)

            repo.save_processing_results(
                doc_id=document_id,
                total_pages=total_pages,
                pages_data=pages_data,
                consensus_data=consolidated_consensus,
                validation_issues=serialized_issues,
                routing_decision=routing_decision
            )

            logger.info(f"[JOB COMPLETE] Document ID: {document_id} processed successfully.")

        except Exception as e:
            logger.exception(f"[JOB ERROR] Document ID: {document_id} failed: {e}")
            try:
                db.rollback()
            except Exception as rollback_err:
                logger.error(f"Rollback failed: {rollback_err}")

            repo.update_status(
                doc_id=document_id,
                status="FAILED",
                progress=100.0,
                error=str(e)
            )
        finally:
            db.close()


# Singleton instance shared across FastAPI
job_manager = JobProcessor()


def process_document_background(doc_id: str = None, document_id: str = None, file_path: str = None, *args, **kwargs):
    """
    Background worker wrapper matching `main.py` signatures.
    Silently consumes extra pipeline service dependencies passed by main.py.
    """
    target_id = doc_id or document_id
    job_manager.process_document_job(document_id=target_id, file_path=file_path)


def process_document_job(doc_id: str = None, document_id: str = None, file_path: str = None, *args, **kwargs):
    """Alias for backwards compatibility."""
    target_id = doc_id or document_id
    job_manager.process_document_job(document_id=target_id, file_path=file_path)