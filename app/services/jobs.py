import logging
from typing import Dict, Any, List, Optional
import cv2
import fitz  # PyMuPDF
import numpy as np
from PIL import Image

from app.core.database import SessionLocal
from app.repositories.document_repository import DocumentRepository
from app.services.preprocessing.quality import QualityAnalyzer
from app.services.preprocessing.pipeline import ImagePreprocessor
from app.services.extraction.parser import FieldExtractor
from app.services.extraction.table import TableExtractor
from app.services.extraction.layout import LayoutAnalyzer

logger = logging.getLogger(__name__)


class JobManager:
    """In-memory file store and job management singleton."""

    def __init__(self):
        self._file_store: Dict[str, bytes] = {}

    def store_file(self, doc_id: str, file_bytes: bytes):
        self._file_store[doc_id] = file_bytes

    def get_file(self, doc_id: str) -> Optional[bytes]:
        return self._file_store.get(doc_id)

    def remove_file(self, doc_id: str):
        self._file_store.pop(doc_id, None)


job_manager = JobManager()


def render_pdf_to_images(file_bytes: bytes) -> List[Image.Image]:
    """Renders PDF bytes directly to PIL Images in-memory using PyMuPDF."""
    images = []
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    for page in doc:
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    doc.close()
    return images


def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    """Converts PIL Image instance to OpenCV (BGR) numpy format."""
    open_cv_image = np.array(pil_image)
    if open_cv_image.ndim == 3 and open_cv_image.shape[2] == 3:
        return cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
    return open_cv_image


def process_document_background(
    doc_id: str,
    poppler_path: Optional[str],
    quality_analyzer: Any,
    image_processor: Any,
    consensus_engine: Any,
    data_validator: Any,
    routing_engine: Any
):
    """Background task executing multi-engine OCR and saving structured results."""
    logger.info(f"[JOB START] Processing Document ID: {doc_id}")

    db = SessionLocal()
    repo = DocumentRepository(db)

    qa = quality_analyzer or QualityAnalyzer()
    ip = image_processor or ImagePreprocessor()
    field_extractor = FieldExtractor()
    table_extractor = TableExtractor()
    layout_analyzer = LayoutAnalyzer()

    try:
        file_bytes = job_manager.get_file(doc_id)
        if not file_bytes:
            raise ValueError(f"No file bytes found for doc_id: {doc_id}")

        pil_images = render_pdf_to_images(file_bytes)
        total_pages = len(pil_images)

        pages_data = []

        for idx, page_img in enumerate(pil_images, start=1):
            quality_report = qa.analyze(page_img)
            recommended_profile = quality_report.get("recommended_profile", "BASIC")

            processed_page_img = ip.preprocess(page_img, profile=recommended_profile)
            engine_outputs = consensus_engine.run_all_engines(processed_page_img)

            extracted_fields_by_engine: Dict[str, Dict[str, Any]] = {}
            for engine_name, output in engine_outputs.items():
                parsed_schema = field_extractor.parse_fields(output)
                extracted_fields_by_engine[engine_name] = parsed_schema.model_dump()

            extracted_line_items_by_engine: Dict[str, List[Dict[str, Any]]] = {}
            for engine_name, output in engine_outputs.items():
                items = table_extractor.extract_line_items_from_engine_output(output)
                extracted_line_items_by_engine[engine_name] = [item.model_dump() for item in items]

            field_consensus = consensus_engine.compute_field_consensus(
                engine_outputs, extracted_fields_by_engine
            )

            pages_data.append({
                "page": idx,
                "quality_report": quality_report,
                "preprocessing_profile_used": recommended_profile,
                "engine_outputs": engine_outputs,
                "parsed_fields": extracted_fields_by_engine,
                "line_items": extracted_line_items_by_engine,
                "field_consensus": field_consensus,
            })

        # Run Data Validation Rules
        validation_issues = data_validator.validate(pages_data)

        # Consolidate Consensus
        consolidated_consensus = {}
        for p in pages_data:
            consolidated_consensus.update(p.get("field_consensus", {}))

        routing_decision = routing_engine.evaluate(consolidated_consensus, validation_issues)

        # Persist structured output to database
        repo.save_processing_results(
            doc_id,
            total_pages,
            pages_data,
            consolidated_consensus,
            validation_issues,
            routing_decision
        )

        logger.info(f"[JOB COMPLETE] Document ID: {doc_id} processed successfully.")

    except Exception as e:
        logger.error(f"[JOB FAILED] Document ID {doc_id}: {str(e)}", exc_info=True)
        repo.update_status(doc_id, status="FAILED", error=str(e))
    finally:
        db.close()
        job_manager.remove_file(doc_id)