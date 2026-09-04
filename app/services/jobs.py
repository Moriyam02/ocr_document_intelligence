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

# Extraction services module imports
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
    """
    Background task executing PDF rendering, quality assessment, image preprocessing, 
    multi-engine OCR, field & table extraction, consensus, validation, and routing.
    """
    logger.info(f"[JOB START] Processing Document ID: {doc_id}")

    db = SessionLocal()
    repo = DocumentRepository(db)

    # Instantiate quality analyzer, image processor, and extraction modules
    qa = quality_analyzer or QualityAnalyzer()
    ip = image_processor or ImagePreprocessor()
    field_extractor = FieldExtractor()
    table_extractor = TableExtractor()
    layout_analyzer = LayoutAnalyzer()

    try:
        file_bytes = job_manager.get_file(doc_id)
        if not file_bytes:
            raise ValueError(f"No file bytes found for doc_id: {doc_id}")

        # 1. Render PDF pages in-memory using PyMuPDF
        logger.info("[INFO] Rendering document pages via PyMuPDF...")
        pil_images = render_pdf_to_images(file_bytes)
        total_pages = len(pil_images)
        logger.info(f"[INFO] Successfully rendered {total_pages} page(s).")

        pages_data = []

        # 2. Process each page
        for idx, page_img in enumerate(pil_images, start=1):
            logger.info(f"[INFO] Assessing Quality and Preprocessing Page {idx}...")

            # Run Image Quality Assessment
            quality_report = qa.analyze(page_img)
            recommended_profile = quality_report.get("recommended_profile", "BASIC")

            logger.info(
                f"[QUALITY] Page {idx} - Label: {quality_report.get('quality_label')} | "
                f"Blur: {quality_report.get('blur_score')} | Recommended Profile: {recommended_profile}"
            )

            # Apply Image Preprocessing Pipeline
            processed_page_img = ip.preprocess(page_img, profile=recommended_profile)

            # Run Multi-Engine OCR Consensus
            logger.info(f"[INFO] Running OCR Consensus on Preprocessed Page {idx}...")
            engine_outputs = consensus_engine.run_all_engines(processed_page_img)

            # Extract extended key-value fields per OCR engine using FieldExtractor
            extracted_fields_by_engine: Dict[str, Dict[str, Any]] = {}
            for engine_name, output in engine_outputs.items():
                parsed_schema = field_extractor.parse_fields(output)
                extracted_fields_by_engine[engine_name] = parsed_schema.model_dump()

            # Extract line-item tables per OCR engine using TableExtractor
            extracted_line_items_by_engine: Dict[str, List[Dict[str, Any]]] = {}
            for engine_name, output in engine_outputs.items():
                items = table_extractor.extract_line_items_from_engine_output(output)
                extracted_line_items_by_engine[engine_name] = [item.model_dump() for item in items]

            # Compute Field Consensus across OCR engines
            field_consensus = consensus_engine.compute_field_consensus(
                engine_outputs, extracted_fields_by_engine
            )

            # Convert page to OpenCV format for visual layout annotation
            cv_img = pil_to_cv2(processed_page_img if isinstance(processed_page_img, Image.Image) else page_img)
            annotations = []
            
            # Draw spatial bounding box overlays from primary engine output tokens
            primary_output = engine_outputs.get("tesseract", list(engine_outputs.values())[0] if engine_outputs else {})
            if isinstance(primary_output, dict):
                tokens = primary_output.get("tokens", primary_output.get("words", []))
                for tok in tokens:
                    bbox = tok.get("bbox")
                    text = tok.get("text", "").strip()
                    if bbox and len(bbox) == 4 and text:
                        norm_bbox = layout_analyzer.normalize_bbox(bbox, (cv_img.shape[1], cv_img.shape[0]))
                        annotations.append(
                            layout_analyzer.FieldAnnotation(
                                field_name="token",
                                text=text,
                                bbox=norm_bbox,
                                confidence=float(tok.get("confidence", 0.9))
                            )
                        )
            
            annotated_img = layout_analyzer.render_bounding_boxes(cv_img, annotations)

            pages_data.append({
                "page": idx,
                "quality_report": quality_report,
                "preprocessing_profile_used": recommended_profile,
                "engine_outputs": engine_outputs,
                "parsed_fields": extracted_fields_by_engine,
                "line_items": extracted_line_items_by_engine,
                "field_consensus": field_consensus,
            })

        # 3. Run Data Validation Rules
        logger.info("[INFO] Running Data Validation rules...")
        validation_issues = data_validator.validate(pages_data)

        # 4. Evaluate Routing & Build Consolidated Consensus Data
        logger.info("[INFO] Evaluating Confidence & Routing Decision...")
        consolidated_consensus = {}
        for p in pages_data:
            consolidated_consensus.update(p.get("field_consensus", {}))

        routing_decision = routing_engine.evaluate(consolidated_consensus, validation_issues)

        # 5. Persist processing output passing all positional arguments
        repo.save_processing_results(
            doc_id,
            total_pages,
            pages_data,
            consolidated_consensus,
            validation_issues,
            routing_decision
        )

        # Update document routing status
        doc = repo.get_document(doc_id)
        if doc:
            doc.status = routing_decision.get("status", "COMPLETED")
            doc.overall_confidence = routing_decision.get("overall_confidence", 0.0)
            doc.routing_reason = routing_decision.get("routing_reason", "")
            db.commit()

        logger.info(
            f"[JOB COMPLETE] Successfully processed Document ID: {doc_id} | "
            f"Status: {routing_decision.get('status')}"
        )

    except Exception as e:
        logger.error(f"[JOB FAILED] Document ID {doc_id} failed with error: {str(e)}", exc_info=True)
        try:
            doc = repo.get_document(doc_id)
            if doc:
                doc.status = "FAILED"
                doc.error_message = str(e)
                db.commit()
        except Exception as db_err:
            logger.error(f"Failed to record failure state in database: {db_err}")
    finally:
        db.close()
        job_manager.remove_file(doc_id)