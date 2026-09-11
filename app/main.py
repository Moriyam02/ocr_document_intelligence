import os
import io
import logging
from typing import Optional, List, Any
import cv2
import numpy as np
import pymupdf as fitz  # for rendering PDF preview pages -- already a project dependency (used in jobs.py)
from PIL import Image
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Depends, Response, status
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import SessionLocal, engine, Base
from app.repositories.document_repository import DocumentRepository
from app.services.jobs import job_manager, process_document_background
from app.services.export_service import ExportService

# Import Document model directly to prevent repository method missing errors
try:
    from app.models.document import Document
except ImportError:
    try:
        from app.models import Document
    except ImportError:
        Document = None

# Service Imports
from app.services.preprocessing.quality import QualityAnalyzer
from app.services.preprocessing.pipeline import ImagePreprocessor
from app.services.ocr.consensus import OCRConsensusEngine
from app.services.validation.rules import DataValidator
from app.services.validation.confidence_router import RoutingEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Asynchronous Document Intelligence API",
    version="1.0.0",
    description="Production-grade OCR service powered by Tesseract, EasyOCR, and PaddleOCR."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Mount Static Directory for Frontend Assets ---
if os.path.exists("app/static"):
    app.mount("/static", StaticFiles(directory="app/static"), name="static")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Initialize processing singletons
quality_analyzer = QualityAnalyzer()
image_processor = ImagePreprocessor()
consensus_engine = OCRConsensusEngine()
data_validator = DataValidator()
routing_engine = RoutingEngine()


# --- Pydantic Schemas ---

class FieldCorrection(BaseModel):
    field_name: str
    new_value: Any
    reason: Optional[str] = None


class LineItemCorrection(BaseModel):
    description: str
    quantity: float
    unit_price: float
    line_total: float


class DocumentReviewRequest(BaseModel):
    approval_status: str = Field(..., description="APPROVED or REJECTED")
    reviewer_id: str = Field("reviewer_01", description="Reviewer identifier")
    field_corrections: Optional[List[FieldCorrection]] = None
    line_item_corrections: Optional[List[LineItemCorrection]] = None


# --- Core API Routes ---

@app.get("/")
def read_root():
    """Serves the Frontend UI dashboard if index.html exists, or fallback JSON.

    FIX: previously declared response_class=FileResponse but fell back to
    returning a plain dict, which is invalid for FileResponse and would raise
    at runtime if index.html were ever missing. Removed the fixed response_class
    and wrapped the fallback in JSONResponse so both branches are valid.
    """
    root_index = os.path.join(os.path.dirname(__file__), "index.html")
    template_index = os.path.join("app", "templates", "index.html")

    if os.path.exists(root_index):
        return FileResponse(root_index)
    elif os.path.exists(template_index):
        return FileResponse(template_index)

    return JSONResponse({"message": "Asynchronous Document Intelligence API is active"})


@app.post("/documents/upload", status_code=status.HTTP_202_ACCEPTED)
async def upload_documents(
    background_tasks: BackgroundTasks,
    file1: UploadFile = File(..., description="Primary file to upload."),
    file2: Optional[UploadFile] = File(None, description="Optional secondary file."),
    file3: Optional[UploadFile] = File(None, description="Optional 3rd file."),
    file4: Optional[UploadFile] = File(None, description="Optional 4th file."),
    file5: Optional[UploadFile] = File(None, description="Optional 5th file."),
    file6: Optional[UploadFile] = File(None, description="Optional 6th file."),
    file7: Optional[UploadFile] = File(None, description="Optional 7th file."),
    file8: Optional[UploadFile] = File(None, description="Optional 8th file."),
    file9: Optional[UploadFile] = File(None, description="Optional 9th file."),
    file10: Optional[UploadFile] = File(None, description="Optional 10th file."),
    auto_process: bool = True,
    db: Session = Depends(get_db)
):
    """
    Upload up to 10 document files simultaneously using native file chooser inputs.
    Triggers background OCR processing automatically when auto_process=True.
    """
    incoming_files = [
        f for f in [file1, file2, file3, file4, file5, file6, file7, file8, file9, file10]
        if f is not None and hasattr(f, "filename") and f.filename and f.filename.strip()
    ]

    if not incoming_files:
        raise HTTPException(
            status_code=400,
            detail="No valid files selected. Please upload at least one document."
        )

    uploaded_docs = []
    repo = DocumentRepository(db)

    for file in incoming_files:
        if not file.filename.lower().endswith((".pdf", ".png", ".jpg", ".jpeg")):
            logger.warning(f"Skipping unsupported file format: {file.filename}")
            continue

        try:
            contents = await file.read()
            if not contents:
                continue

            doc = repo.create_document(
                filename=file.filename,
                content_type=file.content_type,
                file_size=len(contents)
            )

            job_manager.store_file(doc.id, contents)
            initial_status = doc.status

            if auto_process:
                doc.status = "QUEUED"
                db.commit()
                initial_status = "QUEUED"

                background_tasks.add_task(
                    process_document_background,
                    doc_id=doc.id,
                    poppler_path=getattr(settings, "POPPLER_PATH", None),
                    quality_analyzer=quality_analyzer,
                    image_processor=image_processor,
                    consensus_engine=consensus_engine,
                    data_validator=data_validator,
                    routing_engine=routing_engine
                )

            uploaded_docs.append({
                "document_id": doc.id,
                "filename": doc.filename,
                "status": initial_status
            })

        except Exception as e:
            logger.error(f"Failed to process upload for {file.filename}: {e}")

    if not uploaded_docs:
        raise HTTPException(
            status_code=400,
            detail="None of the uploaded files were valid PDFs or images."
        )

    return {
        "total_uploaded": len(uploaded_docs),
        "document_ids": [d["document_id"] for d in uploaded_docs],
        "documents": uploaded_docs,
        "message": f"Successfully uploaded and queued {len(uploaded_docs)} document(s)." if auto_process else "Files uploaded successfully."
    }


@app.post("/documents/{doc_id}/process", status_code=status.HTTP_202_ACCEPTED)
def process_document(
    doc_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Manually start or restart asynchronous background OCR processing pipeline."""
    repo = DocumentRepository(db)
    doc = repo.get_document(doc_id)

    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    file_bytes = job_manager.get_file(doc_id)
    if not file_bytes:
        raise HTTPException(
            status_code=400,
            detail="File bytes missing from memory store. Please re-upload the document."
        )

    doc.status = "QUEUED"
    db.commit()

    background_tasks.add_task(
        process_document_background,
        doc_id=doc_id,
        poppler_path=getattr(settings, "POPPLER_PATH", None),
        quality_analyzer=quality_analyzer,
        image_processor=image_processor,
        consensus_engine=consensus_engine,
        data_validator=data_validator,
        routing_engine=routing_engine
    )

    return {
        "document_id": doc_id,
        "status": "QUEUED",
        "message": "Background OCR job initiated."
    }


@app.get("/documents/{doc_id}/status")
def get_document_status(doc_id: str, db: Session = Depends(get_db)):
    """Read processing status and progress."""
    repo = DocumentRepository(db)
    doc = repo.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    status_lower = (doc.status or "").lower()
    return {
        "document_id": doc.id,
        "filename": doc.filename,
        "status": status_lower,
        "progress": getattr(doc, "progress", 100 if doc.status in ["COMPLETED", "APPROVED", "REJECTED"] else 0),
        "overall_confidence": getattr(doc, "overall_confidence", None),
        "routing_reason": getattr(doc, "routing_reason", None),
        "error_message": getattr(doc, "error_message", None)
    }


@app.get("/documents/{doc_id}/result")
def get_document_result(doc_id: str, db: Session = Depends(get_db)):
    """Get OCR, field extraction, and validation consensus results."""
    repo = DocumentRepository(db)
    doc = repo.get_document(doc_id)

    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    if doc.status == "FAILED":
        raise HTTPException(
            status_code=400,
            detail=f"Job processing failed: {getattr(doc, 'error_message', 'Unknown error')}"
        )

    results = repo.get_processing_results(doc_id)

    # FIX: repo.get_processing_results() stores field data under "consensus_data",
    # not "field_consensus" -- that key never existed, so this was always {}.
    # Fall back to consensus_data so the frontend actually receives field data.
    field_consensus = {}
    if isinstance(results, dict):
        field_consensus = results.get("field_consensus") or results.get("consensus_data") or {}

    return {
        "document_id": doc.id,
        "filename": doc.filename,
        "status": doc.status,
        "progress": getattr(doc, "progress", 100),
        "results": results,
        "field_consensus": field_consensus
    }


@app.get("/documents/{doc_id}/image")
def get_document_image(doc_id: str, page: int = 1, annotated: bool = True, db: Session = Depends(get_db)):
    """Serve original page as a PNG preview -- handles both PDF and raw image uploads.

    FIX: previously this always returned the raw stored bytes labeled as
    "image/png" regardless of actual file type. That's broken for PDF uploads
    (a PDF's bytes are not a valid PNG, so <img> tags render nothing) and was
    also fragile for non-PNG images. Now PDFs are rendered to an actual PNG
    page image via pymupdf, and other images are normalized to PNG via PIL,
    so the response is always a genuinely valid image regardless of source format.
    """
    repo = DocumentRepository(db)
    doc = repo.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    file_bytes = job_manager.get_file(doc_id)

    def _placeholder(message: str) -> Response:
        canvas = np.zeros((800, 600, 3), dtype=np.uint8) + 245
        cv2.putText(canvas, message, (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 2)
        _, encoded_img = cv2.imencode(".png", canvas)
        return Response(content=encoded_img.tobytes(), media_type="image/png")

    if not file_bytes:
        return _placeholder(f"Doc: {doc_id[:8]} - Page {page}")

    try:
        if file_bytes[:4] == b"%PDF":
            pdf_doc = fitz.open(stream=file_bytes, filetype="pdf")
            try:
                page_index = max(0, min(page - 1, len(pdf_doc) - 1))
                pix = pdf_doc[page_index].get_pixmap(dpi=150)
                png_bytes = pix.tobytes("png")
            finally:
                pdf_doc.close()
            return Response(content=png_bytes, media_type="image/png")
        else:
            img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return Response(content=buf.getvalue(), media_type="image/png")
    except Exception as e:
        logger.error(f"Failed to render preview image for document {doc_id}: {e}")
        return _placeholder("Preview unavailable")


@app.put("/documents/{doc_id}/review")
def review_document(doc_id: str, review_data: DocumentReviewRequest, db: Session = Depends(get_db)):
    """Save reviewer human-in-the-loop corrections and approval state."""
    repo = DocumentRepository(db)
    doc = repo.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    doc.status = review_data.approval_status
    if hasattr(doc, "routing_reason"):
        doc.routing_reason = f"Reviewed by {review_data.reviewer_id}"
    db.commit()

    return {
        "document_id": doc_id,
        "status": doc.status,
        "message": f"Document status updated to {review_data.approval_status}."
    }


@app.get("/documents/{doc_id}/export")
def export_document(doc_id: str, format: str = "json", db: Session = Depends(get_db)):
    """Export approved results as JSON or CSV."""
    repo = DocumentRepository(db)
    results = repo.get_processing_results(doc_id)
    if not results:
        raise HTTPException(status_code=404, detail="Processing results not found")

    if format.lower() == "csv":
        payload = results if isinstance(results, dict) else {"document_id": doc_id, "pages": results}
        csv_data = ExportService.generate_csv(payload)
        return StreamingResponse(
            io.StringIO(csv_data),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=document_{doc_id}.csv"}
        )
    elif format.lower() == "json":
        return results
    else:
        raise HTTPException(status_code=400, detail="Invalid export format. Supported formats: 'json', 'csv'.")


@app.get("/documents")
def list_documents(db: Session = Depends(get_db)):
    """List all documents with processing status."""
    repo = DocumentRepository(db)

    if hasattr(repo, "list_documents") and callable(getattr(repo, "list_documents")):
        docs = repo.list_documents()
    elif Document is not None:
        docs = db.query(Document).all()
    else:
        docs = []

    return [
        {
            "document_id": getattr(d, "id", str(d)),
            "filename": getattr(d, "filename", "Unknown"),
            "status": getattr(d, "status", "UNKNOWN"),
            "created_at": str(getattr(d, "created_at", "")) if getattr(d, "created_at", None) else None
        }
        for d in docs
    ]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)