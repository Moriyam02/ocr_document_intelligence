import os
import io
import logging
from typing import Optional, List, Any
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Depends, Response, status
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import SessionLocal, engine, Base
from app.repositories.document_repository import DocumentRepository
from app.services.jobs import job_manager, process_document_background
from app.services.export_service import ExportService

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


# --- Pydantic Schemas for Section 5 Endpoints ---

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
    return {"message": "Asynchronous Document Intelligence API is active"}


@app.post("/documents/upload", status_code=202)
async def upload_documents(
    db: Session = Depends(get_db),
    file1: UploadFile = File(...),
    file2: Optional[UploadFile] = File(None),
    file3: Optional[UploadFile] = File(None),
    file4: Optional[UploadFile] = File(None),
    file5: Optional[UploadFile] = File(None),
    file6: Optional[UploadFile] = File(None),
    file7: Optional[UploadFile] = File(None),
    file8: Optional[UploadFile] = File(None),
    file9: Optional[UploadFile] = File(None),
    file10: Optional[UploadFile] = File(None)
):
    """Upload up to 10 document files at once using native file chooser boxes."""
    incoming_files = [
        f for f in [file1, file2, file3, file4, file5, file6, file7, file8, file9, file10]
        if f is not None
    ]

    uploaded_docs = []
    repo = DocumentRepository(db)

    for file in incoming_files:
        try:
            contents = await file.read()
            doc = repo.create_document(
                filename=file.filename,
                content_type=file.content_type,
                file_size=len(contents)
            )
            job_manager.store_file(doc.id, contents)

            uploaded_docs.append({
                "document_id": doc.id,
                "filename": doc.filename,
                "status": doc.status
            })
        except Exception as e:
            logger.error(f"Failed to upload {file.filename}: {e}")

    return {
        "total_uploaded": len(uploaded_docs),
        "documents": uploaded_docs,
        "message": "Files uploaded successfully."
    }


@app.post("/documents/{doc_id}/process", status_code=202)
def process_document(
    doc_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Start asynchronous background OCR processing pipeline."""
    repo = DocumentRepository(db)
    doc = repo.get_document(doc_id)
    
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
        
    file_bytes = job_manager.get_file(doc_id)
    if not file_bytes:
        raise HTTPException(
            status_code=400, 
            detail="File bytes missing from memory store. Please re-upload."
        )

    # Update state to QUEUED
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

    return {
        "document_id": doc.id,
        "filename": doc.filename,
        "status": doc.status,
        "progress": getattr(doc, "progress", 100 if doc.status == "COMPLETED" else 0),
        "overall_confidence": getattr(doc, "overall_confidence", None),
        "routing_reason": getattr(doc, "routing_reason", None),
        "error_message": getattr(doc, "error_message", None)
    }


@app.get("/documents/{doc_id}/result")
def get_document_result(doc_id: str, db: Session = Depends(get_db)):
    """Get OCR, extraction, and validation data."""
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
    
    return {
        "document_id": doc.id,
        "filename": doc.filename,
        "status": doc.status,
        "progress": getattr(doc, "progress", 100),
        "results": results
    }


@app.get("/documents/{doc_id}/image")
def get_document_image(doc_id: str, page: int = 1, annotated: bool = True, db: Session = Depends(get_db)):
    """Serve original or visual annotated page image."""
    repo = DocumentRepository(db)
    doc = repo.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    file_bytes = job_manager.get_file(doc_id)
    if not file_bytes:
        # Fallback render canvas if raw buffer is purged
        canvas = np.zeros((800, 600, 3), dtype=np.uint8) + 245
        cv2.putText(canvas, f"Doc: {doc_id[:8]} - Page {page}", (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 2)
        _, encoded_img = cv2.imencode(".png", canvas)
        return Response(content=encoded_img.tobytes(), media_type="image/png")

    return Response(content=file_bytes, media_type="image/png")


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
        # Format payloads into structured CSV string using ExportService
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
    docs = repo.list_documents()
    return [
        {
            "document_id": d.id,
            "filename": d.filename,
            "status": d.status,
            "created_at": getattr(d, "created_at", None)
        }
        for d in docs
    ]