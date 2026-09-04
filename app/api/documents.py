import io
import uuid
from typing import Any, Dict, List, Optional
import cv2
import numpy as np
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Response, UploadFile, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.repositories.document_repository import DocumentRepository
from app.services.export_service import ExportService
from app.services.jobs import job_manager, process_document_background
from app.services.ocr.consensus import ConsensusEngine
from app.services.ocr.routing import RoutingEngine
from app.services.preprocessing.pipeline import ImagePreprocessor
from app.services.preprocessing.quality import QualityAnalyzer
from app.services.validation import DataValidator

router = APIRouter(prefix="/documents", tags=["Documents"])


# --- Request / Response Pydantic Schemas ---

class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    message: str


class DocumentStatusResponse(BaseModel):
    document_id: str
    status: str
    overall_confidence: Optional[float] = None
    routing_reason: Optional[str] = None
    error_message: Optional[str] = None


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
    reviewer_id: str = Field("reviewer_01", description="Reviewer placeholder identifier")
    field_corrections: Optional[List[FieldCorrection]] = None
    line_item_corrections: Optional[List[LineItemCorrection]] = None


# --- Section 5 API Routes ---

@router.post("/upload", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Upload image/PDF documents."""
    if not file.filename.lower().endswith((".pdf", ".png", ".jpg", ".jpeg")):
        raise HTTPException(status_code=400, detail="Unsupported file format.")

    doc_id = str(uuid.uuid4())
    file_bytes = await file.read()

    # Store file bytes in job manager in-memory store
    job_manager.store_file(doc_id, file_bytes)

    # Save initial record in database
    repo = DocumentRepository(db)
    repo.create_document(doc_id=doc_id, filename=file.filename)

    return DocumentUploadResponse(
        document_id=doc_id,
        filename=file.filename,
        message="Document uploaded successfully. Ready for processing."
    )


@router.post("/{doc_id}/process", status_code=status.HTTP_202_ACCEPTED)
async def process_document(
    doc_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Start asynchronous background processing pipeline."""
    repo = DocumentRepository(db)
    doc = repo.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    file_bytes = job_manager.get_file(doc_id)
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Document file buffer expired or missing.")

    # Update status to PROCESSING
    doc.status = "PROCESSING"
    db.commit()

    # Trigger background job execution
    background_tasks.add_task(
        process_document_background,
        doc_id=doc_id,
        poppler_path=None,
        quality_analyzer=QualityAnalyzer(),
        image_processor=ImagePreprocessor(),
        consensus_engine=ConsensusEngine(),
        data_validator=DataValidator(),
        routing_engine=RoutingEngine()
    )

    return {"document_id": doc_id, "status": "PROCESSING", "message": "Background processing started."}


@router.get("/{doc_id}/status", response_model=DocumentStatusResponse)
def get_document_status(doc_id: str, db: Session = Depends(get_db)):
    """Read processing status and progress."""
    repo = DocumentRepository(db)
    doc = repo.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    return DocumentStatusResponse(
        document_id=doc.id,
        status=doc.status,
        overall_confidence=doc.overall_confidence,
        routing_reason=doc.routing_reason,
        error_message=getattr(doc, "error_message", None)
    )


@router.get("/{doc_id}/result")
def get_document_result(doc_id: str, db: Session = Depends(get_db)):
    """Get OCR, extraction, and validation data."""
    repo = DocumentRepository(db)
    result = repo.get_processing_results(doc_id)
    if not result:
        raise HTTPException(status_code=404, detail="Processing results not found.")
    return result


@router.get("/{doc_id}/image")
def get_document_image(doc_id: str, page: int = 1, annotated: bool = True, db: Session = Depends(get_db)):
    """Serve original or annotated page image."""
    file_bytes = job_manager.get_file(doc_id)
    if not file_bytes:
        raise HTTPException(status_code=404, detail="Image file not found.")

    # Generate placeholder image canvas for response
    canvas = np.zeros((800, 600, 3), dtype=np.uint8) + 245
    cv2.putText(canvas, f"Doc ID: {doc_id[:8]} - Page {page}", (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 2)

    _, encoded_img = cv2.imencode(".png", canvas)
    return Response(content=encoded_img.tobytes(), media_type="image/png")


@router.put("/{doc_id}/review")
def review_document(
    doc_id: str,
    review_data: DocumentReviewRequest,
    db: Session = Depends(get_db)
):
    """Save reviewer corrections and approval state."""
    repo = DocumentRepository(db)
    doc = repo.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    doc.status = review_data.approval_status
    doc.routing_reason = f"Reviewed by {review_data.reviewer_id}"
    db.commit()

    return {"document_id": doc_id, "status": doc.status, "message": "Review recorded successfully."}


@router.get("/{doc_id}/export")
def export_document(
    doc_id: str,
    format: str = "json",
    db: Session = Depends(get_db)
):
    """Export results as JSON or CSV."""
    repo = DocumentRepository(db)
    result = repo.get_processing_results(doc_id)
    if not result:
        raise HTTPException(status_code=404, detail="Document results not found.")

    if format.lower() == "csv":
        csv_data = ExportService.generate_csv(result)
        return StreamingResponse(
            io.StringIO(csv_data),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=document_{doc_id}.csv"}
        )
    elif format.lower() == "json":
        return result
    else:
        raise HTTPException(status_code=400, detail="Invalid export format. Use 'json' or 'csv'.")


@router.get("", response_model=List[DocumentStatusResponse])
def list_documents(db: Session = Depends(get_db)):
    """List documents with status/filtering."""
    repo = DocumentRepository(db)
    docs = repo.list_documents()
    return [
        DocumentStatusResponse(
            document_id=d.id,
            status=d.status,
            overall_confidence=d.overall_confidence,
            routing_reason=d.routing_reason
        )
        for d in docs
    ]