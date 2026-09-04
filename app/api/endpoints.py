from typing import List, Optional
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Query, UploadFile, status
from fastapi.responses import CSVResponse, JSONResponse
from pydantic import BaseModel, Field

from app.services.ocr.consensus import ConsensusOutput, OCRConsensusEngine

router = APIRouter(prefix="/documents", tags=["documents"])
consensus_engine = OCRConsensusEngine()

# --- In-Memory Database / Cache Simulation ---
# In production, replace with SQLAlchemy / SQLite database sessions per Section 3
db_documents: dict = {}

# --- Request / Response Schemas ---
class UploadResponse(BaseModel):
    document_ids: List[str]
    message: str

class ProcessingStatusResponse(BaseModel):
    document_id: str
    status: str  # pending, processing, completed, failed
    progress: int  # 0 to 100 percentage

class HumanReviewPayload(BaseModel):
    reviewer_id: str
    approved: bool
    edited_fields: dict = Field(default_factory=dict)
    comments: Optional[str] = None

# --- Asynchronous Processing Task ---
def run_ocr_pipeline_task(doc_id: str):
    """Background task simulating document quality check, running 3 OCR engines, 

    and executing consensus evaluation (Section 2 & 4.4).
    """
    db_documents[doc_id]["status"] = "processing"
    db_documents[doc_id]["progress"] = 30

    # Simulated raw OCR outputs from Tesseract, EasyOCR, and PaddleOCR
    mock_ocr_outputs = [
        {
            "engine_name": "Tesseract",
            "confidence": 0.94,
            "raw_text": "Invoice # INV-1024 Total: $150.00 Date: 2026-08-01",
            "processing_time": 0.35,
            "text_coverage": 0.90,
            "validation_readiness": 0.95,
        },
        {
            "engine_name": "EasyOCR",
            "confidence": 0.88,
            "raw_text": "Invoice INV-1024 Total 150.00 Date 2026-08-10",
            "processing_time": 0.85,
            "text_coverage": 0.85,
            "validation_readiness": 0.80,
        },
        {
            "engine_name": "PaddleOCR",
            "confidence": 0.96,
            "raw_text": "Invoice # INV-1024 Total: $150.00 Date: 2026-08-01",
            "processing_time": 0.42,
            "text_coverage": 0.98,
            "validation_readiness": 0.98,
        },
    ]

    # Run Consensus Engine (Option C Implementation)
    consensus_result = consensus_engine.process(doc_id, mock_ocr_outputs)
    
    # Store processed state
    db_documents[doc_id]["result"] = consensus_result.model_dump()
    db_documents[doc_id]["status"] = "completed"
    db_documents[doc_id]["progress"] = 100


# --- Endpoints (Section 5 Spec) ---

@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_documents(files: List[UploadFile] = File(...)):
    """POST /documents/upload - Accepts single, multi-page, or batch document uploads."""
    if not files or len(files) > 10:
        raise HTTPException(
            status_code=400, 
            detail="You must upload between 1 and 10 documents per batch."
        )
    
    uploaded_ids = []
    import uuid
    for file in files:
        doc_id = str(uuid.uuid4())
        db_documents[doc_id] = {
            "id": doc_id,
            "filename": file.filename,
            "status": "uploaded",
            "progress": 0,
            "result": None,
            "review": None
        }
        uploaded_ids.append(doc_id)
        
    return {
        "document_ids": uploaded_ids,
        "message": f"Successfully uploaded {len(uploaded_ids)} file(s). Call /process to start OCR."
    }


@router.post("/{doc_id}/process", status_code=status.HTTP_202_ACCEPTED)
async def start_processing(doc_id: str, background_tasks: BackgroundTasks):
    """POST /documents/{id}/process - Starts asynchronous OCR processing in background."""
    if doc_id not in db_documents:
        raise HTTPException(status_code=404, detail="Document ID not found.")
        
    background_tasks.add_task(run_ocr_pipeline_task, doc_id)
    return {"document_id": doc_id, "message": "Asynchronous OCR processing initiated."}


@router.get("/{doc_id}/status", response_model=ProcessingStatusResponse)
async def get_processing_status(doc_id: str):
    """GET /documents/{id}/status - Retrieves processing state and percentage."""
    if doc_id not in db_documents:
        raise HTTPException(status_code=404, detail="Document ID not found.")
    
    doc = db_documents[doc_id]
    return {
        "document_id": doc_id,
        "status": doc["status"],
        "progress": doc["progress"]
    }


@router.get("/{doc_id}/result", response_model=ConsensusOutput)
async def get_document_result(doc_id: str):
    """GET /documents/{id}/result - Returns consensus OCR, score evaluation, and extracted fields."""
    if doc_id not in db_documents:
        raise HTTPException(status_code=404, detail="Document ID not found.")
        
    doc = db_documents[doc_id]
    if doc["status"] != "completed" or not doc["result"]:
        raise HTTPException(status_code=400, detail="Document processing is not complete yet.")
        
    return doc["result"]


@router.put("/{doc_id}/review")
async def save_human_review(doc_id: str, review: HumanReviewPayload):
    """PUT /documents/{id}/review - Saves reviewer corrections & approval state (Section 4.10)."""
    if doc_id not in db_documents:
        raise HTTPException(status_code=404, detail="Document ID not found.")
        
    db_documents[doc_id]["review"] = review.model_dump()
    return {"document_id": doc_id, "status": "Review saved successfully."}


@router.get("/{doc_id}/export")
async def export_document(doc_id: str, format: str = Query(..., regex="^(json|csv)$")):
    """GET /documents/{id}/export?format=json|csv - Exports finalized structured results."""
    if doc_id not in db_documents or not db_documents[doc_id]["result"]:
        raise HTTPException(status_code=404, detail="Document or completed result not found.")
        
    result_data = db_documents[doc_id]["result"]
    
    if format == "json":
        return JSONResponse(content=result_data)
    elif format == "csv":
        # Basic flattened representation of field consensus for CSV export
        fields = result_data.get("field_consensus", {})
        csv_lines = ["Field Name,Selected Value,Confidence,Discrepancy Flagged\n"]
        for key, val in fields.items():
            csv_lines.append(f"{key},{val['selected_value']},{val['confidence_score']},{val['discrepancy_flagged']}\n")
        
        return CSVResponse(content="".join(csv_lines), media_type="text/csv")