import uuid
import logging
import json
import numpy as np
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
from app.models.document import Document

logger = logging.getLogger(__name__)


# Custom JSON encoder to convert NumPy data types to standard Python types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class DocumentRepository:
    """Handles database CRUD operations for document tracking and processing results."""

    def __init__(self, db: Session):
        self.db = db

    def create_document(self, filename: str, **kwargs) -> Document:
        doc_id = str(uuid.uuid4())
        doc = Document(
            id=doc_id,
            filename=filename,
            file_type=kwargs.get("content_type", "application/pdf"),
            status="PENDING",
            progress=0.0
        )
        self.db.add(doc)
        self.db.commit()
        self.db.refresh(doc)
        return doc

    def get_document(self, doc_id: str) -> Optional[Document]:
        return self.db.query(Document).filter(Document.id == doc_id).first()

    def list_documents(self) -> List[Document]:
        return self.db.query(Document).all()

    def update_status(
        self, 
        doc_id: str, 
        status: str, 
        progress: float = 0.0, 
        error: Optional[str] = None
    ) -> Optional[Document]:
        doc = self.get_document(doc_id)
        if doc:
            doc.status = status
            doc.progress = progress
            if error:
                doc.error_message = error
            self.db.commit()
            self.db.refresh(doc)
        return doc

    def save_processing_results(
        self,
        doc_id: str,
        total_pages: int,
        pages_data: List[Dict[str, Any]],
        consensus_data: Dict[str, Any],
        validation_issues: List[Dict[str, Any]],
        routing_decision: Dict[str, Any]
    ) -> Optional[Document]:
        # Clean numpy types before sending to the database
        pages_data = json.loads(json.dumps(pages_data, cls=NumpyEncoder))
        consensus_data = json.loads(json.dumps(consensus_data, cls=NumpyEncoder))
        validation_issues = json.loads(json.dumps(validation_issues, cls=NumpyEncoder))
        routing_decision = json.loads(json.dumps(routing_decision, cls=NumpyEncoder))

        doc = self.get_document(doc_id)
        if doc:
            doc.total_pages = total_pages
            doc.status = routing_decision.get("status", "COMPLETED")
            doc.progress = 100.0
            doc.pages_data = pages_data
            doc.consensus_data = consensus_data
            doc.validation_issues = validation_issues
            doc.routing_decision = routing_decision
            doc.overall_confidence = routing_decision.get("overall_confidence", routing_decision.get("confidence_score", 0.0))
            doc.routing_reason = routing_decision.get("routing_reason", "")
            self.db.commit()
            self.db.refresh(doc)
        return doc

    def get_processing_results(self, doc_id: str) -> Dict[str, Any]:
        doc = self.get_document(doc_id)
        if not doc:
            return {}

        return {
            "document_id": doc.id,
            "filename": doc.filename,
            "status": doc.status,
            "progress": getattr(doc, "progress", 100.0),
            "overall_confidence": getattr(doc, "overall_confidence", 0.0),
            "routing_reason": getattr(doc, "routing_reason", None),
            "pages_data": getattr(doc, "pages_data", []) or [],
            "consensus_data": getattr(doc, "consensus_data", {}) or {},
            "validation_issues": getattr(doc, "validation_issues", []) or [],
            "routing_decision": getattr(doc, "routing_decision", {}) or {},
            "error_message": getattr(doc, "error_message", None)
        }