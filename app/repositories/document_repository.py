import uuid
import logging
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
from app.models.document import Document

logger = logging.getLogger(__name__)


class DocumentRepository:
    """Handles database CRUD operations for document tracking and processing results."""

    def __init__(self, db: Session):
        self.db = db

    def create_document(self, filename: str, **kwargs) -> Document:
        """Creates a new document record in PENDING status."""
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
        """Retrieves a document by its primary key ID."""
        return self.db.query(Document).filter(Document.id == doc_id).first()

    def list_documents(self) -> List[Document]:
        """Retrieves all documents."""
        return self.db.query(Document).all()

    def update_status(
        self, 
        doc_id: str, 
        status: str, 
        progress: float = 0.0, 
        error: Optional[str] = None
    ) -> Optional[Document]:
        """Updates execution status, progress percentage, and error messages."""
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
        """Saves final 3-engine processing results to the database."""
        doc = self.get_document(doc_id)
        if doc:
            doc.total_pages = total_pages
            doc.status = routing_decision.get("status", "COMPLETED")
            doc.progress = 100.0
            doc.pages_data = pages_data
            doc.consensus_data = consensus_data
            doc.validation_issues = validation_issues
            doc.routing_decision = routing_decision
            doc.overall_confidence = routing_decision.get("overall_confidence", 0.0)
            doc.routing_reason = routing_decision.get("routing_reason", "")
            self.db.commit()
            self.db.refresh(doc)
        return doc

    def get_processing_results(self, doc_id: str) -> Dict[str, Any]:
        """Formats and returns structured results for API response."""
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