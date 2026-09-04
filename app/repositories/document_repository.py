import uuid
import logging
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session

# Safe import for the model class inside app/models/document.py
try:
    from app.models.document import Document
except ImportError:
    from app.models.document import DocumentModel as Document

logger = logging.getLogger(__name__)


class DocumentRepository:
    """Handles database CRUD operations for document tracking and processing results."""

    def __init__(self, db: Session):
        self.db = db

    def create_document(self, filename: str, **kwargs) -> Document:
        """Creates a new document record in PENDING status.
        
        Using **kwargs allows it to safely receive extra params like content_type/file_size.
        """
        doc_id = str(uuid.uuid4())
        doc = Document(
            id=doc_id,
            filename=filename,
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
            doc.status = routing_decision.get("status", "COMPLETED")
            doc.progress = 1.0
            doc.pages_data = pages_data
            doc.consensus_data = consensus_data
            doc.validation_issues = validation_issues
            doc.routing_decision = routing_decision
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
            "progress": doc.progress,
            "pages_data": getattr(doc, "pages_data", []),
            "consensus_data": getattr(doc, "consensus_data", {}),
            "validation_issues": getattr(doc, "validation_issues", []),
            "routing_decision": getattr(doc, "routing_decision", {}),
            "error_message": getattr(doc, "error_message", None)
        }