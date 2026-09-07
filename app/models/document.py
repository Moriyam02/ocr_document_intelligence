import uuid
from datetime import datetime
from sqlalchemy import Column, String, Float, Integer, DateTime, JSON, Text
from sqlalchemy.orm import relationship
from app.core.database import Base


class Document(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String, nullable=False)
    file_path = Column(String, nullable=True)
    file_type = Column(String, nullable=True, default="application/pdf")
    doc_category = Column(String, default="invoice")
    status = Column(String, default="PENDING")
    progress = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # OCR and Analysis Tracking Attributes
    total_pages = Column(Integer, default=1)
    overall_confidence = Column(Float, nullable=True, default=0.0)
    routing_reason = Column(String, nullable=True)
    error_message = Column(Text, nullable=True)

    # Structured Results Persistence (JSON Storage)
    pages_data = Column(JSON, default=list)
    consensus_data = Column(JSON, default=dict)
    validation_issues = Column(JSON, default=list)
    routing_decision = Column(JSON, default=dict)


# Alias for backward compatibility across modules
DocumentModel = Document