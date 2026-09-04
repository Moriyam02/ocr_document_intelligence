import uuid
from datetime import datetime
from sqlalchemy import Column, String, Float, Integer, DateTime, ForeignKey, JSON, Text
from sqlalchemy.orm import relationship
from app.core.database import Base


class DocumentModel(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String, nullable=False)
    status = Column(String, default="QUEUED")
    progress = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    total_pages = Column(Integer, default=1)
    routing_action = Column(String, nullable=True)
    error = Column(Text, nullable=True)

    pages = relationship("DocumentPageModel", back_populates="document", cascade="all, delete-orphan")
    consensus_result = relationship("ConsensusResultModel", uselist=False, back_populates="document", cascade="all, delete-orphan")


class DocumentPageModel(Base):
    __tablename__ = "document_pages"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(String, ForeignKey("documents.id"))
    page_number = Column(Integer, nullable=False)
    quality_label = Column(String, nullable=True)
    blur_score = Column(Float, nullable=True)
    recommended_profile = Column(String, nullable=True)
    ocr_raw_outputs = Column(JSON, nullable=True)

    document = relationship("DocumentModel", back_populates="pages")


class ConsensusResultModel(Base):
    __tablename__ = "consensus_results"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(String, ForeignKey("documents.id"))
    field_consensus = Column(JSON, nullable=True)
    validation_issues = Column(JSON, nullable=True)
    routing_decision = Column(JSON, nullable=True)

    document = relationship("DocumentModel", back_populates="consensus_result")