from sqlalchemy import create_engine, Column, String, Integer, Float, Text, ForeignKey, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime
from app.core.config import settings

engine = create_engine(
    settings.DATABASE_URL, 
    connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Document(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    file_type = Column(String, nullable=False)
    doc_category = Column(String, default="invoice")
    status = Column(String, default="uploaded")
    created_at = Column(DateTime, default=datetime.utcnow)

    quality_reports = relationship("QualityReport", back_populates="document", cascade="all, delete-orphan")
    ocr_results = relationship("OCRResult", back_populates="document", cascade="all, delete-orphan")

class QualityReport(Base):
    __tablename__ = "quality_reports"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(String, ForeignKey("documents.id"), nullable=False)
    width = Column(Integer)
    height = Column(Integer)
    blur_score = Column(Float)
    brightness_score = Column(Float)
    contrast_score = Column(Float)
    quality_label = Column(String)
    recommended_pipeline = Column(String)

    document = relationship("Document", back_populates="quality_reports")

class OCRResult(Base):
    __tablename__ = "ocr_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(String, ForeignKey("documents.id"), nullable=False)
    engine_name = Column(String, nullable=False)
    preprocessing_profile = Column(String)
    raw_text = Column(Text)
    confidence = Column(Float)
    ocr_score = Column(Float)  # Consensus formula score (0.40C + 0.20T + 0.20A + 0.10S + 0.10V)
    processing_duration = Column(Float)
    bounding_boxes = Column(Text)

    document = relationship("Document", back_populates="ocr_results")

def init_db():
    Base.metadata.create_all(bind=engine)

init_db()