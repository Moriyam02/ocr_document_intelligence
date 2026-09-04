from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class QualityReportResponse(BaseModel):
    width: int
    height: int
    blur_score: float
    brightness_score: float
    contrast_score: float
    quality_label: str
    recommended_pipeline: str

    class Config:
        from_attributes = True

class OCRResultResponse(BaseModel):
    engine_name: str
    preprocessing_profile: str
    raw_text: str
    confidence: float
    ocr_score: float
    processing_duration: float
    bounding_boxes: str

    class Config:
        from_attributes = True

class DocumentDetailResponse(BaseModel):
    id: str
    filename: str
    file_type: str
    doc_category: str
    status: str
    created_at: datetime
    quality_reports: List[QualityReportResponse]
    ocr_results: List[OCRResultResponse]

    class Config:
        from_attributes = True

class UploadResponse(BaseModel):
    message: str
    documents: list[dict]

class DocumentStatusResponse(BaseModel):
    id: str
    filename: str
    status: str