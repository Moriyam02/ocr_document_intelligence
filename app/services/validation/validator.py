from typing import Any, Dict, List
from pydantic import BaseModel

ENGINE_KEYS = {"tesseract", "easyocr", "paddleocr", "ocr_results", "engines", "metadata"}


class ValidationIssue(BaseModel):
    field: str
    severity: str
    message: str


class DataValidator:

    def validate_consensus_fields(
        self, field_consensus: Dict[str, Any]
    ) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []

        for field_name, data in field_consensus.items():
            if field_name in ENGINE_KEYS:
                continue

            if hasattr(data, "model_dump"):
                data_dict = data.model_dump()
            elif hasattr(data, "dict"):
                data_dict = data.dict()
            elif isinstance(data, dict):
                data_dict = data
            else:
                continue

            if data_dict.get("discrepancy_flagged"):
                issues.append(
                    ValidationIssue(
                        field=field_name,
                        severity="warning",
                        message=f"Discrepancy detected across OCR engines for field: {field_name}",
                    )
                )

            value = data_dict.get("value")
            if value is None or str(value).strip() == "":
                issues.append(
                    ValidationIssue(
                        field=field_name,
                        severity="error",
                        message=f"Missing or empty value for required field: {field_name}",
                    )
                )

        return issues