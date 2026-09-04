import logging
import re
from typing import Dict, Any, List, Union

logger = logging.getLogger(__name__)


class DataValidator:
    """Validates extracted document consensus fields against deterministic business logic rules."""

    def _extract_consolidated_fields(self, pages_data: Union[List[Any], Dict[str, Any]]) -> Dict[str, Any]:
        """Safely merges field_consensus across single-page dicts or multi-page lists."""
        consolidated_fields = {}

        if isinstance(pages_data, list):
            for page in pages_data:
                if isinstance(page, dict):
                    # Check if page dictionary wraps 'field_consensus' or contains direct keys
                    page_consensus = page.get("field_consensus", page)
                    if isinstance(page_consensus, dict):
                        # Merge page consensus fields; ignore structural metadata keys
                        for k, v in page_consensus.items():
                            if k not in ("page", "quality_metrics", "ocr_results"):
                                consolidated_fields[k] = v
        elif isinstance(pages_data, dict):
            page_consensus = pages_data.get("field_consensus", pages_data)
            if isinstance(page_consensus, dict):
                for k, v in page_consensus.items():
                    if k not in ("page", "quality_metrics", "ocr_results"):
                        consolidated_fields[k] = v

        return consolidated_fields

    def validate(self, pages_data: Union[List[Any], Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Main validation entry point called by the background job pipeline."""
        validation_issues = []

        # 1. Unify multi-page structures into a flat field dictionary safely
        field_consensus = self._extract_consolidated_fields(pages_data)

        if not field_consensus:
            logger.warning("[VALIDATION] No consensus fields extracted to validate.")
            return validation_issues

        # --- Rule 1: Validate Total Amount ---
        total_data = field_consensus.get("total_amount", {})
        if isinstance(total_data, dict):
            raw_val = total_data.get("value")
            if raw_val is not None:
                # Remove currency symbols/commas to verify numeric float validity
                cleaned_val = re.sub(r"[^\d.]", "", str(raw_val))
                try:
                    amount = float(cleaned_val)
                    if amount <= 0:
                        validation_issues.append({
                            "rule": "NON_POSITIVE_TOTAL",
                            "field": "total_amount",
                            "issue": f"Total amount must be greater than zero. Got: {raw_val}"
                        })
                except ValueError:
                    validation_issues.append({
                        "rule": "INVALID_NUMERIC_FORMAT",
                        "field": "total_amount",
                        "issue": f"Could not parse '{raw_val}' as a valid float amount."
                    })

        # --- Rule 2: Validate Invoice / Document Date ---
        date_data = field_consensus.get("invoice_date", {})
        if isinstance(date_data, dict):
            date_val = date_data.get("value")
            if date_val:
                # Basic ISO format / date string format verification check
                date_str = str(date_val).strip()
                if not re.search(r"\d{2,4}[-/\.]\d{1,2}[-/\.]\d{2,4}", date_str):
                    validation_issues.append({
                        "rule": "INVALID_DATE_FORMAT",
                        "field": "invoice_date",
                        "issue": f"Date '{date_str}' does not match expected format (YYYY-MM-DD or DD/MM/YYYY)."
                    })

        # --- Rule 3: Flag High-Risk Unconfirmed Fields ---
        for field_name, field_info in field_consensus.items():
            if isinstance(field_info, dict):
                confidence = field_info.get("confidence", 0.0)
                flagged = field_info.get("flagged_for_review", False)
                if flagged or confidence < 0.60:
                    validation_issues.append({
                        "rule": "LOW_CONFIDENCE_FIELD",
                        "field": field_name,
                        "issue": f"Field '{field_name}' has low OCR confidence ({confidence})."
                    })

        return validation_issues