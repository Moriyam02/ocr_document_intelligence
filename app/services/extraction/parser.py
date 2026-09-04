import logging
import re
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

from app.services.extraction.layout import SpatialLayoutAnalyzer

logger = logging.getLogger(__name__)


class ExtractedFields(BaseModel):
    """Pydantic model representing structured extracted document fields."""
    invoice_number: Optional[str] = None
    tax_id: Optional[str] = None
    currency: Optional[str] = "$"
    subtotal: Optional[float] = None
    tax_amount: Optional[float] = None
    grand_total: Optional[float] = None
    payment_method: Optional[str] = None
    invoice_date: Optional[str] = None


class FieldExtractor:
    """Parses text and tokens to extract key document fields using regex and spatial proximity."""

    def __init__(self):
        self.layout_analyzer = SpatialLayoutAnalyzer()

    # --- Basic Extraction Methods ---

    @staticmethod
    def extract_invoice_number(text: str) -> Optional[str]:
        """Matches invoice number patterns like INV-1024, Invoice # 1024, Invoice: INV1024."""
        match = re.search(
            r"(?:invoice\s*(?:#|num|number)?[:\s]*)([A-Z0-9-]+)",
            text,
            re.IGNORECASE,
        )
        return match.group(1).strip() if match else None

    @staticmethod
    def extract_grand_total(text: str) -> Optional[str]:
        """Matches patterns like Total: $150.00, Grand Total 150.00."""
        match = re.search(
            r"(?:grand\s*total|total)[:\s]*\$?\s*([\d,]+\.\d{2})",
            text,
            re.IGNORECASE,
        )
        return match.group(1).strip() if match else None

    @staticmethod
    def extract_date(text: str) -> Optional[str]:
        """Matches YYYY-MM-DD, MM/DD/YYYY, or DD-MM-YYYY dates."""
        match = re.search(
            r"\b(?:\d{4}[-/]\d{2}[-/]\d{2}|\d{2}[-/]\d{2}[-/]\d{4})\b", text
        )
        return match.group(0).strip() if match else None

    # --- Extended Schema Extraction Methods ---

    @staticmethod
    def extract_tax_id(text: str) -> Optional[str]:
        """Matches Tax ID / VAT / GSTIN numbers."""
        match = re.search(
            r"(?i)(?:tax\s*id|vat\s*no|gstin|gst\s*no|tin\s*no)[\s:]*([A-Z0-9]{8,15})",
            text,
        )
        return match.group(1).strip() if match else None

    @staticmethod
    def extract_currency(text: str) -> Optional[str]:
        """Matches standard currency symbols or codes."""
        match = re.search(r"(\$|€|£|₹|\bUSD\b|\bEUR\b|\bGBP\b|\bINR\b)", text)
        return match.group(1).strip() if match else "$"

    @staticmethod
    def extract_subtotal(text: str) -> Optional[str]:
        """Matches subtotal patterns."""
        match = re.search(
            r"(?i)(?:subtotal|sub\s*total|net\s*amount)[\s:]*[\$€£₹]?\s*([\d,]+\.\d{2})",
            text,
        )
        return match.group(1).strip() if match else None

    @staticmethod
    def extract_tax_amount(text: str) -> Optional[str]:
        """Matches tax / VAT / GST amounts."""
        match = re.search(
            r"(?i)(?:tax|vat|gst|sales\s*tax)[\s:]*[\$€£₹]?\s*([\d,]+\.\d{2})",
            text,
        )
        return match.group(1).strip() if match else None

    @staticmethod
    def extract_payment_method(text: str) -> Optional[str]:
        """Matches payment mode description."""
        match = re.search(
            r"(?i)(?:paid\s*via|payment\s*method|payment\s*mode)[\s:]*([A-Za-z\s]{3,15})",
            text,
        )
        return match.group(1).strip() if match else None

    # --- Unified Extraction Pipeline Methods ---

    def extract_all(self, text: str) -> Dict[str, Optional[str]]:
        """Legacy dict-based extraction method for basic text input."""
        return {
            "invoice_number": self.extract_invoice_number(text),
            "grand_total": self.extract_grand_total(text),
            "invoice_date": self.extract_date(text),
            "tax_id": self.extract_tax_id(text),
            "currency": self.extract_currency(text),
            "subtotal": self.extract_subtotal(text),
            "tax_amount": self.extract_tax_amount(text),
            "payment_method": self.extract_payment_method(text),
        }

    def parse_fields(self, engine_output: Any) -> ExtractedFields:
        """Parses complete extended fields from an engine output dict/string using regex and layout analysis."""
        raw_text = ""
        tokens: List[Dict[str, Any]] = []

        if isinstance(engine_output, dict):
            raw_text = engine_output.get("text", engine_output.get("raw_text", ""))
            tokens = engine_output.get("tokens", engine_output.get("words", []))
        elif isinstance(engine_output, str):
            raw_text = engine_output

        # 1. Regex Extraction
        inv_num = self.extract_invoice_number(raw_text)
        gt_str = self.extract_grand_total(raw_text)
        sub_str = self.extract_subtotal(raw_text)
        tax_str = self.extract_tax_amount(raw_text)

        # 2. Spatial Layout Fallback (if regex fails to find invoice number)
        if not inv_num and tokens:
            inv_num = self.layout_analyzer.find_value_near_labels(
                tokens, ["invoice", "inv", "bill"]
            )

        return ExtractedFields(
            invoice_number=inv_num,
            invoice_date=self.extract_date(raw_text),
            tax_id=self.extract_tax_id(raw_text),
            currency=self.extract_currency(raw_text),
            subtotal=float(sub_str.replace(",", "")) if sub_str else None,
            tax_amount=float(tax_str.replace(",", "")) if tax_str else None,
            grand_total=float(gt_str.replace(",", "")) if gt_str else None,
            payment_method=self.extract_payment_method(raw_text),
        )


# Standalone Execution Block
if __name__ == "__main__":
    sample_text = """
    INVOICE # INV-2026-991
    Date: 2026-08-15
    Tax ID: 27AABCU9603R1ZN
    Paid Via: Bank Transfer
    Subtotal: $450.00
    Tax: $50.00
    Grand Total: $500.00
    """

    extractor = FieldExtractor()
    extracted_data = extractor.parse_fields(sample_text)
    print(extracted_data.model_dump_json(indent=2))