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
    currency: Optional[str] = "₹"
    subtotal: Optional[float] = None
    tax_amount: Optional[float] = None
    grand_total: Optional[float] = None
    payment_method: Optional[str] = None
    invoice_date: Optional[str] = None


class FieldExtractor:
    """Parses text and tokens to extract key document fields using regex and spatial proximity."""

    def __init__(self):
        self.layout_analyzer = SpatialLayoutAnalyzer()

    # --- Robust Regex Extraction Methods ---

    @staticmethod
    def extract_invoice_number(text: str) -> Optional[str]:
        """Matches invoice number patterns while skipping filler words like 'No', 'Num', or '#'."""
        pattern = r"(?:Invoice|INV|Bill)\s*(?:No|Num|#|\.)?\s*:?\s*([A-Za-z0-9\-_]{4,20})"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            val = match.group(1).strip()
            # Ensure it didn't just match the literal word "No" or "Number"
            if val.lower() not in ("no", "num", "number"):
                return val
        return None

    @staticmethod
    def extract_grand_total(text: str) -> Optional[float]:
        """Matches final total while explicitly ignoring Sub-Total or line totals."""
        pattern = r"(?<!Sub-)(?:Grand\s*)?Total\s*:?\s*[^\d]*([\d,]+\.\d{2})"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1).replace(",", ""))
            except ValueError:
                return None
        return None

    @staticmethod
    def extract_date(text: str) -> Optional[str]:
        """Matches YYYY-MM-DD, MM/DD/YYYY, or DD/MM/YYYY dates."""
        match = re.search(
            r"\b(?:\d{4}[-/]\d{2}[-/]\d{2}|\d{2}[-/]\d{2}[-/]\d{4})\b", text
        )
        return match.group(0).strip() if match else None

    @staticmethod
    def extract_tax_id(text: str) -> Optional[str]:
        """Matches GSTIN / Tax ID / VAT numbers (including OCR-scrambled GSTINs)."""
        pattern = r"(?:GSTIN|Tax\s*ID|VAT\s*No|TIN)[\s:]*([A-Z0-9]{8,15})"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # Fallback for Indian GSTIN format without explicit label
        gstin_match = re.search(r"\b\d{2}[A-Z]{5}\d{4}[A-Z]{1}[A-Z0-9]{3}\b", text)
        return gstin_match.group(0).strip() if gstin_match else None

    @staticmethod
    def extract_currency(text: str) -> str:
        """Determines currency based on GSTIN/CGST markers or symbols."""
        if any(marker in text.upper() for marker in ["GSTIN", "CGST", "SGST", "DELHI", "GUJARAT", "INR"]):
            return "₹"
        match = re.search(r"(\$|€|£|₹|\bUSD\b|\bEUR\b|\bGBP\b|\bINR\b)", text)
        return match.group(1).strip() if match else "$"

    @staticmethod
    def extract_subtotal(text: str) -> Optional[float]:
        """Matches Sub-Total amounts."""
        pattern = r"Sub-?Total\s*:?\s*[^\d]*([\d,]+\.\d{2})"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1).replace(",", ""))
            except ValueError:
                return None
        return None

    @staticmethod
    def extract_tax_amount(text: str) -> Optional[float]:
        """Calculates total tax by summing CGST, SGST, VAT, or general tax lines."""
        tax_matches = re.findall(r"(?:CGST|SGST|VAT|Tax)\b.*?\b([\d,]+\.\d{2})", text, re.IGNORECASE)
        if tax_matches:
            try:
                total_tax = sum(float(val.replace(",", "")) for val in tax_matches)
                return round(total_tax, 2)
            except ValueError:
                pass
        return None

    @staticmethod
    def extract_payment_method(text: str) -> Optional[str]:
        """Matches payment mode (e.g. Card, Cash, UPI, Bank Transfer)."""
        pattern = r"(?:Mode|Paid\s*via|Payment\s*Method)\s*:?\s*([A-Za-z\s]{3,15})"
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(1).strip().capitalize() if match else None

    # --- Unified Extraction Pipeline Methods ---

    def parse_fields(self, engine_output: Any) -> ExtractedFields:
        """Parses complete extended fields from an engine output dict or raw text."""
        raw_text = ""
        tokens: List[Dict[str, Any]] = []

        if isinstance(engine_output, dict):
            raw_text = engine_output.get("text", engine_output.get("raw_text", ""))
            tokens = engine_output.get("tokens", engine_output.get("words", []))
        elif isinstance(engine_output, str):
            raw_text = engine_output

        # 1. Regex Extractions
        inv_num = self.extract_invoice_number(raw_text)
        grand_total = self.extract_grand_total(raw_text)
        subtotal = self.extract_subtotal(raw_text)
        tax_amount = self.extract_tax_amount(raw_text)

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
            subtotal=subtotal,
            tax_amount=tax_amount,
            grand_total=grand_total,
            payment_method=self.extract_payment_method(raw_text),
        )


# Standalone Verification Execution
if __name__ == "__main__":
    sample_text = """
    Sunrise Foods Pvt Ltd
    9 Palm Court, Delhi, Gujarat 856604
    RECEIPT
    Name: Pooja Iyer Invoice No: INV-2026-0423
    Table: #02 Date: 12/02/2026

    Sub-Total: 25,186.00
    CGST: SGST: 2.5% 129.65
    SGST: SGST: 2.5% 129.65

    Mode: card Total: 5,445.30
    GSTIN: 30XICTI5508S8Z5
    """

    extractor = FieldExtractor()
    extracted_data = extractor.parse_fields(sample_text)
    print(extracted_data.model_dump_json(indent=2))