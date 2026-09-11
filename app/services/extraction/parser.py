import logging
import re
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from app.services.extraction.layout import SpatialLayoutAnalyzer

logger = logging.getLogger(__name__)


class LineItem(BaseModel):
    """Pydantic model for structured line items."""
    description: str
    quantity: float = 1.0
    unit_price: float = 0.0
    line_total: float = 0.0


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
    line_items: List[LineItem] = Field(default_factory=list)


class FieldExtractor:
    """Parses text and tokens to extract key document fields using regex and spatial proximity."""

    def __init__(self):
        self.layout_analyzer = SpatialLayoutAnalyzer()

    @staticmethod
    def extract_invoice_number(text: str) -> Optional[str]:
        # Priority 1: Explicit Invoice No / Bill No with labeled prefix
        pattern_explicit = r"(?:Invoice\s*(?:No|Num|#|\.)?|INV|Bill\s*(?:No|Num|#|\.)?)\s*:?\s*([A-Za-z0-9\-_]{3,25})"
        matches = re.findall(pattern_explicit, text, re.IGNORECASE)
        for val in matches:
            val_clean = val.strip()
            if val_clean.lower() not in ("no", "num", "number", "date", "table", "receipt") and not val_clean.startswith("---"):
                return val_clean

        # Priority 2: Generic Receipt/Check/Token fallback
        pattern_generic = r"(?:Receipt|Order|Token|Check)\s*(?:No|Num|#|\.)?\s*:?\s*([A-Za-z0-9\-_]{2,20})"
        match = re.search(pattern_generic, text, re.IGNORECASE)
        if match:
            val = match.group(1).strip()
            if val.lower() not in ("no", "num", "number", "date", "table") and not val.startswith("---"):
                return val

        return None

    @staticmethod
    def extract_grand_total(text: str) -> Optional[float]:
        # Clean currency symbols and unusual OCR noise from amounts like $5,445.30 or <5,445.30
        pattern = r"(?<!Sub-)(?:Grand\s*|Net\s*|Mode:.*?\s*)?Total\s*:?\s*[\$€£₹<>\s]*([\d,]+(?:\.\d{1,2})?)"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                val = float(match.group(1).replace(",", ""))
                return val if val > 0 else None
            except ValueError:
                return None
        return None

    @staticmethod
    def extract_date(text: str) -> Optional[str]:
        # Look for explicit labeled Date first
        labeled_match = re.search(r"Date\s*:?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{2}[-/]\d{2})", text, re.IGNORECASE)
        if labeled_match:
            return labeled_match.group(1).strip()

        # Fallback to standard date string matching
        match = re.search(r"\b(?:\d{4}[-/]\d{2}[-/]\d{2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b", text)
        return match.group(0).strip() if match else None

    @staticmethod
    def extract_tax_id(text: str) -> Optional[str]:
        pattern = r"(?:GSTIN|Tax\s*ID|VAT\s*No|TIN)[\s:]*([A-Z0-9]{8,15})"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        gstin_match = re.search(r"\b\d{2}[A-Z]{5}\d{4}[A-Z]{1}[A-Z0-9]{3}\b", text)
        return gstin_match.group(0).strip() if gstin_match else None

    @staticmethod
    def extract_currency(text: str) -> str:
        if any(marker in text.upper() for marker in ["GSTIN", "CGST", "SGST", "DELHI", "GUJARAT", "INR", "RS", "RUPEES"]):
            return "₹"
        match = re.search(r"(\$|€|£|₹|\bUSD\b|\bEUR\b|\bGBP\b|\bINR\b)", text)
        return match.group(1).strip() if match else "₹"

    @staticmethod
    def extract_subtotal(text: str) -> Optional[float]:
        pattern = r"Sub-?Total\s*:?\s*[\$€£₹%<>\s]*([\d,]+(?:\.\d{1,2})?)"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1).replace(",", ""))
            except ValueError:
                return None
        return None

    @staticmethod
    def extract_tax_amount(text: str) -> Optional[float]:
        # Extract CGST / SGST / VAT numeric amounts
        tax_matches = re.findall(r"(?:CGST|SGST|VAT|Tax)\b.*?(?:[\d\.]+%|\s)*?[\$€£₹<>\s]*([\d,]+(?:\.\d{1,2})?)", text, re.IGNORECASE)
        if tax_matches:
            try:
                valid_taxes = []
                for val in tax_matches:
                    f_val = float(val.replace(",", ""))
                    # Exclude rates like 2.5 if it's accompanied by percentage markers in raw text
                    if f_val > 0:
                        valid_taxes.append(f_val)
                if valid_taxes:
                    return round(sum(valid_taxes), 2)
            except ValueError:
                pass
        return None

    @staticmethod
    def extract_payment_method(text: str) -> Optional[str]:
        # Capture labeled payment mode (e.g. Mode: card)
        pattern = r"(?:Mode|Paid\s*via|Payment\s*Method|Pay\s*Mode)\s*:?\s*([A-Za-z]{3,15})"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            val = match.group(1).strip().capitalize()
            if val.lower() not in ("total", "subtotal", "amount"):
                return val

        upper_text = text.upper()
        for mode in ["UPI", "CASH", "CREDIT CARD", "DEBIT CARD", "CARD", "PAYTM", "GPAY"]:
            if re.search(r"\b" + mode + r"\b", upper_text):
                return mode.title()
        return None

    @staticmethod
    def extract_line_items(text: str) -> List[LineItem]:
        """Parses structured receipt item rows (e.g., 'Masala Dosa 280 4 1120')."""
        line_items: List[LineItem] = []
        lines = text.split("\n")

        # Typical line item pattern: Item Description | Price | Qty | Line Total
        item_pattern = re.compile(
            r"^(?P<desc>[A-Za-z\s]{3,30})\s+(?P<price>[\d,]+(?:\.\d{1,2})?)\s+(?P<qty>\d{1,3})\s+(?P<total>[\d,]+(?:\.\d{1,2})?)$"
        )

        for line in lines:
            line_clean = line.strip()
            # Ignore headers and summary totals
            if any(k in line_clean.lower() for k in ["sub-total", "total", "cgst", "sgst", "tax", "mode", "gstin", "thank"]):
                continue

            match = item_pattern.match(line_clean)
            if match:
                try:
                    desc = match.group("desc").strip()
                    price = float(match.group("price").replace(",", ""))
                    qty = float(match.group("qty"))
                    total = float(match.group("total").replace(",", ""))

                    line_items.append(LineItem(
                        description=desc,
                        quantity=qty,
                        unit_price=price,
                        line_total=total
                    ))
                except ValueError:
                    continue

        return line_items

    def parse_fields(self, engine_output: Any) -> ExtractedFields:
        raw_text = ""
        tokens: List[Dict[str, Any]] = []

        if isinstance(engine_output, dict):
            raw_text = engine_output.get("text", engine_output.get("raw_text", ""))
            tokens = engine_output.get("tokens", engine_output.get("words", []))
        elif isinstance(engine_output, str):
            raw_text = engine_output

        inv_num = self.extract_invoice_number(raw_text)
        grand_total = self.extract_grand_total(raw_text)
        subtotal = self.extract_subtotal(raw_text)
        tax_amount = self.extract_tax_amount(raw_text)
        line_items = self.extract_line_items(raw_text)

        if not inv_num and tokens:
            inv_num = self.layout_analyzer.find_value_near_labels(
                tokens, ["invoice", "inv", "bill", "receipt", "order"]
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
            line_items=line_items,
        )

    def build_candidate_dict_from_engines(self, engine_outputs: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        candidates: Dict[str, Dict[str, Any]] = {
            "invoice_number": {},
            "invoice_date": {},
            "tax_id": {},
            "currency": {},
            "subtotal": {},
            "tax_amount": {},
            "grand_total": {},
            "payment_method": {},
            "line_items": {},
        }

        for engine_name, output in engine_outputs.items():
            parsed: ExtractedFields = self.parse_fields(output)
            fields_dict = parsed.model_dump()

            for field_key, val in fields_dict.items():
                if val is not None and field_key in candidates:
                    # Convert list of Pydantic LineItem models to dicts for candidate dictionary serialization
                    if field_key == "line_items" and isinstance(val, list):
                        candidates[field_key][engine_name] = [item.model_dump() if hasattr(item, "model_dump") else item for item in val]
                    else:
                        candidates[field_key][engine_name] = val

        return candidates