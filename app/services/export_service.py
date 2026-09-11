import csv
import io
import logging
from typing import Any, Dict, List, Union

logger = logging.getLogger(__name__)


class ExportService:
    """Service to convert structured document processing results into CSV or formatted JSON."""

    @staticmethod
    def _extract_field_value(raw_val: Any) -> Any:
        """Helper to extract primitive value from nested consensus dict or raw scalar."""
        if isinstance(raw_val, dict):
            return raw_val.get("value")
        return raw_val

    @staticmethod
    def _to_dict(document_data: Union[Dict[str, Any], Any]) -> Dict[str, Any]:
        """Converts SQLAlchemy Document models or Pydantic instances to a Python dictionary."""
        if isinstance(document_data, dict):
            return document_data
        
        # Handle SQLAlchemy ORM or Pydantic objects
        res = {}
        for attr in ["id", "document_id", "status", "total_pages", "consensus_data", 
                     "pages_data", "routing_decision", "validation_issues", "field_corrections"]:
            if hasattr(document_data, attr):
                res[attr] = getattr(document_data, attr)
        
        # Ensure document_id primary fallback
        if "document_id" not in res or not res["document_id"]:
            res["document_id"] = getattr(document_data, "id", "")
            
        return res

    @classmethod
    def generate_csv(cls, document_data: Union[Dict[str, Any], Any]) -> str:
        """Flattens document-level fields, human corrections, and page-level extractions into CSV."""
        data = cls._to_dict(document_data)
        
        output = io.StringIO()
        writer = csv.writer(output)

        # ---------------------------------------------------------
        # 1. Write Metadata Header
        # ---------------------------------------------------------
        writer.writerow(["=== DOCUMENT METADATA ==="])
        doc_id = data.get("document_id") or data.get("id", "")
        status = data.get("status", "PROCESSING")
        
        pages = data.get("pages_data") or data.get("pages", [])
        total_pages = data.get("total_pages") or (len(pages) if isinstance(pages, list) else 1)
        
        writer.writerow(["Document ID", str(doc_id)])
        writer.writerow(["Status", str(status)])
        writer.writerow(["Total Pages", total_pages])
        writer.writerow([])

        # ---------------------------------------------------------
        # 2. Collect All Extracted & Corrected Fields
        # ---------------------------------------------------------
        writer.writerow(["=== EXTRACTED & REVIEWED FIELDS ==="])
        writer.writerow(["Field Name", "Value", "Confidence", "Sources"])

        extracted_fields: Dict[str, Any] = {}

        # Look for top-level consensus structures
        if "consensus_data" in data and isinstance(data["consensus_data"], dict):
            extracted_fields.update(data["consensus_data"])
        elif "field_consensus" in data and isinstance(data["field_consensus"], dict):
            extracted_fields.update(data["field_consensus"])
        elif "consensus" in data and isinstance(data["consensus"], dict):
            extracted_fields.update(data["consensus"])
        elif "extracted_data" in data and isinstance(data["extracted_data"], dict):
            extracted_fields.update(data["extracted_data"])

        # Look for page-level extractions if top-level is empty
        if not extracted_fields and isinstance(pages, list):
            for page_info in pages:
                if isinstance(page_info, dict):
                    fc = page_info.get("consensus") or page_info.get("field_consensus", {})
                    if fc:
                        extracted_fields.update(fc)

        # Apply Human Review Corrections on top of OCR consensus
        field_corrections = data.get("field_corrections", data.get("corrections", []))
        if isinstance(field_corrections, list):
            for correction in field_corrections:
                if isinstance(correction, dict) and "field_name" in correction:
                    extracted_fields[correction["field_name"]] = correction.get("new_value")
        elif isinstance(field_corrections, dict):
            extracted_fields.update(field_corrections)

        # Write Extracted Fields to CSV
        has_written_field = False
        if extracted_fields:
            for field_key, val in extracted_fields.items():
                if field_key == "line_items":
                    continue  # Line items handled in Section 3

                field_value = cls._extract_field_value(val)
                
                # Format confidence and sources if available from consensus engine
                confidence_str = ""
                sources_str = ""
                if isinstance(val, dict):
                    conf = val.get("confidence")
                    if isinstance(conf, (int, float)):
                        confidence_str = f"{conf:.2f}"
                    sources = val.get("sources", [])
                    if isinstance(sources, list):
                        sources_str = ", ".join(sources)

                if field_value is not None and not isinstance(field_value, (list, dict)):
                    writer.writerow([field_key, field_value, confidence_str, sources_str])
                    has_written_field = True

        if not has_written_field:
            writer.writerow(["No extracted fields found", "", "", ""])

        writer.writerow([])

        # ---------------------------------------------------------
        # 3. Process Line Items
        # ---------------------------------------------------------
        writer.writerow(["=== LINE ITEMS ==="])
        writer.writerow(["Description", "Quantity", "Unit Price", "Line Total"])

        line_items: List[Dict[str, Any]] = []

        # Check consensus line items
        raw_line_items = extracted_fields.get("line_items")
        if isinstance(raw_line_items, dict):
            line_items = raw_line_items.get("value", [])
        elif isinstance(raw_line_items, list):
            line_items = raw_line_items
        elif "line_items" in data and isinstance(data["line_items"], list):
            line_items = data["line_items"]

        # Page-level fallback for line items
        if not line_items and isinstance(pages, list):
            for page_info in pages:
                if isinstance(page_info, dict):
                    page_consensus = page_info.get("consensus", {})
                    p_items = page_consensus.get("line_items", {}).get("value", [])
                    if isinstance(p_items, list) and p_items:
                        line_items = p_items
                        break

        # Write line items table
        if line_items and isinstance(line_items, list):
            for item in line_items:
                if isinstance(item, dict):
                    writer.writerow([
                        item.get("description", ""),
                        item.get("quantity", 1.0),
                        item.get("unit_price", 0.0),
                        item.get("line_total", 0.0)
                    ])
                elif hasattr(item, "model_dump"):
                    item_dict = item.model_dump()
                    writer.writerow([
                        item_dict.get("description", ""),
                        item_dict.get("quantity", 1.0),
                        item_dict.get("unit_price", 0.0),
                        item_dict.get("line_total", 0.0)
                    ])
        else:
            writer.writerow(["No line items detected.", "", "", ""])

        return output.getvalue()