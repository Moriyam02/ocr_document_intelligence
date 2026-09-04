import csv
import io
from typing import Any, Dict, List


class ExportService:
    """Service to convert structured document processing results into CSV or formatted JSON."""

    @staticmethod
    def generate_csv(document_data: Dict[str, Any]) -> str:
        """Flattens page-level parsed fields and line-item tables into a downloadable CSV string."""
        output = io.StringIO()
        writer = csv.writer(output)

        # Write Header Section
        writer.writerow(["=== DOCUMENT METADATA ==="])
        writer.writerow(["Document ID", document_data.get("document_id", "")])
        writer.writerow(["Status", document_data.get("status", "")])
        writer.writerow(["Total Pages", document_data.get("total_pages", 0)])
        writer.writerow([])

        # Process Each Page
        pages = document_data.get("pages", [])
        for page_info in pages:
            page_num = page_info.get("page", 1)
            writer.writerow([f"=== PAGE {page_num} - EXTRACTED FIELDS ==="])
            writer.writerow(["Field Name", "Extracted Value"])

            # Field Consensus or Parsed Fields
            field_consensus = page_info.get("field_consensus", {})
            if field_consensus:
                for field_key, val in field_consensus.items():
                    writer.writerow([field_key, val])
            else:
                parsed_fields = page_info.get("parsed_fields", {})
                # Use primary engine output if consensus is not available
                first_engine_fields = list(parsed_fields.values())[0] if parsed_fields else {}
                for field_key, val in first_engine_fields.items():
                    writer.writerow([field_key, val])

            writer.writerow([])
            writer.writerow([f"=== PAGE {page_num} - LINE ITEMS ==="])
            writer.writerow(["Description", "Quantity", "Unit Price", "Line Total"])

            line_items_by_engine = page_info.get("line_items", {})
            # Select line items from primary engine output
            primary_items: List[Dict[str, Any]] = (
                list(line_items_by_engine.values())[0] if line_items_by_engine else []
            )

            if primary_items:
                for item in primary_items:
                    writer.writerow([
                        item.get("description", ""),
                        item.get("quantity", 0),
                        item.get("unit_price", 0.0),
                        item.get("line_total", 0.0)
                    ])
            else:
                writer.writerow(["No line items detected.", "", "", ""])

            writer.writerow([])

        return output.getvalue()