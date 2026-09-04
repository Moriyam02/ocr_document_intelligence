import re
import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class LineItem(BaseModel):
    description: str
    quantity: float
    unit_price: float
    line_total: float


class TableExtractor:

    def __init__(self, y_tolerance: float = 10.0):
        self.y_tolerance = y_tolerance
        # Regex pattern matching: Description + Quantity + Unit Price + Line Total
        self.line_pattern = re.compile(
            r"^\s*(.*?)\s+(\d+(?:\.\d+)?)\s+[\$€£₹]?\s*([\d,]+\.\d{2})\s+[\$€£₹]?\s*([\d,]+\.\d{2})\s*$"
        )

    def extract_rows_from_tokens(
        self, tokens: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        """Groups text tokens into rows based on vertical (Y) proximity."""
        if not tokens:
            return []

        # Filter and sort tokens top-to-bottom
        valid_tokens = [t for t in tokens if "bbox" in t or "text" in t]
        sorted_tokens = sorted(
            valid_tokens, key=lambda t: t.get("bbox", [0, 0, 0, 0])[1]
        )

        rows: List[List[Dict[str, Any]]] = []
        current_row: List[Dict[str, Any]] = []
        current_y = None

        for token in sorted_tokens:
            token_y = token.get("bbox", [0, 0, 0, 0])[1]

            if current_y is None or abs(token_y - current_y) <= self.y_tolerance:
                current_row.append(token)
                if current_y is None:
                    current_y = token_y
            else:
                # Sort row left-to-right (X coordinate)
                current_row.sort(key=lambda t: t.get("bbox", [0, 0, 0, 0])[0])
                rows.append(current_row)
                current_row = [token]
                current_y = token_y

        if current_row:
            current_row.sort(key=lambda t: t.get("bbox", [0, 0, 0, 0])[0])
            rows.append(current_row)

        return rows

    def parse_line_items(self, rows: List[List[Dict[str, Any]]]) -> List[LineItem]:
        """Parses structured line items from spatial token rows using regex and positional fallbacks."""
        line_items: List[LineItem] = []

        for row in rows:
            text = " ".join([t.get("text", "") for t in row if t.get("text")]).strip()
            if not text:
                continue

            # Approach 1: Regex Pattern Match (Handles currency symbols & formatting)
            match = self.line_pattern.search(text)
            if match:
                desc, qty, unit_price, total = match.groups()
                try:
                    line_items.append(
                        LineItem(
                            description=desc.strip(),
                            quantity=float(qty),
                            unit_price=float(unit_price.replace(",", "")),
                            line_total=float(total.replace(",", "")),
                        )
                    )
                    continue
                except ValueError:
                    pass

            # Approach 2: Split Fallback for non-standard spaced rows
            parts = text.split()
            if len(parts) >= 4:
                try:
                    qty = float(parts[-3].replace(",", ""))
                    unit_price = float(re.sub(r"[^\d.]", "", parts[-2]))
                    total = float(re.sub(r"[^\d.]", "", parts[-1]))
                    desc = " ".join(parts[:-3]).strip()
                    if desc:
                        line_items.append(
                            LineItem(
                                description=desc,
                                quantity=qty,
                                unit_price=unit_price,
                                line_total=total,
                            )
                        )
                except (ValueError, IndexError):
                    continue

        return line_items

    def extract_line_items_from_engine_output(
        self, engine_output: Any
    ) -> List[LineItem]:
        """Wrapper method to handle direct OCR engine dictionary outputs or raw text."""
        if isinstance(engine_output, dict):
            tokens = engine_output.get("tokens", engine_output.get("words", []))
            if tokens:
                rows = self.extract_rows_from_tokens(tokens)
                return self.parse_line_items(rows)
            raw_text = engine_output.get("text", engine_output.get("raw_text", ""))
        elif isinstance(engine_output, str):
            raw_text = engine_output
        else:
            raw_text = ""

        # Raw string extraction fallback
        lines = [[{"text": line}] for line in raw_text.splitlines() if line.strip()]
        return self.parse_line_items(lines)


# Standalone Test Execution Block
if __name__ == "__main__":
    # Mock token data with bounding boxes [x, y, w, h]
    mock_tokens = [
        {"text": "Widget", "bbox": [10, 100, 50, 20]},
        {"text": "A", "bbox": [65, 102, 20, 20]},
        {"text": "2", "bbox": [200, 99, 10, 20]},
        {"text": "$25.00", "bbox": [250, 101, 40, 20]},
        {"text": "$50.00", "bbox": [320, 100, 40, 20]},
        {"text": "Gadget", "bbox": [10, 140, 50, 20]},
        {"text": "B", "bbox": [65, 141, 20, 20]},
        {"text": "1", "bbox": [200, 139, 10, 20]},
        {"text": "100.00", "bbox": [250, 140, 40, 20]},
        {"text": "100.00", "bbox": [320, 142, 40, 20]},
    ]

    extractor = TableExtractor()
    grouped_rows = extractor.extract_rows_from_tokens(mock_tokens)
    line_items = extractor.parse_line_items(grouped_rows)

    print(f"Extracted {len(line_items)} Line Item(s):")
    for item in line_items:
        print(item.model_dump_json(indent=2))