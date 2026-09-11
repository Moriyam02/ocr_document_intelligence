import logging
from typing import Dict, Any, List
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ValidationIssue(BaseModel):
    field: str
    issue_type: str
    message: str
    severity: str = "WARNING"


class BusinessRulesValidator:
    """Validates extracted document fields against business rules."""

    def validate(self, consensus_data: Dict[str, Any]) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []

        # Extract values
        grand_total = consensus_data.get("grand_total", {}).get("value")
        subtotal = consensus_data.get("subtotal", {}).get("value")
        tax_amount = consensus_data.get("tax_amount", {}).get("value")
        invoice_number = consensus_data.get("invoice_number", {}).get("value")
        invoice_date = consensus_data.get("invoice_date", {}).get("value")

        # Check required field: Invoice Number
        if not invoice_number:
            issues.append(ValidationIssue(
                field="invoice_number",
                issue_type="MISSING_FIELD",
                message="Invoice number could not be detected.",
                severity="HIGH"
            ))

        # Check required field: Invoice Date
        if not invoice_date:
            issues.append(ValidationIssue(
                field="invoice_date",
                issue_type="MISSING_FIELD",
                message="Invoice date is missing.",
                severity="MEDIUM"
            ))

        # Math Tally Check: subtotal + tax_amount == grand_total
        if grand_total is not None and subtotal is not None and tax_amount is not None:
            expected_total = round(float(subtotal) + float(tax_amount), 2)
            actual_total = round(float(grand_total), 2)
            if abs(expected_total - actual_total) > 0.05:
                issues.append(ValidationIssue(
                    field="grand_total",
                    issue_type="MATH_MISMATCH",
                    message=f"Subtotal ({subtotal}) + Tax ({tax_amount}) = {expected_total}, which does not match Grand Total ({actual_total}).",
                    severity="HIGH"
                ))

        return issues


# Aliases for 100% backward/forward compatibility across main.py and jobs.py
DataValidator = BusinessRulesValidator
DocumentValidator = BusinessRulesValidator
RulesValidator = BusinessRulesValidator