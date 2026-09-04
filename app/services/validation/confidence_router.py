import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class RoutingEngine:
    """Evaluates multi-engine confidence scores, agreement ratios, and validation issues
    to route documents to AUTO_APPROVED or HUMAN_REVIEW_REQUIRED status.
    """

    def __init__(self, confidence_threshold: float = 0.85):
        self.confidence_threshold = confidence_threshold

    def evaluate(
        self,
        consensus_data: Dict[str, Any],
        validation_issues: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculates overall consensus confidence and determines HITL routing status."""
        # Handles both consensus_data={"field_consensus": {...}} and direct field_consensus dictionary
        if "field_consensus" in consensus_data and isinstance(consensus_data["field_consensus"], dict):
            field_consensus = consensus_data["field_consensus"]
        else:
            field_consensus = consensus_data

        if not field_consensus or not isinstance(field_consensus, dict):
            return {
                "status": "HUMAN_REVIEW_REQUIRED",
                "overall_confidence": 0.0,
                "routing_reason": "NO_FIELDS_EXTRACTED",
                "flagged_fields": [],
                "validation_issues_count": len(validation_issues)
            }

        flagged_fields = []
        confidence_scores = []

        for field_name, info in field_consensus.items():
            if not isinstance(info, dict):
                continue

            conf = info.get("confidence", 0.0)
            confidence_scores.append(conf)

            if conf < self.confidence_threshold or info.get("flagged_for_review", False):
                flagged_fields.append(field_name)

        overall_confidence = (
            round(sum(confidence_scores) / len(confidence_scores), 2)
            if confidence_scores else 0.0
        )

        needs_human_review = len(flagged_fields) > 0 or len(validation_issues) > 0

        reasons = []
        if flagged_fields:
            reasons.append(f"Low confidence on fields: {', '.join(flagged_fields)}")
        if validation_issues:
            reasons.append(f"Validation failed on {len(validation_issues)} rule(s)")

        routing_reason = " | ".join(reasons) if reasons else "Passed all confidence and rule checks"

        return {
            "status": "HUMAN_REVIEW_REQUIRED" if needs_human_review else "AUTO_APPROVED",
            "overall_confidence": overall_confidence,
            "routing_reason": routing_reason,
            "flagged_fields": flagged_fields,
            "validation_issues_count": len(validation_issues)
        }