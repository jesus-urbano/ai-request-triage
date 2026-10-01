"""Data model for a triaged request.

Every LLM response is validated against this schema before it is used,
so malformed or out-of-range output is caught instead of silently passed on.
"""

import json
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

Category = Literal["IT Support", "Access Management", "HR", "Finance", "Security", "Operations", "Other"]
Urgency = Literal["High", "Medium", "Low"]

# Which team owns each category. Kept in code (not left to the LLM) so routing
# is deterministic and easy to audit.
ROUTING = {
    "IT Support": "IT Service Desk",
    "Access Management": "Identity & Access Team",
    "HR": "HR Operations",
    "Finance": "Accounts Payable / Finance Ops",
    "Security": "Security Operations",
    "Operations": "Business Operations",
    "Other": "Operations Intake",
}


class TriageResult(BaseModel):
    category: Category
    urgency: Urgency
    summary: str = Field(description="One-sentence summary of the request")
    key_details: dict[str, str] = Field(
        default_factory=dict,
        description="Extracted fields such as deadline, amount, system, person, invoice number. "
                    "Only include details that appear in the request.",
    )
    draft_reply: str = Field(description="Short, polite first reply to the requester")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the classification, 0 to 1")
    reasoning: str = Field(description="Brief explanation of the category and urgency choice")

    @field_validator("key_details", mode="before")
    @classmethod
    def stringify_values(cls, value: Any) -> Any:
        """Models sometimes return numbers (e.g. amount: 4500) or a JSON string. Accept both instead of failing."""
        if isinstance(value, str):
            try:
                value = json.loads(value) if value.strip() else {}
            except json.JSONDecodeError:
                return {"note": value}
        if isinstance(value, dict):
            return {str(k): "" if v is None else str(v) for k, v in value.items()}
        return value
