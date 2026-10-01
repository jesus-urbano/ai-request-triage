"""Rules that decide how much the AI is trusted on each request.

Two separate questions:
  1. Does a person need to review this before anything happens?  (human review)
  2. Does this need to jump the queue?                            (escalation)

Urgent requests are escalated, not held for review: holding an outage in a review
queue would slow down exactly the requests that need speed. Review is reserved for
cases where the AI's decision itself shouldn't be trusted on its own.

These checks run in code, outside the model, so they still apply if the model is
wrong or has been manipulated by the request text.
"""

import re

CONFIDENCE_THRESHOLD = 0.7
ALWAYS_REVIEW_CATEGORIES = {"Security"}  # never auto-handle security issues
SENSITIVE_KEYWORDS = ["bank account", "direct deposit", "terminate", "password", "ssn", "wire transfer", "admin access"]

# Text that tries to instruct the AI instead of describing a problem.
INJECTION_PATTERNS = [
    r"ignore (all |any |your )?(previous |prior )?(instructions|rules)",
    r"disregard (the |your )?(instructions|rules|system prompt)",
    r"you are now",
    r"(mark|classify|label|set|flag) (this|it|me) (as|to) (low|medium|high|urgent|not urgent)",
    r"(no|without|skip( the)?) (approval|review) (needed|required)",
    r"system prompt",
]


def looks_like_injection(text: str) -> bool:
    lower = text.lower()
    return any(re.search(pattern, lower) for pattern in INJECTION_PATTERNS)


def review_reasons(result, text: str, error: str | None) -> list[str]:
    """Reasons this request needs human review. Empty list means it can be auto-routed."""
    if error:
        return [error]

    reasons = []
    if looks_like_injection(text):
        reasons.append("Possible prompt injection: request tries to instruct the AI")
    if result.confidence < CONFIDENCE_THRESHOLD:
        reasons.append(f"Low confidence ({result.confidence:.2f})")
    if result.category in ALWAYS_REVIEW_CATEGORIES:
        reasons.append(f"{result.category} requests always get human review")
    if result.category == "Other":
        reasons.append("Could not be categorized")
    lower = text.lower()
    sensitive = [word for word in SENSITIVE_KEYWORDS if word in lower]
    if sensitive:
        reasons.append("Sensitive topic: " + ", ".join(sensitive))
    return reasons


def needs_escalation(result, error: str | None) -> bool:
    """High-urgency requests (and anything that failed) are pushed to the top of the queue."""
    return error is not None or result.urgency == "High"
