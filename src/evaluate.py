"""Measure triage quality against hand-labeled expected answers.

An automation is only as trustworthy as its measured accuracy. If the input file
has `expected_category` and `expected_urgency` columns, these metrics are reported:

- category accuracy and urgency accuracy
- missed urgent: truly High-urgency requests the system did NOT mark High.
  This is the most costly error (an outage or security issue sitting in a normal queue),
  so it's tracked separately.
- a list of every mistake, so they can be reviewed and used to improve the prompt or rules
"""

import pandas as pd


def evaluate(results: pd.DataFrame, labels: pd.DataFrame) -> dict | None:
    if not {"expected_category", "expected_urgency"}.issubset(labels.columns):
        return None

    df = results.merge(labels[["request_id", "expected_category", "expected_urgency"]], on="request_id")
    truly_high = df["expected_urgency"] == "High"
    errors = df[(df["category"] != df["expected_category"]) | (df["urgency"] != df["expected_urgency"])]

    return {
        "n": len(df),
        "category_accuracy": (df["category"] == df["expected_category"]).mean(),
        "urgency_accuracy": (df["urgency"] == df["expected_urgency"]).mean(),
        "missed_urgent": int((truly_high & (df["urgency"] != "High")).sum()),
        "total_urgent": int(truly_high.sum()),
        "errors": errors[["request_id", "category", "expected_category", "urgency", "expected_urgency"]],
    }


def print_evaluation(metrics: dict) -> None:
    print("\n=== Evaluation vs. labeled answers ===")
    print(f"Category accuracy: {metrics['category_accuracy']:.0%} ({metrics['n']} requests)")
    print(f"Urgency accuracy:  {metrics['urgency_accuracy']:.0%}")
    print(f"Missed urgent:     {metrics['missed_urgent']} of {metrics['total_urgent']} truly urgent requests")
    if len(metrics["errors"]):
        print("\nMistakes to review:")
        print(metrics["errors"].to_string(index=False))
