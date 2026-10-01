"""AI request triage: turn a pile of messy employee requests into a routed, prioritized queue.

Usage:
    python src/triage.py              # uses Claude (needs ANTHROPIC_API_KEY in .env)
    python src/triage.py --baseline   # offline keyword baseline, no API key needed
    python src/triage.py --input my.csv --output results.csv
"""

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
ROOT = Path(__file__).resolve().parent.parent

try:  # load .env before importing modules that read settings from it (e.g. TRIAGE_MODEL)
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from classifier import DEFAULT_MODEL, Usage, safe_classify  # noqa: E402
from evaluate import evaluate, print_evaluation  # noqa: E402
from guardrails import needs_escalation, review_reasons  # noqa: E402
from report import build_report  # noqa: E402
from schema import ROUTING  # noqa: E402

MANUAL_MINUTES_PER_REQUEST = 5  # assumption: time for a person to read, categorize, route and reply
URGENCY_ORDER = {"High": 0, "Medium": 1, "Low": 2}


def get_client(baseline: bool):
    if baseline:
        return None
    if not os.getenv("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set. Add it to .env, or run with --baseline.")
    import anthropic
    return anthropic.Anthropic()


def format_details(details: dict) -> str:
    return "; ".join(f"{k}: {v}" for k, v in details.items())


def triage(requests: pd.DataFrame, client=None, model: str = DEFAULT_MODEL) -> tuple[pd.DataFrame, Usage]:
    rows, total = [], Usage()
    for _, req in requests.iterrows():
        result, error, usage = safe_classify(req["text"], client, model)
        total.input_tokens += usage.input_tokens
        total.output_tokens += usage.output_tokens
        reasons = review_reasons(result, req["text"], error)
        rows.append({
            "request_id": req["request_id"],
            "submitted_by": req["submitted_by"],
            "category": result.category if result else "Unclassified",
            "urgency": result.urgency if result else "High",
            "escalate": needs_escalation(result, error),
            "routed_to": ROUTING[result.category] if result else "Operations Intake",
            "needs_human_review": bool(reasons),
            "review_reasons": "; ".join(reasons),
            "confidence": round(result.confidence, 2) if result else None,
            "summary": result.summary if result else "",
            "key_details": format_details(result.key_details) if result else "",
            "draft_reply": result.draft_reply if result else "",
            "reasoning": result.reasoning if result else error,
        })
    df = pd.DataFrame(rows)
    df = df.sort_values("urgency", key=lambda s: s.map(URGENCY_ORDER), kind="stable").reset_index(drop=True)
    return df, total


def print_summary(df: pd.DataFrame, mode: str, stats: dict) -> None:
    n = len(df)
    auto = int((~df["needs_human_review"]).sum())
    print("\n=== AI Request Triage Summary ===")
    print(f"Mode: {mode}")
    print(f"Requests processed: {n} in {stats['seconds']:.1f}s")
    if stats["input_tokens"]:
        print(f"Tokens used: {stats['input_tokens']:,} input / {stats['output_tokens']:,} output")
    print(f"Auto-routed: {auto} ({auto / n:.0%}) | Human review: {n - auto} | Escalated as urgent: {int(df['escalate'].sum())}")
    print("\nBy category:\n" + df["category"].value_counts().to_string())
    print("\nBy urgency:\n" + df["urgency"].value_counts().to_string())
    print(f"\nEstimated manual triage time saved: ~{stats['minutes_saved']} minutes "
          f"(auto-routed requests x {MANUAL_MINUTES_PER_REQUEST} min each; reviewed requests still arrive "
          f"pre-categorized with a draft reply)")


def main():
    parser = argparse.ArgumentParser(description="AI-powered triage of business requests")
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "sample_requests.csv")
    parser.add_argument("--output", type=Path, default=None, help="CSV path; an HTML report is saved alongside it")
    parser.add_argument("--baseline", action="store_true", help="Run offline with the keyword baseline")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    requests = pd.read_csv(args.input)
    client = get_client(args.baseline)
    mode = "Offline keyword baseline" if args.baseline else f"Claude ({args.model})"
    suffix = ("baseline" if args.baseline else "claude") + ("" if "sample" in args.input.stem else f"_{args.input.stem}")
    output = args.output or ROOT / "outputs" / f"results_{suffix}.csv"

    start = time.time()
    results, usage = triage(requests, client, args.model)
    failed = results["reasoning"].astype(str).str.startswith("Classification failed")
    if failed.all():
        # e.g. a bad API key or model name: stop instead of writing a report full of failures
        sys.exit(f"Every request failed, so no results were saved. First error: {results['reasoning'].iloc[0]}")
    stats = {
        "seconds": time.time() - start,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "minutes_saved": int((~results["needs_human_review"]).sum()) * MANUAL_MINUTES_PER_REQUEST,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)
    metrics = evaluate(results, requests)
    report_path = output.with_suffix(".html")
    report_path.write_text(build_report(results, mode, metrics, stats), encoding="utf-8")

    print_summary(results, mode, stats)
    if metrics:
        print_evaluation(metrics)
        method = "Keyword baseline" if args.baseline else "Claude"
        dataset = "Main set" if "sample" in args.input.stem else "Held-out set"
        print(f"\nREADME results row:\n| {method} | {dataset} | {metrics['category_accuracy']:.0%} | "
              f"{metrics['urgency_accuracy']:.0%} | {metrics['missed_urgent']} of {metrics['total_urgent']} |")
    print(f"\nResults: {output}\nReport:  {report_path}  (open in a browser)")


if __name__ == "__main__":
    main()
