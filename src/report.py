"""Builds a self-contained HTML report so non-technical stakeholders can see the results in a browser."""

from html import escape

import pandas as pd

CSS = """
:root { --ink:#1b1f24; --muted:#5b6470; --line:#e3e6ea; --bg:#f6f7f9; --card:#fff;
        --high:#b42318; --med:#b54708; --low:#067647; --accent:#1d4ed8; }
* { box-sizing:border-box; }
body { margin:0; font:14px/1.5 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
       color:var(--ink); background:var(--bg); }
main { max-width:1100px; margin:0 auto; padding:32px 20px 48px; }
h1 { font-size:24px; margin:0 0 4px; } h2 { font-size:16px; margin:32px 0 12px; }
.sub { color:var(--muted); margin:0 0 24px; }
.stats { display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:12px; }
.stat { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px 16px; }
.stat b { display:block; font-size:24px; } .stat span { color:var(--muted); font-size:12px; }
.table-wrap { overflow-x:auto; background:var(--card); border:1px solid var(--line); border-radius:10px; }
table { width:100%; border-collapse:collapse; min-width:760px; }
th, td { text-align:left; padding:10px 12px; border-bottom:1px solid var(--line); vertical-align:top; }
th { font-size:12px; color:var(--muted); font-weight:600; background:#fafbfc; }
tr:last-child td { border-bottom:none; }
td:first-child, th:first-child { white-space:nowrap; }
.pill { display:inline-block; padding:2px 8px; border-radius:999px; font-size:12px; font-weight:600; }
.High { background:#fee4e2; color:var(--high); } .Medium { background:#fef0c7; color:var(--med); }
.Low { background:#dcfae6; color:var(--low); }
.review { color:var(--high); font-size:12px; } .auto { color:var(--low); font-size:12px; }
.small { color:var(--muted); font-size:12px; }
"""


def build_report(df: pd.DataFrame, mode: str, metrics: dict | None, run_stats: dict) -> str:
    n = len(df)
    auto = int((~df["needs_human_review"]).sum())
    cards = [
        (n, "requests triaged"),
        (f"{auto / n:.0%}", "auto-routed"),
        (int(df["needs_human_review"].sum()), "sent to human review"),
        (int(df["escalate"].sum()), "escalated as urgent"),
        (f"~{run_stats['minutes_saved']} min", "manual triage time saved"),
    ]
    if metrics:
        cards += [
            (f"{metrics['category_accuracy']:.0%}", "category accuracy"),
            (f"{metrics['missed_urgent']} of {metrics['total_urgent']}", "urgent requests missed"),
        ]
    stat_html = "".join(f'<div class="stat"><b>{escape(str(v))}</b><span>{escape(k)}</span></div>' for v, k in cards)

    rows = []
    for _, r in df.iterrows():
        decision = (f'<span class="review">Human review: {escape(r["review_reasons"])}</span>'
                    if r["needs_human_review"] else '<span class="auto">Auto-routed</span>')
        rows.append(
            f"<tr><td>{escape(r['request_id'])}</td>"
            f"<td><span class='pill {escape(str(r['urgency']))}'>{escape(str(r['urgency']))}</span></td>"
            f"<td>{escape(r['category'])}<div class='small'>{escape(r['routed_to'])}</div></td>"
            f"<td>{escape(r['summary'])}<div class='small'>{escape(r['key_details'])}</div></td>"
            f"<td>{decision}</td></tr>"
        )

    tokens = ""
    if run_stats.get("input_tokens"):
        tokens = f" · {run_stats['input_tokens'] + run_stats['output_tokens']:,} tokens"

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI Request Triage Report</title><style>{CSS}</style></head>
<body><main>
<h1>AI Request Triage Report</h1>
<p class="sub">{escape(mode)} · {n} requests in {run_stats['seconds']:.1f}s{tokens}</p>
<div class="stats">{stat_html}</div>
<h2>Prioritized queue</h2>
<div class="table-wrap"><table>
<thead><tr><th>ID</th><th>Urgency</th><th>Category / routed to</th><th>Summary / key details</th><th>Decision</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>
</main></body></html>"""
