"""
visualize_results.py

Builds an interactive HTML dashboard from the JSON files run_simulation.py
saves for a run:
    - metrics_{PROVIDER}.json  -> per-round trend lines
    - results_{PROVIDER}.json  -> per-capability/per-agent breakdowns,
                                   entropy, and the final_summary block

Called automatically at the end of run_simulation.py's main(), writing
dashboard_{PROVIDER}.html into that run's own runs/run_<timestamp>_<provider>/
folder. Can also be run standalone against any run folder:

Usage:
    python visualize_results.py                       # uses the most recent runs/ folder
    python visualize_results.py --run-dir runs/run_20260720_143205_groq
    python visualize_results.py --metrics m.json --results r.json --output d.html

Requires: pip install plotly numpy
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


RUNS_BASE_DIR = "runs"

# (json key, display title, is_fraction_to_percent)
ROUND_METRICS = [
    ("task_success_rate", "Task Success Rate (%)", True),
    ("mean_task_quality", "Mean Task Quality", False),
    ("capability_match_score", "Capability Match Score", False),
    ("agent_utility", "Agent Utility", False),
    ("load_distribution", "Agent Load", False),
    ("task_allocation_delay", "Allocation Delay (s)", False),
    ("task_bid_rate", "Task Bid Rate (%)", True),
    ("mean_bidders", "Mean Bidders", False),
    ("average_reputation", "Reputation", False),
    ("average_capability", "Average Capability", False),
    ("capability_entropy", "Mean Capability Entropy (bits)", False),
]


def _find_latest_run_dir(base: str = RUNS_BASE_DIR) -> str | None:
    if not os.path.isdir(base):
        return None
    candidates = [
        os.path.join(base, d) for d in os.listdir(base)
        if os.path.isdir(os.path.join(base, d))
    ]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def load_json(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"[!] Not found, skipping: {path}")
        return None
    except json.JSONDecodeError as e:
        print(f"[!] Invalid JSON in {path}: {e}")
        return None


def _linear_trend(x, y):
    """Least-squares line + R^2, matching the paper's Fig. 3 style."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~np.isnan(y)
    if mask.sum() < 2:
        return None, None

    coeffs = np.polyfit(x[mask], y[mask], 1)
    fit = np.poly1d(coeffs)
    y_pred = fit(x[mask])

    ss_res = np.sum((y[mask] - y_pred) ** 2)
    ss_tot = np.sum((y[mask] - np.mean(y[mask])) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return fit(x), r2


def build_round_metrics_figure(metrics_history):
    """One subplot per round-level metric, each auto-scaled to its own
    data range (paper-style zoomed axes) with a fitted trend line + R^2."""
    rounds = [m["round"] for m in metrics_history]

    present_metrics = [
        (key, title, to_pct)
        for key, title, to_pct in ROUND_METRICS
        if any(key in m for m in metrics_history)
    ]

    cols = 2
    rows = math.ceil(len(present_metrics) / cols)

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[title for _, title, _ in present_metrics],
        vertical_spacing=0.08,
        horizontal_spacing=0.09,
    )

    for i, (key, title, to_pct) in enumerate(present_metrics):
        row = i // cols + 1
        col = i % cols + 1

        y_raw = [m.get(key) for m in metrics_history]
        y = [v * 100 if (to_pct and v is not None) else v for v in y_raw]

        fig.add_trace(
            go.Scatter(
                x=rounds,
                y=y,
                mode="lines+markers",
                name=title,
                line=dict(width=1.5),
                marker=dict(size=4),
                showlegend=False,
                hovertemplate=f"Round %{{x}}<br>{title}: %{{y:.3f}}<extra></extra>",
            ),
            row=row,
            col=col,
        )

        trend_y, r2 = _linear_trend(rounds, y)
        if trend_y is not None:
            fig.add_trace(
                go.Scatter(
                    x=rounds,
                    y=trend_y,
                    mode="lines",
                    name=f"Trend (R²={r2:.2f})",
                    line=dict(dash="dash", width=1.5, color="firebrick"),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row,
                col=col,
            )
            fig.add_annotation(
                text=f"R²={r2:.2f}",
                xref=f"x{'' if i == 0 else i + 1} domain",
                yref=f"y{'' if i == 0 else i + 1} domain",
                x=0.98, y=0.02,
                showarrow=False,
                font=dict(size=10, color="firebrick"),
                row=row, col=col,
            )

        fig.update_yaxes(title_text=title, row=row, col=col)
        fig.update_xaxes(title_text="Round", row=row, col=col)

    fig.update_layout(
        height=320 * rows,
        title_text="Simulation Metrics Over Rounds (paper-style zoomed axes)",
        hovermode="x unified",
        template="plotly_white",
        showlegend=False,
    )

    return fig


def build_capability_figure(results):
    """Bar chart: pass rate per required-capability tag."""
    capability_success = results.get("capability_success", {})
    if not capability_success:
        return None

    tags = list(capability_success.keys())
    success_rates = [capability_success[t]["success_rate"] * 100 for t in tags]
    attempted = [capability_success[t]["attempted"] for t in tags]

    order = sorted(range(len(tags)), key=lambda i: success_rates[i])
    tags = [tags[i] for i in order]
    success_rates = [success_rates[i] for i in order]
    attempted = [attempted[i] for i in order]

    fig = go.Figure(
        go.Bar(
            x=success_rates,
            y=tags,
            orientation="h",
            text=[f"{r:.0f}% (n={a})" for r, a in zip(success_rates, attempted)],
            textposition="outside",
            hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(
        title_text="Success Rate by Required Capability Tag",
        xaxis_title="Success Rate (%)",
        xaxis=dict(range=[0, 115]),
        template="plotly_white",
        height=max(320, 40 * len(tags)),
    )
    return fig


def build_agent_figure(results):
    """Bar chart: success rate and reputation per agent."""
    agent_stats = results.get("agent_stats", [])
    if not agent_stats:
        return None

    agent_stats = sorted(agent_stats, key=lambda a: a["agent_id"])
    agent_ids = [f"Agent {a['agent_id']}" for a in agent_stats]
    success_rates = [a["success_rate"] * 100 for a in agent_stats]
    attempted = [a["tasks_attempted"] for a in agent_stats]
    reputation = [a["reputation"] for a in agent_stats]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=agent_ids,
            y=success_rates,
            name="Success Rate (%)",
            hovertemplate="%{x}<br>Success Rate: %{y:.1f}%<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=agent_ids,
            y=reputation,
            name="Final Reputation",
            mode="markers",
            marker=dict(size=9, symbol="diamond"),
            hovertemplate="%{x}<br>Reputation: %{y:.3f}<extra></extra>",
        ),
        secondary_y=True,
    )

    rep_lo = max(0.0, min(reputation) - 0.03)
    rep_hi = min(1.0, max(reputation) + 0.03)

    fig.update_yaxes(title_text="Success Rate (%)", range=[0, 100], secondary_y=False)
    fig.update_yaxes(title_text="Final Reputation", range=[rep_lo, rep_hi], secondary_y=True)
    fig.update_layout(
        title_text="Per-Agent Success Rate vs Final Reputation "
        f"(bars show tasks attempted: {min(attempted)}-{max(attempted)} range)",
        template="plotly_white",
        height=420,
    )
    return fig


def build_entropy_figure(results):
    """Per-agent final capability entropy, mirroring paper Fig. 4(b)."""
    final_entropy = results.get("final_agent_entropy")
    if not final_entropy:
        return None

    agent_ids = sorted(final_entropy.keys(), key=lambda k: int(k))
    values = [final_entropy[a] for a in agent_ids]
    labels = [f"Agent {a}" for a in agent_ids]

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker_color="mediumseagreen",
            hovertemplate="%{x}<br>Entropy: %{y:.3f} bits<extra></extra>",
        )
    )
    mean_e = float(np.mean(values))
    fig.add_hline(
        y=mean_e,
        line_dash="dash",
        line_color="firebrick",
        annotation_text=f"mean = {mean_e:.3f} bits",
        annotation_position="top left",
    )
    fig.update_layout(
        title_text="Final Capability Entropy per Agent (cf. paper Fig. 4(b))",
        yaxis_title="Entropy (bits)",
        template="plotly_white",
        height=380,
    )
    return fig


def build_reflection_arc_figure(metrics_history):
    """
    Dedicated figure for the reflection module (novel contribution, NOT
    part of the base paper's mechanism). Shows:
      1. Raw first-attempt success rate vs. effective success rate
         (first attempt OR a validated reflection-guided retry), with the
         gap between them shaded - that gap IS what reflection is buying you.
      2. The lift (effective - raw) per round, as a bar chart with trend.
      3. What fraction of that round's failures got recovered via retry.
      4. Cumulative lessons stored over the run (memory growth).
    """
    if not metrics_history or "effective_task_success_rate" not in metrics_history[0]:
        return None

    rounds = [m["round"] for m in metrics_history]
    raw = [m.get("task_success_rate", 0) * 100 for m in metrics_history]
    effective = [m.get("effective_task_success_rate", 0) * 100 for m in metrics_history]
    lift = [e - r for e, r in zip(effective, raw)]

    recovery_rounds, recovery_vals = [], []
    for m in metrics_history:
        rr = m.get("round_retry_recovery_rate")
        if rr is not None:
            recovery_rounds.append(m["round"])
            recovery_vals.append(rr * 100)

    cumulative_lessons = [m.get("cumulative_lessons_stored") for m in metrics_history]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Raw vs. Effective (Reflection-Recovered) Success Rate",
            "Reflection Lift per Round (Effective − Raw, pp)",
            "Retry Recovery Rate (% of that round's failures fixed)",
            "Cumulative Lessons Stored",
        ],
        vertical_spacing=0.14,
        horizontal_spacing=0.09,
    )

    # Panel 1: raw vs effective, shaded gap = reflection's contribution
    fig.add_trace(
        go.Scatter(
            x=rounds, y=raw, name="Raw Success Rate",
            mode="lines", line=dict(color="lightslategray", width=1.5),
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=rounds, y=effective, name="Effective Success Rate (with reflection)",
            mode="lines", line=dict(color="seagreen", width=2),
            fill="tonexty", fillcolor="rgba(46,139,87,0.15)",
        ),
        row=1, col=1,
    )
    fig.update_yaxes(title_text="Success Rate (%)", row=1, col=1)
    fig.update_xaxes(title_text="Round", row=1, col=1)

    # Panel 2: lift bar + trend
    fig.add_trace(
        go.Bar(x=rounds, y=lift, name="Lift (pp)", marker_color="mediumpurple", showlegend=False),
        row=1, col=2,
    )
    trend_y, r2 = _linear_trend(rounds, lift)
    if trend_y is not None:
        fig.add_trace(
            go.Scatter(
                x=rounds, y=trend_y, mode="lines", showlegend=False,
                line=dict(dash="dash", width=1.5, color="firebrick"),
                hoverinfo="skip",
            ),
            row=1, col=2,
        )
        fig.add_annotation(
            text=f"R²={r2:.2f}", xref="x2 domain", yref="y2 domain",
            x=0.98, y=0.02, showarrow=False,
            font=dict(size=10, color="firebrick"), row=1, col=2,
        )
    fig.update_yaxes(title_text="Lift (percentage points)", row=1, col=2)
    fig.update_xaxes(title_text="Round", row=1, col=2)

    # Panel 3: retry recovery rate (only rounds that actually had failures)
    fig.add_trace(
        go.Scatter(
            x=recovery_rounds, y=recovery_vals, name="Retry Recovery Rate",
            mode="lines+markers", line=dict(color="darkorange", width=1.5),
            showlegend=False,
        ),
        row=2, col=1,
    )
    fig.update_yaxes(title_text="Recovery Rate (%)", range=[0, 100], row=2, col=1)
    fig.update_xaxes(title_text="Round", row=2, col=1)

    # Panel 4: cumulative lessons stored (memory growth)
    fig.add_trace(
        go.Scatter(
            x=rounds, y=cumulative_lessons, name="Cumulative Lessons",
            mode="lines", line=dict(color="steelblue", width=2),
            fill="tozeroy", fillcolor="rgba(70,130,180,0.15)",
            showlegend=False,
        ),
        row=2, col=2,
    )
    fig.update_yaxes(title_text="Lessons Stored (cumulative)", row=2, col=2)
    fig.update_xaxes(title_text="Round", row=2, col=2)

    fig.update_layout(
        height=700,
        title_text="Reflection Module Arc (novel contribution - not present in the base paper)",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="center", x=0.22),
    )

    return fig


def _fmt_pct(x):
    return f"{x * 100:.1f}%" if isinstance(x, (int, float)) else "n/a"


def _fmt_num(x, d=3):
    return f"{x:.{d}f}" if isinstance(x, (int, float)) else "n/a"


def build_summary_header_html(results):
    """Renders results['final_summary'] (if present) as an HTML block at
    the top of the dashboard, so the headline numbers are visible without
    opening the JSON."""
    summary = results.get("final_summary") if results else None
    if not summary:
        return ""

    overall = summary.get("overall", {})
    reflection = summary.get("reflection", {})
    rep = summary.get("final_agent_reputation", {})
    ent = summary.get("final_agent_capability_entropy", {})

    def row(*cells):
        tds = "".join(
            f'<td style="padding:4px 12px;{"font-weight:600;" if i % 2 == 0 else ""}">{c}</td>'
            for i, c in enumerate(cells)
        )
        return f"<tr>{tds}</tr>"

    html = f"""
    <div style="font-family: -apple-system, sans-serif; background:#f7f7f9;
                border-radius:8px; padding:16px 24px; margin-bottom:24px; max-width:1000px;">
      <h2 style="margin-top:0;">Final Summary ({summary.get('rounds_completed', '?')} rounds)</h2>
      <table style="border-collapse: collapse; width:100%;">
        {row("Overall Task Success Rate", _fmt_pct(overall.get('task_success_rate')),
             "Overall Effective Success Rate", _fmt_pct(overall.get('effective_task_success_rate')))}
        {row("Mean Task Quality", _fmt_num(overall.get('mean_task_quality')),
             "Capability Match Score", _fmt_num(overall.get('capability_match_score')))}
        {row("Agent Utility", _fmt_num(overall.get('agent_utility')),
             "Allocation Delay (s)", _fmt_num(overall.get('task_allocation_delay')))}
        {row("Final Reputation (mean ± std)",
             f"{_fmt_num(rep.get('mean'))} ± {_fmt_num(rep.get('std'))}",
             "Final Capability Entropy (mean ± std)",
             f"{_fmt_num(ent.get('mean'))} ± {_fmt_num(ent.get('std'))} bits")}
      </table>
      <h3 style="margin-bottom:6px;">Reflection Module (novel contribution)</h3>
      <table style="border-collapse: collapse; width:100%;">
        {row("Total Failures", reflection.get('total_failures', 'n/a'),
             "Retries Attempted", reflection.get('total_retries_attempted', 'n/a'))}
        {row("Retries Succeeded", reflection.get('total_retries_succeeded', 'n/a'),
             "Failure Recovery Rate", _fmt_pct(reflection.get('failure_recovery_rate')))}
        {row("Lessons Stored", reflection.get('total_lessons_stored', 'n/a'),
             "Raw → Effective Success Lift", _fmt_pct(reflection.get('raw_vs_effective_success_lift')))}
      </table>
    </div>
    """
    return html


def build_dashboard(metrics_path, results_path, output_path):
    metrics_history = load_json(metrics_path)
    results = load_json(results_path)

    figures = []

    if metrics_history:
        figures.append(build_round_metrics_figure(metrics_history))
    else:
        print(f"[!] No round metrics loaded from {metrics_path}; skipping trend charts.")

    if results:
        cap_fig = build_capability_figure(results)
        if cap_fig:
            figures.append(cap_fig)

        agent_fig = build_agent_figure(results)
        if agent_fig:
            figures.append(agent_fig)

        entropy_fig = build_entropy_figure(results)
        if entropy_fig:
            figures.append(entropy_fig)
    else:
        print(f"[!] No results loaded from {results_path}; skipping capability/agent/entropy charts.")

    if metrics_history:
        reflection_fig = build_reflection_arc_figure(metrics_history)
        if reflection_fig:
            figures.append(reflection_fig)

    if not figures:
        print("[!] Nothing to plot. Check your file paths.")
        sys.exit(1)

    with open(output_path, "w") as f:
        f.write("<html><head><title>Simulation Dashboard</title></head><body>\n")
        f.write("<h1 style='font-family: sans-serif;'>Multi-Agent Simulation Dashboard</h1>\n")
        if results:
            f.write(build_summary_header_html(results))
        for i, fig in enumerate(figures):
            f.write(fig.to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False))
        f.write("</body></html>")

    print(f"[+] Dashboard written to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build an interactive dashboard from simulation output JSON.")

    try:
        from config import PROVIDER
    except ImportError:
        PROVIDER = "groq"

    parser.add_argument(
        "--run-dir", default=None,
        help="Path to a run folder under runs/. Defaults to the most recently created run.",
    )
    parser.add_argument("--metrics", default=None)
    parser.add_argument("--results", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    run_dir = args.run_dir or _find_latest_run_dir()

    if run_dir is None and (args.metrics is None or args.results is None):
        print(
            f"[!] No '{RUNS_BASE_DIR}/' folder found and no explicit --metrics/--results given. "
            "Run run_simulation.py first, or pass --metrics/--results explicitly."
        )
        sys.exit(1)

    metrics_path = args.metrics or os.path.join(run_dir, f"metrics_{PROVIDER}.json")
    results_path = args.results or os.path.join(run_dir, f"results_{PROVIDER}.json")
    output_path = args.output or (
        os.path.join(run_dir, f"dashboard_{PROVIDER}.html") if run_dir
        else f"dashboard_{PROVIDER}.html"
    )

    if run_dir:
        print(f"[*] Using run folder: {run_dir}")

    build_dashboard(metrics_path, results_path, output_path)