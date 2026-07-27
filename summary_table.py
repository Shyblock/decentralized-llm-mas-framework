"""
summary_table.py

Computes a Table-V-style summary and prints it next to the paper's
reported numbers (Qi et al., "Toward Transparent and Incentive-Compatible
Collaboration in Decentralized LLM Multi-Agent Systems", IEEE TNSE 2026),
plus your reflection module's own arc numbers (not present in the paper).

Prefers results.json's embedded "final_summary" block (written by
run_simulation.py) when present; falls back to recomputing from the raw
per-round histories for older runs that predate that block.

Usage:
    python summary_table.py                          # uses the most recent runs/ folder
    python summary_table.py --run-dir runs/run_20260720_143205_groq
    python summary_table.py --metrics m.json --results r.json
    python summary_table.py --csv out.csv             # also write a CSV
"""

import argparse
import csv
import json
import os
import sys

import numpy as np


RUNS_BASE_DIR = "runs"

# Paper's reported numbers (Section VI-A/B, Table V, Fig. 3/4).
PAPER = {
    "task_success_rate_avg":      0.8677,
    "task_success_rate_early":    0.8015,
    "task_success_rate_late":     0.9449,
    "mean_task_quality_avg":      0.82,
    "mean_task_quality_early":    0.75,
    "mean_task_quality_late":     0.89,
    "quality_std_early":          0.061,
    "quality_std_late":           0.030,
    "capmatch_early":             0.69,
    "capmatch_round25":           0.735,
    "capmatch_round50":           0.765,
    "capmatch_max":               0.794,
    "agent_utility_round1":       3.52,
    "agent_utility_final":        6.43,
    "agent_utility_avg":          5.02,
    "allocation_delay_avg":       0.99,
    "allocation_delay_min":       0.43,
    "allocation_delay_max":       1.56,
    "bid_rate_round1":            0.92,
    "bid_rate_round20":           0.729,
    "bid_rate_round40plus":       0.55,
    "final_mean_reputation":      0.488,
    "final_reputation_std":       0.044,
    "final_reputation_min":       0.429,
    "final_reputation_max":       0.558,
    "capability_entropy_init":    2.31,
    "capability_entropy_final":   2.315,
    "capability_entropy_std":     0.073,
}


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
        print(f"[!] Not found: {path}")
        return None
    except json.JSONDecodeError as e:
        print(f"[!] Invalid JSON in {path}: {e}")
        return None


def _early_late_split(values, frac=0.2):
    values = [v for v in values if v is not None]
    if not values:
        return None, None
    n = max(1, int(len(values) * frac))
    return float(np.mean(values[:n])), float(np.mean(values[-n:]))


def compute_paper_comparison(metrics_history, results):
    """
    Builds the PAPER-comparable metric dict. Prefers results['final_summary']
    (written directly by run_simulation.py) when available; falls back to
    recomputing from raw histories for older run folders that predate it.
    """
    out = {}
    final_summary = (results or {}).get("final_summary")

    def series(key):
        return [m.get(key) for m in metrics_history] if metrics_history else []

    if final_summary:
        overall = final_summary.get("overall", {})
        early_late = final_summary.get("early_vs_late", {})
        rep = final_summary.get("final_agent_reputation", {})
        ent = final_summary.get("final_agent_capability_entropy", {})

        out["task_success_rate_avg"] = overall.get("task_success_rate")
        out["task_success_rate_early"] = early_late.get("task_success_rate", {}).get("early")
        out["task_success_rate_late"] = early_late.get("task_success_rate", {}).get("late")

        out["mean_task_quality_avg"] = overall.get("mean_task_quality")
        out["mean_task_quality_early"] = early_late.get("mean_task_quality", {}).get("early")
        out["mean_task_quality_late"] = early_late.get("mean_task_quality", {}).get("late")

        out["agent_utility_avg"] = overall.get("agent_utility")
        out["allocation_delay_avg"] = overall.get("task_allocation_delay")

        out["final_mean_reputation"] = rep.get("mean")
        out["final_reputation_std"] = rep.get("std")
        out["final_reputation_min"] = rep.get("min")
        out["final_reputation_max"] = rep.get("max")

        out["capability_entropy_final"] = ent.get("mean")
        out["capability_entropy_std"] = ent.get("std")

        cm = series("capability_match_score")
        if cm:
            out["capmatch_round25"] = cm[min(24, len(cm) - 1)]
            out["capmatch_round50"] = cm[min(49, len(cm) - 1)]
            out["capmatch_max"] = float(np.max(cm))

        u = series("agent_utility")
        if u:
            out["agent_utility_round1"] = u[0]
            out["agent_utility_final"] = u[-1]

        d = series("task_allocation_delay")
        if d:
            out["allocation_delay_min"] = float(np.min(d))
            out["allocation_delay_max"] = float(np.max(d))

        br = series("task_bid_rate")
        if br:
            out["bid_rate_round1"] = br[0]
            out["bid_rate_round20"] = br[min(19, len(br) - 1)]
            n40 = [v for i, v in enumerate(br) if i >= 39]
            out["bid_rate_round40plus"] = float(np.mean(n40)) if n40 else br[-1]

        eh = series("capability_entropy")
        if eh:
            out["capability_entropy_init"] = eh[0]

        return out

    # ---- Fallback: recompute from raw histories (pre-final_summary runs) ----
    sr = series("task_success_rate")
    if sr:
        out["task_success_rate_avg"] = float(np.mean(sr))
        e, l = _early_late_split(sr)
        out["task_success_rate_early"], out["task_success_rate_late"] = e, l

    q = series("mean_task_quality")
    if q:
        out["mean_task_quality_avg"] = float(np.mean(q))
        e, l = _early_late_split(q)
        out["mean_task_quality_early"], out["mean_task_quality_late"] = e, l
        n = max(1, int(len(q) * 0.2))
        out["quality_std_early"] = float(np.std(q[:n]))
        out["quality_std_late"] = float(np.std(q[-n:]))

    cm = series("capability_match_score")
    if cm:
        n = max(1, int(len(cm) * 0.2))
        out["capmatch_early"] = float(np.mean(cm[:n]))
        out["capmatch_round25"] = cm[min(24, len(cm) - 1)]
        out["capmatch_round50"] = cm[min(49, len(cm) - 1)]
        out["capmatch_max"] = float(np.max(cm))

    u = series("agent_utility")
    if u:
        out["agent_utility_round1"] = u[0]
        out["agent_utility_final"] = u[-1]
        out["agent_utility_avg"] = float(np.mean(u))

    d = series("task_allocation_delay")
    if d:
        out["allocation_delay_avg"] = float(np.mean(d))
        out["allocation_delay_min"] = float(np.min(d))
        out["allocation_delay_max"] = float(np.max(d))

    br = series("task_bid_rate")
    if br:
        out["bid_rate_round1"] = br[0]
        out["bid_rate_round20"] = br[min(19, len(br) - 1)]
        n40 = [v for i, v in enumerate(br) if i >= 39]
        out["bid_rate_round40plus"] = float(np.mean(n40)) if n40 else br[-1]

    if results and results.get("agent_stats"):
        reps = [a["reputation"] for a in results["agent_stats"]]
        out["final_mean_reputation"] = float(np.mean(reps))
        out["final_reputation_std"] = float(np.std(reps))
        out["final_reputation_min"] = float(np.min(reps))
        out["final_reputation_max"] = float(np.max(reps))

    eh = series("capability_entropy")
    if eh:
        out["capability_entropy_init"] = eh[0]
        out["capability_entropy_final"] = eh[-1]
        out["capability_entropy_std"] = float(np.std(eh))
    elif results and results.get("final_agent_entropy"):
        vals = list(results["final_agent_entropy"].values())
        out["capability_entropy_final"] = float(np.mean(vals))
        out["capability_entropy_std"] = float(np.std(vals))

    return out


def _fmt(v):
    if v is None:
        return "n/a"
    return f"{v:.3f}"


def _verdict(key, yours, papers):
    if yours is None or papers is None:
        return ""
    lower_is_better = any(s in key for s in ["delay", "std"])
    if lower_is_better:
        return "▲ better" if yours < papers else ("▼ worse" if yours > papers else "= tie")
    return "▲ better" if yours > papers else ("▼ worse" if yours < papers else "= tie")


def print_paper_comparison(summary, csv_path=None):
    rows = []
    header = f"{'Metric':<28} {'Yours':>10} {'Paper':>10}   {'Verdict'}"
    print("=" * len(header))
    print("PAPER COMPARISON")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for key, paper_val in PAPER.items():
        yours = summary.get(key)
        line = f"{key:<28} {_fmt(yours):>10} {_fmt(paper_val):>10}   {_verdict(key, yours, paper_val)}"
        print(line)
        rows.append([key, yours, paper_val])

    if csv_path:
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "your_value", "paper_value"])
            writer.writerows(rows)
        print(f"\n[+] CSV written to {csv_path}")


def print_reflection_summary(results):
    """Prints the reflection module's own numbers - these have no paper
    equivalent (the reflection/retry mechanism is your novel contribution
    on top of the base paper's incentive mechanism)."""
    final_summary = (results or {}).get("final_summary")
    if not final_summary:
        print("\n[!] No final_summary.reflection block found (older run format) - skipping.")
        return

    r = final_summary.get("reflection", {})

    print()
    print("=" * 60)
    print("REFLECTION MODULE (novel contribution - no paper baseline)")
    print("=" * 60)
    print(f"{'Total failures':<32} {r.get('total_failures', 'n/a')}")
    print(f"{'Retries attempted':<32} {r.get('total_retries_attempted', 'n/a')}")
    print(f"{'Retries succeeded':<32} {r.get('total_retries_succeeded', 'n/a')}")

    rrr = r.get("retry_recovery_rate")
    frr = r.get("failure_recovery_rate")
    lift = r.get("raw_vs_effective_success_lift")

    print(f"{'Retry recovery rate':<32} {f'{rrr*100:.1f}%' if rrr is not None else 'n/a'}")
    print(f"{'Failure recovery rate':<32} {f'{frr*100:.1f}%' if frr is not None else 'n/a'}")
    print(f"{'Lessons stored':<32} {r.get('total_lessons_stored', 'n/a')}")
    print(f"{'Raw -> Effective success lift':<32} {f'{lift*100:.1f}pp' if lift is not None else 'n/a'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare your simulation run against the paper's reported results.")

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
    parser.add_argument("--csv", default=None, help="Optional path to also write a CSV")
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

    if run_dir:
        print(f"[*] Using run folder: {run_dir}\n")

    metrics_history = load_json(metrics_path)
    results = load_json(results_path)

    if not metrics_history and not results:
        print("[!] Nothing loaded. Check file paths.")
        sys.exit(1)

    summary = compute_paper_comparison(metrics_history or [], results or {})
    print_paper_comparison(summary, csv_path=args.csv)
    print_reflection_summary(results or {})