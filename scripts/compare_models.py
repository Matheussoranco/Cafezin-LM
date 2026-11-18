#!/usr/bin/env python
"""Compare baseline and candidate models on Portuguese benchmarks.

This script runs evaluation on both a baseline model and a candidate model,
then produces a comparative report showing:
- Per-task metrics for both models
- Multilingual retention/regression analysis
- Bits-per-UTF-8-byte comparison
- Statistical significance (where applicable)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def run_evaluation(
    model_path: str,
    tasks: list[str],
    output_dir: Path,
    device: str = "cuda",
    batch_size: str | int = "auto",
    limit: int | None = None,
    num_fewshot: int | None = None,
    include_path: str | None = None,
) -> dict[str, Any]:
    """Run lm_eval and return parsed results."""
    cmd = [
        "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_path},trust_remote_code=True,dtype=bfloat16",
        "--tasks", ",".join(tasks),
        "--device", device,
        "--batch_size", str(batch_size),
        "--output_path", str(output_dir),
    ]

    if limit is not None:
        cmd.extend(["--limit", str(limit)])

    if num_fewshot is not None:
        cmd.extend(["--num_fewshot", str(num_fewshot)])

    if include_path:
        cmd.extend(["--include_path", include_path])

    print(f"Running evaluation for {model_path}...")
    print(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Evaluation failed: {result.stderr}")
        return {}

    # Parse results
    results = {}
    for result_file in output_dir.rglob("results_*.json"):
        with open(result_file) as f:
            data = json.load(f)
            results.update(data.get("results", {}))

    return results


def compare_results(
    baseline_results: dict[str, Any],
    candidate_results: dict[str, Any],
    threshold: float = 0.01,
) -> dict[str, Any]:
    """Compare baseline and candidate results."""
    comparison = {
        "task_comparison": {},
        "summary": {
            "improved": 0,
            "regressed": 0,
            "unchanged": 0,
            "total": 0,
        },
        "multilingual_retention": {},
    }

    all_tasks = set(baseline_results.keys()) | set(candidate_results.keys())

    for task in sorted(all_tasks):
        base = baseline_results.get(task, {})
        cand = candidate_results.get(task, {})

        if not base or not cand:
            comparison["task_comparison"][task] = {
                "status": "missing",
                "baseline": base,
                "candidate": cand,
            }
            continue

        # Compare primary metrics
        task_diff = {}
        for metric in set(base.keys()) | set(cand.keys()):
            base_val = base.get(metric)
            cand_val = cand.get(metric)

            if isinstance(base_val, (int, float)) and isinstance(cand_val, (int, float)):
                diff = cand_val - base_val
                rel_diff = diff / abs(base_val) if base_val != 0 else float("inf")
                task_diff[metric] = {
                    "baseline": base_val,
                    "candidate": cand_val,
                    "absolute_diff": diff,
                    "relative_diff": rel_diff,
                    "improved": diff > threshold if "loss" not in metric.lower() and "perplexity" not in metric.lower() else diff < -threshold,
                    "regressed": diff < -threshold if "loss" not in metric.lower() and "perplexity" not in metric.lower() else diff > threshold,
                }

                # Update summary
                if task_diff[metric]["improved"]:
                    comparison["summary"]["improved"] += 1
                elif task_diff[metric]["regressed"]:
                    comparison["summary"]["regressed"] += 1
                else:
                    comparison["summary"]["unchanged"] += 1
                comparison["summary"]["total"] += 1

        comparison["task_comparison"][task] = {
            "status": "compared",
            "metrics": task_diff,
        }

    # Multilingual retention analysis
    # Group tasks by language/category
    pt_tasks = [t for t in all_tasks if any(x in t for x in ["pt", "por", "assin", "faquad", "flores_pt", "belebele_por"])]
    other_tasks = [t for t in all_tasks if t not in pt_tasks]

    for group_name, group_tasks in [("portuguese", pt_tasks), ("other", other_tasks)]:
        group_improved = 0
        group_regressed = 0
        group_total = 0

        for task in group_tasks:
            if task in comparison["task_comparison"]:
                for metric_data in comparison["task_comparison"][task].get("metrics", {}).values():
                    if metric_data.get("improved"):
                        group_improved += 1
                    elif metric_data.get("regressed"):
                        group_regressed += 1
                    group_total += 1

        if group_total > 0:
            comparison["multilingual_retention"][group_name] = {
                "improved_rate": group_improved / group_total,
                "regressed_rate": group_regressed / group_total,
                "total_metrics": group_total,
            }

    return comparison


def generate_report(
    comparison: dict[str, Any],
    baseline_name: str,
    candidate_name: str,
    output_path: Path,
) -> None:
    """Generate a human-readable comparison report."""
    with open(output_path, "w") as f:
        f.write(f"# Model Comparison Report\n\n")
        f.write(f"**Baseline:** {baseline_name}\n")
        f.write(f"**Candidate:** {candidate_name}\n\n")

        f.write("## Summary\n\n")
        summary = comparison["summary"]
        f.write(f"- Total metrics compared: {summary['total']}\n")
        f.write(f"- Improved: {summary['improved']}\n")
        f.write(f"- Regressed: {summary['regressed']}\n")
        f.write(f"- Unchanged: {summary['unchanged']}\n\n")

        f.write("## Multilingual Retention/Regression\n\n")
        for group, stats in comparison["multilingual_retention"].items():
            f.write(f"### {group.capitalize()}\n")
            f.write(f"- Improved rate: {stats['improved_rate']:.1%}\n")
            f.write(f"- Regressed rate: {stats['regressed_rate']:.1%}\n")
            f.write(f"- Total metrics: {stats['total_metrics']}\n\n")

        f.write("## Per-Task Comparison\n\n")
        for task, data in comparison["task_comparison"].items():
            f.write(f"### {task}\n")
            if data["status"] == "missing":
                f.write(f"- Status: Missing in one or both models\n")
                f.write(f"- Baseline: {data['baseline']}\n")
                f.write(f"- Candidate: {data['candidate']}\n\n")
            else:
                for metric, metrics in data["metrics"].items():
                    status = "↑" if metrics["improved"] else ("↓" if metrics["regressed"] else "→")
                    f.write(
                        f"- {metric}: {metrics['baseline']:.4f} → {metrics['candidate']:.4f} "
                        f"({metrics['absolute_diff']:+.4f}, {metrics['relative_diff']:+.1%}) {status}\n"
                    )
                f.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare baseline and candidate models")
    parser.add_argument("--baseline-model", required=True, help="Path to baseline model")
    parser.add_argument("--candidate-model", required=True, help="Path to candidate model")
    parser.add_argument("--baseline-name", default="baseline", help="Baseline model name")
    parser.add_argument("--candidate-name", default="candidate", help="Candidate model name")
    parser.add_argument("--output-dir", default="results/comparison", help="Output directory")
    parser.add_argument("--tasks", help="Comma-separated task names")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--batch-size", default="auto", help="Batch size")
    parser.add_argument("--limit", type=int, help="Limit examples per task")
    parser.add_argument("--num-fewshot", type=int, help="Number of few-shot examples")
    parser.add_argument("--include-path", help="Additional task include path")
    parser.add_argument("--threshold", type=float, default=0.01, help="Significance threshold")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_dir = output_dir / "baseline"
    candidate_dir = output_dir / "candidate"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir.mkdir(parents=True, exist_ok=True)

    # Determine tasks
    if args.tasks:
        tasks = [t.strip() for t in args.tasks.split(",")]
    else:
        # Default Portuguese tasks
        tasks = [
            "assin_paraphrase",
            "assin_entailment",
            "assin2_rte",
            "assin2_sts",
            "faquad_nli",
            "xnli",
            "belebele_por_Latn",
            "flores_pt-en",
            "flores_en-pt",
            "wikitext_pt",
        ]

    # Run evaluations
    print("=" * 60)
    print("Evaluating baseline model")
    print("=" * 60)
    baseline_results = run_evaluation(
        args.baseline_model,
        tasks,
        baseline_dir,
        device=args.device,
        batch_size=args.batch_size,
        limit=args.limit,
        num_fewshot=args.num_fewshot,
        include_path=args.include_path,
    )

    print("\n" + "=" * 60)
    print("Evaluating candidate model")
    print("=" * 60)
    candidate_results = run_evaluation(
        args.candidate_model,
        tasks,
        candidate_dir,
        device=args.device,
        batch_size=args.batch_size,
        limit=args.limit,
        num_fewshot=args.num_fewshot,
        include_path=args.include_path,
    )

    # Compare results
    print("\n" + "=" * 60)
    print("Comparing results")
    print("=" * 60)
    comparison = compare_results(baseline_results, candidate_results, threshold=args.threshold)

    # Save detailed results
    with open(output_dir / "baseline_results.json", "w") as f:
        json.dump(baseline_results, f, indent=2)
    with open(output_dir / "candidate_results.json", "w") as f:
        json.dump(candidate_results, f, indent=2)
    with open(output_dir / "comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    # Generate report
    generate_report(
        comparison,
        args.baseline_name,
        args.candidate_name,
        output_dir / "comparison_report.md",
    )

    print(f"\nResults saved to {output_dir}")
    print(f"Report: {output_dir / 'comparison_report.md'}")

    # Print summary
    summary = comparison["summary"]
    print(f"\nSummary: {summary['improved']} improved, {summary['regressed']} regressed, {summary['unchanged']} unchanged")

    return 0


if __name__ == "__main__":
    sys.exit(main())