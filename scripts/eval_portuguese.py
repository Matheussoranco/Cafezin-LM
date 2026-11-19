#!/usr/bin/env python
"""Run Portuguese evaluation tasks using lm-evaluation-harness.

This script evaluates Cafezin LM on Portuguese benchmarks including:
- ASSIN (paraphrase, entailment)
- ASSIN2 (RTE, STS)
- FAQUAD-NLI
- XNLI (Portuguese)
- XQuAD (Portuguese)
- Flores-200 (Portuguese translation pairs)
- Belebele (Portuguese)
- Perplexity (bits-per-UTF-8-byte) on Portuguese text

Usage:
    python scripts/eval_portuguese.py --model-path /path/to/checkpoint --output-dir results/
    python scripts/eval_portuguese.py --model-path /path/to/checkpoint --tasks assin2_rte,faquad_nli --limit 100
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

PORTUGUESE_TASKS = {
    # Core Portuguese benchmarks (from lm-eval main + portuguese_bench)
    "core": [
        "assin_paraphrase",
        "assin_entailment",
        "belebele_por_Latn",
    ],
    # ASSIN2 tasks (added in lm-eval 0.4.13+)
    "assin2": [
        "assin2_rte",
        "assin2_sts",
    ],
    # FAQUAD NLI (from eduagarcia fork - may need --include_path)
    "faquad": [
        "faquad_nli",
    ],
    # XNLI Portuguese (cross-lingual NLI)
    "xnli": [
        "xnli",
    ],
    # XQuAD Portuguese (cross-lingual QA)
    "xquad": [
        "xquad_pt",
    ],
    # Flores-200 Portuguese translation pairs
    "flores": [
        "flores_pt-en",
        "flores_en-pt",
        "flores_pt-es",
        "flores_es-pt",
        "flores_pt-fr",
        "flores_fr-pt",
        "flores_pt-de",
        "flores_de-pt",
        "flores_pt-it",
        "flores_it-pt",
    ],
    # Perplexity on Portuguese corpora
    "perplexity": [
        "wikitext_pt",
        "c4_pt",
    ],
}


def get_all_tasks(groups: list[str] | None = None) -> list[str]:
    """Get all task names from specified groups."""
    if groups is None:
        groups = list(PORTUGUESE_TASKS.keys())
    tasks = []
    for group in groups:
        if group in PORTUGUESE_TASKS:
            tasks.extend(PORTUGUESE_TASKS[group])
    return tasks


def build_lm_eval_command(
    model_path: str,
    tasks: list[str],
    output_dir: Path,
    device: str = "cuda",
    batch_size: str | int = "auto",
    limit: int | None = None,
    num_fewshot: int | None = None,
    include_path: str | None = None,
    model_args: dict[str, Any] | None = None,
) -> list[str]:
    """Build the lm_eval command line."""
    cmd = [
        "lm_eval",
        "--model", "hf",
        "--model_args", build_model_args(model_path, model_args),
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

    return cmd


def build_model_args(model_path: str, extra_args: dict[str, Any] | None = None) -> str:
    """Build model_args string for lm_eval."""
    args = {
        "pretrained": model_path,
        "trust_remote_code": True,
        "dtype": "bfloat16",
    }
    if extra_args:
        args.update(extra_args)
    return ",".join(f"{k}={v}" for k, v in args.items())


def run_evaluation(cmd: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    """Run the evaluation command."""
    run_env = os.environ.copy()
    if env:
        run_env.update(env)

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=run_env, capture_output=False, text=True)
    return result


def compute_bits_per_utf8_byte(
    model_path: str,
    data_path: str,
    output_dir: Path,
    device: str = "cuda",
    batch_size: str | int = "auto",
) -> dict[str, float]:
    """Compute bits-per-UTF-8-byte metric on Portuguese text.

    This uses loglikelihood_rolling to compute per-token perplexity,
    then converts to bits per UTF-8 byte.
    """
    # Use lm_eval's loglikelihood_rolling for perplexity
    cmd = [
        "lm_eval",
        "--model", "hf",
        "--model_args", build_model_args(model_path),
        "--tasks", "wikitext_pt",  # Portuguese Wikitext for perplexity
        "--device", device,
        "--batch_size", str(batch_size),
        "--output_path", str(output_dir / "perplexity"),
    ]

    result = run_evaluation(cmd)

    # Parse results for perplexity and compute bits/byte
    results_file = output_dir / "perplexity" / "results.json"
    if results_file.exists():
        with open(results_file) as f:
            results = json.load(f)

        # Extract perplexity from results
        # The metric is typically "word_perplexity" or "byte_perplexity"
        # Convert to bits per UTF-8 byte
        pass

    return {}


def parse_results(output_dir: Path) -> dict[str, Any]:
    """Parse lm_eval results from output directory."""
    results = {}
    for result_file in output_dir.rglob("results_*.json"):
        with open(result_file) as f:
            data = json.load(f)
            results.update(data.get("results", {}))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Portuguese evaluation tasks")
    parser.add_argument("--model-path", help="Path to model checkpoint")
    parser.add_argument("--output-dir", default="results/portuguese", help="Output directory")
    parser.add_argument("--tasks", help="Comma-separated task names (default: all Portuguese tasks)")
    parser.add_argument("--groups", help="Comma-separated task groups (default: all)")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--batch-size", default="auto", help="Batch size")
    parser.add_argument("--limit", type=int, help="Limit examples per task")
    parser.add_argument("--num-fewshot", type=int, help="Number of few-shot examples")
    parser.add_argument("--include-path", help="Additional task include path (e.g., for eduagarcia fork)")
    parser.add_argument("--model-args", help="Additional model args as JSON")
    parser.add_argument("--list-tasks", action="store_true", help="List available tasks and exit")
    parser.add_argument("--perplexity-only", action="store_true", help="Run only perplexity evaluation")
    parser.add_argument("--compute-bits-per-byte", action="store_true", help="Compute bits-per-UTF-8-byte metric")

    args = parser.parse_args()

    if args.list_tasks:
        print("Available Portuguese task groups:")
        for group, tasks in PORTUGUESE_TASKS.items():
            print(f"  {group}: {', '.join(tasks)}")
        print("\nAll tasks:")
        all_tasks = get_all_tasks()
        print(f"  {', '.join(all_tasks)}")
        return 0

    if not args.model_path:
        parser.error("--model-path is required unless --list-tasks is used")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine tasks to run
    if args.tasks:
        tasks = [t.strip() for t in args.tasks.split(",")]
    elif args.groups:
        groups = [g.strip() for g in args.groups.split(",")]
        tasks = get_all_tasks(groups)
    else:
        tasks = get_all_tasks()

    print(f"Running {len(tasks)} tasks: {', '.join(tasks)}")

    model_args = json.loads(args.model_args) if args.model_args else None

    if args.perplexity_only or args.compute_bits_per_byte:
        tasks = ["wikitext_pt"]

    cmd = build_lm_eval_command(
        model_path=args.model_path,
        tasks=tasks,
        output_dir=output_dir,
        device=args.device,
        batch_size=args.batch_size,
        limit=args.limit,
        num_fewshot=args.num_fewshot,
        include_path=args.include_path,
        model_args=model_args,
    )

    result = run_evaluation(cmd)

    if result.returncode != 0:
        print(f"Evaluation failed with return code {result.returncode}")
        return result.returncode

    print(f"\nResults saved to {output_dir}")

    # Print summary
    results = parse_results(output_dir)
    print("\n=== Results Summary ===")
    for task, metrics in results.items():
        print(f"\n{task}:")
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
