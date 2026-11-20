#!/usr/bin/env python
"""Main evaluation runner for Portuguese LM evaluation.

This script orchestrates the full evaluation pipeline:
1. Records evaluation metadata (versions, splits, contamination)
2. Runs Portuguese benchmark tasks
3. Computes bits-per-UTF-8-byte metric
4. Compares baseline and candidate models
5. Runs generative evaluation (MT-Bench/LLM-judge style)
6. Generates comprehensive report

Usage:
    python scripts/run_eval.py --baseline-model /path/to/baseline --candidate-model /path/to/candidate --output-dir results/
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run_script(script: str, args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    """Run a Python script with arguments."""
    cmd = [sys.executable, script, *args]
    print(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=cwd, capture_output=False, text=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run full Portuguese LM evaluation pipeline")
    parser.add_argument("--baseline-model", required=True, help="Path to baseline model checkpoint")
    parser.add_argument("--candidate-model", required=True, help="Path to candidate model checkpoint")
    parser.add_argument("--tokenizer-path", required=True, help="Path to tokenizer.json")
    parser.add_argument("--corpus-path", required=True, help="Path to Portuguese corpus for perplexity")
    parser.add_argument("--output-dir", default="results/portuguese_eval", help="Output directory")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--batch-size", default="auto", help="Batch size")
    parser.add_argument("--limit", type=int, help="Limit examples per task (for testing)")
    parser.add_argument("--num-fewshot", type=int, default=0, help="Number of few-shot examples")
    parser.add_argument("--include-path", help="Additional task include path (e.g., eduagarcia fork)")
    parser.add_argument("--skip-metadata", action="store_true", help="Skip metadata recording")
    parser.add_argument("--skip-perplexity", action="store_true", help="Skip perplexity computation")
    parser.add_argument("--skip-comparison", action="store_true", help="Skip model comparison")
    parser.add_argument("--skip-generative", action="store_true", help="Skip generative evaluation")
    parser.add_argument("--generative-prompts", default="eval/generative/prompts_v1.json", help="Path to generative prompts JSON")
    parser.add_argument("--generative-rubric", default="eval/generative/rubric_v1.json", help="Path to generative rubric JSON")
    parser.add_argument("--judge-model", default="heuristic", help="Judge model for generative eval (heuristic, gpt-4o, etc.)")
    parser.add_argument("--gen-temperature", type=float, default=0.7, help="Generation temperature for generative eval")
    parser.add_argument("--gen-top-p", type=float, default=0.9, help="Generation top-p for generative eval")
    parser.add_argument("--gen-max-tokens", type=int, default=1024, help="Max tokens for generative eval")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scripts_dir = Path(__file__).parent
    project_root = scripts_dir.parent

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
        "flores_pt-es",
        "flores_es-pt",
        "flores_pt-fr",
        "flores_fr-pt",
        "flores_pt-de",
        "flores_de-pt",
        "flores_pt-it",
        "flores_it-pt",
        "wikitext_pt",
    ]

    task_list = ",".join(tasks)

    # Step 1: Record metadata
    if not args.skip_metadata:
        print("\n" + "=" * 60)
        print("Step 1: Recording evaluation metadata")
        print("=" * 60)
        result = run_script(
            scripts_dir / "record_eval_metadata.py",
            [
                "--output", str(output_dir / "eval_metadata.json"),
                "--tasks", task_list,
            ] + (["--include-path", args.include_path] if args.include_path else []),
            cwd=project_root,
        )
        if result.returncode != 0:
            print("Warning: Metadata recording failed")

    # Step 2: Run baseline evaluation
    print("\n" + "=" * 60)
    print("Step 2: Evaluating baseline model")
    print("=" * 60)
    baseline_dir = output_dir / "baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    result = run_script(
        scripts_dir / "eval_portuguese.py",
        [
            "--model-path", args.baseline_model,
            "--output-dir", str(baseline_dir),
            "--tasks", task_list,
            "--device", args.device,
            "--batch-size", args.batch_size,
            "--num-fewshot", str(args.num_fewshot),
        ] + (["--limit", str(args.limit)] if args.limit else [])
        + (["--include-path", args.include_path] if args.include_path else []),
        cwd=project_root,
    )
    if result.returncode != 0:
        print("Baseline evaluation failed!")
        return result.returncode

    # Step 3: Run candidate evaluation
    print("\n" + "=" * 60)
    print("Step 3: Evaluating candidate model")
    print("=" * 60)
    candidate_dir = output_dir / "candidate"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    result = run_script(
        scripts_dir / "eval_portuguese.py",
        [
            "--model-path", args.candidate_model,
            "--output-dir", str(candidate_dir),
            "--tasks", task_list,
            "--device", args.device,
            "--batch-size", args.batch_size,
            "--num-fewshot", str(args.num_fewshot),
        ] + (["--limit", str(args.limit)] if args.limit else [])
        + (["--include-path", args.include_path] if args.include_path else []),
        cwd=project_root,
    )
    if result.returncode != 0:
        print("Candidate evaluation failed!")
        return result.returncode

    # Step 4: Compute bits-per-UTF-8-byte for both models
    if not args.skip_perplexity:
        print("\n" + "=" * 60)
        print("Step 4: Computing bits-per-UTF-8-byte metric")
        print("=" * 60)

        for name, model_path in [("baseline", args.baseline_model), ("candidate", args.candidate_model)]:
            perp_dir = output_dir / "perplexity" / name
            perp_dir.mkdir(parents=True, exist_ok=True)
            result = run_script(
                scripts_dir / "compute_bits_per_byte.py",
                [
                    "--model-path", model_path,
                    "--tokenizer-path", args.tokenizer_path,
                    "--data-path", args.corpus_path,
                    "--output", str(perp_dir / "bits_per_byte.json"),
                    "--device", args.device,
                ],
                cwd=project_root,
            )
            if result.returncode != 0:
                print(f"Warning: Perplexity computation failed for {name}")

    # Step 5: Compare models
    if not args.skip_comparison:
        print("\n" + "=" * 60)
        print("Step 5: Comparing models")
        print("=" * 60)
        comparison_dir = output_dir / "comparison"
        comparison_dir.mkdir(parents=True, exist_ok=True)
        result = run_script(
            scripts_dir / "compare_models.py",
            [
                "--baseline-model", args.baseline_model,
                "--candidate-model", args.candidate_model,
                "--baseline-name", "baseline",
                "--candidate-name", "candidate",
                "--output-dir", str(comparison_dir),
                "--tasks", task_list,
                "--device", args.device,
                "--batch-size", args.batch_size,
                "--num-fewshot", str(args.num_fewshot),
            ] + (["--limit", str(args.limit)] if args.limit else [])
            + (["--include-path", args.include_path] if args.include_path else []),
            cwd=project_root,
        )
        if result.returncode != 0:
            print("Warning: Model comparison failed")

    # Step 6: Generative evaluation (MT-Bench/LLM-judge style)
    if not args.skip_generative:
        print("\n" + "=" * 60)
        print("Step 6: Generative evaluation (Brazilian Portuguese)")
        print("=" * 60)
        generative_dir = output_dir / "generative"
        generative_dir.mkdir(parents=True, exist_ok=True)
        result = run_script(
            scripts_dir / "eval_generative_br.py",
            [
                "--baseline-model", args.baseline_model,
                "--candidate-model", args.candidate_model,
                "--tokenizer-path", args.tokenizer_path,
                "--output-dir", str(generative_dir),
                "--prompts", args.generative_prompts,
                "--rubric", args.generative_rubric,
                "--judge-model", args.judge_model,
                "--device", args.device,
                "--temperature", str(args.gen_temperature),
                "--top-p", str(args.gen_top_p),
                "--max-tokens", str(args.gen_max_tokens),
            ] + (["--limit", str(args.limit)] if args.limit else []),
            cwd=project_root,
        )
        if result.returncode != 0:
            print("Warning: Generative evaluation failed")

    # Step 7: Generate final summary report
    print("\n" + "=" * 60)
    print("Step 7: Generating final report")
    print("=" * 60)

    summary = {
        "evaluation": {
            "baseline_model": args.baseline_model,
            "candidate_model": args.candidate_model,
            "tokenizer": args.tokenizer_path,
            "corpus": args.corpus_path,
            "tasks": tasks,
            "device": args.device,
            "batch_size": args.batch_size,
            "num_fewshot": args.num_fewshot,
            "limit": args.limit,
            "generative_eval": not args.skip_generative,
            "generative_prompts": args.generative_prompts,
            "generative_rubric": args.generative_rubric,
            "judge_model": args.judge_model,
        },
        "outputs": {
            "metadata": str(output_dir / "eval_metadata.json"),
            "baseline_results": str(baseline_dir),
            "candidate_results": str(candidate_dir),
            "perplexity": str(output_dir / "perplexity"),
            "comparison": str(output_dir / "comparison" / "comparison_report.md"),
            "generative": str(output_dir / "generative" / "generative_eval_report.md") if not args.skip_generative else None,
        },
    }

    with open(output_dir / "evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 60}")
    print("EVALUATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"Results saved to: {output_dir}")
    print(f"Summary: {output_dir / 'evaluation_summary.json'}")
    print(f"Comparison report: {output_dir / 'comparison' / 'comparison_report.md'}")
    if not args.skip_generative:
        print(f"Generative report: {output_dir / 'generative' / 'generative_eval_report.md'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
