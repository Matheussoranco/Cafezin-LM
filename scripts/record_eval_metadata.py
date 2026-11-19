#!/usr/bin/env python
"""Record evaluation metadata: versions, splits, contamination checks.

This script documents the exact configuration used for evaluation to ensure
reproducibility as required by the acceptance criteria.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def get_git_info() -> dict[str, str]:
    """Get git commit and branch info."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True
        ).stdout.strip()
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, check=True
        ).stdout.strip()
        return {"commit": commit, "branch": branch}
    except Exception:
        return {"commit": "unknown", "branch": "unknown"}


def get_package_versions(packages: list[str]) -> dict[str, str]:
    """Get installed package versions."""
    versions = {}
    for pkg in packages:
        try:
            result = subprocess.run(
                ["pip", "show", pkg],
                capture_output=True, text=True, check=True
            )
            for line in result.stdout.split("\n"):
                if line.startswith("Version:"):
                    versions[pkg] = line.split(":", 1)[1].strip()
                    break
            else:
                versions[pkg] = "not found"
        except Exception:
            versions[pkg] = "error"
    return versions


def get_task_metadata(task_names: list[str], include_path: str | None = None) -> dict[str, Any]:
    """Get metadata for each task (splits, dataset versions, etc.)."""
    metadata = {}

    # Use lm_eval to get task info
    for task in task_names:
        try:
            cmd = ["lm_eval", "--tasks", task, "--show_task_metadata"]
            if include_path:
                cmd.extend(["--include_path", include_path])

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                # Parse the metadata output
                metadata[task] = {"raw_output": result.stdout}
            else:
                metadata[task] = {"error": result.stderr}
        except Exception as e:
            metadata[task] = {"error": str(e)}

    return metadata


def get_contamination_info(task_names: list[str], include_path: str | None = None) -> dict[str, Any]:
    """Get contamination check information for tasks."""
    contamination = {}

    for task in task_names:
        try:
            cmd = ["lm_eval", "--tasks", task, "--show_contamination"]
            if include_path:
                cmd.extend(["--include_path", include_path])

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                contamination[task] = {"raw_output": result.stdout}
            else:
                contamination[task] = {"error": result.stderr}
        except Exception as e:
            contamination[task] = {"error": str(e)}

    return contamination


def main() -> int:
    parser = argparse.ArgumentParser(description="Record evaluation metadata")
    parser.add_argument("--output", default="eval_metadata.json", help="Output JSON file")
    parser.add_argument("--tasks", help="Comma-separated task names")
    parser.add_argument("--include-path", help="Additional task include path")
    parser.add_argument("--model-path", help="Model path (for model metadata)")

    args = parser.parse_args()

    # Determine tasks
    if args.tasks:
        tasks = [t.strip() for t in args.tasks.split(",")]
    else:
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

    metadata = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
        },
        "git": get_git_info(),
        "packages": get_package_versions([
            "lm-eval",
            "transformers",
            "torch",
            "accelerate",
            "datasets",
            "tokenizers",
            "evaluate",
        ]),
        "tasks": {},
    }

    if args.model_path:
        metadata["model_path"] = args.model_path

    # Get task metadata
    print("Fetching task metadata...")
    task_metadata = get_task_metadata(tasks, args.include_path)
    for task, meta in task_metadata.items():
        metadata["tasks"][task] = meta

    # Get contamination info
    print("Fetching contamination info...")
    contamination = get_contamination_info(tasks, args.include_path)
    for task, contam in contamination.items():
        if task in metadata["tasks"]:
            metadata["tasks"][task]["contamination"] = contam
        else:
            metadata["tasks"][task] = {"contamination": contam}

    # Save metadata
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata saved to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
