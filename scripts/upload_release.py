#!/usr/bin/env python
"""Upload Cafezin LM release to Hugging Face Hub.

This script:
1. Validates an exported checkpoint directory (safetensors, configs, tokenizer assets)
2. Generates/validates a complete model card
3. Confirms provenance, redistribution terms, and required notices
4. Reloads the bundle in a clean environment and smoke-tests generation
5. Uploads to HF Hub via huggingface_hub (explicit, not CI)
6. Records the repository and revision

Usage:
    python scripts/upload_release.py \
        --checkpoint-dir /path/to/exported/checkpoint \
        --model-card /path/to/MODEL_CARD.md \
        --repo-id cafezin-lm/cafezin-lm-br \
        --token $HF_TOKEN \
        [--revision main] \
        [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cafezin_lm.checkpoint import (
    ASSET_FILES,
    checkpoint_assets,
    import_config,
    load_hf_checkpoint,
    save_hf_checkpoint,
)
from cafezin_lm.config import ModelConfig
from cafezin_lm.model import build_model
from cafezin_lm.tokenizer import Tokenizer


REQUIRED_FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)

REQUIRED_NOTICES = ("LICENSE", "NOTICE")


def validate_checkpoint_dir(checkpoint_dir: Path) -> dict[str, Any]:
    """Validate that the checkpoint directory has all required files."""
    missing = []
    for f in REQUIRED_FILES:
        if not (checkpoint_dir / f).exists():
            missing.append(f)

    # Check for at least one safetensors shard
    shards = list(checkpoint_dir.glob("model-*.safetensors"))
    if not shards:
        missing.append("model-*.safetensors (at least one shard)")

    if missing:
        raise FileNotFoundError(
            f"Checkpoint directory {checkpoint_dir} missing required files: {missing}"
        )

    # Load and validate config
    config = import_config(checkpoint_dir)
    config_path = checkpoint_dir / "config.json"
    with config_path.open() as f:
        config_json = json.load(f)

    # Load generation config
    gen_path = checkpoint_dir / "generation_config.json"
    with gen_path.open() as f:
        generation_config = json.load(f)

    # Load tokenizer
    tokenizer = Tokenizer(checkpoint_dir / "tokenizer.json")

    # Load assets
    assets = checkpoint_assets(checkpoint_dir)

    print(f"✓ Checkpoint validation passed")
    print(f"  Config: {config_path}")
    print(f"  Shards: {len(shards)}")
    print(f"  Tokenizer vocab size: {tokenizer.vocab_size}")
    print(f"  Assets found: {list(assets.keys())}")

    return {
        "config": config,
        "config_json": config_json,
        "generation_config": generation_config,
        "tokenizer": tokenizer,
        "assets": assets,
        "shards": shards,
    }


def validate_model_card(model_card_path: Path) -> str:
    """Validate that model card exists and has required sections."""
    if not model_card_path.exists():
        raise FileNotFoundError(f"Model card not found: {model_card_path}")

    content = model_card_path.read_text(encoding="utf-8")

    required_sections = [
        "Model Summary",
        "Parent Checkpoint",
        "Data Provenance",
        "Tokenizer Changes",
        "Training Settings",
        "Evaluation Results",
        "Limitations",
        "License",
    ]

    missing = [s for s in required_sections if s not in content]
    if missing:
        raise ValueError(f"Model card missing required sections: {missing}")

    # Check for license and notice references
    if "Apache-2.0" not in content:
        raise ValueError("Model card must reference Apache-2.0 license")
    if "NOTICE" not in content:
        raise ValueError("Model card must reference NOTICE file")

    print(f"✓ Model card validation passed: {model_card_path}")
    return content


def validate_notices(project_root: Path) -> dict[str, str]:
    """Validate LICENSE and NOTICE files exist and return their content."""
    notices = {}
    for notice in REQUIRED_NOTICES:
        path = project_root / notice
        if not path.exists():
            raise FileNotFoundError(f"Required notice file not found: {path}")
        notices[notice] = path.read_text(encoding="utf-8")
        print(f"✓ {notice} found")
    return notices


def confirm_provenance_and_terms(
    checkpoint_info: dict[str, Any],
    model_card_content: str,
    notices: dict[str, str],
) -> bool:
    """Interactive confirmation of provenance, redistribution terms, and notices."""
    print("\n" + "=" * 70)
    print("PROVENANCE AND REDISTRIBUTION CONFIRMATION")
    print("=" * 70)
    print()
    print("Parent checkpoint: maritaca-ai/S5-nano @ fe3a3581c33cb22917770ef1987b7c6e30929425")
    print(f"License: Apache-2.0 (inherited from parent)")
    print(f"Model ID: cafezin-lm/cafezin-lm-br")
    print()
    print("This release bundles:")
    print(f"  - {len(checkpoint_info['shards'])} SafeTensors shards")
    print(f"  - Tokenizer (vocab_size={checkpoint_info['tokenizer'].vocab_size})")
    print(f"  - Model config (config.json)")
    print(f"  - Generation config (generation_config.json)")
    print(f"  - Model card (MODEL_CARD.md)")
    print(f"  - LICENSE and NOTICE files")
    print()
    print("Required notices included:")
    for notice, content in notices.items():
        print(f"  - {notice}: {len(content)} chars")
    print()
    print("By confirming, you assert:")
    print("  1. You have the right to redistribute the parent checkpoint weights")
    print("     under Apache-2.0 (S5-nano is Apache-2.0)")
    print("  2. Training data provenance is documented in the model card")
    print("  3. All required notices (LICENSE, NOTICE) are included")
    print("  4. The model card accurately describes the model")
    print()

    # In non-interactive mode (--yes), auto-confirm
    if os.environ.get("CAFEZIN_UPLOAD_YES") == "1":
        print("Auto-confirming via CAFEZIN_UPLOAD_YES=1")
        return True

    try:
        response = input("Confirm upload? [y/N]: ").strip().lower()
        return response == "y" or response == "yes"
    except (EOFError, KeyboardInterrupt):
        print("\nAborted.")
        return False


def smoke_test_reload(
    checkpoint_dir: Path,
    prompt: str = "Olá, mundo!",
    max_new: int = 32,
) -> str:
    """Reload the checkpoint in a clean environment and test generation."""
    print("\n" + "=" * 70)
    print("SMOKE TEST: Clean reload and generation")
    print("=" * 70)

    with tempfile.TemporaryDirectory(prefix="cafezin-smoke-") as tmp:
        # Copy the entire checkpoint to a clean temp directory
        clean_dir = Path(tmp) / "clean_checkpoint"
        shutil.copytree(checkpoint_dir, clean_dir)

        print(f"  Copied to clean env: {clean_dir}")

        # Load tokenizer from clean dir
        tokenizer = Tokenizer(clean_dir / "tokenizer.json")
        print(f"  Tokenizer loaded: vocab_size={tokenizer.vocab_size}")

        # Load config from clean dir
        config = import_config(clean_dir)
        print(f"  Config loaded: {config.text.vocab_size} vocab, {config.text.num_hidden_layers} layers")

        # Build model on meta device
        with torch.device("meta"):
            model = build_model(config)
        print(f"  Model built on meta device")

        # Load weights (materializes on CPU)
        load_hf_checkpoint(model, clean_dir, assign=True)
        print(f"  Weights loaded")

        # Recreate RoPE buffers (not in checkpoint)
        from cafezin_lm.model import TextRotaryEmbedding, VisionRotaryEmbedding
        model.model.language_model.rotary_emb = TextRotaryEmbedding(config.text)
        model.model.visual.rotary_pos_emb = VisionRotaryEmbedding(config.vision)

        # Move to CPU and eval
        model = model.to("cpu")
        model.eval()
        print(f"  Model on CPU, eval mode")

        # Generate
        input_ids = torch.tensor([tokenizer.encode(prompt, eos=False)], dtype=torch.long)
        eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else config.text.eos_token_id

        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=max_new,
                temperature=0.0,
                eos_token_id=eos,
                vocab_limit=tokenizer.max_token_id + 1,
            )

        generated = tokenizer.decode(out[0].tolist())
        print(f"  Generation successful!")
        print(f"  Prompt: {prompt!r}")
        print(f"  Output: {generated!r}")

        return generated


def upload_to_hf(
    checkpoint_dir: Path,
    model_card_content: str,
    notices: dict[str, str],
    repo_id: str,
    revision: str,
    token: str | None,
    dry_run: bool = False,
) -> str:
    """Upload the release to Hugging Face Hub."""
    print("\n" + "=" * 70)
    print(f"UPLOAD TO HF HUB: {repo_id} @ {revision}")
    print("=" * 70)

    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        raise RuntimeError("huggingface_hub not installed. Install with: pip install huggingface_hub")

    api = HfApi(token=token)

    # Create repo if it doesn't exist
    if not dry_run:
        try:
            create_repo(repo_id, token=token, exist_ok=True, private=False)
            print(f"  Repository ensured: {repo_id}")
        except Exception as e:
            print(f"  Warning: Could not create repo (may exist): {e}")

    # Prepare files to upload
    files_to_upload = []

    # All files in checkpoint directory
    for f in checkpoint_dir.rglob("*"):
        if f.is_file():
            rel_path = f.relative_to(checkpoint_dir)
            files_to_upload.append((str(rel_path), f))

    # Model card
    files_to_upload.append(("README.md", model_card_content))

    # Notices
    for notice, content in notices.items():
        files_to_upload.append((notice, content))

    print(f"  Files to upload: {len(files_to_upload)}")
    for rel_path, _ in files_to_upload:
        print(f"    - {rel_path}")

    if dry_run:
        print("  DRY RUN: Skipping actual upload")
        return "dry-run-revision"

    # Upload all files
    commit_info = api.upload_folder(
        folder_path=str(checkpoint_dir),
        repo_id=repo_id,
        revision=revision,
        token=token,
        commit_message=f"Release Cafezin LM BR model",
    )

    # Upload model card and notices separately (they're not in the checkpoint dir)
    api.upload_file(
        path_or_fileobj=model_card_content.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        revision=revision,
        token=token,
        commit_message="Add model card",
    )

    for notice, content in notices.items():
        api.upload_file(
            path_or_fileobj=content.encode("utf-8"),
            path_in_repo=notice,
            repo_id=repo_id,
            revision=revision,
            token=token,
            commit_message=f"Add {notice}",
        )

    revision_hash = commit_info.oid
    print(f"  ✓ Upload complete!")
    print(f"  Revision: {revision_hash}")
    print(f"  URL: https://huggingface.co/{repo_id}/tree/{revision_hash}")

    return revision_hash


def verify_upload(repo_id: str, revision: str, token: str | None) -> dict[str, Any]:
    """Verify the uploaded files are accessible."""
    print("\n" + "=" * 70)
    print(f"VERIFYING UPLOAD: {repo_id} @ {revision}")
    print("=" * 70)

    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise RuntimeError("huggingface_hub not installed")

    api = HfApi(token=token)

    # List files in the repo at the revision
    files = api.list_repo_files(repo_id=repo_id, revision=revision, token=token)

    print(f"  Files in repo ({len(files)}):")
    for f in sorted(files):
        print(f"    - {f}")

    # Check required files
    required_in_repo = {
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "README.md",
        "LICENSE",
        "NOTICE",
    }

    missing = required_in_repo - set(files)
    if missing:
        print(f"  ⚠ Missing required files in repo: {missing}")
    else:
        print(f"  ✓ All required files present")

    # Check for safetensors shards
    shards = [f for f in files if f.startswith("model-") and f.endswith(".safetensors")]
    print(f"  Safetensors shards: {len(shards)}")

    return {
        "files": files,
        "shards": shards,
        "missing_required": list(missing),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload Cafezin LM release to HF Hub")
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        help="Path to exported checkpoint directory (output of cafezin-lm export)",
    )
    parser.add_argument(
        "--model-card",
        required=True,
        help="Path to MODEL_CARD.md",
    )
    parser.add_argument(
        "--repo-id",
        default="cafezin-lm/cafezin-lm-br",
        help="HF Hub repository ID (default: cafezin-lm/cafezin-lm-br)",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Git revision/branch to upload to (default: main)",
    )
    parser.add_argument(
        "--token",
        help="HF Hub token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate everything but don't upload",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip interactive confirmation (set CAFEZIN_UPLOAD_YES=1)",
    )
    parser.add_argument(
        "--smoke-test-prompt",
        default="Olá, mundo!",
        help="Prompt for smoke test generation",
    )
    parser.add_argument(
        "--smoke-test-max-new",
        type=int,
        default=32,
        help="Max new tokens for smoke test",
    )

    args = parser.parse_args()

    if args.yes:
        os.environ["CAFEZIN_UPLOAD_YES"] = "1"

    token = args.token or os.environ.get("HF_TOKEN")
    if not token and not args.dry_run:
        print("Error: HF token required (--token or HF_TOKEN env var)")
        return 1

    project_root = Path(__file__).parent.parent
    checkpoint_dir = Path(args.checkpoint_dir)
    model_card_path = Path(args.model_card)

    if not checkpoint_dir.exists():
        print(f"Error: Checkpoint directory not found: {checkpoint_dir}")
        return 1

    if not model_card_path.exists():
        print(f"Error: Model card not found: {model_card_path}")
        return 1

    print("=" * 70)
    print("CAFEZIN LM RELEASE UPLOAD")
    print("=" * 70)
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"Model card: {model_card_path}")
    print(f"Repo: {args.repo_id} @ {args.revision}")
    print(f"Dry run: {args.dry_run}")

    try:
        # Step 1: Validate checkpoint
        checkpoint_info = validate_checkpoint_dir(checkpoint_dir)

        # Step 2: Validate model card
        model_card_content = validate_model_card(model_card_path)

        # Step 3: Validate notices
        notices = validate_notices(project_root)

        # Step 4: Confirm provenance and terms
        if not confirm_provenance_and_terms(checkpoint_info, model_card_content, notices):
            print("Upload cancelled.")
            return 1

        # Step 5: Smoke test reload
        smoke_test_reload(
            checkpoint_dir,
            prompt=args.smoke_test_prompt,
            max_new=args.smoke_test_max_new,
        )

        # Step 6: Upload to HF Hub
        revision_hash = upload_to_hf(
            checkpoint_dir=checkpoint_dir,
            model_card_content=model_card_content,
            notices=notices,
            repo_id=args.repo_id,
            revision=args.revision,
            token=token,
            dry_run=args.dry_run,
        )

        # Step 7: Verify upload
        if not args.dry_run:
            verify_upload(args.repo_id, revision_hash, token)

        print("\n" + "=" * 70)
        print("RELEASE UPLOAD COMPLETE")
        print("=" * 70)
        print(f"Repository: {args.repo_id}")
        print(f"Revision: {revision_hash}")
        print(f"URL: https://huggingface.co/{args.repo_id}/tree/{revision_hash}")

        # Record the release info
        release_info = {
            "repo_id": args.repo_id,
            "revision": revision_hash,
            "checkpoint_dir": str(checkpoint_dir),
            "model_card": str(model_card_path),
            "uploaded_files": len(list(checkpoint_dir.rglob("*"))) + len(notices) + 1,  # +1 for README
        }

        # Save release record
        record_path = project_root / "release_record.json"
        record_path.write_text(json.dumps(release_info, indent=2), encoding="utf-8")
        print(f"Release record saved to: {record_path}")

        return 0

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())