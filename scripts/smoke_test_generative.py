#!/usr/bin/env python
"""Smoke test for generative evaluation using tiny model fixtures.

This script creates minimal model checkpoints and runs the generative
evaluation pipeline to verify it works end-to-end.
"""

from __future__ import annotations

import json
import sys
import tempfile
import torch
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cafezin_lm import (
    Tokenizer, build_model, tiny_config, extend_tokenizer,
    grow_embeddings, mean_decomposition_init
)
from tokenizers import Tokenizer as HFTokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

# Import the evaluation module directly
from scripts.eval_generative_br import run_generative_evaluation, GenerationConfig


def create_test_tokenizer(tmpdir: Path) -> Path:
    """Create a minimal test tokenizer with Portuguese vocabulary."""
    tok = HFTokenizer(BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(vocab_size=500, special_tokens=["[UNK]", "[EOS]", "[PAD]"])
    
    # Portuguese-heavy training data
    corpus = [
        "olá mundo como vai você " * 20,
        "bom dia boa tarde boa noite " * 20,
        "brasil brasileiro são paulo rio de janeiro " * 15,
        "carnaval samba futebol praia sol " * 15,
        "café pão na chapa manteiga pingado " * 15,
        "amazônia cerrado pantanal caatinga " * 15,
        "sus saúde educação infraestrutura " * 15,
        "jeitinho brasileiro cultura região " * 15,
    ]
    tok.train_from_iterator(corpus, trainer=trainer)
    
    tok_path = tmpdir / "tokenizer.json"
    tok.save(str(tok_path))
    return tok_path


def create_test_model(tokenizer_path: Path, model_path: Path, seed: int = 42) -> None:
    """Create a tiny test model checkpoint."""
    from cafezin_lm import Tokenizer
    
    tokenizer = Tokenizer(str(tokenizer_path))
    cfg = tiny_config(vocab_size=tokenizer.vocab_size)
    model = build_model(cfg, seed=seed)
    
    torch.save(model.state_dict(), model_path)


def create_test_candidate(tokenizer_path: Path, model_path: Path, seed: int = 123) -> None:
    """Create a slightly different candidate model (different seed)."""
    from cafezin_lm import Tokenizer
    
    tokenizer = Tokenizer(str(tokenizer_path))
    cfg = tiny_config(vocab_size=tokenizer.vocab_size)
    model = build_model(cfg, seed=seed)
    
    torch.save(model.state_dict(), model_path)


def run_smoke_test() -> bool:
    """Run the complete smoke test."""
    print("=" * 60)
    print("GENERATIVE EVALUATION SMOKE TEST")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        
        # Create test artifacts
        print("\n1. Creating test tokenizer...")
        tokenizer_path = create_test_tokenizer(tmpdir)
        print(f"   Tokenizer saved to {tokenizer_path}")
        
        print("\n2. Creating baseline model...")
        baseline_path = tmpdir / "baseline.pt"
        create_test_model(tokenizer_path, baseline_path, seed=42)
        print(f"   Baseline model saved to {baseline_path}")
        
        print("\n3. Creating candidate model...")
        candidate_path = tmpdir / "candidate.pt"
        create_test_candidate(tokenizer_path, candidate_path, seed=123)
        print(f"   Candidate model saved to {candidate_path}")
        
        print("\n4. Running generative evaluation (limited to 3 prompts)...")
        output_dir = tmpdir / "generative_results"
        
        prompts_path = Path(__file__).parent.parent / "eval" / "generative" / "prompts_v1.json"
        rubric_path = Path(__file__).parent.parent / "eval" / "generative" / "rubric_v1.json"
        
        generation_config = GenerationConfig(
            temperature=0.7,
            top_p=0.9,
            max_tokens=128  # Shorter for smoke test
        )
        
        try:
            results = run_generative_evaluation(
                baseline_model_path=str(baseline_path),
                candidate_model_path=str(candidate_path),
                tokenizer_path=str(tokenizer_path),
                prompts_path=prompts_path,
                rubric_path=rubric_path,
                output_dir=output_dir,
                judge_model="heuristic",
                generation_config=generation_config,
                device="cpu",  # Use CPU for smoke test
                config_name="tiny",
                limit=3  # Only first 3 prompts
            )
            
            print("\n5. Checking results...")
            results_file = output_dir / "generative_eval_results.json"
            if results_file.exists():
                with open(results_file) as f:
                    data = json.load(f)
                
                print(f"   Results saved successfully")
                print(f"   Examples evaluated: {data['metadata']['num_prompts']}")
                print(f"   Baseline mean: {data['aggregate']['overall']['baseline_mean']:.2f}")
                print(f"   Candidate mean: {data['aggregate']['overall']['candidate_mean']:.2f}")
                print(f"   Difference: {data['aggregate']['overall']['difference']:+.2f}")
                
                # Check report
                report_file = output_dir / "generative_eval_report.md"
                if report_file.exists():
                    print(f"   Report generated: {report_file}")
                
                print("\n" + "=" * 60)
                print("SMOKE TEST PASSED")
                print("=" * 60)
                return True
            else:
                print("   ERROR: Results file not created")
                return False
                
        except Exception as e:
            print(f"\n   ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)