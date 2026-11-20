#!/usr/bin/env python3
"""CLI for Portuguese corpus curation with extended tokenizer.

This script demonstrates the full data curation pipeline:
1. Assemble Portuguese corpus from multiple sources (with fixture for demo)
2. Filter, deduplicate, and split train/eval
3. Apply 50/50 -> 90/10 curriculum by token count
4. Tokenize with extended tokenizer
5. Record realized token counts and curriculum
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Add src to path for imports
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from data_curation import (  # noqa: E402
    CorpusProvenance,
    CurriculumSchedule,
    assemble_portuguese_corpus,
    default_portuguese_sources,
    filter_and_deduplicate,
    multilingual_sources,
    prepare_curriculum_data,
    record_token_counts,
    split_train_eval,
)

from cafezin_lm.tokenizer import Tokenizer  # noqa: E402
from cafezin_lm.tokenizer_extension import (  # noqa: E402
    extend_tokenizer,
    mine_candidate_tokens,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_fixture_corpus() -> tuple[list[str], list[str]]:
    """Create a small Portuguese/English fixture corpus for demonstration.

    Returns:
        Tuple of (portuguese_docs, multilingual_docs).
    """
    pt_docs = [
        "O café é uma bebida muito popular no Brasil. Os brasileiros consomem café "
        "diariamente, seja no café da manhã ou depois do almoço. Existem muitos tipos "
        "de café, como o café coado, o espresso e o café com leite.",

        "São Paulo é a maior cidade do Brasil e da América do Sul. A cidade é conhecida "
        "pela sua diversidade cultural, gastronomia rica e vida noturna agitada. "
        "Milhões de pessoas visitam São Paulo todos os anos.",

        "A língua portuguesa é falada por mais de 250 milhões de pessoas no mundo. "
        "O Brasil é o maior país lusófono, com mais de 200 milhões de falantes. "
        "O português do Brasil tem características únicas de pronúncia e vocabulário.",

        "O Carnaval do Rio de Janeiro é uma das festas mais famosas do mundo. "
        "Durante quatro dias, as ruas se enchem de blocos, desfiles e muita música. "
        "Turistas de todo o planeta vêm assistir aos desfiles das escolas de samba.",

        "A Amazônia brasileira é a maior floresta tropical do planeta. Ela abriga "
        "uma biodiversidade incrível, com milhões de espécies de plantas e animais. "
        "A preservação da Amazônia é essencial para o equilíbrio climático global.",

        "O futebol é a maior paixão nacional no Brasil. A seleção brasileira já ganhou "
        "a Copa do Mundo cinco vezes. Pelé, Ronaldo e Neymar são alguns dos maiores "
        "jogadores da história do futebol brasileiro.",

        "A culinária brasileira é muito diversa, refletindo a mistura de culturas "
        "indígena, africana e europeia. Pratos como feijoada, acarajé, moqueca e "
        "pão de queijo são conhecidos internacionalmente.",

        "O português do Brasil difere do português de Portugal em vários aspectos: "
        "pronúncia, vocabulário, gramática e até ortografia. Apesar das diferenças, "
        "falantes das duas variantes se entendem perfeitamente.",

        "A música brasileira é rica e variada: samba, bossa nova, MPB, forró, "
        "funk carioca, sertanejo e muitos outros gêneros. Artistas como Tom Jobim, "
        "Chico Buarque e Gilberto Gil são ícones da música popular brasileira.",

        "O Brasil tem um litoral de mais de 7.000 quilômetros, com praias famosas "
        "como Copacabana, Ipanema, Jericoacoara e Fernando de Noronha. O turismo "
        "de praia é uma das principais atividades econômicas do país.",
    ]

    multi_docs = [
        "The quick brown fox jumps over the lazy dog. This pangram contains every "
        "letter of the English alphabet. It is often used for font testing and "
        "keyboard practice.",

        "Machine learning is a subset of artificial intelligence that enables "
        "systems to learn from data without explicit programming. Deep learning "
        "uses neural networks with multiple layers to model complex patterns.",

        "Le français est une langue romane parlée par des millions de personnes "
        "dans le monde. C'est la langue officielle de la France, du Canada, de la "
        "Belgique, de la Suisse et de nombreux pays africains.",

        "El español es la segunda lengua más hablada del mundo por número de "
        "hablantes nativos. Se habla en España, México, Argentina, Colombia y "
        "muchos otros países de América Latina.",

        "Deutsch ist eine germanische Sprache, die hauptsächlich in Deutschland, "
        "Österreich und der Schweiz gesprochen wird. Sie hat vier Fälle: Nominativ, "
        "Genitiv, Dativ und Akkusativ.",

        "中文是世界上使用人数最多的语言之一。汉字是表意文字，每个汉字代表一个"
        "音节和意义。普通话是中国的官方语言。",

        "日本語は日本で話される言語です。ひらがな、カタカナ、漢字の3種類の文字"
        "を使います。日本語の文法は主語・目的語・動詞の順序が特徴的です。",

        "한국어는 한반도에서 사용되는 언어입니다. 한글은 세종대왕이 창제한 과학적인"
        "문자 체계입니다. 한국어는 교착어적 특성을 가지고 있습니다.",

        "Русский язык — это восточнославянский язык, официальный в России, "
        "Беларуси, Казахстане и Киргизии. Использует кириллицу.",

        "العربية هي لغة سامية يتحدث بها مئات الملايين من الناس في العالم العربي "
        "والإسلامي. تكتب من اليمين إلى اليسار بخط عربي جميل.",
    ]

    return pt_docs, multi_docs


def extend_tokenizer_for_portuguese(
    base_tokenizer_path: str,
    pt_corpus: list[str],
    output_path: str,
    min_frequency: int = 2,
    max_candidates: int = 100,
) -> Tokenizer:
    """Extend tokenizer with Portuguese-specific tokens.

    Args:
        base_tokenizer_path: Path to base tokenizer.json.
        pt_corpus: Portuguese corpus for mining candidates.
        output_path: Output path for extended tokenizer.
        min_frequency: Minimum frequency for candidate tokens.
        max_candidates: Maximum candidates to add.

    Returns:
        Extended tokenizer.
    """
    logger.info(f"Loading base tokenizer from {base_tokenizer_path}")
    base_tok = Tokenizer(base_tokenizer_path)

    logger.info(f"Mining candidate tokens from {len(pt_corpus)} Portuguese documents")
    candidates = mine_candidate_tokens(
        base_tok,
        pt_corpus,
        min_frequency=min_frequency,
        max_candidates=max_candidates,
    )
    logger.info(f"Found {len(candidates)} candidate tokens")

    if not candidates:
        logger.warning("No candidates found; returning base tokenizer")
        return base_tok

    # Show top candidates
    logger.info("Top 20 candidates:")
    for token, gain in candidates[:20]:
        logger.info(f"  {token!r}: gain={gain}")

    new_tokens = [c[0] for c in candidates]
    logger.info(f"Extending tokenizer with {len(new_tokens)} new tokens")
    extended_tok = extend_tokenizer(base_tok, new_tokens, output_path)

    logger.info(f"Extended tokenizer saved to {output_path}")
    logger.info(f"  Original vocab size: {base_tok.vocab_size}")
    logger.info(f"  Extended vocab size: {extended_tok.vocab_size}")
    logger.info(f"  New tokens added: {extended_tok.vocab_size - base_tok.vocab_size}")

    return extended_tok


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Portuguese corpus curation with extended tokenizer"
    )
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="Path to base tokenizer.json",
    )
    parser.add_argument(
        "--output-dir",
        default="data/portuguese_corpus",
        help="Output directory for curated corpus",
    )
    parser.add_argument(
        "--extended-tokenizer",
        default="tokenizer_extended.json",
        help="Output path for extended tokenizer",
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=2,
        help="Minimum frequency for tokenizer extension candidates",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=100,
        help="Maximum candidates for tokenizer extension",
    )
    parser.add_argument(
        "--eval-fraction",
        type=float,
        default=0.01,
        help="Fraction of documents for eval split",
    )
    parser.add_argument(
        "--transition-tokens",
        type=int,
        default=100_000_000_000,
        help="Tokens over which to anneal curriculum (default 100B)",
    )
    parser.add_argument(
        "--total-training-tokens",
        type=int,
        default=200_000_000_000,
        help="Total training tokens (default 200B)",
    )
    parser.add_argument(
        "--use-fixture",
        action="store_true",
        help="Use built-in fixture corpus instead of loading from HF datasets",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Maximum documents per source (for testing)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without writing files",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Load or create corpus
    if args.use_fixture:
        logger.info("Using fixture corpus for demonstration")
        pt_docs, multi_docs = create_fixture_corpus()
        logger.info(f"Fixture: {len(pt_docs)} PT docs, {len(multi_docs)} multilingual docs")
        provenance = CorpusProvenance(corpus_name="portuguese-pretrain-fixture")
    else:
        logger.info("Loading Portuguese corpus from HF datasets...")
        # Assemble Portuguese corpus
        pt_sources = default_portuguese_sources(max_docs=args.max_docs)
        provenance = CorpusProvenance(corpus_name="portuguese-pretrain")
        pt_iterator, provenance = assemble_portuguese_corpus(pt_sources, provenance)
        pt_docs = list(pt_iterator)
        logger.info(f"Loaded {len(pt_docs)} Portuguese documents")

        # Assemble multilingual corpus
        multi_sources = multilingual_sources(max_docs=args.max_docs)
        multi_iterator, _ = assemble_portuguese_corpus(multi_sources, provenance)
        multi_docs = list(multi_iterator)
        logger.info(f"Loaded {len(multi_docs)} multilingual documents")

        if not args.dry_run:
            provenance.save(output_dir / "provenance.json")

    if not pt_docs or not multi_docs:
        logger.error("Failed to load corpus documents")
        return 1

    # Filter and deduplicate
    logger.info("Filtering and deduplicating Portuguese corpus...")
    pt_docs, provenance = filter_and_deduplicate(iter(pt_docs), provenance)
    logger.info(f"PT after filter/dedup: {len(pt_docs)} docs")

    logger.info("Filtering and deduplicating multilingual corpus...")
    multi_docs, provenance = filter_and_deduplicate(iter(multi_docs), provenance)
    logger.info(f"Multilingual after filter/dedup: {len(multi_docs)} docs")

    # Split train/eval
    logger.info(f"Splitting train/eval (eval_fraction={args.eval_fraction})...")
    pt_train, pt_eval, provenance = split_train_eval(pt_docs, provenance, args.eval_fraction)
    multi_train, multi_eval, provenance = split_train_eval(multi_docs, provenance, args.eval_fraction)

    logger.info(f"PT train: {len(pt_train)}, PT eval: {len(pt_eval)}")
    logger.info(f"Multi train: {len(multi_train)}, Multi eval: {len(multi_eval)}")

    # Extend tokenizer
    logger.info("Extending tokenizer for Portuguese...")
    extended_tokenizer = extend_tokenizer_for_portuguese(
        args.tokenizer,
        pt_train,
        args.extended_tokenizer,
        min_frequency=args.min_frequency,
        max_candidates=args.max_candidates,
    )

    # Create curriculum schedule
    schedule = CurriculumSchedule(
        initial_pt_ratio=0.5,
        final_pt_ratio=0.9,
        transition_tokens=args.transition_tokens,
        total_training_tokens=args.total_training_tokens,
    )
    logger.info(f"Curriculum schedule: {schedule.to_dict()}")

    # Record token counts
    logger.info("Recording token counts...")
    pt_report, multi_report, combined_report = record_token_counts(
        pt_train, multi_train, extended_tokenizer, schedule, tokens_seen=0
    )

    logger.info(f"PT train tokens: {pt_report.total_tokens:,}")
    logger.info(f"Multilingual train tokens: {multi_report.total_tokens:,}")
    logger.info(f"Combined train tokens: {combined_report.total_tokens:,}")
    logger.info(f"PT ratio: {combined_report.pt_ratio:.3f}")
    logger.info(f"Chars/token: {combined_report.chars_per_token:.3f}")

    # Prepare curriculum data package
    if not args.dry_run:
        logger.info(f"Preparing curriculum data package in {output_dir}...")
        result = prepare_curriculum_data(
            pt_train, pt_eval, multi_train, multi_eval,
            extended_tokenizer, schedule, output_dir
        )
        logger.info("Curriculum data package created:")
        for k, v in result.items():
            if isinstance(v, dict):
                logger.info(f"  {k}: {json.dumps(v, indent=4)}")
            else:
                logger.info(f"  {k}: {v}")

        # Save tokenization reports
        pt_report.save(output_dir / "tokenization_reports" / "pt_train_report.json")
        multi_report.save(output_dir / "tokenization_reports" / "multi_train_report.json")
        combined_report.save(output_dir / "tokenization_reports" / "combined_train_report.json")

        # Save provenance
        provenance.tokenizer_info = {
            "base_tokenizer": args.tokenizer,
            "extended_tokenizer": args.extended_tokenizer,
            "vocab_size": extended_tokenizer.vocab_size,
        }
        provenance.curriculum_schedule = schedule.to_dict()
        provenance.save(output_dir / "provenance.json")

        logger.info("Done!")
    else:
        logger.info("Dry run complete - no files written")

    return 0


if __name__ == "__main__":
    sys.exit(main())
