#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

DEFAULT_TOP_N = 25
SECTION_LABELS = (
    "SCENE_OVERVIEW",
    "VISUAL_DETAILS",
    "DIALOGUE",
    "OTHER_SOUNDS",
)
SECTION_LINE_PATTERN = re.compile(
    rf"^(?P<label>{'|'.join(re.escape(label) for label in SECTION_LABELS)})\s*:?\s*(?P<body>.*)$",
    re.IGNORECASE,
)
TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)*")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect one or more final JSONL caption manifests such as LTX exports "
            "and report the most common words and 2-word combinations."
        )
    )
    parser.add_argument(
        "manifest_paths",
        nargs="+",
        type=Path,
        help="One or more JSONL manifests containing a caption field.",
    )
    parser.add_argument(
        "--top-n",
        type=positive_int,
        default=DEFAULT_TOP_N,
        help=f"Number of top words and bigrams to print. Defaults to {DEFAULT_TOP_N}.",
    )
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    del parser
    for manifest_path in args.manifest_paths:
        if not manifest_path.exists() or not manifest_path.is_file():
            raise ValueError(f"{manifest_path} must exist and be a file")


def iter_manifest_rows(manifest_path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{manifest_path}:{line_number} contains invalid JSON: {exc.msg}") from exc

            if not isinstance(record, dict):
                raise ValueError(f"{manifest_path}:{line_number} must contain JSON objects")
            yield line_number, record


def normalize_text_block(text: str) -> tuple[str, ...]:
    return tuple(token for token in TOKEN_PATTERN.findall(text.lower()) if token != "none")


def normalize_caption_blocks(caption: str) -> tuple[tuple[str, ...], ...]:
    blocks: list[tuple[str, ...]] = []
    for raw_line in caption.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        label_match = SECTION_LINE_PATTERN.fullmatch(line)
        if label_match is not None:
            body = label_match.group("body").strip()
            if not body:
                continue
            line = body

        tokens = normalize_text_block(line)
        if tokens:
            blocks.append(tokens)
    return tuple(blocks)


def collect_statistics(manifest_paths: list[Path]) -> tuple[int, Counter[str], Counter[str]]:
    caption_rows = 0
    word_counts: Counter[str] = Counter()
    bigram_counts: Counter[str] = Counter()

    for manifest_path in manifest_paths:
        for line_number, record in iter_manifest_rows(manifest_path):
            caption = record.get("caption")
            if not isinstance(caption, str) or not caption.strip():
                raise ValueError(f"{manifest_path}:{line_number} is missing a non-empty caption")

            caption_rows += 1
            for tokens in normalize_caption_blocks(caption):
                word_counts.update(tokens)
                bigram_counts.update(
                    " ".join((tokens[index], tokens[index + 1]))
                    for index in range(len(tokens) - 1)
                )

    return caption_rows, word_counts, bigram_counts


def top_counts(counter: Counter[str], limit: int) -> list[tuple[str, int]]:
    return sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:limit]


def format_top_counts(title: str, counter: Counter[str], limit: int) -> list[str]:
    lines = [title]
    entries = top_counts(counter, limit)
    if not entries:
        lines.append("(none)")
        return lines

    for index, (text, count) in enumerate(entries, start=1):
        lines.append(f"{index}. {text} ({count})")
    return lines


def render_report(
    manifest_paths: list[Path],
    caption_rows: int,
    word_counts: Counter[str],
    bigram_counts: Counter[str],
    top_n: int,
) -> str:
    lines = [
        (
            f"manifests={len(manifest_paths)} caption_rows={caption_rows} "
            f"unique_words={len(word_counts)} unique_bigrams={len(bigram_counts)}"
        ),
        "",
        *format_top_counts(f"Top {top_n} words", word_counts, top_n),
        "",
        *format_top_counts(f"Top {top_n} bigrams", bigram_counts, top_n),
    ]
    return "\n".join(lines)


def run(args: argparse.Namespace) -> int:
    try:
        caption_rows, word_counts, bigram_counts = collect_statistics(args.manifest_paths)
        if not word_counts:
            raise ValueError("Dataset yielded zero usable tokens after normalization")
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(
        render_report(
            manifest_paths=args.manifest_paths,
            caption_rows=caption_rows,
            word_counts=word_counts,
            bigram_counts=bigram_counts,
            top_n=args.top_n,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_args(parser, args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
