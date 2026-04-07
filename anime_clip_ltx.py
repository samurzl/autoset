#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, TextIO

VIDEO_EXTENSIONS = {".avi", ".mkv", ".mov", ".mp4", ".webm"}
DEFAULT_CAPTIONS_FILENAME = "captions.jsonl"
DEFAULT_OUTPUT_FILENAME = "ltx.jsonl"
TRAIN_SPLIT_NAME = "train"
VALIDATION_SPLIT_NAME = "val"
SPLIT_NAMES = (TRAIN_SPLIT_NAME, VALIDATION_SPLIT_NAME)


@dataclass(frozen=True)
class ExportRow:
    media_path: str
    caption: str


@dataclass(frozen=True)
class SplitExport:
    split_name: str
    output_file: Path
    rows: tuple[ExportRow, ...]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert subset train/val clip directories plus caption manifests into "
            "LTX-ready JSONL metadata files."
        )
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Subset root containing train/ and val/ directories.",
    )
    parser.add_argument(
        "--captions-filename",
        default=DEFAULT_CAPTIONS_FILENAME,
        help=(
            "Per-split caption manifest filename. "
            f"Defaults to {DEFAULT_CAPTIONS_FILENAME}."
        ),
    )
    parser.add_argument(
        "--output-filename",
        default=DEFAULT_OUTPUT_FILENAME,
        help=f"Per-split LTX manifest filename. Defaults to {DEFAULT_OUTPUT_FILENAME}.",
    )
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not args.input_dir.exists() or not args.input_dir.is_dir():
        parser.error("--input-dir must exist and be a directory")

    if not args.captions_filename.strip():
        parser.error("--captions-filename must not be empty")
    if not args.output_filename.strip():
        parser.error("--output-filename must not be empty")


def is_relative_to(path: Path, other: Path) -> bool:
    try:
        path.relative_to(other)
    except ValueError:
        return False
    return True


def collect_video_files(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def iter_manifest_rows(manifest_path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{manifest_path}:{line_number} must contain JSON objects")
            yield line_number, record


def resolve_media_relpath(
    record: dict[str, Any],
    split_dir: Path,
    split_name: str,
    known_relpaths: set[str],
) -> str:
    clip_path_text = record.get("clip_path")
    if not isinstance(clip_path_text, str) or not clip_path_text.strip():
        raise ValueError("caption record is missing clip_path")

    raw_path = Path(clip_path_text.strip())
    resolved_split_dir = split_dir.resolve()
    candidates: list[Path] = []
    if raw_path.is_absolute():
        candidates.append(raw_path.resolve())
    else:
        candidates.append((resolved_split_dir / raw_path).resolve())
        candidates.append((resolved_split_dir.parent / raw_path).resolve())

    matched_relpaths: list[str] = []
    inside_relpaths: list[str] = []
    for candidate in candidates:
        if not is_relative_to(candidate, resolved_split_dir):
            continue

        relpath = str(candidate.relative_to(resolved_split_dir))
        if relpath not in inside_relpaths:
            inside_relpaths.append(relpath)
        if relpath in known_relpaths and relpath not in matched_relpaths:
            matched_relpaths.append(relpath)

    if len(matched_relpaths) == 1:
        return matched_relpaths[0]
    if len(matched_relpaths) > 1:
        raise ValueError(f"caption record clip_path is ambiguous within {split_name}: {clip_path_text}")
    if len(inside_relpaths) == 1:
        return inside_relpaths[0]
    if len(inside_relpaths) > 1:
        raise ValueError(f"caption record clip_path is ambiguous within {split_name}: {clip_path_text}")
    raise ValueError(f"caption record clip_path points outside {split_name}: {clip_path_text}")


def build_split_export(
    input_dir: Path,
    split_name: str,
    captions_filename: str,
    output_filename: str,
) -> SplitExport:
    split_dir = input_dir / split_name
    if not split_dir.exists() or not split_dir.is_dir():
        raise FileNotFoundError(f"Missing split directory: {split_dir}")

    videos = collect_video_files(split_dir)
    if not videos:
        raise ValueError(f"No supported videos found in {split_dir}")

    output_file = split_dir / output_filename
    if output_file.exists() and output_file.is_dir():
        raise ValueError(f"Output path is a directory: {output_file}")

    captions_file = split_dir / captions_filename
    if not captions_file.exists() or not captions_file.is_file():
        raise FileNotFoundError(f"Missing captions manifest: {captions_file}")

    video_relpaths = sorted(str(path.relative_to(split_dir)) for path in videos)
    known_relpaths = set(video_relpaths)
    rows_by_media_path: dict[str, ExportRow] = {}

    for line_number, record in iter_manifest_rows(captions_file):
        status = record.get("status")
        if status != "captioned":
            raise ValueError(
                f"{captions_file}:{line_number} has status {status!r}; expected 'captioned'"
            )

        caption = record.get("caption")
        if not isinstance(caption, str) or not caption.strip():
            raise ValueError(f"{captions_file}:{line_number} is missing a non-empty caption")

        media_path = resolve_media_relpath(
            record,
            split_dir=split_dir,
            split_name=split_name,
            known_relpaths=known_relpaths,
        )
        if media_path not in known_relpaths:
            raise ValueError(
                f"{captions_file}:{line_number} references a clip that is not present in {split_name}: "
                f"{media_path}"
            )
        if media_path in rows_by_media_path:
            raise ValueError(f"{captions_file}:{line_number} duplicates clip {media_path}")

        rows_by_media_path[media_path] = ExportRow(media_path=media_path, caption=caption)

    missing_relpaths = [relpath for relpath in video_relpaths if relpath not in rows_by_media_path]
    if missing_relpaths:
        raise ValueError(
            f"{captions_file} is missing successful captions for {len(missing_relpaths)} clip(s); "
            f"first missing: {missing_relpaths[0]}"
        )

    rows = tuple(rows_by_media_path[relpath] for relpath in sorted(rows_by_media_path))
    if not rows:
        raise ValueError(f"Split {split_name} has zero usable samples")

    return SplitExport(split_name=split_name, output_file=output_file, rows=rows)


def write_export_row(handle: TextIO, row: ExportRow) -> None:
    handle.write(json.dumps({"caption": row.caption, "media_path": row.media_path}))
    handle.write("\n")
    handle.flush()


def write_split_export(export: SplitExport) -> None:
    export.output_file.parent.mkdir(parents=True, exist_ok=True)
    with export.output_file.open("w", encoding="utf-8") as handle:
        for row in export.rows:
            write_export_row(handle, row)


def summarize_results(exports: tuple[SplitExport, ...]) -> str:
    parts: list[str] = []
    for export in exports:
        parts.append(f"{export.split_name}={len(export.rows)}")
    for export in exports:
        parts.append(f"{export.split_name}_manifest={export.output_file}")
    return " ".join(parts)


def run(args: argparse.Namespace) -> int:
    try:
        exports = tuple(
            build_split_export(
                input_dir=args.input_dir,
                split_name=split_name,
                captions_filename=args.captions_filename,
                output_filename=args.output_filename,
            )
            for split_name in SPLIT_NAMES
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    try:
        for export in exports:
            write_split_export(export)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(summarize_results(exports))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
