#!/usr/bin/env python3

from __future__ import annotations

import argparse
import contextlib
import io
import json
import random
import tempfile
import time
from pathlib import Path

import anime_clip_subset


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def positive_fraction(value: str) -> float:
    parsed = float(value)
    if parsed <= 0 or parsed > 1:
        raise argparse.ArgumentTypeError("value must be greater than 0 and less than or equal to 1")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic tagged clip dataset and time anime_clip_subset.py against it."
    )
    parser.add_argument("--clips", type=positive_int, required=True, help="Number of synthetic clips to generate.")
    parser.add_argument(
        "--clips-per-source",
        type=positive_int,
        default=5,
        help="How many clips share the same synthetic source video. Default: 5.",
    )
    parser.add_argument(
        "--tags-pool",
        type=positive_int,
        default=80,
        help="Number of unique synthetic tags to sample from. Default: 80.",
    )
    parser.add_argument(
        "--tags-per-clip",
        type=positive_int,
        default=5,
        help="Number of unique tags attached to each synthetic clip. Default: 5.",
    )
    parser.add_argument(
        "--fraction",
        type=positive_fraction,
        default=0.10,
        help="Validation fraction passed to anime_clip_subset.py. Default: 0.10.",
    )
    parser.add_argument(
        "--train-count",
        type=non_negative_int,
        default=0,
        help="Requested train count. Use 0 to omit --train-count. Default: 0.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for synthetic tag assignment.")
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep the generated temp directory instead of deleting it after the benchmark.",
    )
    parser.add_argument(
        "--show-run-output",
        action="store_true",
        help="Print anime_clip_subset.py selection output instead of suppressing it.",
    )
    return parser


def populate_dataset(root: Path, clips: int, clips_per_source: int, tag_pool_size: int, tags_per_clip: int, seed: int) -> Path:
    if tags_per_clip > tag_pool_size:
        raise ValueError("--tags-per-clip must be less than or equal to --tags-pool")

    random.seed(seed)
    input_dir = root / "clips"
    input_dir.mkdir(parents=True, exist_ok=True)
    tags_path = input_dir / "tags.jsonl"
    manifest_path = input_dir / "manifest.jsonl"
    tag_pool = [f"tag{i}" for i in range(tag_pool_size)]

    with tags_path.open("w", encoding="utf-8") as tags_handle, manifest_path.open("w", encoding="utf-8") as manifest_handle:
        for index in range(clips):
            clip_path = input_dir / f"clip_{index:05d}.mp4"
            clip_path.write_bytes(b"x")
            source_path = f"/dataset/source_{index // clips_per_source:05d}.mp4"
            manifest_handle.write(json.dumps({"clip_path": str(clip_path), "source_path": source_path}) + "\n")
            tags_handle.write(
                json.dumps(
                    {
                        "status": "tagged",
                        "clip_path": str(clip_path),
                        "clip_relpath": clip_path.name,
                        "source_path": source_path,
                        "all_tags": random.sample(tag_pool, k=tags_per_clip),
                    }
                )
                + "\n"
            )

    return input_dir


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    temp_dir_manager = tempfile.TemporaryDirectory(prefix="subset_bench_")
    cleanup_temp_dir = not args.keep_temp
    try:
        root = Path(temp_dir_manager.name)
        input_dir = populate_dataset(
            root=root,
            clips=args.clips,
            clips_per_source=args.clips_per_source,
            tag_pool_size=args.tags_pool,
            tags_per_clip=args.tags_per_clip,
            seed=args.seed,
        )

        subset_parser = anime_clip_subset.build_parser()
        subset_argv = [
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(root / "subset"),
            "--fraction",
            str(args.fraction),
        ]
        if args.train_count > 0:
            subset_argv.extend(["--train-count", str(args.train_count)])
        subset_args = subset_parser.parse_args(subset_argv)
        anime_clip_subset.validate_args(subset_parser, subset_args)

        start = time.perf_counter()
        if args.show_run_output:
            exit_code = anime_clip_subset.run(subset_args)
        else:
            with contextlib.redirect_stdout(io.StringIO()):
                exit_code = anime_clip_subset.run(subset_args)
        elapsed = time.perf_counter() - start

        train_count_text = "all remaining" if args.train_count == 0 else str(args.train_count)
        print(
            " ".join(
                [
                    f"clips={args.clips}",
                    f"clips_per_source={args.clips_per_source}",
                    f"fraction={args.fraction}",
                    f"train_count={train_count_text}",
                    f"seed={args.seed}",
                    f"seconds={elapsed:.3f}",
                    f"exit_code={exit_code}",
                    f"temp_dir={root}",
                ]
            )
        )
        return exit_code
    finally:
        if cleanup_temp_dir:
            temp_dir_manager.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
