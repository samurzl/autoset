#!/usr/bin/env python3

from __future__ import annotations

import argparse
import io
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

try:
    from PIL import Image
except ModuleNotFoundError:
    Image = None

try:
    from wdtagger import Tagger
except ModuleNotFoundError:
    Tagger = None

VIDEO_EXTENSIONS = {".avi", ".mkv", ".mov", ".mp4", ".webm"}
DEFAULT_OUTPUT_FILENAME = "tags.jsonl"
DEFAULT_MODEL_REPO = "SmilingWolf/wd-swinv2-tagger-v3"
DEFAULT_GENERAL_THRESHOLD = 0.35
DEFAULT_CHARACTER_THRESHOLD = 0.9
DEFAULT_BATCH_SIZE = 8
FRAME_TIME_EPSILON = 1e-3


@dataclass(frozen=True)
class VideoMetadata:
    duration: float


@dataclass(frozen=True)
class PendingClip:
    clip_path: Path
    clip_relpath: str
    duration: float
    frame_time: float
    image: Any


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def probability_float(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("value must be between 0 and 1")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate wd-tagger tags for one representative frame from every video clip "
            "in a directory."
        )
    )
    parser.add_argument("--input-dir", required=True, type=Path, help="Directory containing video clips.")
    parser.add_argument(
        "--output-file",
        type=Path,
        help=f"JSONL manifest path. Defaults to <input-dir>/{DEFAULT_OUTPUT_FILENAME}.",
    )
    parser.add_argument(
        "--model-repo",
        default=DEFAULT_MODEL_REPO,
        help=f"Hugging Face repo for wd-tagger. Defaults to {DEFAULT_MODEL_REPO}.",
    )
    parser.add_argument(
        "--general-threshold",
        type=probability_float,
        default=DEFAULT_GENERAL_THRESHOLD,
        help=f"Threshold for general tags. Defaults to {DEFAULT_GENERAL_THRESHOLD}.",
    )
    parser.add_argument(
        "--character-threshold",
        type=probability_float,
        default=DEFAULT_CHARACTER_THRESHOLD,
        help=f"Threshold for character tags. Defaults to {DEFAULT_CHARACTER_THRESHOLD}.",
    )
    parser.add_argument(
        "--batch-size",
        type=positive_int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size for wd-tagger inference. Defaults to {DEFAULT_BATCH_SIZE}.",
    )
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not args.input_dir.exists() or not args.input_dir.is_dir():
        parser.error("--input-dir must exist and be a directory")

    if args.output_file is None:
        args.output_file = args.input_dir / DEFAULT_OUTPUT_FILENAME

    if args.output_file.exists() and args.output_file.is_dir():
        parser.error("--output-file must be a file path")


def collect_video_files(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def require_runtime() -> None:
    missing_tools = [tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None]
    if missing_tools:
        raise SystemExit(
            f"Missing required system tool(s): {', '.join(missing_tools)}. "
            "Install FFmpeg so both ffmpeg and ffprobe are available on PATH."
        )
    if Tagger is None or Image is None:
        raise SystemExit(
            "Missing dependencies for tagging. Install them with: "
            "python -m pip install -r requirements.txt"
        )


def safe_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def probe_video(video_path: Path) -> VideoMetadata:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_format",
        "-show_streams",
        "-of",
        "json",
        str(video_path),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(completed.stdout)
    streams = payload.get("streams", [])
    video_stream = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    if video_stream is None:
        raise ValueError(f"No video stream found in {video_path}")

    duration = safe_float(payload.get("format", {}).get("duration"))
    if duration is None:
        duration = safe_float(video_stream.get("duration"))
    if duration is None or duration <= 0:
        raise ValueError(f"Could not determine duration for {video_path}")

    return VideoMetadata(duration=duration)


def compute_representative_frame_time(duration: float) -> float:
    if duration <= 0:
        raise ValueError("duration must be greater than 0")
    upper_bound = max(duration - FRAME_TIME_EPSILON, 0.0)
    return min(duration * 0.5, upper_bound)


def build_frame_command(clip_path: Path, frame_time: float) -> list[str]:
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(clip_path),
        "-ss",
        f"{frame_time:.6f}",
        "-frames:v",
        "1",
        "-f",
        "image2pipe",
        "-vcodec",
        "png",
        "-",
    ]


def extract_frame_image(clip_path: Path, frame_time: float) -> Any:
    assert Image is not None
    completed = subprocess.run(build_frame_command(clip_path, frame_time), check=True, capture_output=True)
    if not completed.stdout:
        raise ValueError(f"ffmpeg did not produce a frame for {clip_path}")
    image = Image.open(io.BytesIO(completed.stdout))
    image.load()
    return image


def normalize_score_mapping(values: dict[str, Any]) -> dict[str, float]:
    return {str(key): float(value) for key, value in values.items()}


def dedupe_tags(tags: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for tag in tags:
        if tag not in seen:
            seen.add(tag)
            ordered.append(tag)
    return ordered


def make_manifest_record(clip_path: Path, clip_relpath: str, model_repo: str) -> dict[str, Any]:
    return {
        "clip_path": str(clip_path),
        "clip_relpath": clip_relpath,
        "model_repo": model_repo,
        "duration": None,
        "frame_time": None,
        "rating": None,
        "rating_scores": {},
        "general_tags": [],
        "general_tag_scores": {},
        "character_tags": [],
        "character_tag_scores": {},
        "all_tags": [],
        "status": "failed",
        "error": "",
    }


def build_success_record(pending: PendingClip, result: Any, model_repo: str) -> dict[str, Any]:
    rating_scores = normalize_score_mapping(result.rating_data)
    general_tag_scores = normalize_score_mapping(result.general_tag_data)
    character_tag_scores = normalize_score_mapping(result.character_tag_data)
    return {
        "clip_path": str(pending.clip_path),
        "clip_relpath": pending.clip_relpath,
        "model_repo": model_repo,
        "duration": pending.duration,
        "frame_time": pending.frame_time,
        "rating": str(result.rating),
        "rating_scores": rating_scores,
        "general_tags": list(result.general_tags),
        "general_tag_scores": general_tag_scores,
        "character_tags": list(result.character_tags),
        "character_tag_scores": character_tag_scores,
        "all_tags": dedupe_tags([str(tag) for tag in result.all_tags]),
        "status": "tagged",
        "error": "",
    }


def build_failure_record(
    clip_path: Path,
    clip_relpath: str,
    model_repo: str,
    error: str,
    duration: float | None = None,
    frame_time: float | None = None,
) -> dict[str, Any]:
    record = make_manifest_record(clip_path, clip_relpath, model_repo)
    record["duration"] = duration
    record["frame_time"] = frame_time
    record["error"] = error
    return record


def normalize_tag_results(results: Any) -> list[Any]:
    if isinstance(results, list):
        return results
    if isinstance(results, tuple):
        return list(results)
    return [results]


def tag_pending_batch(
    tagger: Any,
    pending_batch: list[PendingClip],
    model_repo: str,
    general_threshold: float,
    character_threshold: float,
) -> list[dict[str, Any]]:
    images = [item.image for item in pending_batch]
    results = tagger.tag(
        images,
        general_threshold=general_threshold,
        character_threshold=character_threshold,
    )
    normalized_results = normalize_tag_results(results)
    if len(normalized_results) != len(pending_batch):
        raise ValueError("wd-tagger returned an unexpected number of results")
    return [
        build_success_record(pending, result, model_repo)
        for pending, result in zip(pending_batch, normalized_results, strict=True)
    ]


def close_pending_images(pending_batch: list[PendingClip]) -> None:
    for pending in pending_batch:
        close = getattr(pending.image, "close", None)
        if callable(close):
            close()


def flush_pending_batch(
    tagger: Any,
    pending_batch: list[PendingClip],
    model_repo: str,
    general_threshold: float,
    character_threshold: float,
) -> list[dict[str, Any]]:
    try:
        return tag_pending_batch(
            tagger=tagger,
            pending_batch=pending_batch,
            model_repo=model_repo,
            general_threshold=general_threshold,
            character_threshold=character_threshold,
        )
    except Exception:
        records: list[dict[str, Any]] = []
        for pending in pending_batch:
            try:
                record = tag_pending_batch(
                    tagger=tagger,
                    pending_batch=[pending],
                    model_repo=model_repo,
                    general_threshold=general_threshold,
                    character_threshold=character_threshold,
                )[0]
            except Exception as exc:
                record = build_failure_record(
                    clip_path=pending.clip_path,
                    clip_relpath=pending.clip_relpath,
                    model_repo=model_repo,
                    duration=pending.duration,
                    frame_time=pending.frame_time,
                    error=str(exc),
                )
            records.append(record)
        return records
    finally:
        close_pending_images(pending_batch)


def write_manifest_row(handle: TextIO, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record))
    handle.write("\n")
    handle.flush()


def summarize_results(total_clips: int, tagged_clips: int, failed_clips: int, output_file: Path) -> str:
    return " ".join(
        [
            f"clips={total_clips}",
            f"tagged={tagged_clips}",
            f"failed={failed_clips}",
            f"manifest={output_file}",
        ]
    )


def run(args: argparse.Namespace) -> int:
    clips = collect_video_files(args.input_dir)
    if not clips:
        print(f"No supported videos found in {args.input_dir}", file=sys.stderr)
        return 1

    require_runtime()
    assert Tagger is not None
    tagger = Tagger(model_repo=args.model_repo)

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    tagged_clips = 0
    failed_clips = 0
    pending_batch: list[PendingClip] = []

    with args.output_file.open("w", encoding="utf-8") as manifest_handle:
        for clip_path in clips:
            clip_relpath = str(clip_path.relative_to(args.input_dir))
            try:
                metadata = probe_video(clip_path)
                frame_time = compute_representative_frame_time(metadata.duration)
                image = extract_frame_image(clip_path, frame_time)
                pending_batch.append(
                    PendingClip(
                        clip_path=clip_path,
                        clip_relpath=clip_relpath,
                        duration=metadata.duration,
                        frame_time=frame_time,
                        image=image,
                    )
                )
            except Exception as exc:
                failed_clips += 1
                record = build_failure_record(
                    clip_path=clip_path,
                    clip_relpath=clip_relpath,
                    model_repo=args.model_repo,
                    error=str(exc),
                )
                write_manifest_row(manifest_handle, record)
                print(f"failed: {clip_path} ({record['error']})", file=sys.stderr)
                continue

            if len(pending_batch) < args.batch_size:
                continue

            records = flush_pending_batch(
                tagger=tagger,
                pending_batch=pending_batch,
                model_repo=args.model_repo,
                general_threshold=args.general_threshold,
                character_threshold=args.character_threshold,
            )
            pending_batch = []
            for record in records:
                write_manifest_row(manifest_handle, record)
                if record["status"] == "tagged":
                    tagged_clips += 1
                    print(f"tagged: {record['clip_path']}")
                else:
                    failed_clips += 1
                    print(f"failed: {record['clip_path']} ({record['error']})", file=sys.stderr)

        if pending_batch:
            records = flush_pending_batch(
                tagger=tagger,
                pending_batch=pending_batch,
                model_repo=args.model_repo,
                general_threshold=args.general_threshold,
                character_threshold=args.character_threshold,
            )
            for record in records:
                write_manifest_row(manifest_handle, record)
                if record["status"] == "tagged":
                    tagged_clips += 1
                    print(f"tagged: {record['clip_path']}")
                else:
                    failed_clips += 1
                    print(f"failed: {record['clip_path']} ({record['error']})", file=sys.stderr)

    print(
        summarize_results(
            total_clips=len(clips),
            tagged_clips=tagged_clips,
            failed_clips=failed_clips,
            output_file=args.output_file,
        )
    )
    return 1 if failed_clips > 0 or tagged_clips == 0 else 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
