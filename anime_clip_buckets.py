#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

VIDEO_EXTENSIONS = {".avi", ".mkv", ".mov", ".mp4", ".webm"}
DEFAULT_MAX_BUCKETS = 30
DIMENSION_MULTIPLE = 32
FRAME_STEP = 8
FRAME_OFFSET = 1
DIMENSION_SEARCH_RADIUS = 16
EPSILON = 1e-12


@dataclass(frozen=True)
class ClipMetadata:
    width: int
    height: int
    duration: float
    fps: float
    frame_count: int

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height


@dataclass(frozen=True)
class FeaturePoint:
    width: int
    height: int
    frame_count: int
    weight: int

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height


@dataclass(frozen=True, order=True)
class Bucket:
    width: int
    height: int
    frames: int

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height

    def to_text(self) -> str:
        return f"{self.width}x{self.height}x{self.frames}"


def positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Scan video clips and print up to 30 semicolon-delimited "
            "widthxheightxframes buckets derived from the dataset."
        )
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Directory containing video clips.",
    )
    parser.add_argument(
        "--resolution",
        required=True,
        type=positive_integer,
        help="Square-root area target N, so projected bucket areas approximate N^2.",
    )
    parser.add_argument(
        "--max-buckets",
        type=positive_integer,
        default=DEFAULT_MAX_BUCKETS,
        help=f"Maximum number of buckets to print. Defaults to {DEFAULT_MAX_BUCKETS}.",
    )
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not args.input_dir.exists() or not args.input_dir.is_dir():
        parser.error("--input-dir must exist and be a directory")


def collect_video_files(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def require_runtime() -> None:
    if shutil.which("ffprobe") is None:
        raise SystemExit(
            "Missing required system tool: ffprobe. Install FFmpeg so ffprobe is available on PATH."
        )


def safe_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def safe_int(value: Any) -> int | None:
    try:
        text = str(value).strip()
        if not text or text in {"N/A", "nan"}:
            return None
        parsed = int(text)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def parse_fraction(value: str | None) -> float | None:
    if not value or value in {"0/0", "N/A"}:
        return None
    if "/" in value:
        numerator_text, denominator_text = value.split("/", maxsplit=1)
        numerator = safe_float(numerator_text)
        denominator = safe_float(denominator_text)
        if numerator is None or denominator in {None, 0.0}:
            return None
        return numerator / denominator
    return safe_float(value)


def round_frame_count(value: float) -> int:
    return max(1, int(round(value)))


def probe_video(video_path: Path) -> ClipMetadata:
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

    width = safe_int(video_stream.get("width"))
    height = safe_int(video_stream.get("height"))
    if width is None or height is None:
        raise ValueError(f"Could not determine resolution for {video_path}")

    duration = safe_float(payload.get("format", {}).get("duration"))
    if duration is None:
        duration = safe_float(video_stream.get("duration"))
    if duration is None or duration <= 0:
        raise ValueError(f"Could not determine duration for {video_path}")

    fps = parse_fraction(video_stream.get("avg_frame_rate")) or parse_fraction(video_stream.get("r_frame_rate"))
    if fps is None or fps <= 0:
        raise ValueError(f"Could not determine FPS for {video_path}")

    frame_count = safe_int(video_stream.get("nb_frames"))
    if frame_count is None:
        frame_count = round_frame_count(duration * fps)

    return ClipMetadata(width=width, height=height, duration=duration, fps=fps, frame_count=frame_count)


def build_feature_points(metadata_items: list[ClipMetadata]) -> list[FeaturePoint]:
    counts: dict[tuple[int, int, int], int] = {}
    for metadata in metadata_items:
        key = (metadata.width, metadata.height, metadata.frame_count)
        counts[key] = counts.get(key, 0) + 1

    return [
        FeaturePoint(width=width, height=height, frame_count=frame_count, weight=weight)
        for (width, height, frame_count), weight in sorted(counts.items())
    ]


def log_distance(aspect_ratio_a: float, frame_count_a: int, aspect_ratio_b: float, frame_count_b: int) -> float:
    return abs(math.log(aspect_ratio_a / aspect_ratio_b)) + abs(math.log(frame_count_a / frame_count_b))


def bucket_distance(bucket: Bucket, feature: FeaturePoint) -> float:
    return log_distance(bucket.aspect_ratio, bucket.frames, feature.aspect_ratio, feature.frame_count)


def feature_distance(left: FeaturePoint, right: FeaturePoint) -> float:
    return log_distance(left.aspect_ratio, left.frame_count, right.aspect_ratio, right.frame_count)


def quantize_frame_count(frame_count: int) -> int:
    if frame_count <= FRAME_OFFSET:
        return FRAME_OFFSET

    base_index = max(0, (frame_count - FRAME_OFFSET) // FRAME_STEP)
    lower = FRAME_OFFSET + (base_index * FRAME_STEP)
    upper = lower + FRAME_STEP
    if abs(frame_count - lower) <= abs(upper - frame_count):
        return lower
    return upper


def candidate_dimension_values(ideal_dimension: float) -> range:
    center = max(1, int(round(ideal_dimension / DIMENSION_MULTIPLE)))
    start = max(1, center - DIMENSION_SEARCH_RADIUS)
    stop = center + DIMENSION_SEARCH_RADIUS
    return range(start, stop + 1)


def project_bucket_dimensions(aspect_ratio: float, resolution: int) -> tuple[int, int]:
    sqrt_ratio = math.sqrt(aspect_ratio)
    ideal_width = resolution * sqrt_ratio
    ideal_height = resolution / sqrt_ratio
    target_area = float(resolution * resolution)

    best_key: tuple[float, float, int, int] | None = None
    best_dimensions: tuple[int, int] | None = None
    for width_units in candidate_dimension_values(ideal_width):
        width = width_units * DIMENSION_MULTIPLE
        for height_units in candidate_dimension_values(ideal_height):
            height = height_units * DIMENSION_MULTIPLE
            projected_aspect = width / height
            aspect_error = abs(math.log(projected_aspect / aspect_ratio))
            area_error = abs((width * height) - target_area)
            candidate_key = (aspect_error, area_error, width, height)
            if best_key is None or candidate_key < best_key:
                best_key = candidate_key
                best_dimensions = (width, height)

    assert best_dimensions is not None
    return best_dimensions


def project_bucket(feature: FeaturePoint, resolution: int) -> Bucket:
    width, height = project_bucket_dimensions(feature.aspect_ratio, resolution)
    return Bucket(width=width, height=height, frames=quantize_frame_count(feature.frame_count))


def weighted_total_cost(candidate: FeaturePoint, features: list[FeaturePoint]) -> float:
    return sum(feature_distance(candidate, feature) * feature.weight for feature in features)


def greedy_select_buckets(features: list[FeaturePoint], resolution: int, max_buckets: int) -> list[Bucket]:
    if not features:
        return []

    sorted_features = sorted(features, key=lambda feature: (feature.aspect_ratio, feature.frame_count, feature.width, feature.height))
    best_seed = min(
        sorted_features,
        key=lambda candidate: (
            weighted_total_cost(candidate, sorted_features),
            candidate.aspect_ratio,
            candidate.frame_count,
            candidate.width,
            candidate.height,
        ),
    )
    selected_features = [best_seed]
    selected_buckets = {project_bucket(best_seed, resolution)}
    best_costs = [feature_distance(best_seed, feature) for feature in sorted_features]

    while len(selected_buckets) < max_buckets:
        current_total_cost = sum(cost * feature.weight for cost, feature in zip(best_costs, sorted_features))
        best_candidate: FeaturePoint | None = None
        best_candidate_bucket: Bucket | None = None
        best_candidate_costs: list[float] | None = None
        best_improvement = 0.0

        for candidate in sorted_features:
            if candidate in selected_features:
                continue

            candidate_bucket = project_bucket(candidate, resolution)
            if candidate_bucket in selected_buckets:
                continue

            candidate_costs = [
                min(current_cost, feature_distance(candidate, feature))
                for current_cost, feature in zip(best_costs, sorted_features)
            ]
            candidate_total_cost = sum(cost * feature.weight for cost, feature in zip(candidate_costs, sorted_features))
            improvement = current_total_cost - candidate_total_cost
            if improvement <= EPSILON:
                continue

            if (
                best_candidate is None
                or improvement > best_improvement + EPSILON
                or (
                    abs(improvement - best_improvement) <= EPSILON
                    and (
                        candidate_bucket.aspect_ratio,
                        candidate_bucket.frames,
                        candidate_bucket.width,
                        candidate_bucket.height,
                        candidate.aspect_ratio,
                        candidate.frame_count,
                        candidate.width,
                        candidate.height,
                    )
                    < (
                        best_candidate_bucket.aspect_ratio,
                        best_candidate_bucket.frames,
                        best_candidate_bucket.width,
                        best_candidate_bucket.height,
                        best_candidate.aspect_ratio,
                        best_candidate.frame_count,
                        best_candidate.width,
                        best_candidate.height,
                    )
                )
            ):
                best_candidate = candidate
                best_candidate_bucket = candidate_bucket
                best_candidate_costs = candidate_costs
                best_improvement = improvement

        if best_candidate is None or best_candidate_bucket is None or best_candidate_costs is None:
            break

        selected_features.append(best_candidate)
        selected_buckets.add(best_candidate_bucket)
        best_costs = best_candidate_costs

    return sorted(selected_buckets, key=lambda bucket: (bucket.aspect_ratio, bucket.frames, bucket.width, bucket.height))


def assign_features_to_buckets(features: list[FeaturePoint], buckets: list[Bucket]) -> dict[Bucket, int]:
    assignments = {bucket: 0 for bucket in buckets}
    for feature in features:
        best_bucket = min(
            buckets,
            key=lambda bucket: (
                bucket_distance(bucket, feature),
                bucket.aspect_ratio,
                bucket.frames,
                bucket.width,
                bucket.height,
            ),
        )
        assignments[best_bucket] += feature.weight
    return assignments


def format_buckets(buckets: list[Bucket]) -> str:
    return ";".join(bucket.to_text() for bucket in buckets)


def run(args: argparse.Namespace) -> int:
    video_paths = collect_video_files(args.input_dir)
    if not video_paths:
        print(f"No supported videos found in {args.input_dir}", file=sys.stderr)
        return 1

    require_runtime()

    metadata_items: list[ClipMetadata] = []
    for video_path in video_paths:
        try:
            metadata_items.append(probe_video(video_path))
        except Exception as exc:
            print(f"failed: {video_path} ({exc})", file=sys.stderr)
            return 1

    feature_points = build_feature_points(metadata_items)
    buckets = greedy_select_buckets(feature_points, resolution=args.resolution, max_buckets=args.max_buckets)
    if not buckets:
        print("Could not derive any buckets from the input clips", file=sys.stderr)
        return 1

    _ = assign_features_to_buckets(feature_points, buckets)
    print(format_buckets(buckets))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
