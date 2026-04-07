#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Iterable, TextIO

import numpy as np

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

try:
    import torch
except ModuleNotFoundError:
    torch = None

try:
    from transnetv2_pytorch import TransNetV2
except ModuleNotFoundError:
    TransNetV2 = None

VIDEO_EXTENSIONS = {".avi", ".mkv", ".mov", ".mp4", ".webm"}
DEFAULT_OUTPUT_DIR = Path("clips")
DEFAULT_MIN_DURATION = 1.0
DEFAULT_MAX_DURATION = 5.0
DEFAULT_BOUNDARY_THRESHOLD = 0.25
DEFAULT_BOUNDARY_PADDING = 0.75
DEFAULT_BOUNDARY_MODE = "high_recall_any"
DEFAULT_MIN_GAP = 0.25
EPSILON = 1e-9


@dataclass(frozen=True)
class VideoProbe:
    duration: float
    fps: float


@dataclass(frozen=True)
class ClipWindow:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class VideoProcessResult:
    video_path: Path
    clips_created: int


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than 0")
    return parsed


def non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract recall-biased anime-safe 1s-5s clips from downloaded videos, "
            "avoiding scene boundaries detected by TransNetV2."
        )
    )
    parser.add_argument("--input-dir", required=True, type=Path, help="Directory containing input videos.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for extracted clips. Defaults to {DEFAULT_OUTPUT_DIR}/.",
    )
    parser.add_argument(
        "--min-duration",
        type=positive_float,
        default=DEFAULT_MIN_DURATION,
        help=f"Minimum clip duration in seconds. Defaults to {DEFAULT_MIN_DURATION}.",
    )
    parser.add_argument(
        "--max-duration",
        type=positive_float,
        default=DEFAULT_MAX_DURATION,
        help=f"Maximum clip duration in seconds. Defaults to {DEFAULT_MAX_DURATION}.",
    )
    parser.add_argument(
        "--boundary-threshold",
        type=positive_float,
        default=DEFAULT_BOUNDARY_THRESHOLD,
        help=(
            "Boundary probability threshold. Lower values are more recall-biased and "
            f"exclude more ambiguous cuts. Defaults to {DEFAULT_BOUNDARY_THRESHOLD}."
        ),
    )
    parser.add_argument(
        "--boundary-padding",
        type=non_negative_float,
        default=DEFAULT_BOUNDARY_PADDING,
        help=(
            "Seconds to pad around detected boundaries. Larger values are more "
            f"recall-biased. Defaults to {DEFAULT_BOUNDARY_PADDING}."
        ),
    )
    parser.add_argument(
        "--min-gap",
        type=non_negative_float,
        default=DEFAULT_MIN_GAP,
        help=f"Seconds to leave between extracted clips. Defaults to {DEFAULT_MIN_GAP}.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="cpu",
        help="Device to use for TransNetV2 inference. Defaults to cpu.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Global seed for deterministic per-video clip sampling. Defaults to 0.",
    )
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not args.input_dir.exists() or not args.input_dir.is_dir():
        parser.error("--input-dir must exist and be a directory")
    if args.max_duration + EPSILON < args.min_duration:
        parser.error("--max-duration must be greater than or equal to --min-duration")


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
    if TransNetV2 is None or torch is None:
        raise SystemExit(
            "Missing dependency 'transnetv2-pytorch'. Install it with: "
            "python -m pip install -r requirements.txt"
        )


def safe_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


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


def probe_video(video_path: Path) -> VideoProbe:
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

    fps = parse_fraction(video_stream.get("avg_frame_rate")) or parse_fraction(video_stream.get("r_frame_rate"))
    if fps is None or fps <= 0:
        raise ValueError(f"Could not determine FPS for {video_path}")

    return VideoProbe(duration=duration, fps=fps)


def compute_frame_guard(fps: float) -> float:
    return 1.0 / fps


def load_model(device: str) -> Any:
    assert TransNetV2 is not None
    assert torch is not None

    model = TransNetV2(device=device)
    weights_resource = resources.files("transnetv2_pytorch").joinpath("transnetv2-pytorch-weights.pth")
    with resources.as_file(weights_resource) as weights_path:
        state_dict = torch.load(weights_path, map_location=model.device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def to_numpy_1d(values: Any) -> np.ndarray:
    if hasattr(values, "detach"):
        values = values.detach()
    if hasattr(values, "cpu"):
        values = values.cpu()
    if hasattr(values, "numpy"):
        values = values.numpy()
    return np.asarray(values, dtype=float).reshape(-1)


def combine_boundary_scores(single_frame_predictions: Any, all_frame_predictions: Any) -> np.ndarray:
    single_scores = to_numpy_1d(single_frame_predictions)
    all_scores = to_numpy_1d(all_frame_predictions)
    if single_scores.shape != all_scores.shape:
        raise ValueError("TransNetV2 prediction shapes do not match")
    return np.maximum(single_scores, all_scores)


def merge_spans(spans: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    ordered = sorted((start, end) for start, end in spans if end - start > EPSILON)
    if not ordered:
        return []

    merged: list[list[float]] = [[ordered[0][0], ordered[0][1]]]
    for start, end in ordered[1:]:
        previous = merged[-1]
        if start <= previous[1] + EPSILON:
            previous[1] = max(previous[1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def build_unsafe_spans(
    boundary_scores: np.ndarray,
    fps: float,
    duration: float,
    threshold: float,
    padding: float,
    frame_guard: float = 0.0,
) -> list[tuple[float, float]]:
    unsafe_spans: list[tuple[float, float]] = []
    start_index: int | None = None
    effective_padding = padding + frame_guard

    for index, score in enumerate(boundary_scores):
        if score >= threshold:
            if start_index is None:
                start_index = index
            continue

        if start_index is not None:
            span_start = max(0.0, (start_index / fps) - effective_padding)
            span_end = min(duration, (index / fps) + effective_padding)
            unsafe_spans.append((span_start, span_end))
            start_index = None

    if start_index is not None:
        span_start = max(0.0, (start_index / fps) - effective_padding)
        span_end = min(duration, (len(boundary_scores) / fps) + effective_padding)
        unsafe_spans.append((span_start, span_end))

    return merge_spans(unsafe_spans)


def invert_spans(spans: Iterable[tuple[float, float]], duration: float) -> list[tuple[float, float]]:
    safe_spans: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in merge_spans(spans):
        if start > cursor + EPSILON:
            safe_spans.append((cursor, start))
        cursor = max(cursor, end)
    if duration > cursor + EPSILON:
        safe_spans.append((cursor, duration))
    return safe_spans


def filter_safe_spans(spans: Iterable[tuple[float, float]], min_duration: float) -> list[tuple[float, float]]:
    return [(start, end) for start, end in spans if (end - start) + EPSILON >= min_duration]


def select_safe_spans(
    boundary_scores: np.ndarray,
    fps: float,
    duration: float,
    threshold: float,
    padding: float,
    min_duration: float,
    frame_guard: float = 0.0,
) -> list[tuple[float, float]]:
    unsafe_spans = build_unsafe_spans(
        boundary_scores,
        fps,
        duration,
        threshold,
        padding,
        frame_guard=frame_guard,
    )
    safe_spans = invert_spans(unsafe_spans, duration)
    return filter_safe_spans(safe_spans, min_duration)


def make_video_rng(seed: int, video_path: Path) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{video_path.resolve()}".encode("utf-8")).digest()
    derived_seed = int.from_bytes(digest[:8], "big")
    return random.Random(derived_seed)


def tile_safe_spans(
    safe_spans: Iterable[tuple[float, float]],
    min_duration: float,
    max_duration: float,
    min_gap: float,
    rng: random.Random,
) -> list[ClipWindow]:
    clips: list[ClipWindow] = []
    for span_start, span_end in safe_spans:
        cursor = span_start
        while (span_end - cursor) + EPSILON >= min_duration:
            max_clip_duration = min(max_duration, span_end - cursor)
            if max_clip_duration + EPSILON < min_duration:
                break
            if abs(max_clip_duration - min_duration) <= EPSILON:
                clip_duration = min_duration
            else:
                clip_duration = rng.uniform(min_duration, max_clip_duration)
            clip_end = min(span_end, cursor + clip_duration)
            clips.append(ClipWindow(start=cursor, end=clip_end))
            cursor = clip_end + min_gap
    return clips


def make_clip_output_path(output_dir: Path, video_path: Path, clip_index: int, clip: ClipWindow) -> Path:
    start_ms = round(clip.start * 1000)
    end_ms = round(clip.end * 1000)
    return output_dir / video_path.stem / f"clip_{clip_index}_{start_ms}_{end_ms}.mp4"


def build_ffmpeg_command(source_path: Path, clip: ClipWindow, output_path: Path) -> list[str]:
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source_path),
        "-ss",
        f"{clip.start:.6f}",
        "-t",
        f"{clip.duration:.6f}",
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        str(output_path),
    ]


def extract_clip(source_path: Path, clip: ClipWindow, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(build_ffmpeg_command(source_path, clip, output_path), check=True)


def write_manifest_row(
    handle: TextIO,
    source_path: Path,
    clip_path: Path,
    clip: ClipWindow,
    threshold: float,
    padding: float,
    seed: int,
) -> None:
    record = {
        "source_path": str(source_path),
        "clip_path": str(clip_path),
        "start": clip.start,
        "end": clip.end,
        "duration": clip.duration,
        "boundary_threshold": threshold,
        "boundary_padding": padding,
        "boundary_mode": DEFAULT_BOUNDARY_MODE,
        "seed": seed,
    }
    handle.write(json.dumps(record))
    handle.write("\n")
    handle.flush()


def analyze_video(model: Any, video_path: Path) -> np.ndarray:
    _, single_frame_predictions, all_frame_predictions = model.predict_video(str(video_path), quiet=True)
    return combine_boundary_scores(single_frame_predictions, all_frame_predictions)


def process_video(
    model: Any,
    video_path: Path,
    output_dir: Path,
    min_duration: float,
    max_duration: float,
    threshold: float,
    padding: float,
    min_gap: float,
    seed: int,
    manifest_handle: TextIO,
) -> VideoProcessResult:
    probe = probe_video(video_path)
    boundary_scores = analyze_video(model, video_path)
    analysis_duration = min(probe.duration, len(boundary_scores) / probe.fps)
    frame_guard = compute_frame_guard(probe.fps)
    safe_spans = select_safe_spans(
        boundary_scores=boundary_scores,
        fps=probe.fps,
        duration=analysis_duration,
        threshold=threshold,
        padding=padding,
        min_duration=min_duration,
        frame_guard=frame_guard,
    )
    rng = make_video_rng(seed, video_path)
    clips = tile_safe_spans(
        safe_spans=safe_spans,
        min_duration=min_duration,
        max_duration=max_duration,
        min_gap=min_gap,
        rng=rng,
    )

    for clip_index, clip in enumerate(clips, start=1):
        output_path = make_clip_output_path(output_dir, video_path, clip_index, clip)
        extract_clip(video_path, clip, output_path)
        write_manifest_row(
            manifest_handle,
            source_path=video_path,
            clip_path=output_path,
            clip=clip,
            threshold=threshold,
            padding=padding,
            seed=seed,
        )

    return VideoProcessResult(video_path=video_path, clips_created=len(clips))


def summarize_results(
    total_videos: int,
    failed_videos: int,
    total_clips: int,
    manifest_path: Path,
    output_dir: Path,
) -> str:
    return " ".join(
        [
            f"videos={total_videos}",
            f"failed={failed_videos}",
            f"clips={total_clips}",
            f"manifest={manifest_path}",
            f"output={output_dir}",
        ]
    )


def run(args: argparse.Namespace) -> int:
    videos = collect_video_files(args.input_dir)
    if not videos:
        print(f"No supported videos found in {args.input_dir}", file=sys.stderr)
        return 1

    require_runtime()
    model = load_model(args.device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.jsonl"
    total_clips = 0
    failed_videos = 0

    with manifest_path.open("w", encoding="utf-8") as manifest_handle:
        for video_path in videos:
            try:
                result = process_video(
                    model=model,
                    video_path=video_path,
                    output_dir=args.output_dir,
                    min_duration=args.min_duration,
                    max_duration=args.max_duration,
                    threshold=args.boundary_threshold,
                    padding=args.boundary_padding,
                    min_gap=args.min_gap,
                    seed=args.seed,
                    manifest_handle=manifest_handle,
                )
            except Exception as exc:
                failed_videos += 1
                print(f"failed: {video_path} ({exc})", file=sys.stderr)
                continue

            total_clips += result.clips_created
            print(f"processed: {video_path} clips={result.clips_created}")

    print(
        summarize_results(
            total_videos=len(videos),
            failed_videos=failed_videos,
            total_clips=total_clips,
            manifest_path=manifest_path,
            output_dir=args.output_dir,
        )
    )
    return 1 if failed_videos > 0 or total_clips == 0 else 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
