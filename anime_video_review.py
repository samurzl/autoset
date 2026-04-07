#!/usr/bin/env python3

from __future__ import annotations

import argparse
import io
import json
import shutil
import subprocess
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

try:
    import tkinter as tk
except ModuleNotFoundError:
    tk = None

try:
    from PIL import Image, ImageChops, ImageSequence, ImageStat, ImageTk
except ModuleNotFoundError:
    Image = None
    ImageChops = None
    ImageSequence = None
    ImageStat = None
    ImageTk = None

VIDEO_EXTENSIONS = {".avi", ".mkv", ".mov", ".mp4", ".webm"}
DEFAULT_KEEP_SUFFIX = "_keep"
DEFAULT_REJECT_SUFFIX = "_reject"
DEFAULT_REVIEW_LOG_SUFFIX = "_review.jsonl"
DEFAULT_PREVIEW_DURATION = 2.0
DEFAULT_PREVIEW_FPS = 12
DEFAULT_PREVIEW_WIDTH = 640
DEFAULT_PREVIEW_CACHE_SIZE = 8
DEFAULT_WATCH_POLL_MS = 1000
DEFAULT_UI_POLL_MS = 100
STATUS_PENDING = "pending"
STATUS_ACCEPTED = "accepted"
STATUS_REJECTED = "rejected"
STATUS_ERROR = "error"
VALID_STATUSES = {STATUS_PENDING, STATUS_ACCEPTED, STATUS_REJECTED, STATUS_ERROR}


@dataclass(frozen=True)
class VideoMetadata:
    duration: float


@dataclass(frozen=True)
class PreviewWindow:
    start: float
    duration: float


@dataclass(frozen=True)
class PreviewBundle:
    frames: list[Any]
    preview_start: float
    preview_duration: float
    motion_score: float
    frame_delay_ms: int


@dataclass(frozen=True)
class PreviewTask:
    video_id: str
    path: Path
    future: Future


@dataclass(frozen=True)
class ResolvedPreview:
    video_id: str
    path: Path
    bundle: PreviewBundle | None = None
    error: Exception | None = None


@dataclass
class VideoReviewState:
    video_id: str
    original_path: Path
    current_path: Path
    relative_path: Path
    status: str = STATUS_PENDING
    preview_start: float | None = None
    preview_duration: float | None = None
    motion_score: float | None = None
    timestamp: str | None = None
    error: str = ""
    discovered_order: int = 0
    seen_order: int | None = None

    @property
    def is_seen(self) -> bool:
        return self.seen_order is not None


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than 0")
    return parsed


def normalize_path(path: Path) -> Path:
    return path.expanduser().resolve()


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def is_relative_to(path: Path, other: Path) -> bool:
    try:
        path.relative_to(other)
    except ValueError:
        return False
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Manually review downloaded source videos with a looping middle preview "
            "and split them into keep/reject folders."
        )
    )
    parser.add_argument("--input-dir", required=True, type=Path, help="Directory containing downloaded videos.")
    parser.add_argument("--keep-dir", type=Path, help="Directory for accepted videos.")
    parser.add_argument("--reject-dir", type=Path, help="Directory for rejected videos.")
    parser.add_argument("--review-log", type=Path, help="JSONL log used to resume decisions and preview metadata.")
    parser.add_argument(
        "--preview-duration",
        type=positive_float,
        default=DEFAULT_PREVIEW_DURATION,
        help=f"Loop duration in seconds. Defaults to {DEFAULT_PREVIEW_DURATION}.",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Keep polling the input directory for newly completed downloads.",
    )
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    args.input_dir = normalize_path(args.input_dir)
    if not args.input_dir.exists() or not args.input_dir.is_dir():
        parser.error("--input-dir must exist and be a directory")

    if args.keep_dir is None:
        args.keep_dir = args.input_dir.with_name(f"{args.input_dir.name}{DEFAULT_KEEP_SUFFIX}")
    if args.reject_dir is None:
        args.reject_dir = args.input_dir.with_name(f"{args.input_dir.name}{DEFAULT_REJECT_SUFFIX}")
    if args.review_log is None:
        args.review_log = args.input_dir.with_name(f"{args.input_dir.name}{DEFAULT_REVIEW_LOG_SUFFIX}")

    args.keep_dir = normalize_path(args.keep_dir)
    args.reject_dir = normalize_path(args.reject_dir)
    args.review_log = normalize_path(args.review_log)

    if args.keep_dir == args.reject_dir:
        parser.error("--keep-dir and --reject-dir must be different")
    if args.keep_dir == args.input_dir:
        parser.error("--keep-dir must be different from --input-dir")
    if args.reject_dir == args.input_dir:
        parser.error("--reject-dir must be different from --input-dir")
    if args.review_log.exists() and args.review_log.is_dir():
        parser.error("--review-log must be a file path")
    if args.keep_dir.exists() and args.keep_dir.is_file():
        parser.error("--keep-dir must be a directory path")
    if args.reject_dir.exists() and args.reject_dir.is_file():
        parser.error("--reject-dir must be a directory path")


def collect_video_files(input_dir: Path, exclude_dirs: tuple[Path, ...] = ()) -> list[Path]:
    normalized_excludes = tuple(normalize_path(path) for path in exclude_dirs)
    videos: list[Path] = []
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file():
            continue
        resolved = normalize_path(path)
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        if any(is_relative_to(resolved, exclude_dir) for exclude_dir in normalized_excludes):
            continue
        videos.append(resolved)
    return videos


def require_runtime() -> None:
    missing_tools = [tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None]
    if missing_tools:
        raise SystemExit(
            f"Missing required system tool(s): {', '.join(missing_tools)}. "
            "Install FFmpeg so ffmpeg and ffprobe are available on PATH."
        )
    if tk is None:
        raise SystemExit("Missing tkinter runtime. Use a Python build that includes Tk support.")
    if Image is None or ImageChops is None or ImageSequence is None or ImageStat is None or ImageTk is None:
        raise SystemExit(
            "Missing Pillow runtime with ImageTk support. Install it with: "
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


def compute_preview_window(duration: float, preview_duration: float = DEFAULT_PREVIEW_DURATION) -> PreviewWindow:
    if duration <= 0:
        raise ValueError("duration must be greater than 0")
    if duration <= preview_duration:
        return PreviewWindow(start=0.0, duration=duration)

    midpoint = duration * 0.5
    start = max(0.0, midpoint - (preview_duration * 0.5))
    max_start = max(0.0, duration - preview_duration)
    return PreviewWindow(start=min(start, max_start), duration=preview_duration)


def build_preview_command(
    video_path: Path,
    preview_start: float,
    preview_duration: float,
    preview_fps: int = DEFAULT_PREVIEW_FPS,
    preview_width: int = DEFAULT_PREVIEW_WIDTH,
) -> list[str]:
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-ss",
        f"{preview_start:.6f}",
        "-t",
        f"{preview_duration:.6f}",
        "-an",
        "-vf",
        f"fps={preview_fps},scale={preview_width}:-2:flags=lanczos",
        "-f",
        "gif",
        "-",
    ]


def decode_gif_frames(gif_bytes: bytes) -> list[Any]:
    if Image is None or ImageSequence is None:
        raise RuntimeError("Pillow is not available")
    if not gif_bytes:
        raise ValueError("ffmpeg did not produce preview frames")

    image = Image.open(io.BytesIO(gif_bytes))
    frames = [frame.convert("RGB").copy() for frame in ImageSequence.Iterator(image)]
    if not frames:
        raise ValueError("Preview contained no frames")
    return frames


def compute_motion_score(frames: list[Any]) -> float:
    if ImageChops is None or ImageStat is None:
        raise RuntimeError("Pillow is not available")
    if len(frames) < 2:
        return 0.0

    deltas: list[float] = []
    for previous, current in zip(frames, frames[1:]):
        diff = ImageChops.difference(previous, current).convert("L")
        mean_delta = ImageStat.Stat(diff).mean[0] / 255.0
        deltas.append(mean_delta)
    return sum(deltas) / len(deltas)


def extract_preview(
    video_path: Path,
    preview_duration: float = DEFAULT_PREVIEW_DURATION,
    preview_fps: int = DEFAULT_PREVIEW_FPS,
    preview_width: int = DEFAULT_PREVIEW_WIDTH,
) -> PreviewBundle:
    metadata = probe_video(video_path)
    preview_window = compute_preview_window(metadata.duration, preview_duration=preview_duration)
    command = build_preview_command(
        video_path=video_path,
        preview_start=preview_window.start,
        preview_duration=preview_window.duration,
        preview_fps=preview_fps,
        preview_width=preview_width,
    )
    completed = subprocess.run(command, check=True, capture_output=True)
    frames = decode_gif_frames(completed.stdout)
    frame_delay_ms = max(40, int(round((preview_window.duration / max(len(frames), 1)) * 1000)))
    return PreviewBundle(
        frames=frames,
        preview_start=preview_window.start,
        preview_duration=preview_window.duration,
        motion_score=compute_motion_score(frames),
        frame_delay_ms=frame_delay_ms,
    )


def default_relative_path(original_path: Path, input_dir: Path) -> Path:
    if is_relative_to(original_path, input_dir):
        return original_path.relative_to(input_dir)
    return Path(original_path.name)


def coerce_status(value: Any) -> str:
    text = str(value or STATUS_PENDING)
    if text not in VALID_STATUSES:
        return STATUS_PENDING
    return text


class ReviewSession:
    def __init__(
        self,
        input_dir: Path,
        keep_dir: Path,
        reject_dir: Path,
        review_log: Path,
    ) -> None:
        self.input_dir = normalize_path(input_dir)
        self.keep_dir = normalize_path(keep_dir)
        self.reject_dir = normalize_path(reject_dir)
        self.review_log = normalize_path(review_log)
        self.entries: dict[str, VideoReviewState] = {}
        self.discovery_counter = 0
        self.seen_counter = 0

    def load(self) -> None:
        if not self.review_log.exists():
            return

        with self.review_log.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                video_id = str(payload.get("video_id") or "").strip()
                if not video_id:
                    continue

                original_path = normalize_path(Path(str(payload.get("original_path") or payload.get("current_path"))))
                current_path = normalize_path(Path(str(payload.get("current_path") or original_path)))
                relative_path_text = payload.get("relative_path")
                relative_path = (
                    Path(str(relative_path_text))
                    if isinstance(relative_path_text, str) and relative_path_text.strip()
                    else default_relative_path(original_path, self.input_dir)
                )
                discovered_order = int(payload.get("discovered_order") or 0)
                seen_order = payload.get("seen_order")
                seen_value = int(seen_order) if seen_order is not None else None
                entry = self.entries.get(video_id)
                if entry is None:
                    entry = VideoReviewState(
                        video_id=video_id,
                        original_path=original_path,
                        current_path=current_path,
                        relative_path=relative_path,
                    )
                    self.entries[video_id] = entry

                entry.original_path = original_path
                entry.current_path = current_path
                entry.relative_path = relative_path
                entry.status = coerce_status(payload.get("status"))
                entry.preview_start = safe_float(payload.get("preview_start"))
                entry.preview_duration = safe_float(payload.get("preview_duration"))
                entry.motion_score = safe_float(payload.get("motion_score"))
                entry.timestamp = str(payload.get("timestamp")) if payload.get("timestamp") else None
                entry.error = str(payload.get("error") or "")
                entry.discovered_order = discovered_order or entry.discovered_order or self._next_discovery_order()
                entry.seen_order = seen_value
                self.discovery_counter = max(self.discovery_counter, entry.discovered_order)
                if entry.seen_order is not None:
                    self.seen_counter = max(self.seen_counter, entry.seen_order)

    def _next_discovery_order(self) -> int:
        self.discovery_counter += 1
        return self.discovery_counter

    def _next_seen_order(self) -> int:
        self.seen_counter += 1
        return self.seen_counter

    def _entry_destination(self, entry: VideoReviewState, destination_root: Path) -> Path:
        return destination_root / entry.relative_path

    def _resolve_existing_source_path(self, entry: VideoReviewState) -> Path:
        candidates = [
            entry.current_path,
            self.input_dir / entry.relative_path,
            self.keep_dir / entry.relative_path,
            self.reject_dir / entry.relative_path,
            entry.original_path,
        ]
        for candidate in candidates:
            if candidate.exists():
                return normalize_path(candidate)
        raise FileNotFoundError(f"Could not find file for {entry.video_id}")

    def _move_to_destination(self, entry: VideoReviewState, destination_root: Path) -> Path:
        destination = self._entry_destination(entry, destination_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            return normalize_path(destination)

        source = self._resolve_existing_source_path(entry)
        if source == destination:
            return normalize_path(destination)

        try:
            moved = source.rename(destination)
        except OSError:
            moved = Path(shutil.move(str(source), str(destination)))
        return normalize_path(moved)

    def _write_log_row(self, entry: VideoReviewState) -> None:
        self.review_log.parent.mkdir(parents=True, exist_ok=True)
        entry.timestamp = utc_now_iso()
        record = {
            "video_id": entry.video_id,
            "original_path": str(entry.original_path),
            "current_path": str(entry.current_path),
            "relative_path": str(entry.relative_path),
            "status": entry.status,
            "preview_start": entry.preview_start,
            "preview_duration": entry.preview_duration,
            "motion_score": entry.motion_score,
            "timestamp": entry.timestamp,
            "error": entry.error,
            "discovered_order": entry.discovered_order,
            "seen_order": entry.seen_order,
        }
        with self.review_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record))
            handle.write("\n")
            handle.flush()

    def add_discovered_video(self, video_path: Path) -> VideoReviewState:
        resolved = normalize_path(video_path)
        video_id = resolved.stem
        entry = self.entries.get(video_id)
        if entry is not None:
            if entry.current_path != resolved and resolved.exists() and entry.status == STATUS_PENDING:
                entry.current_path = resolved
            return entry

        entry = VideoReviewState(
            video_id=video_id,
            original_path=resolved,
            current_path=resolved,
            relative_path=default_relative_path(resolved, self.input_dir),
            discovered_order=self._next_discovery_order(),
        )
        self.entries[video_id] = entry
        self._write_log_row(entry)
        return entry

    def scan_input_dir(self) -> list[VideoReviewState]:
        discovered: list[VideoReviewState] = []
        for video_path in collect_video_files(self.input_dir, exclude_dirs=(self.keep_dir, self.reject_dir)):
            entry = self.entries.get(video_path.stem)
            if entry is None:
                discovered.append(self.add_discovered_video(video_path))
                continue
            if entry.status == STATUS_PENDING and entry.current_path != video_path:
                entry.current_path = video_path
        return discovered

    def ordered_video_ids(self) -> list[str]:
        seen_entries = sorted(
            (entry for entry in self.entries.values() if entry.seen_order is not None),
            key=lambda entry: (entry.seen_order or 0, entry.discovered_order, entry.video_id),
        )
        unseen_entries = sorted(
            (entry for entry in self.entries.values() if entry.seen_order is None),
            key=lambda entry: (
                entry.motion_score is None,
                entry.motion_score if entry.motion_score is not None else float("inf"),
                entry.discovered_order,
                entry.video_id,
            ),
        )
        return [entry.video_id for entry in seen_entries] + [entry.video_id for entry in unseen_entries]

    def choose_start_video_id(self) -> str | None:
        pending_unseen = [
            entry
            for entry in self.entries.values()
            if entry.status == STATUS_PENDING and entry.seen_order is None
        ]
        if pending_unseen:
            pending_unseen.sort(
                key=lambda entry: (
                    entry.motion_score is None,
                    entry.motion_score if entry.motion_score is not None else float("inf"),
                    entry.discovered_order,
                    entry.video_id,
                )
            )
            return pending_unseen[0].video_id

        pending_seen = [entry for entry in self.entries.values() if entry.status == STATUS_PENDING]
        if pending_seen:
            pending_seen.sort(key=lambda entry: (entry.seen_order or float("inf"), entry.discovered_order, entry.video_id))
            return pending_seen[0].video_id

        ordered = self.ordered_video_ids()
        return ordered[-1] if ordered else None

    def mark_viewed(self, video_id: str) -> VideoReviewState:
        entry = self.entries[video_id]
        if entry.seen_order is None:
            entry.seen_order = self._next_seen_order()
            self._write_log_row(entry)
        return entry

    def update_preview(self, video_id: str, preview: PreviewBundle) -> None:
        entry = self.entries[video_id]
        changed = (
            entry.preview_start != preview.preview_start
            or entry.preview_duration != preview.preview_duration
            or entry.motion_score != preview.motion_score
        )
        entry.preview_start = preview.preview_start
        entry.preview_duration = preview.preview_duration
        entry.motion_score = preview.motion_score
        if changed:
            self._write_log_row(entry)

    def set_error(self, video_id: str, error: str) -> None:
        entry = self.entries[video_id]
        entry.status = STATUS_ERROR
        entry.error = error
        self._write_log_row(entry)

    def accept(self, video_id: str) -> VideoReviewState:
        entry = self.entries[video_id]
        entry.current_path = self._move_to_destination(entry, self.keep_dir)
        entry.status = STATUS_ACCEPTED
        entry.error = ""
        self._write_log_row(entry)
        return entry

    def reject(self, video_id: str) -> VideoReviewState:
        entry = self.entries[video_id]
        entry.current_path = self._move_to_destination(entry, self.reject_dir)
        entry.status = STATUS_REJECTED
        entry.error = ""
        self._write_log_row(entry)
        return entry


class ReviewController:
    def __init__(self, session: ReviewSession) -> None:
        self.session = session
        self.current_video_id: str | None = None

    def refresh_queue(self) -> list[VideoReviewState]:
        discovered = self.session.scan_input_dir()
        if self.current_video_id is None:
            start_video_id = self.session.choose_start_video_id()
            if start_video_id is not None:
                self.current_video_id = start_video_id
                self.session.mark_viewed(start_video_id)
        return discovered

    def get_current_entry(self) -> VideoReviewState | None:
        if self.current_video_id is None:
            return None
        return self.session.entries.get(self.current_video_id)

    def get_visual_status(self) -> str:
        entry = self.get_current_entry()
        if entry is None:
            return STATUS_PENDING
        if entry.status == STATUS_PENDING and entry.is_seen:
            return STATUS_ACCEPTED
        return entry.status

    def _can_auto_accept(self, entry: VideoReviewState | None) -> bool:
        return entry is not None and entry.status == STATUS_PENDING and entry.is_seen

    def _move_to_neighbor(self, ordered_ids: list[str], target_index: int) -> VideoReviewState | None:
        if not ordered_ids:
            self.current_video_id = None
            return None
        self.current_video_id = ordered_ids[target_index]
        return self.session.mark_viewed(self.current_video_id)

    def navigate(self, delta: int) -> VideoReviewState | None:
        entry = self.get_current_entry()
        ordered_ids = self.session.ordered_video_ids()
        if entry is None or not ordered_ids:
            return entry

        current_index = ordered_ids.index(entry.video_id)
        target_index = max(0, min(len(ordered_ids) - 1, current_index + delta))
        if target_index == current_index:
            return entry

        if self._can_auto_accept(entry):
            self.session.accept(entry.video_id)
        return self._move_to_neighbor(ordered_ids, target_index)

    def advance_after_decision(self, video_id: str) -> VideoReviewState | None:
        ordered_ids = self.session.ordered_video_ids()
        if not ordered_ids:
            self.current_video_id = None
            return None

        current_index = ordered_ids.index(video_id)
        if current_index < len(ordered_ids) - 1:
            return self._move_to_neighbor(ordered_ids, current_index + 1)
        return self.session.entries[video_id]

    def accept_current(self) -> VideoReviewState | None:
        entry = self.get_current_entry()
        if entry is None:
            return None
        self.session.accept(entry.video_id)
        return self.advance_after_decision(entry.video_id)

    def reject_current(self) -> VideoReviewState | None:
        entry = self.get_current_entry()
        if entry is None:
            return None
        self.session.reject(entry.video_id)
        return self.advance_after_decision(entry.video_id)


class PreviewManager:
    def __init__(
        self,
        loader: Callable[[Path], PreviewBundle],
        max_workers: int = 2,
        cache_size: int = DEFAULT_PREVIEW_CACHE_SIZE,
    ) -> None:
        self.loader = loader
        self.max_workers = max_workers
        self.cache_size = max(1, cache_size)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.cache: OrderedDict[str, PreviewBundle] = OrderedDict()
        self.inflight: dict[str, PreviewTask] = {}
        self.pending: OrderedDict[str, Path] = OrderedDict()

    def close(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)

    def get_cached(self, video_id: str) -> PreviewBundle | None:
        cached = self.cache.get(video_id)
        if cached is not None:
            self.cache.move_to_end(video_id)
        return cached

    def request(self, video_id: str, path: Path) -> None:
        self.request_many([(video_id, path)])

    def request_many(self, requests: list[tuple[str, Path]]) -> None:
        prioritized_pending: OrderedDict[str, Path] = OrderedDict()
        for video_id, path in requests:
            if video_id in self.cache:
                continue
            resolved_path = normalize_path(path)
            existing = self.inflight.get(video_id)
            if existing is not None and existing.path == resolved_path:
                continue
            prioritized_pending[video_id] = resolved_path

        self.pending = prioritized_pending
        self._fill_workers()

    def _fill_workers(self) -> None:
        while len(self.inflight) < self.max_workers:
            next_request = next(
                ((video_id, path) for video_id, path in self.pending.items() if video_id not in self.inflight),
                None,
            )
            if next_request is None:
                break
            video_id, path = next_request
            del self.pending[video_id]
            self.inflight[video_id] = PreviewTask(
                video_id=video_id,
                path=path,
                future=self.executor.submit(self.loader, path),
            )

    def _store_cache(self, video_id: str, bundle: PreviewBundle) -> None:
        self.cache[video_id] = bundle
        self.cache.move_to_end(video_id)
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)

    def drain_finished(self) -> list[ResolvedPreview]:
        resolved: list[ResolvedPreview] = []
        for video_id, task in list(self.inflight.items()):
            if not task.future.done():
                continue
            del self.inflight[video_id]
            exception = task.future.exception()
            if exception is not None:
                resolved.append(ResolvedPreview(video_id=video_id, path=task.path, error=exception))
                continue
            bundle = task.future.result()
            self._store_cache(video_id, bundle)
            resolved.append(ResolvedPreview(video_id=video_id, path=task.path, bundle=bundle))
        if resolved:
            self._fill_workers()
        return resolved


class VideoReviewApp:
    STATUS_COLORS = {
        STATUS_PENDING: "#6b7280",
        STATUS_ACCEPTED: "#166534",
        STATUS_REJECTED: "#991b1b",
        STATUS_ERROR: "#92400e",
    }

    def __init__(
        self,
        root: Any,
        controller: ReviewController,
        preview_manager: PreviewManager,
        watch: bool,
    ) -> None:
        self.root = root
        self.controller = controller
        self.preview_manager = preview_manager
        self.watch = watch
        self.current_tk_frames: list[Any] = []
        self.current_frame_index = 0
        self.current_animation_job: str | None = None
        self.rendered_video_id: str | None = None
        self.last_watch_scan_ms = 0

        self.root.title("Anime Video Review")
        self.root.geometry("920x760")
        self.root.minsize(760, 620)

        self.preview_label = tk.Label(
            self.root,
            text="Waiting for videos...",
            bg="#111827",
            fg="#f9fafb",
            width=80,
            height=24,
            anchor="center",
            justify="center",
        )
        self.preview_label.pack(fill="both", expand=True, padx=16, pady=(16, 12))

        self.status_badge = tk.Label(
            self.root,
            text=STATUS_PENDING.upper(),
            bg=self.STATUS_COLORS[STATUS_PENDING],
            fg="#ffffff",
            padx=12,
            pady=6,
            font=("Helvetica", 14, "bold"),
        )
        self.status_badge.pack(anchor="w", padx=16)

        self.filename_label = tk.Label(self.root, text="", anchor="w", justify="left", font=("Helvetica", 15, "bold"))
        self.filename_label.pack(fill="x", padx=16, pady=(12, 4))

        self.folder_label = tk.Label(self.root, text="", anchor="w", justify="left")
        self.folder_label.pack(fill="x", padx=16)

        self.motion_label = tk.Label(self.root, text="", anchor="w", justify="left")
        self.motion_label.pack(fill="x", padx=16, pady=(4, 0))

        self.queue_label = tk.Label(self.root, text="", anchor="w", justify="left")
        self.queue_label.pack(fill="x", padx=16, pady=(4, 0))

        self.help_label = tk.Label(
            self.root,
            text="Right: next  Left: previous  A: accept  R: reject",
            anchor="w",
            justify="left",
            fg="#374151",
        )
        self.help_label.pack(fill="x", padx=16, pady=(8, 16))

        self.root.bind("<Right>", self._on_next)
        self.root.bind("<Left>", self._on_previous)
        self.root.bind("<KeyPress-a>", self._on_accept)
        self.root.bind("<KeyPress-A>", self._on_accept)
        self.root.bind("<KeyPress-r>", self._on_reject)
        self.root.bind("<KeyPress-R>", self._on_reject)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.controller.refresh_queue()
        self._refresh_view()
        self.root.after(DEFAULT_UI_POLL_MS, self._poll)

    def close(self) -> None:
        if self.current_animation_job is not None:
            self.root.after_cancel(self.current_animation_job)
            self.current_animation_job = None
        self.preview_manager.close()
        self.root.destroy()

    def _candidate_preview_ids(self) -> list[str]:
        ordered_ids = self.controller.session.ordered_video_ids()
        current_entry = self.controller.get_current_entry()
        current_video_id = current_entry.video_id if current_entry is not None else None
        preload_limit = max(1, self.preview_manager.cache_size)
        preload_ids = [
            video_id
            for video_id in ordered_ids
            if self.controller.session.entries[video_id].status != STATUS_ERROR
        ]
        if current_video_id is None or current_video_id not in preload_ids:
            return preload_ids[:preload_limit]

        current_index = preload_ids.index(current_video_id)
        prioritized = [current_video_id]
        for offset in range(1, len(preload_ids)):
            next_index = current_index + offset
            previous_index = current_index - offset
            if next_index < len(preload_ids):
                prioritized.append(preload_ids[next_index])
                if len(prioritized) >= preload_limit:
                    break
            if previous_index >= 0:
                prioritized.append(preload_ids[previous_index])
                if len(prioritized) >= preload_limit:
                    break
        return prioritized[:preload_limit]

    def _ensure_preview_requests(self) -> None:
        requests = [
            (video_id, self.controller.session.entries[video_id].current_path)
            for video_id in self._candidate_preview_ids()
        ]
        if requests:
            self.preview_manager.request_many(requests)

    def _drain_preview_updates(self) -> bool:
        changed = False
        for resolved in self.preview_manager.drain_finished():
            entry = self.controller.session.entries.get(resolved.video_id)
            if entry is None:
                continue
            if resolved.error is not None:
                if normalize_path(entry.current_path) != resolved.path:
                    self.preview_manager.request(entry.video_id, entry.current_path)
                    continue
                self.controller.session.set_error(entry.video_id, str(resolved.error))
                changed = True
                continue
            assert resolved.bundle is not None
            self.controller.session.update_preview(entry.video_id, resolved.bundle)
            changed = True
        return changed

    def _render_preview(self, video_id: str, preview: PreviewBundle) -> None:
        if ImageTk is None:
            raise RuntimeError("Pillow ImageTk support is unavailable")
        if self.current_animation_job is not None:
            self.root.after_cancel(self.current_animation_job)
            self.current_animation_job = None

        self.rendered_video_id = video_id
        self.current_tk_frames = [ImageTk.PhotoImage(frame) for frame in preview.frames]
        self.current_frame_index = 0
        self.preview_label.config(image=self.current_tk_frames[0], text="")
        self._animate_preview(preview.frame_delay_ms)

    def _animate_preview(self, delay_ms: int) -> None:
        if not self.current_tk_frames:
            return
        self.current_frame_index = (self.current_frame_index + 1) % len(self.current_tk_frames)
        self.preview_label.config(image=self.current_tk_frames[self.current_frame_index])
        self.current_animation_job = self.root.after(delay_ms, self._animate_preview, delay_ms)

    def _set_preview_placeholder(self, message: str, background: str = "#111827") -> None:
        if self.current_animation_job is not None:
            self.root.after_cancel(self.current_animation_job)
            self.current_animation_job = None
        self.current_tk_frames = []
        self.rendered_video_id = None
        self.preview_label.config(image="", text=message, bg=background, fg="#f9fafb")

    def _refresh_view(self) -> None:
        self._ensure_preview_requests()
        entry = self.controller.get_current_entry()
        if entry is None:
            self.status_badge.config(text=STATUS_PENDING.upper(), bg=self.STATUS_COLORS[STATUS_PENDING])
            self.filename_label.config(text="No videos in the review queue yet.")
            self.folder_label.config(text=str(self.controller.session.input_dir))
            self.motion_label.config(text="Waiting for new downloads...")
            self.queue_label.config(text="")
            self._set_preview_placeholder("Waiting for videos...")
            return

        visual_status = self.controller.get_visual_status()
        self.status_badge.config(text=visual_status.upper(), bg=self.STATUS_COLORS[visual_status])
        self.filename_label.config(text=entry.current_path.name)
        self.folder_label.config(text=f"Current path: {entry.current_path}")

        motion_text = "Motion score: pending"
        if entry.motion_score is not None:
            motion_text = f"Motion score: {entry.motion_score:.4f}"
        if entry.preview_start is not None and entry.preview_duration is not None:
            motion_text += f" | Preview: {entry.preview_start:.3f}s + {entry.preview_duration:.3f}s"
        self.motion_label.config(text=motion_text)

        ordered_ids = self.controller.session.ordered_video_ids()
        queue_index = ordered_ids.index(entry.video_id) + 1 if entry.video_id in ordered_ids else 1
        self.queue_label.config(text=f"Queue position: {queue_index}/{len(ordered_ids)}")

        if entry.status == STATUS_ERROR:
            error_text = entry.error or "Preview generation failed."
            self._set_preview_placeholder(error_text, background="#7c2d12")
            return

        cached_preview = self.preview_manager.get_cached(entry.video_id)
        if cached_preview is None:
            self._set_preview_placeholder("Loading preview...")
            return

        if self.rendered_video_id != entry.video_id:
            self._render_preview(entry.video_id, cached_preview)

    def _move_or_refresh(self, action: Callable[[], VideoReviewState | None]) -> None:
        action()
        self._refresh_view()

    def _on_next(self, _event: Any) -> None:
        self._move_or_refresh(lambda: self.controller.navigate(1))

    def _on_previous(self, _event: Any) -> None:
        self._move_or_refresh(lambda: self.controller.navigate(-1))

    def _on_accept(self, _event: Any) -> None:
        self._move_or_refresh(self.controller.accept_current)

    def _on_reject(self, _event: Any) -> None:
        self._move_or_refresh(self.controller.reject_current)

    def _poll(self) -> None:
        changed = self._drain_preview_updates()
        if self.watch:
            self.last_watch_scan_ms += DEFAULT_UI_POLL_MS
            if self.last_watch_scan_ms >= DEFAULT_WATCH_POLL_MS:
                self.last_watch_scan_ms = 0
                discovered = self.controller.refresh_queue()
                changed = changed or bool(discovered)
        if changed:
            self._refresh_view()
        self._ensure_preview_requests()
        self.root.after(DEFAULT_UI_POLL_MS, self._poll)


def run(args: argparse.Namespace) -> int:
    require_runtime()

    session = ReviewSession(
        input_dir=args.input_dir,
        keep_dir=args.keep_dir,
        reject_dir=args.reject_dir,
        review_log=args.review_log,
    )
    session.load()
    controller = ReviewController(session)
    preview_manager = PreviewManager(
        loader=lambda path: extract_preview(path, preview_duration=args.preview_duration),
    )

    root = tk.Tk()
    VideoReviewApp(root=root, controller=controller, preview_manager=preview_manager, watch=args.watch)
    root.mainloop()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
