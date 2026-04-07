#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

try:
    import torch
except ModuleNotFoundError:
    torch = None

try:
    import torchcodec  # noqa: F401
except ModuleNotFoundError:
    torchcodec = None

try:
    from transformers import AutoModelForMultimodalLM, AutoProcessor
except ModuleNotFoundError:
    AutoModelForMultimodalLM = None
    AutoProcessor = None

VIDEO_EXTENSIONS = {".avi", ".mkv", ".mov", ".mp4", ".webm"}
DEFAULT_DEVICE = "cuda"
DEFAULT_DTYPE = "auto"
DEFAULT_NUM_FRAMES = 12
DEFAULT_MAX_NEW_TOKENS = 768
DEFAULT_OUTPUT_FILENAME = "captions.jsonl"
DEFAULT_TAGS_FILENAME = "tags.jsonl"
VIDEO_MAX_DURATION = 60.0
AUDIO_MAX_DURATION = 30.0
RATING_TAGS = {"general", "sensitive", "questionable", "explicit"}
MAX_GENERAL_HELPER_TAGS = 16
MAX_CHARACTER_HELPER_TAGS = 8
SECTION_SCENE_OVERVIEW = "SCENE_OVERVIEW"
SECTION_VISUAL_DETAILS = "VISUAL_DETAILS"
SECTION_DIALOGUE = "DIALOGUE"
SECTION_OTHER_SOUNDS = "OTHER_SOUNDS"
LEAKED_SECTION_LABELS = (
    SECTION_SCENE_OVERVIEW,
    SECTION_VISUAL_DETAILS,
    SECTION_DIALOGUE,
    SECTION_OTHER_SOUNDS,
    "VISUAL",
    "AUDIO",
)
LEAKED_SECTION_PATTERN = re.compile(
    rf"^\s*(?P<label>{'|'.join(re.escape(label) for label in LEAKED_SECTION_LABELS)})\s*:?\s*(?P<body>.*)$"
)
BULLET_PREFIX_PATTERN = re.compile(r"^(?:[-*+]\s+|\d+\.\s+)")
MARKDOWN_FENCE_PATTERN = re.compile(r"^`{3,}\s*$")
WHITESPACE_PATTERN = re.compile(r"\s+")


@dataclass(frozen=True)
class VideoMetadata:
    duration: float
    has_audio_track: bool


@dataclass(frozen=True)
class TagHints:
    general_tags: tuple[str, ...]
    character_tags: tuple[str, ...]


@dataclass(frozen=True)
class TagHintLookup:
    by_clip_path: dict[str, TagHints]
    by_clip_relpath: dict[str, TagHints]


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Caption short video clips with a Gemma 4 multimodal model hosted on "
            "Hugging Face."
        )
    )
    parser.add_argument("--input-dir", required=True, type=Path, help="Directory containing video clips.")
    parser.add_argument(
        "--model-repo",
        required=True,
        help="Hugging Face repo for the Gemma 4 multimodal model to use.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        help=f"JSONL manifest path. Defaults to <input-dir>/{DEFAULT_OUTPUT_FILENAME}.",
    )
    parser.add_argument(
        "--tags-file",
        type=Path,
        help=(
            "Optional JSONL tag manifest to use as helper scene hints. "
            f"Defaults to <input-dir>/{DEFAULT_TAGS_FILENAME} when that file exists."
        ),
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default=DEFAULT_DEVICE,
        help=f"Device to use for inference. Defaults to {DEFAULT_DEVICE}.",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default=DEFAULT_DTYPE,
        help=f"Model dtype hint. Defaults to {DEFAULT_DTYPE}.",
    )
    parser.add_argument(
        "--num-frames",
        type=positive_int,
        default=DEFAULT_NUM_FRAMES,
        help=f"Number of sampled frames per clip. Defaults to {DEFAULT_NUM_FRAMES}.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=positive_int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help=f"Maximum generated tokens per clip. Defaults to {DEFAULT_MAX_NEW_TOKENS}.",
    )
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not args.input_dir.exists() or not args.input_dir.is_dir():
        parser.error("--input-dir must exist and be a directory")

    if args.output_file is None:
        args.output_file = args.input_dir / DEFAULT_OUTPUT_FILENAME

    if args.output_file.exists() and args.output_file.is_dir():
        parser.error("--output-file must be a file path")

    tags_file = getattr(args, "tags_file", None)
    if tags_file is None:
        default_tags_file = args.input_dir / DEFAULT_TAGS_FILENAME
        if default_tags_file.exists():
            tags_file = default_tags_file
    args.tags_file = tags_file

    if args.tags_file is not None and (not args.tags_file.exists() or not args.tags_file.is_file()):
        parser.error("--tags-file must exist and be a file")


def collect_video_files(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def normalize_lookup_path(path: Path) -> str:
    return str(path.expanduser().resolve(strict=False))


def normalize_tag_list(values: Any) -> tuple[str, ...]:
    if not isinstance(values, list):
        return ()

    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return tuple(ordered)


def extract_tag_hints(record: dict[str, Any]) -> TagHints:
    general_tags = normalize_tag_list(record.get("general_tags"))
    character_tags = normalize_tag_list(record.get("character_tags"))
    if not general_tags and not character_tags:
        general_tags = tuple(
            tag for tag in normalize_tag_list(record.get("all_tags")) if tag.lower() not in RATING_TAGS
        )

    return TagHints(
        general_tags=general_tags[:MAX_GENERAL_HELPER_TAGS],
        character_tags=character_tags[:MAX_CHARACTER_HELPER_TAGS],
    )


def load_tag_hints(tags_file: Path, input_dir: Path) -> TagHintLookup:
    normalized_input_dir = input_dir.expanduser().resolve(strict=False)
    by_clip_path: dict[str, TagHints] = {}
    by_clip_relpath: dict[str, TagHints] = {}

    with tags_file.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{tags_file}:{line_number} contains invalid JSON") from exc

            if not isinstance(payload, dict):
                raise ValueError(f"{tags_file}:{line_number} must contain a JSON object")
            if payload.get("status") != "tagged":
                continue

            tag_hints = extract_tag_hints(payload)
            if not tag_hints.general_tags and not tag_hints.character_tags:
                continue

            clip_path_value = payload.get("clip_path")
            normalized_clip_path: Path | None = None
            if isinstance(clip_path_value, str) and clip_path_value.strip():
                normalized_clip_path = Path(clip_path_value.strip())
                if not normalized_clip_path.is_absolute():
                    normalized_clip_path = normalized_input_dir / normalized_clip_path
                by_clip_path[normalize_lookup_path(normalized_clip_path)] = tag_hints

            clip_relpath_value = payload.get("clip_relpath")
            if isinstance(clip_relpath_value, str) and clip_relpath_value.strip():
                by_clip_relpath[Path(clip_relpath_value.strip()).as_posix()] = tag_hints
            elif normalized_clip_path is not None:
                try:
                    inferred_relpath = normalized_clip_path.relative_to(normalized_input_dir).as_posix()
                except ValueError:
                    continue
                by_clip_relpath[inferred_relpath] = tag_hints

    return TagHintLookup(by_clip_path=by_clip_path, by_clip_relpath=by_clip_relpath)


def find_tag_hints(tag_lookup: TagHintLookup | None, clip_path: Path, input_dir: Path) -> TagHints | None:
    if tag_lookup is None:
        return None

    direct_match = tag_lookup.by_clip_path.get(normalize_lookup_path(clip_path))
    if direct_match is not None:
        return direct_match

    try:
        clip_relpath = clip_path.expanduser().resolve(strict=False).relative_to(
            input_dir.expanduser().resolve(strict=False)
        )
    except ValueError:
        return None

    return tag_lookup.by_clip_relpath.get(clip_relpath.as_posix())


def require_runtime(device: str) -> None:
    if shutil.which("ffprobe") is None:
        raise SystemExit(
            "Missing required system tool: ffprobe. Install FFmpeg so ffprobe is available on PATH."
        )
    if torch is None or AutoProcessor is None or AutoModelForMultimodalLM is None or torchcodec is None:
        raise SystemExit(
            "Missing dependencies for captioning. Install them with: "
            "python -m pip install -r requirements.txt"
        )
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA device requested but torch.cuda.is_available() is false.")


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

    has_audio_track = any(stream.get("codec_type") == "audio" for stream in streams)
    return VideoMetadata(duration=duration, has_audio_track=has_audio_track)


def resolve_torch_dtype(dtype_name: str) -> Any:
    if dtype_name == "auto":
        return "auto"
    assert torch is not None
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[dtype_name]


def read_config_value(config: Any, name: str) -> Any:
    if isinstance(config, dict):
        return config.get(name)
    return getattr(config, name, None)


def model_supports_audio(model: Any) -> bool:
    config = getattr(model, "config", model)
    for attribute in (
        "audio_config",
        "audio_encoder_config",
        "audio_tower_config",
        "audio_backbone_config",
        "speech_config",
    ):
        if read_config_value(config, attribute) is not None:
            return True

    supported_modalities = read_config_value(config, "supported_modalities")
    if supported_modalities is None:
        supported_modalities = read_config_value(config, "modalities")

    if supported_modalities is None:
        return False

    if isinstance(supported_modalities, str):
        values = [supported_modalities]
    else:
        values = list(supported_modalities)
    return any(str(value).lower() == "audio" for value in values)


def load_captioning_components(model_repo: str, device: str, dtype_name: str) -> tuple[Any, Any, bool]:
    assert AutoProcessor is not None
    assert AutoModelForMultimodalLM is not None
    processor = AutoProcessor.from_pretrained(model_repo)
    model = AutoModelForMultimodalLM.from_pretrained(
        model_repo,
        torch_dtype=resolve_torch_dtype(dtype_name),
    )
    if hasattr(model, "to"):
        model = model.to(device)
    if hasattr(model, "eval"):
        model.eval()
    return processor, model, model_supports_audio(model)


def determine_audio_plan(metadata: VideoMetadata, audio_supported: bool) -> tuple[bool, str | None]:
    if not metadata.has_audio_track:
        return False, "clip has no audio track"
    if not audio_supported:
        return False, "model does not support audio input"
    if metadata.duration > AUDIO_MAX_DURATION:
        return False, f"clip audio exceeds {AUDIO_MAX_DURATION:.0f}s limit"
    return True, None


def build_prompt(include_audio: bool, tag_hints: TagHints | None = None) -> str:
    lines = [
        "Caption this single short video clip.",
        "Write exactly one plain paragraph of continuous text.",
        (
            "Do not use labels like SCENE_OVERVIEW, VISUAL_DETAILS, DIALOGUE, OTHER_SOUNDS, "
            "VISUAL, or AUDIO."
        ),
        "Do not add any prefatory text, bullet points, markdown fences, or extra formatting.",
    ]

    if tag_hints is not None and (tag_hints.general_tags or tag_hints.character_tags):
        lines.extend(
            [
                "",
                "Helper scene hints from a separate tagger are provided below.",
                "Use them only as soft guidance. Trust the actual video first and ignore any hint that does not match what you can see.",
            ]
        )
        if tag_hints.character_tags:
            lines.append(f"Character hints: {', '.join(tag_hints.character_tags)}")
        if tag_hints.general_tags:
            lines.append(f"Visual hints: {', '.join(tag_hints.general_tags)}")

    lines.extend(
        [
            "",
            (
                "Describe the visible scene factually: the main subjects, the setting, actions in order, "
                "camera framing and movement, lighting, colors, overall visual style, and if applicable the "
                "animation style such as anime, held keyframes, limited animation, smears, or similar cues."
            ),
        ]
    )

    if include_audio:
        lines.extend(
            [
                "",
                (
                    "Blend any clearly audible spoken dialogue, music, ambience, crowd noise, or sound effects "
                    "naturally into the same paragraph when they are relevant."
                ),
            ]
        )
    else:
        lines.extend(
            [
                "",
                "Only describe visual content. Do not speculate about unheard dialogue or sounds.",
            ]
        )

    return "\n".join(lines)


def build_messages(clip_path: Path, include_audio: bool, tag_hints: TagHints | None = None) -> list[dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are a precise video captioning assistant. Keep the content factual and respond "
                        "with one plain paragraph of unstructured text."
                    ),
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "video", "video": str(clip_path)},
                {"type": "text", "text": build_prompt(include_audio, tag_hints=tag_hints)},
            ],
        },
    ]


def prepare_inputs(processor: Any, messages: list[dict[str, Any]], include_audio: bool, num_frames: int) -> Any:
    kwargs = {
        "tokenize": True,
        "return_dict": True,
        "return_tensors": "pt",
        "add_generation_prompt": True,
        "enable_thinking": False,
        "do_sample_frames": True,
        "num_frames": num_frames,
        "load_audio_from_video": include_audio,
    }
    try:
        return processor.apply_chat_template(messages, **kwargs)
    except TypeError as exc:
        if "enable_thinking" not in str(exc):
            raise
        kwargs.pop("enable_thinking")
        return processor.apply_chat_template(messages, **kwargs)


def get_model_device(model: Any) -> Any:
    device = getattr(model, "device", None)
    if device is not None:
        return device

    parameters = getattr(model, "parameters", None)
    if callable(parameters):
        try:
            first_parameter = next(parameters())
        except (StopIteration, TypeError):
            first_parameter = None
        if first_parameter is not None:
            return first_parameter.device

    return "cpu"


def get_input_length(inputs: Any) -> int:
    input_ids = inputs["input_ids"]
    shape = getattr(input_ids, "shape", None)
    if shape is not None:
        return int(shape[-1])
    if input_ids and isinstance(input_ids[0], (list, tuple)):
        return len(input_ids[0])
    return len(input_ids)


def decode_generated_text(processor: Any, generated_ids: Any) -> str:
    if hasattr(processor, "decode"):
        return processor.decode(generated_ids, skip_special_tokens=True)
    if hasattr(processor, "batch_decode"):
        return processor.batch_decode([generated_ids], skip_special_tokens=True)[0]
    raise TypeError("Processor does not support decode() or batch_decode()")


def generate_caption_text(
    processor: Any,
    model: Any,
    clip_path: Path,
    include_audio: bool,
    num_frames: int,
    max_new_tokens: int,
    tag_hints: TagHints | None = None,
) -> str:
    messages = build_messages(clip_path, include_audio, tag_hints=tag_hints)
    inputs = prepare_inputs(processor, messages, include_audio=include_audio, num_frames=num_frames)
    if hasattr(inputs, "to"):
        inputs = inputs.to(get_model_device(model))

    generated = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    input_length = get_input_length(inputs)
    generated_text_ids = generated[0][input_length:]
    return decode_generated_text(processor, generated_text_ids)


def normalize_caption_line(raw_line: str) -> str:
    line = raw_line.strip()
    if not line or MARKDOWN_FENCE_PATTERN.fullmatch(line):
        return ""

    label_match = LEAKED_SECTION_PATTERN.fullmatch(line)
    if label_match is not None:
        line = label_match.group("body").strip()
    line = BULLET_PREFIX_PATTERN.sub("", line).strip()
    return line.replace("`", "")


def normalize_caption_text(text: str) -> str:
    cleaned_lines = [normalize_caption_line(raw_line) for raw_line in text.splitlines()]
    cleaned_parts = [line for line in cleaned_lines if line]
    cleaned_text = WHITESPACE_PATTERN.sub(" ", " ".join(cleaned_parts)).strip()
    if not cleaned_text:
        raise ValueError("Model response produced an empty caption after normalization")
    return cleaned_text


def make_manifest_record(clip_path: Path, model_repo: str) -> dict[str, Any]:
    return {
        "clip_path": str(clip_path),
        "model_repo": model_repo,
        "duration": None,
        "has_audio_track": None,
        "used_audio": False,
        "audio_skip_reason": None,
        "scene_overview": None,
        "visual_details": None,
        "dialogue": None,
        "other_sounds": None,
        "caption": None,
        "status": "failed",
        "error": "",
    }


def process_video(
    processor: Any,
    model: Any,
    clip_path: Path,
    model_repo: str,
    num_frames: int,
    max_new_tokens: int,
    audio_supported: bool,
    tag_hints: TagHints | None = None,
) -> dict[str, Any]:
    record = make_manifest_record(clip_path, model_repo)
    try:
        metadata = probe_video(clip_path)
        record["duration"] = metadata.duration
        record["has_audio_track"] = metadata.has_audio_track

        if metadata.duration > VIDEO_MAX_DURATION:
            record["error"] = f"clip duration exceeds {VIDEO_MAX_DURATION:.0f}s video limit"
            return record

        used_audio, audio_skip_reason = determine_audio_plan(metadata, audio_supported=audio_supported)
        record["used_audio"] = used_audio
        record["audio_skip_reason"] = audio_skip_reason

        raw_caption = generate_caption_text(
            processor=processor,
            model=model,
            clip_path=clip_path,
            include_audio=used_audio,
            num_frames=num_frames,
            max_new_tokens=max_new_tokens,
            tag_hints=tag_hints,
        )
        record.update(
            {
                "caption": normalize_caption_text(raw_caption),
                "status": "captioned",
                "error": "",
            }
        )
    except Exception as exc:
        record["error"] = str(exc)

    return record


def write_manifest_row(handle: TextIO, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record))
    handle.write("\n")
    handle.flush()


def summarize_results(
    total_videos: int,
    captioned_videos: int,
    failed_videos: int,
    output_file: Path,
) -> str:
    return " ".join(
        [
            f"videos={total_videos}",
            f"captioned={captioned_videos}",
            f"failed={failed_videos}",
            f"manifest={output_file}",
        ]
    )


def run(args: argparse.Namespace) -> int:
    videos = collect_video_files(args.input_dir)
    if not videos:
        print(f"No supported videos found in {args.input_dir}", file=sys.stderr)
        return 1

    require_runtime(args.device)
    processor, model, audio_supported = load_captioning_components(
        model_repo=args.model_repo,
        device=args.device,
        dtype_name=args.dtype,
    )
    tags_file = getattr(args, "tags_file", None)
    if tags_file is None:
        default_tags_file = args.input_dir / DEFAULT_TAGS_FILENAME
        if default_tags_file.exists():
            tags_file = default_tags_file
    tag_lookup = load_tag_hints(tags_file, args.input_dir) if tags_file is not None else None

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    captioned_videos = 0
    failed_videos = 0

    with args.output_file.open("w", encoding="utf-8") as manifest_handle:
        for clip_path in videos:
            record = process_video(
                processor=processor,
                model=model,
                clip_path=clip_path,
                model_repo=args.model_repo,
                num_frames=args.num_frames,
                max_new_tokens=args.max_new_tokens,
                audio_supported=audio_supported,
                tag_hints=find_tag_hints(tag_lookup, clip_path=clip_path, input_dir=args.input_dir),
            )
            write_manifest_row(manifest_handle, record)

            if record["status"] == "captioned":
                captioned_videos += 1
                print(f"captioned: {clip_path} audio={record['used_audio']}")
            else:
                failed_videos += 1
                print(f"failed: {clip_path} ({record['error']})", file=sys.stderr)

    print(
        summarize_results(
            total_videos=len(videos),
            captioned_videos=captioned_videos,
            failed_videos=failed_videos,
            output_file=args.output_file,
        )
    )
    return 1 if failed_videos > 0 or captioned_videos == 0 else 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
