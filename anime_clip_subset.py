#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable, TextIO

VIDEO_EXTENSIONS = {".avi", ".mkv", ".mov", ".mp4", ".webm"}
DEFAULT_TAGS_FILENAME = "tags.jsonl"
DEFAULT_SOURCE_MANIFEST_FILENAME = "manifest.jsonl"
DEFAULT_OUTPUT_SUFFIX = "_subset"
TRAIN_SPLIT_NAME = "train"
VALIDATION_SPLIT_NAME = "val"


@dataclass(frozen=True)
class TaggedClip:
    clip_path: Path
    clip_relpath: str
    source_id: str
    tags: tuple[str, ...]
    record: dict[str, Any]
    tag_set: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if not self.tag_set:
            object.__setattr__(self, "tag_set", frozenset(self.tags))


@dataclass(frozen=True)
class SelectionResult:
    tagged_clip: TaggedClip
    selection_rank: int
    new_tags_added: int
    balance_penalty: int
    distribution_distance: float | None = None


@dataclass(frozen=True)
class SourceStats:
    source_id: str
    clips: tuple[TaggedClip, ...]
    clip_count: int
    tag_counts: Counter[str]
    unique_tags: frozenset[str]
    tag_mass: int


@dataclass(frozen=True)
class PreparedSelectionData:
    source_stats_by_id: dict[str, SourceStats]
    source_ids: tuple[str, ...]
    global_tag_counts: Counter[str]
    global_tag_mass: int


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def positive_fraction(value: str) -> float:
    parsed = float(value)
    if parsed <= 0 or parsed > 1:
        raise argparse.ArgumentTypeError("value must be greater than 0 and less than or equal to 1")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create source-disjoint train/val clip splits from a wd-tagger manifest by "
            "keeping val diverse while matching a balanced train tag distribution."
        )
    )
    parser.add_argument("--input-dir", required=True, type=Path, help="Directory containing the tagged clips.")
    parser.add_argument(
        "--tags-file",
        type=Path,
        help=f"JSONL tag manifest path. Defaults to <input-dir>/{DEFAULT_TAGS_FILENAME}.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=f"Split output directory. Defaults to a sibling named <input-dir>{DEFAULT_OUTPUT_SUFFIX}.",
    )
    parser.add_argument(
        "--train-count",
        type=positive_int,
        help="Exact number of tagged clips to copy into the train split. Defaults to all eligible remaining clips.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--count", type=positive_int, help="Exact number of tagged clips to copy into the val split.")
    group.add_argument(
        "--fraction",
        type=positive_fraction,
        help="Fraction of successfully tagged clips to include in the val split.",
    )
    return parser


def is_relative_to(path: Path, other: Path) -> bool:
    try:
        path.relative_to(other)
    except ValueError:
        return False
    return True


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not args.input_dir.exists() or not args.input_dir.is_dir():
        parser.error("--input-dir must exist and be a directory")

    if args.tags_file is None:
        args.tags_file = args.input_dir / DEFAULT_TAGS_FILENAME
    if not args.tags_file.exists() or not args.tags_file.is_file():
        parser.error("--tags-file must exist and be a file")

    if args.output_dir is None:
        args.output_dir = args.input_dir.with_name(f"{args.input_dir.name}{DEFAULT_OUTPUT_SUFFIX}")
    if args.output_dir.exists() and args.output_dir.is_file():
        parser.error("--output-dir must be a directory path")

    resolved_input = args.input_dir.resolve()
    resolved_output = args.output_dir.resolve()
    if resolved_output == resolved_input:
        parser.error("--output-dir must be different from --input-dir")
    if is_relative_to(resolved_output, resolved_input):
        parser.error("--output-dir must not be inside --input-dir")


def collect_video_files(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def load_manifest_rows(tags_file: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with tags_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_source_lookup(input_dir: Path) -> dict[str, str]:
    manifest_path = input_dir / DEFAULT_SOURCE_MANIFEST_FILENAME
    if not manifest_path.exists():
        return {}

    source_lookup: dict[str, str] = {}
    for record in load_manifest_rows(manifest_path):
        source_path = record.get("source_path")
        if not isinstance(source_path, str) or not source_path.strip():
            continue
        clip_relpath = resolve_clip_relpath(record, input_dir)
        source_lookup[clip_relpath] = source_path.strip()
    return source_lookup


def dedupe_tags(tags: Iterable[Any]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for tag in tags:
        text = str(tag).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return tuple(ordered)


def extract_all_tags(record: dict[str, Any]) -> tuple[str, ...]:
    all_tags = record.get("all_tags")
    if isinstance(all_tags, list):
        return dedupe_tags(all_tags)

    tags: list[Any] = []
    rating = record.get("rating")
    if isinstance(rating, str) and rating.strip():
        tags.append(rating)
    general_tags = record.get("general_tags")
    if isinstance(general_tags, list):
        tags.extend(general_tags)
    character_tags = record.get("character_tags")
    if isinstance(character_tags, list):
        tags.extend(character_tags)
    return dedupe_tags(tags)


def resolve_clip_relpath(record: dict[str, Any], input_dir: Path) -> str:
    clip_relpath = record.get("clip_relpath")
    if isinstance(clip_relpath, str) and clip_relpath.strip():
        return clip_relpath

    clip_path_text = record.get("clip_path")
    if not isinstance(clip_path_text, str) or not clip_path_text.strip():
        raise ValueError("record is missing clip_path")
    clip_path = Path(clip_path_text)
    if not clip_path.is_absolute():
        return str(clip_path)
    return str(clip_path.relative_to(input_dir))


def resolve_source_id(record: dict[str, Any], clip_relpath: str, source_lookup: dict[str, str]) -> str:
    source_path = record.get("source_path")
    if isinstance(source_path, str) and source_path.strip():
        return source_path.strip()
    if clip_relpath in source_lookup:
        return source_lookup[clip_relpath]
    return clip_relpath


def build_tagged_clip(
    record: dict[str, Any],
    input_dir: Path,
    source_lookup: dict[str, str],
) -> TaggedClip | None:
    if record.get("status") != "tagged":
        return None

    clip_relpath = resolve_clip_relpath(record, input_dir)
    clip_path = input_dir / clip_relpath
    if not clip_path.exists():
        raise FileNotFoundError(f"Tagged clip is missing on disk: {clip_path}")
    source_id = resolve_source_id(record, clip_relpath, source_lookup)
    normalized_record = dict(record)
    if "source_path" not in normalized_record and clip_relpath in source_lookup:
        normalized_record["source_path"] = source_lookup[clip_relpath]
    return TaggedClip(
        clip_path=clip_path,
        clip_relpath=clip_relpath,
        source_id=source_id,
        tags=extract_all_tags(record),
        record=normalized_record,
    )


def load_tagged_clips(tags_file: Path, input_dir: Path) -> list[TaggedClip]:
    source_lookup = load_source_lookup(input_dir)
    tagged_clips: list[TaggedClip] = []
    for record in load_manifest_rows(tags_file):
        tagged_clip = build_tagged_clip(record, input_dir, source_lookup)
        if tagged_clip is not None:
            tagged_clips.append(tagged_clip)
    return sorted(tagged_clips, key=lambda clip: clip.clip_relpath)


def compute_balance_penalty(tag_counts: Counter[str]) -> int:
    return sum(count * count for count in tag_counts.values())


def compute_selection_metrics(
    tagged_clip: TaggedClip,
    covered_tags: set[str],
    tag_counts: Counter[str],
    current_balance_penalty: int,
) -> tuple[int, int, int]:
    new_tags_added = 0
    overlap = 0
    delta_penalty = 0
    for tag in tagged_clip.tag_set:
        if tag in covered_tags:
            overlap += 1
        else:
            new_tags_added += 1
        delta_penalty += (2 * tag_counts[tag]) + 1
    return new_tags_added, current_balance_penalty + delta_penalty, overlap


def build_tag_counts(tagged_clips: Iterable[TaggedClip]) -> Counter[str]:
    tag_counts: Counter[str] = Counter()
    for tagged_clip in tagged_clips:
        for tag in tagged_clip.tag_set:
            tag_counts[tag] += 1
    return tag_counts


def compute_distribution_distance_to_target(
    val_tag_counts: Counter[str],
    val_total: int,
    target_total_mass: float,
    target_frequency_for_tag: Callable[[str], float],
) -> float:
    if val_total <= 0:
        return float("inf")

    inverse_total = 1.0 / val_total
    distance = target_total_mass
    for tag, count in val_tag_counts.items():
        target_frequency = target_frequency_for_tag(tag)
        distance += abs((count * inverse_total) - target_frequency) - target_frequency
    return distance


def compute_distribution_distance_after_adding_clip(
    current_val_tag_counts: Counter[str],
    current_val_count: int,
    tagged_clip: TaggedClip,
    target_total_mass: float,
    target_frequency_for_tag: Callable[[str], float],
) -> float:
    prospective_total = current_val_count + 1
    if prospective_total <= 0:
        return float("inf")

    distance = target_total_mass
    inverse_total = 1.0 / prospective_total
    clip_tags = tagged_clip.tag_set
    seen_tags: set[str] = set()
    for tag, count in current_val_tag_counts.items():
        prospective_count = count + (1 if tag in clip_tags else 0)
        target_frequency = target_frequency_for_tag(tag)
        distance += abs((prospective_count * inverse_total) - target_frequency) - target_frequency
        seen_tags.add(tag)
    for tag in clip_tags:
        if tag in seen_tags:
            continue
        target_frequency = target_frequency_for_tag(tag)
        distance += abs(inverse_total - target_frequency) - target_frequency
    return distance


def compute_tag_distribution_distance(val_clips: list[TaggedClip], train_clips: list[TaggedClip]) -> float:
    if not val_clips or not train_clips:
        return float("inf")

    train_tag_counts = build_tag_counts(train_clips)
    train_total = len(train_clips)
    inverse_train_total = 1.0 / train_total
    return compute_distribution_distance_to_target(
        val_tag_counts=build_tag_counts(val_clips),
        val_total=len(val_clips),
        target_total_mass=sum(train_tag_counts.values()) * inverse_train_total,
        target_frequency_for_tag=lambda tag: train_tag_counts.get(tag, 0) * inverse_train_total,
    )


def compute_unique_tag_count(tagged_clips: Iterable[TaggedClip]) -> int:
    return len({tag for tagged_clip in tagged_clips for tag in tagged_clip.tag_set})


def select_balanced_subset(tagged_clips: list[TaggedClip], target_count: int) -> list[SelectionResult]:
    if target_count > len(tagged_clips):
        raise ValueError(
            f"requested balanced subset size ({target_count}) exceeds available clips ({len(tagged_clips)})"
        )
    remaining = list(tagged_clips)
    covered_tags: set[str] = set()
    tag_counts: Counter[str] = Counter()
    selections: list[SelectionResult] = []
    current_balance_penalty = 0

    for rank in range(1, target_count + 1):
        best_clip: TaggedClip | None = None
        best_metrics: tuple[int, int, int] | None = None
        best_sort_key: tuple[int, int, int, str] | None = None

        for tagged_clip in remaining:
            metrics = compute_selection_metrics(
                tagged_clip,
                covered_tags=covered_tags,
                tag_counts=tag_counts,
                current_balance_penalty=current_balance_penalty,
            )
            sort_key = (-metrics[0], metrics[1], metrics[2], tagged_clip.clip_relpath)
            if best_sort_key is None or sort_key < best_sort_key:
                best_clip = tagged_clip
                best_metrics = metrics
                best_sort_key = sort_key

        assert best_clip is not None
        assert best_metrics is not None
        selections.append(
            SelectionResult(
                tagged_clip=best_clip,
                selection_rank=rank,
                new_tags_added=best_metrics[0],
                balance_penalty=best_metrics[1],
            )
        )
        remaining.remove(best_clip)
        current_balance_penalty = best_metrics[1]
        covered_tags.update(best_clip.tag_set)
        for tag in best_clip.tag_set:
            tag_counts[tag] += 1

    return selections


def build_source_stats(tagged_clips: list[TaggedClip]) -> PreparedSelectionData:
    clips_by_source: dict[str, list[TaggedClip]] = {}
    for tagged_clip in tagged_clips:
        clips_by_source.setdefault(tagged_clip.source_id, []).append(tagged_clip)

    global_tag_counts = build_tag_counts(tagged_clips)
    source_stats_by_id: dict[str, SourceStats] = {}
    for source_id, clips in clips_by_source.items():
        source_tag_counts = build_tag_counts(clips)
        source_stats_by_id[source_id] = SourceStats(
            source_id=source_id,
            clips=tuple(clips),
            clip_count=len(clips),
            tag_counts=source_tag_counts,
            unique_tags=frozenset(source_tag_counts),
            tag_mass=sum(source_tag_counts.values()),
        )

    return PreparedSelectionData(
        source_stats_by_id=source_stats_by_id,
        source_ids=tuple(sorted(source_stats_by_id)),
        global_tag_counts=global_tag_counts,
        global_tag_mass=sum(global_tag_counts.values()),
    )


def subtract_tag_counts(current_counts: Counter[str], counts_to_remove: Counter[str]) -> None:
    for tag, count in counts_to_remove.items():
        next_count = current_counts.get(tag, 0) - count
        if next_count > 0:
            current_counts[tag] = next_count
            continue
        current_counts.pop(tag, None)


def compute_fixed_target_source_costs(
    selection_data: PreparedSelectionData,
    train_target_count: int,
) -> tuple[float, Callable[[str], float], dict[str, tuple[int, float, str]]]:
    draft_train_selections = select_balanced_subset(
        [clip for source_id in selection_data.source_ids for clip in selection_data.source_stats_by_id[source_id].clips],
        target_count=train_target_count,
    )
    draft_train_clips = [selection.tagged_clip for selection in draft_train_selections]
    draft_train_tag_counts = build_tag_counts(draft_train_clips)
    inverse_total = 1.0 / train_target_count
    target_frequencies = {tag: count * inverse_total for tag, count in draft_train_tag_counts.items()}
    draft_train_hits_by_source = Counter(tagged_clip.source_id for tagged_clip in draft_train_clips)
    source_open_costs = {
        source_id: (
            draft_train_hits_by_source[source_id],
            sum(target_frequencies.get(tag, 0.0) for tag in source_stats.unique_tags),
            source_id,
        )
        for source_id, source_stats in selection_data.source_stats_by_id.items()
    }
    return (
        sum(target_frequencies.values()),
        lambda tag: target_frequencies.get(tag, 0.0),
        source_open_costs,
    )


def score_validation_candidate(
    tagged_clip: TaggedClip,
    covered_tags: set[str],
    val_tag_counts: Counter[str],
    current_balance_penalty: int,
    current_val_count: int,
    target_total_mass: float,
    target_frequency_for_tag: Callable[[str], float],
    source_open_cost: tuple[int, float, str],
    is_new_source: bool,
) -> tuple[tuple[float, int, float, str, int, int, int, int, str], tuple[int, int]]:
    new_tags_added, next_balance_penalty, overlap = compute_selection_metrics(
        tagged_clip,
        covered_tags=covered_tags,
        tag_counts=val_tag_counts,
        current_balance_penalty=current_balance_penalty,
    )
    distribution_distance = compute_distribution_distance_after_adding_clip(
        current_val_tag_counts=val_tag_counts,
        current_val_count=current_val_count,
        tagged_clip=tagged_clip,
        target_total_mass=target_total_mass,
        target_frequency_for_tag=target_frequency_for_tag,
    )
    sort_key = (
        distribution_distance,
        source_open_cost[0],
        source_open_cost[1],
        source_open_cost[2],
        -new_tags_added,
        next_balance_penalty,
        1 if is_new_source else 0,
        overlap,
        tagged_clip.clip_relpath,
    )
    return sort_key, (new_tags_added, next_balance_penalty)


def select_validation_subset(
    tagged_clips: list[TaggedClip],
    target_count: int,
    train_target_count: int | None = None,
) -> list[SelectionResult]:
    if target_count > len(tagged_clips):
        raise ValueError(f"requested val size ({target_count}) exceeds available clips ({len(tagged_clips)})")

    selection_data = build_source_stats(tagged_clips)
    remaining_clips_by_source = {
        source_id: list(selection_data.source_stats_by_id[source_id].clips)
        for source_id in selection_data.source_ids
    }
    opened_source_ids: set[str] = set()
    covered_tags: set[str] = set()
    val_tag_counts: Counter[str] = Counter()
    current_balance_penalty = 0
    current_val_count = 0
    current_remaining_train_clip_count = len(tagged_clips)
    current_remaining_train_tag_counts = Counter(selection_data.global_tag_counts)
    current_remaining_train_tag_mass = selection_data.global_tag_mass
    fixed_target_total_mass = 0.0
    fixed_target_frequency_for_tag: Callable[[str], float] | None = None
    fixed_source_open_costs: dict[str, tuple[int, float, str]] = {}
    selections: list[SelectionResult] = []

    if train_target_count is not None:
        (
            fixed_target_total_mass,
            fixed_target_frequency_for_tag,
            fixed_source_open_costs,
        ) = compute_fixed_target_source_costs(selection_data, train_target_count)

    for rank in range(1, target_count + 1):
        best_clip: TaggedClip | None = None
        best_metrics: tuple[int, int] | None = None
        best_sort_key: tuple[float, int, float, str, int, int, int, int, str] | None = None

        for source_id in selection_data.source_ids:
            remaining_clips = remaining_clips_by_source[source_id]
            if not remaining_clips:
                continue

            source_stats = selection_data.source_stats_by_id[source_id]
            is_new_source = source_id not in opened_source_ids

            if is_new_source:
                prospective_train_clip_count = current_remaining_train_clip_count - source_stats.clip_count
                if prospective_train_clip_count <= 0:
                    continue
                if train_target_count is not None and prospective_train_clip_count < train_target_count:
                    continue

                if train_target_count is None:
                    inverse_total = 1.0 / prospective_train_clip_count
                    target_total_mass = (current_remaining_train_tag_mass - source_stats.tag_mass) * inverse_total

                    def target_frequency_for_tag(
                        tag: str,
                        remaining_counts: Counter[str] = current_remaining_train_tag_counts,
                        source_tag_counts: Counter[str] = source_stats.tag_counts,
                        inverse_total: float = inverse_total,
                    ) -> float:
                        return max(remaining_counts.get(tag, 0) - source_tag_counts.get(tag, 0), 0) * inverse_total

                    source_open_cost = (
                        0,
                        sum(target_frequency_for_tag(tag) for tag in source_stats.unique_tags),
                        source_id,
                    )
                else:
                    assert fixed_target_frequency_for_tag is not None
                    target_total_mass = fixed_target_total_mass
                    target_frequency_for_tag = fixed_target_frequency_for_tag
                    source_open_cost = fixed_source_open_costs[source_id]

                best_source_clip: TaggedClip | None = None
                best_source_metrics: tuple[int, int] | None = None
                best_source_sort_key: tuple[float, int, float, str, int, int, int, int, str] | None = None
                for tagged_clip in remaining_clips:
                    sort_key, metrics = score_validation_candidate(
                        tagged_clip=tagged_clip,
                        covered_tags=covered_tags,
                        val_tag_counts=val_tag_counts,
                        current_balance_penalty=current_balance_penalty,
                        current_val_count=current_val_count,
                        target_total_mass=target_total_mass,
                        target_frequency_for_tag=target_frequency_for_tag,
                        source_open_cost=source_open_cost,
                        is_new_source=True,
                    )
                    if best_source_sort_key is None or sort_key < best_source_sort_key:
                        best_source_clip = tagged_clip
                        best_source_metrics = metrics
                        best_source_sort_key = sort_key

                if best_source_clip is None or best_source_metrics is None or best_source_sort_key is None:
                    continue
                if best_sort_key is None or best_source_sort_key < best_sort_key:
                    best_clip = best_source_clip
                    best_metrics = best_source_metrics
                    best_sort_key = best_source_sort_key
                continue

            if train_target_count is None:
                inverse_total = 1.0 / current_remaining_train_clip_count
                target_total_mass = current_remaining_train_tag_mass * inverse_total

                def target_frequency_for_tag(
                    tag: str,
                    remaining_counts: Counter[str] = current_remaining_train_tag_counts,
                    inverse_total: float = inverse_total,
                ) -> float:
                    return remaining_counts.get(tag, 0) * inverse_total

            else:
                assert fixed_target_frequency_for_tag is not None
                target_total_mass = fixed_target_total_mass
                target_frequency_for_tag = fixed_target_frequency_for_tag

            source_open_cost = (-1, 0.0, source_id)
            for tagged_clip in remaining_clips:
                sort_key, metrics = score_validation_candidate(
                    tagged_clip=tagged_clip,
                    covered_tags=covered_tags,
                    val_tag_counts=val_tag_counts,
                    current_balance_penalty=current_balance_penalty,
                    current_val_count=current_val_count,
                    target_total_mass=target_total_mass,
                    target_frequency_for_tag=target_frequency_for_tag,
                    source_open_cost=source_open_cost,
                    is_new_source=False,
                )
                if best_sort_key is None or sort_key < best_sort_key:
                    best_clip = tagged_clip
                    best_metrics = metrics
                    best_sort_key = sort_key

        if best_clip is None or best_metrics is None:
            if train_target_count is not None:
                raise ValueError(
                    "could not construct source-disjoint train/val splits with "
                    f"val={target_count} and train={train_target_count}"
                )
            raise ValueError("could not construct a source-disjoint validation split from the tagged clips")

        selections.append(
            SelectionResult(
                tagged_clip=best_clip,
                selection_rank=rank,
                new_tags_added=best_metrics[0],
                balance_penalty=best_metrics[1],
            )
        )
        remaining_clips_by_source[best_clip.source_id].remove(best_clip)
        if best_clip.source_id not in opened_source_ids:
            opened_source_ids.add(best_clip.source_id)
            source_stats = selection_data.source_stats_by_id[best_clip.source_id]
            current_remaining_train_clip_count -= source_stats.clip_count
            subtract_tag_counts(current_remaining_train_tag_counts, source_stats.tag_counts)
            current_remaining_train_tag_mass -= source_stats.tag_mass

        current_balance_penalty = best_metrics[1]
        current_val_count += 1
        covered_tags.update(best_clip.tag_set)
        for tag in best_clip.tag_set:
            val_tag_counts[tag] += 1

    return selections


def compute_max_validation_count(tagged_clips: list[TaggedClip]) -> int:
    source_counts = Counter(tagged_clip.source_id for tagged_clip in tagged_clips)
    if len(source_counts) < 2:
        return 0
    return len(tagged_clips) - min(source_counts.values())


def resolve_target_count(
    count: int | None,
    fraction: float | None,
    usable_clips: int,
    max_validation_count: int,
) -> int:
    if usable_clips <= 0:
        raise ValueError("no successfully tagged clips are available")
    if max_validation_count <= 0:
        raise ValueError("at least two distinct source videos are required to build source-disjoint train/val splits")
    if count is not None:
        if count > max_validation_count:
            raise ValueError(
                f"--count ({count}) must be less than or equal to the maximum feasible val size ({max_validation_count})"
            )
        return count

    assert fraction is not None
    resolved = max(1, round(fraction * usable_clips))
    if resolved > max_validation_count:
        resolved = max_validation_count
    return resolved


def resolve_train_target_count(
    train_count: int | None,
    usable_clips: int,
    validation_count: int,
) -> int | None:
    if train_count is None:
        return None
    if train_count + validation_count > usable_clips:
        raise ValueError(
            f"--train-count ({train_count}) plus val size ({validation_count}) "
            f"must be less than or equal to usable clips ({usable_clips})"
        )
    return train_count


def ensure_output_dir_is_empty(output_dir: Path) -> None:
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        return

    if collect_video_files(output_dir):
        raise ValueError(f"Output directory already contains video clips: {output_dir}")

    manifest_paths = sorted(path for path in output_dir.rglob(DEFAULT_TAGS_FILENAME) if path.is_file())
    if manifest_paths:
        raise ValueError(f"Output directory already contains a tag manifest: {manifest_paths[0]}")


def copy_selected_clip(source_path: Path, output_dir: Path, clip_relpath: str) -> Path:
    destination = output_dir / clip_relpath
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination)
    return destination


def write_manifest_row(handle: TextIO, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record))
    handle.write("\n")
    handle.flush()


def resolve_selection_count(requested_count: int | None, available_count: int) -> int:
    if requested_count is None:
        return available_count
    return requested_count


def summarize_selection_set(
    selections: list[SelectionResult],
) -> tuple[list[TaggedClip], int, int]:
    selected_clips = [selection.tagged_clip for selection in selections]
    tag_counts = build_tag_counts(selected_clips)
    return selected_clips, compute_unique_tag_count(selected_clips), compute_balance_penalty(tag_counts)


def annotate_validation_distribution_distances(
    validation_selections: list[SelectionResult],
    train_selections: list[SelectionResult],
) -> list[SelectionResult]:
    train_clips = [selection.tagged_clip for selection in train_selections]
    if not validation_selections or not train_clips:
        return validation_selections

    train_tag_counts = build_tag_counts(train_clips)
    inverse_train_total = 1.0 / len(train_clips)
    target_total_mass = sum(train_tag_counts.values()) * inverse_train_total
    target_frequency_for_tag = lambda tag: train_tag_counts.get(tag, 0) * inverse_train_total
    validation_tag_counts: Counter[str] = Counter()
    annotated: list[SelectionResult] = []

    for selection in validation_selections:
        for tag in selection.tagged_clip.tag_set:
            validation_tag_counts[tag] += 1
        annotated.append(
            replace(
                selection,
                distribution_distance=compute_distribution_distance_to_target(
                    val_tag_counts=validation_tag_counts,
                    val_total=selection.selection_rank,
                    target_total_mass=target_total_mass,
                    target_frequency_for_tag=target_frequency_for_tag,
                ),
            )
        )

    return annotated


def split_tagged_clips(
    tagged_clips: list[TaggedClip],
    validation_count: int,
    train_count: int | None = None,
) -> tuple[list[SelectionResult], list[SelectionResult]]:
    validation_selections = select_validation_subset(
        tagged_clips,
        target_count=validation_count,
        train_target_count=train_count,
    )
    validation_source_ids = {selection.tagged_clip.source_id for selection in validation_selections}
    train_clips = [tagged_clip for tagged_clip in tagged_clips if tagged_clip.source_id not in validation_source_ids]
    if not train_clips:
        raise ValueError("validation split consumed every source video; no train clips remain")
    resolved_train_count = resolve_selection_count(train_count, len(train_clips))
    if resolved_train_count > len(train_clips):
        raise ValueError(
            f"requested train size ({resolved_train_count}) exceeds available source-disjoint train clips ({len(train_clips)})"
        )
    train_selections = select_balanced_subset(train_clips, target_count=resolved_train_count)
    return train_selections, annotate_validation_distribution_distances(validation_selections, train_selections)


def write_split(
    output_dir: Path,
    split_name: str,
    selections: list[SelectionResult],
) -> Path:
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    output_manifest = split_dir / DEFAULT_TAGS_FILENAME
    with output_manifest.open("w", encoding="utf-8") as manifest_handle:
        for selection in selections:
            copied_path = copy_selected_clip(
                source_path=selection.tagged_clip.clip_path,
                output_dir=split_dir,
                clip_relpath=selection.tagged_clip.clip_relpath,
            )
            record = dict(selection.tagged_clip.record)
            record["clip_path"] = str(copied_path)
            record["split"] = split_name
            record["source_id"] = selection.tagged_clip.source_id
            record["selection_rank"] = selection.selection_rank
            record["new_tags_added"] = selection.new_tags_added
            record["balance_penalty"] = selection.balance_penalty
            if selection.distribution_distance is not None:
                record["distribution_distance"] = selection.distribution_distance
            write_manifest_row(manifest_handle, record)
            print(f"selected[{split_name}]: {copied_path}")
    return output_manifest


def summarize_results(
    total_clips: int,
    train_clips: int,
    validation_clips: int,
    output_dir: Path,
    train_manifest: Path,
    validation_manifest: Path,
) -> str:
    return " ".join(
        [
            f"usable={total_clips}",
            f"train={train_clips}",
            f"val={validation_clips}",
            f"output={output_dir}",
            f"train_manifest={train_manifest}",
            f"val_manifest={validation_manifest}",
        ]
    )


def run(args: argparse.Namespace) -> int:
    try:
        tagged_clips = load_tagged_clips(args.tags_file, args.input_dir)
        max_validation_count = compute_max_validation_count(tagged_clips)
        target_count = resolve_target_count(args.count, args.fraction, len(tagged_clips), max_validation_count)
        train_count = resolve_train_target_count(getattr(args, "train_count", None), len(tagged_clips), target_count)
        ensure_output_dir_is_empty(args.output_dir)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    try:
        train_selections, validation_selections = split_tagged_clips(
            tagged_clips,
            validation_count=target_count,
            train_count=train_count,
        )
        train_manifest = write_split(args.output_dir, TRAIN_SPLIT_NAME, train_selections)
        validation_manifest = write_split(args.output_dir, VALIDATION_SPLIT_NAME, validation_selections)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(
        summarize_results(
            total_clips=len(tagged_clips),
            train_clips=len(train_selections),
            validation_clips=len(validation_selections),
            output_dir=args.output_dir,
            train_manifest=train_manifest,
            validation_manifest=validation_manifest,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
