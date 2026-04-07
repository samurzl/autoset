from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

import anime_clip_subset


def make_record(
    clip_path: Path,
    clip_relpath: str,
    tags: list[str],
    status: str = "tagged",
    source_path: str | None = None,
) -> dict[str, object]:
    rating = tags[0] if tags else None
    record: dict[str, object] = {
        "clip_path": str(clip_path),
        "clip_relpath": clip_relpath,
        "model_repo": "SmilingWolf/wd-swinv2-tagger-v3",
        "duration": 2.0,
        "frame_time": 1.0,
        "rating": rating,
        "rating_scores": {"general": 0.9},
        "general_tags": tags[1:],
        "general_tag_scores": {tag: 0.8 for tag in tags[1:]},
        "character_tags": [],
        "character_tag_scores": {},
        "all_tags": tags,
        "status": status,
        "error": "" if status == "tagged" else "boom",
    }
    if source_path is not None:
        record["source_path"] = source_path
    return record


class AnimeClipSubsetHelperTests(unittest.TestCase):
    def test_validate_args_sets_default_paths(self) -> None:
        parser = anime_clip_subset.build_parser()

        with tempfile.TemporaryDirectory() as temp_dir:
            tags_file = Path(temp_dir) / anime_clip_subset.DEFAULT_TAGS_FILENAME
            tags_file.write_text("", encoding="utf-8")
            args = parser.parse_args(["--input-dir", temp_dir, "--count", "1"])
            anime_clip_subset.validate_args(parser, args)

            self.assertEqual(args.tags_file, tags_file)
            self.assertEqual(args.output_dir, Path(temp_dir).with_name(f"{Path(temp_dir).name}_subset"))
            self.assertIsNone(args.train_count)

    def test_resolve_target_count_rejects_count_larger_than_usable_pool(self) -> None:
        with self.assertRaises(ValueError):
            anime_clip_subset.resolve_target_count(
                count=3,
                fraction=None,
                usable_clips=4,
                max_validation_count=2,
            )

    def test_resolve_train_target_count_rejects_combined_size_larger_than_usable_pool(self) -> None:
        with self.assertRaises(ValueError):
            anime_clip_subset.resolve_train_target_count(
                train_count=3,
                usable_clips=4,
                validation_count=2,
            )

    def test_load_tagged_clips_skips_failed_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            clip = root / "a.mp4"
            clip.write_bytes(b"a")
            tags_file = root / "tags.jsonl"
            lines = [
                json.dumps(make_record(clip, "a.mp4", ["general", "sky"], status="failed")),
                json.dumps(make_record(clip, "a.mp4", ["general", "sky"], status="tagged")),
            ]
            tags_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

            tagged_clips = anime_clip_subset.load_tagged_clips(tags_file, root)

        self.assertEqual(len(tagged_clips), 1)
        self.assertEqual(tagged_clips[0].tags, ("general", "sky"))
        self.assertEqual(tagged_clips[0].source_id, "a.mp4")

    def test_load_tagged_clips_uses_extract_manifest_for_source_lookup(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            clip = root / "a.mp4"
            clip.write_bytes(b"a")
            tags_file = root / "tags.jsonl"
            tags_file.write_text(
                json.dumps(make_record(clip, "a.mp4", ["general", "sky"])) + "\n",
                encoding="utf-8",
            )
            manifest_file = root / "manifest.jsonl"
            manifest_file.write_text(
                json.dumps({"clip_path": str(clip), "source_path": "/dataset/source_a.mp4"}) + "\n",
                encoding="utf-8",
            )

            tagged_clips = anime_clip_subset.load_tagged_clips(tags_file, root)

        self.assertEqual(tagged_clips[0].source_id, "/dataset/source_a.mp4")
        self.assertEqual(tagged_clips[0].record["source_path"], "/dataset/source_a.mp4")

    def test_select_balanced_subset_maximizes_new_tag_coverage(self) -> None:
        candidates = [
            anime_clip_subset.TaggedClip(
                clip_path=Path("/tmp/a.mp4"),
                clip_relpath="a.mp4",
                source_id="a.mp4",
                tags=("general", "fight"),
                record={"clip_path": "/tmp/a.mp4"},
            ),
            anime_clip_subset.TaggedClip(
                clip_path=Path("/tmp/b.mp4"),
                clip_relpath="b.mp4",
                source_id="b.mp4",
                tags=("general", "smile", "crowd"),
                record={"clip_path": "/tmp/b.mp4"},
            ),
            anime_clip_subset.TaggedClip(
                clip_path=Path("/tmp/c.mp4"),
                clip_relpath="c.mp4",
                source_id="c.mp4",
                tags=("general", "fight"),
                record={"clip_path": "/tmp/c.mp4"},
            ),
        ]

        selections = anime_clip_subset.select_balanced_subset(candidates, target_count=2)

        self.assertEqual([selection.tagged_clip.clip_relpath for selection in selections], ["b.mp4", "a.mp4"])

    def test_select_balanced_subset_prefers_lower_balance_penalty_on_tie(self) -> None:
        candidates = [
            anime_clip_subset.TaggedClip(
                clip_path=Path("/tmp/a.mp4"),
                clip_relpath="a.mp4",
                source_id="a.mp4",
                tags=("general", "x", "y"),
                record={"clip_path": "/tmp/a.mp4"},
            ),
            anime_clip_subset.TaggedClip(
                clip_path=Path("/tmp/b.mp4"),
                clip_relpath="b.mp4",
                source_id="b.mp4",
                tags=("general", "x", "z"),
                record={"clip_path": "/tmp/b.mp4"},
            ),
            anime_clip_subset.TaggedClip(
                clip_path=Path("/tmp/c.mp4"),
                clip_relpath="c.mp4",
                source_id="c.mp4",
                tags=("general", "z"),
                record={"clip_path": "/tmp/c.mp4"},
            ),
        ]

        selections = anime_clip_subset.select_balanced_subset(candidates, target_count=2)

        self.assertEqual([selection.tagged_clip.clip_relpath for selection in selections], ["a.mp4", "c.mp4"])
        self.assertEqual(selections[1].balance_penalty, 7)

    def test_select_balanced_subset_uses_lexicographic_tie_break(self) -> None:
        candidates = [
            anime_clip_subset.TaggedClip(
                clip_path=Path("/tmp/b.mp4"),
                clip_relpath="b.mp4",
                source_id="b.mp4",
                tags=("general", "wave"),
                record={"clip_path": "/tmp/b.mp4"},
            ),
            anime_clip_subset.TaggedClip(
                clip_path=Path("/tmp/a.mp4"),
                clip_relpath="a.mp4",
                source_id="a.mp4",
                tags=("general", "wave"),
                record={"clip_path": "/tmp/a.mp4"},
            ),
        ]

        selections = anime_clip_subset.select_balanced_subset(candidates, target_count=1)

        self.assertEqual(selections[0].tagged_clip.clip_relpath, "a.mp4")

    def test_select_validation_subset_trainless_path_is_deterministic(self) -> None:
        candidates = [
            anime_clip_subset.TaggedClip(
                clip_path=Path("/tmp/a.mp4"),
                clip_relpath="a.mp4",
                source_id="source_one",
                tags=("general", "x"),
                record={"clip_path": "/tmp/a.mp4"},
            ),
            anime_clip_subset.TaggedClip(
                clip_path=Path("/tmp/b.mp4"),
                clip_relpath="b.mp4",
                source_id="source_one",
                tags=("general", "x"),
                record={"clip_path": "/tmp/b.mp4"},
            ),
            anime_clip_subset.TaggedClip(
                clip_path=Path("/tmp/c.mp4"),
                clip_relpath="c.mp4",
                source_id="source_two",
                tags=("general", "x"),
                record={"clip_path": "/tmp/c.mp4"},
            ),
            anime_clip_subset.TaggedClip(
                clip_path=Path("/tmp/d.mp4"),
                clip_relpath="d.mp4",
                source_id="source_three",
                tags=("general", "y"),
                record={"clip_path": "/tmp/d.mp4"},
            ),
        ]

        first_selection = anime_clip_subset.select_validation_subset(candidates, target_count=1)
        second_selection = anime_clip_subset.select_validation_subset(candidates, target_count=1)

        self.assertEqual(
            [selection.tagged_clip.clip_relpath for selection in first_selection],
            [selection.tagged_clip.clip_relpath for selection in second_selection],
        )
        self.assertEqual(
            [selection.tagged_clip.source_id for selection in first_selection],
            [selection.tagged_clip.source_id for selection in second_selection],
        )

    def test_select_validation_subset_prefers_reusing_open_source(self) -> None:
        candidates = [
            anime_clip_subset.TaggedClip(
                clip_path=Path("/tmp/a.mp4"),
                clip_relpath="a.mp4",
                source_id="source_one",
                tags=("general", "x", "y"),
                record={"clip_path": "/tmp/a.mp4"},
            ),
            anime_clip_subset.TaggedClip(
                clip_path=Path("/tmp/b.mp4"),
                clip_relpath="b.mp4",
                source_id="source_one",
                tags=("general", "z"),
                record={"clip_path": "/tmp/b.mp4"},
            ),
            anime_clip_subset.TaggedClip(
                clip_path=Path("/tmp/c.mp4"),
                clip_relpath="c.mp4",
                source_id="source_two",
                tags=("general", "x"),
                record={"clip_path": "/tmp/c.mp4"},
            ),
            anime_clip_subset.TaggedClip(
                clip_path=Path("/tmp/d.mp4"),
                clip_relpath="d.mp4",
                source_id="source_three",
                tags=("general", "y"),
                record={"clip_path": "/tmp/d.mp4"},
            ),
        ]

        selections = anime_clip_subset.select_validation_subset(candidates, target_count=2)

        self.assertEqual([selection.tagged_clip.source_id for selection in selections], ["source_one", "source_one"])

    def test_select_validation_subset_rejects_new_source_when_requested_train_size_would_become_infeasible(self) -> None:
        candidates = [
            anime_clip_subset.TaggedClip(
                clip_path=Path("/tmp/a.mp4"),
                clip_relpath="a.mp4",
                source_id="source_one",
                tags=("general", "fight"),
                record={"clip_path": "/tmp/a.mp4"},
            ),
            anime_clip_subset.TaggedClip(
                clip_path=Path("/tmp/b.mp4"),
                clip_relpath="b.mp4",
                source_id="source_one",
                tags=("general", "smear"),
                record={"clip_path": "/tmp/b.mp4"},
            ),
            anime_clip_subset.TaggedClip(
                clip_path=Path("/tmp/c.mp4"),
                clip_relpath="c.mp4",
                source_id="source_two",
                tags=("general", "smile"),
                record={"clip_path": "/tmp/c.mp4"},
            ),
            anime_clip_subset.TaggedClip(
                clip_path=Path("/tmp/d.mp4"),
                clip_relpath="d.mp4",
                source_id="source_two",
                tags=("general", "glow"),
                record={"clip_path": "/tmp/d.mp4"},
            ),
        ]

        selections = anime_clip_subset.select_validation_subset(
            candidates,
            target_count=2,
            train_target_count=2,
        )

        self.assertEqual(len({selection.tagged_clip.source_id for selection in selections}), 1)


class AnimeClipSubsetRunTests(unittest.TestCase):
    def test_run_writes_source_disjoint_train_and_val_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "clips"
            input_dir.mkdir()
            clip_a = input_dir / "a.mp4"
            clip_b = input_dir / "b.mp4"
            clip_c = input_dir / "c.mp4"
            clip_a.write_bytes(b"a")
            clip_b.write_bytes(b"b")
            clip_c.write_bytes(b"c")
            tags_file = input_dir / "tags.jsonl"
            source_manifest = input_dir / "manifest.jsonl"
            output_dir = root / "subset"
            source_manifest.write_text(
                "\n".join(
                    [
                        json.dumps({"clip_path": str(clip_a), "source_path": "/dataset/source_one.mp4"}),
                        json.dumps({"clip_path": str(clip_b), "source_path": "/dataset/source_one.mp4"}),
                        json.dumps({"clip_path": str(clip_c), "source_path": "/dataset/source_two.mp4"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            tags_file.write_text(
                "\n".join(
                    [
                        json.dumps(make_record(clip_a, "a.mp4", ["general", "fight", "smear"])),
                        json.dumps(make_record(clip_b, "b.mp4", ["general", "fight"])),
                        json.dumps(make_record(clip_c, "c.mp4", ["general", "smear"])),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            args = argparse.Namespace(
                input_dir=input_dir,
                tags_file=tags_file,
                output_dir=output_dir,
                train_count=None,
                count=1,
                fraction=None,
            )

            exit_code = anime_clip_subset.run(args)

            self.assertEqual(exit_code, 0)
            val_records = [
                json.loads(line)
                for line in (output_dir / anime_clip_subset.VALIDATION_SPLIT_NAME / "tags.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            train_records = [
                json.loads(line)
                for line in (output_dir / anime_clip_subset.TRAIN_SPLIT_NAME / "tags.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(len(val_records), 1)
            self.assertEqual(len(train_records), 1)
            self.assertTrue(Path(val_records[0]["clip_path"]).exists())
            self.assertTrue(Path(train_records[0]["clip_path"]).exists())
            self.assertNotEqual(val_records[0]["source_id"], train_records[0]["source_id"])
            self.assertEqual(val_records[0]["split"], anime_clip_subset.VALIDATION_SPLIT_NAME)
            self.assertEqual(train_records[0]["split"], anime_clip_subset.TRAIN_SPLIT_NAME)
            self.assertIsInstance(val_records[0]["distribution_distance"], float)

            rerun_output_dir = root / "subset_again"
            rerun_args = argparse.Namespace(
                input_dir=input_dir,
                tags_file=tags_file,
                output_dir=rerun_output_dir,
                train_count=None,
                count=1,
                fraction=None,
            )
            rerun_exit_code = anime_clip_subset.run(rerun_args)
            self.assertEqual(rerun_exit_code, 0)
            rerun_val_records = [
                json.loads(line)
                for line in (rerun_output_dir / anime_clip_subset.VALIDATION_SPLIT_NAME / "tags.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            rerun_train_records = [
                json.loads(line)
                for line in (rerun_output_dir / anime_clip_subset.TRAIN_SPLIT_NAME / "tags.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]

            def normalize_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
                normalized: list[dict[str, object]] = []
                for record in records:
                    normalized_record = dict(record)
                    normalized_record["clip_path"] = Path(str(normalized_record["clip_path"])).name
                    normalized.append(normalized_record)
                return normalized

            self.assertEqual(normalize_records(val_records), normalize_records(rerun_val_records))
            self.assertEqual(normalize_records(train_records), normalize_records(rerun_train_records))

    def test_run_rejects_validation_size_when_only_one_source_video_exists(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "clips"
            input_dir.mkdir()
            clip_a = input_dir / "a.mp4"
            clip_b = input_dir / "b.mp4"
            clip_a.write_bytes(b"a")
            clip_b.write_bytes(b"b")
            (input_dir / "manifest.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"clip_path": str(clip_a), "source_path": "/dataset/source_one.mp4"}),
                        json.dumps({"clip_path": str(clip_b), "source_path": "/dataset/source_one.mp4"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (input_dir / "tags.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(make_record(clip_a, "a.mp4", ["general", "fight"])),
                        json.dumps(make_record(clip_b, "b.mp4", ["general", "smear"])),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            args = argparse.Namespace(
                input_dir=input_dir,
                tags_file=input_dir / "tags.jsonl",
                output_dir=root / "subset",
                train_count=None,
                count=1,
                fraction=None,
            )

            exit_code = anime_clip_subset.run(args)

            self.assertEqual(exit_code, 1)

    def test_run_limits_train_subset_to_requested_size(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "clips"
            input_dir.mkdir()
            clip_a = input_dir / "a.mp4"
            clip_b = input_dir / "b.mp4"
            clip_c = input_dir / "c.mp4"
            clip_d = input_dir / "d.mp4"
            clip_a.write_bytes(b"a")
            clip_b.write_bytes(b"b")
            clip_c.write_bytes(b"c")
            clip_d.write_bytes(b"d")
            (input_dir / "manifest.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"clip_path": str(clip_a), "source_path": "/dataset/source_one.mp4"}),
                        json.dumps({"clip_path": str(clip_b), "source_path": "/dataset/source_two.mp4"}),
                        json.dumps({"clip_path": str(clip_c), "source_path": "/dataset/source_three.mp4"}),
                        json.dumps({"clip_path": str(clip_d), "source_path": "/dataset/source_four.mp4"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (input_dir / "tags.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(make_record(clip_a, "a.mp4", ["general", "fight", "smear"])),
                        json.dumps(make_record(clip_b, "b.mp4", ["general", "fight", "smile"])),
                        json.dumps(make_record(clip_c, "c.mp4", ["general", "smear", "glow"])),
                        json.dumps(make_record(clip_d, "d.mp4", ["general", "smile", "glow"])),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            args = argparse.Namespace(
                input_dir=input_dir,
                tags_file=input_dir / "tags.jsonl",
                output_dir=root / "subset",
                train_count=2,
                count=1,
                fraction=None,
            )

            exit_code = anime_clip_subset.run(args)

            self.assertEqual(exit_code, 0)
            train_manifest = args.output_dir / anime_clip_subset.TRAIN_SPLIT_NAME / "tags.jsonl"
            train_records = [json.loads(line) for line in train_manifest.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(train_records), 2)
            train_source_ids = {record["source_id"] for record in train_records}
            val_manifest = args.output_dir / anime_clip_subset.VALIDATION_SPLIT_NAME / "tags.jsonl"
            val_records = [json.loads(line) for line in val_manifest.read_text(encoding="utf-8").splitlines()]
            self.assertTrue(train_source_ids.isdisjoint({record["source_id"] for record in val_records}))

    def test_run_rejects_infeasible_train_size_for_source_disjoint_split(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "clips"
            input_dir.mkdir()
            clip_a = input_dir / "a.mp4"
            clip_b = input_dir / "b.mp4"
            clip_c = input_dir / "c.mp4"
            clip_d = input_dir / "d.mp4"
            clip_a.write_bytes(b"a")
            clip_b.write_bytes(b"b")
            clip_c.write_bytes(b"c")
            clip_d.write_bytes(b"d")
            (input_dir / "manifest.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"clip_path": str(clip_a), "source_path": "/dataset/source_one.mp4"}),
                        json.dumps({"clip_path": str(clip_b), "source_path": "/dataset/source_one.mp4"}),
                        json.dumps({"clip_path": str(clip_c), "source_path": "/dataset/source_two.mp4"}),
                        json.dumps({"clip_path": str(clip_d), "source_path": "/dataset/source_two.mp4"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (input_dir / "tags.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(make_record(clip_a, "a.mp4", ["general", "fight"])),
                        json.dumps(make_record(clip_b, "b.mp4", ["general", "smear"])),
                        json.dumps(make_record(clip_c, "c.mp4", ["general", "smile"])),
                        json.dumps(make_record(clip_d, "d.mp4", ["general", "glow"])),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            args = argparse.Namespace(
                input_dir=input_dir,
                tags_file=input_dir / "tags.jsonl",
                output_dir=root / "subset",
                train_count=3,
                count=1,
                fraction=None,
            )

            exit_code = anime_clip_subset.run(args)

            self.assertEqual(exit_code, 1)


if __name__ == "__main__":
    unittest.main()
