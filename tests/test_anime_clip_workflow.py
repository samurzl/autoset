from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import anime_clip_ltx
import anime_clip_subset
import anime_clip_tag


class FakeTagResult:
    def __init__(self, rating: str, general_tags: dict[str, float], character_tags: dict[str, float]) -> None:
        self.rating = rating
        self.rating_data = {
            "general": 0.9 if rating == "general" else 0.05,
            "sensitive": 0.9 if rating == "sensitive" else 0.03,
            "questionable": 0.01,
            "explicit": 0.01,
        }
        self.general_tag_data = general_tags
        self.character_tag_data = character_tags

    @property
    def general_tags(self) -> tuple[str, ...]:
        return tuple(self.general_tag_data)

    @property
    def character_tags(self) -> tuple[str, ...]:
        return tuple(self.character_tag_data)

    @property
    def all_tags(self) -> list[str]:
        return [self.rating, *self.character_tags, *self.general_tags]


class FakeTagger:
    def __init__(self, responses: list[object]):
        self.responses = list(responses)

    def tag(self, images, general_threshold: float, character_threshold: float):
        del images, general_threshold, character_threshold
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class AnimeClipWorkflowTests(unittest.TestCase):
    def test_tag_then_subset_workflow_creates_train_and_val_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "clips"
            input_dir.mkdir()
            first = input_dir / "a.mp4"
            second = input_dir / "b.mp4"
            first.write_bytes(b"a")
            second.write_bytes(b"b")

            tag_args = argparse.Namespace(
                input_dir=input_dir,
                output_file=input_dir / "tags.jsonl",
                model_repo="SmilingWolf/wd-swinv2-tagger-v3",
                general_threshold=0.35,
                character_threshold=0.9,
                batch_size=2,
            )
            fake_tagger = FakeTagger(
                [
                    [
                        FakeTagResult("general", {"fight": 0.88}, {"hero": 0.97}),
                        FakeTagResult("general", {"fight": 0.9, "smear": 0.86}, {"hero": 0.98}),
                    ]
                ]
            )

            with patch("anime_clip_tag.require_runtime"), patch.object(
                anime_clip_tag, "Tagger", side_effect=lambda model_repo: fake_tagger
            ), patch(
                "anime_clip_tag.probe_video",
                return_value=anime_clip_tag.VideoMetadata(duration=2.0),
            ), patch(
                "anime_clip_tag.extract_frame_image",
                return_value=Image.new("RGB", (4, 4), "white"),
            ):
                tag_exit_code = anime_clip_tag.run(tag_args)

            self.assertEqual(tag_exit_code, 0)
            manifest_lines = tag_args.output_file.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(manifest_lines), 2)

            subset_args = argparse.Namespace(
                input_dir=input_dir,
                tags_file=tag_args.output_file,
                output_dir=root / "subset",
                train_count=None,
                count=1,
                fraction=None,
            )

            subset_exit_code = anime_clip_subset.run(subset_args)

            self.assertEqual(subset_exit_code, 0)
            val_manifest = subset_args.output_dir / anime_clip_subset.VALIDATION_SPLIT_NAME / "tags.jsonl"
            train_manifest = subset_args.output_dir / anime_clip_subset.TRAIN_SPLIT_NAME / "tags.jsonl"
            val_records = [json.loads(line) for line in val_manifest.read_text(encoding="utf-8").splitlines()]
            train_records = [json.loads(line) for line in train_manifest.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(val_records), 1)
            self.assertEqual(len(train_records), 1)
            self.assertEqual(val_records[0]["clip_relpath"], "b.mp4")
            self.assertEqual(train_records[0]["clip_relpath"], "a.mp4")
            self.assertTrue((subset_args.output_dir / anime_clip_subset.VALIDATION_SPLIT_NAME / "b.mp4").exists())
            self.assertTrue((subset_args.output_dir / anime_clip_subset.TRAIN_SPLIT_NAME / "a.mp4").exists())
            self.assertEqual(val_records[0]["new_tags_added"], 4)

    def test_tag_then_subset_then_ltx_workflow_creates_split_local_exports(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "clips"
            input_dir.mkdir()
            first = input_dir / "a.mp4"
            second = input_dir / "b.mp4"
            first.write_bytes(b"a")
            second.write_bytes(b"b")

            tag_args = argparse.Namespace(
                input_dir=input_dir,
                output_file=input_dir / "tags.jsonl",
                model_repo="SmilingWolf/wd-swinv2-tagger-v3",
                general_threshold=0.35,
                character_threshold=0.9,
                batch_size=2,
            )
            fake_tagger = FakeTagger(
                [
                    [
                        FakeTagResult("general", {"fight": 0.88}, {"hero": 0.97}),
                        FakeTagResult("general", {"fight": 0.9, "smear": 0.86}, {"hero": 0.98}),
                    ]
                ]
            )

            with patch("anime_clip_tag.require_runtime"), patch.object(
                anime_clip_tag, "Tagger", side_effect=lambda model_repo: fake_tagger
            ), patch(
                "anime_clip_tag.probe_video",
                return_value=anime_clip_tag.VideoMetadata(duration=2.0),
            ), patch(
                "anime_clip_tag.extract_frame_image",
                return_value=Image.new("RGB", (4, 4), "white"),
            ):
                tag_exit_code = anime_clip_tag.run(tag_args)

            self.assertEqual(tag_exit_code, 0)

            subset_args = argparse.Namespace(
                input_dir=input_dir,
                tags_file=tag_args.output_file,
                output_dir=root / "subset",
                train_count=None,
                count=1,
                fraction=None,
            )

            subset_exit_code = anime_clip_subset.run(subset_args)

            self.assertEqual(subset_exit_code, 0)
            train_dir = subset_args.output_dir / anime_clip_subset.TRAIN_SPLIT_NAME
            val_dir = subset_args.output_dir / anime_clip_subset.VALIDATION_SPLIT_NAME
            train_clip = train_dir / "a.mp4"
            val_clip = val_dir / "b.mp4"
            (train_dir / anime_clip_ltx.DEFAULT_CAPTIONS_FILENAME).write_text(
                json.dumps(
                    {
                        "clip_path": str(train_clip),
                        "caption": "Train clip",
                        "status": "captioned",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (val_dir / anime_clip_ltx.DEFAULT_CAPTIONS_FILENAME).write_text(
                json.dumps(
                    {
                        "clip_path": "val/b.mp4",
                        "caption": "Val clip",
                        "status": "captioned",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            ltx_args = argparse.Namespace(
                input_dir=subset_args.output_dir,
                captions_filename=anime_clip_ltx.DEFAULT_CAPTIONS_FILENAME,
                output_filename=anime_clip_ltx.DEFAULT_OUTPUT_FILENAME,
            )

            ltx_exit_code = anime_clip_ltx.run(ltx_args)

            self.assertEqual(ltx_exit_code, 0)
            train_rows = [
                json.loads(line)
                for line in (train_dir / anime_clip_ltx.DEFAULT_OUTPUT_FILENAME).read_text(encoding="utf-8").splitlines()
            ]
            val_rows = [
                json.loads(line)
                for line in (val_dir / anime_clip_ltx.DEFAULT_OUTPUT_FILENAME).read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                train_rows,
                [{"caption": "Train clip", "media_path": "a.mp4"}],
            )
            self.assertEqual(
                val_rows,
                [{"caption": "Val clip", "media_path": "b.mp4"}],
            )


if __name__ == "__main__":
    unittest.main()
