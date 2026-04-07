from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import anime_clip_tag


class FakeTagResult:
    def __init__(
        self,
        rating: str = "general",
        rating_data: dict[str, float] | None = None,
        general_tag_data: dict[str, float] | None = None,
        character_tag_data: dict[str, float] | None = None,
    ) -> None:
        self.rating = rating
        self.rating_data = rating_data or {
            "general": 0.9,
            "sensitive": 0.05,
            "questionable": 0.03,
            "explicit": 0.02,
        }
        self.general_tag_data = general_tag_data or {"running": 0.8, "sky": 0.7}
        self.character_tag_data = character_tag_data or {"character_a": 0.95}

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
        self.calls: list[dict[str, object]] = []

    def tag(self, images, general_threshold: float, character_threshold: float):
        self.calls.append(
            {
                "count": len(images),
                "general_threshold": general_threshold,
                "character_threshold": character_threshold,
            }
        )
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class AnimeClipTagHelperTests(unittest.TestCase):
    def test_validate_args_sets_default_output_file(self) -> None:
        parser = anime_clip_tag.build_parser()

        with tempfile.TemporaryDirectory() as temp_dir:
            args = parser.parse_args(["--input-dir", temp_dir])
            anime_clip_tag.validate_args(parser, args)

            self.assertEqual(args.output_file, Path(temp_dir) / anime_clip_tag.DEFAULT_OUTPUT_FILENAME)

    def test_require_runtime_reports_missing_python_dependency(self) -> None:
        with patch("anime_clip_tag.shutil.which", return_value="/usr/bin/ffmpeg"), patch.object(
            anime_clip_tag, "Tagger", None
        ):
            with self.assertRaises(SystemExit) as context:
                anime_clip_tag.require_runtime()

        self.assertIn("Missing dependencies for tagging", str(context.exception))

    def test_compute_representative_frame_time_uses_midpoint_and_clamps_tiny_clips(self) -> None:
        self.assertEqual(anime_clip_tag.compute_representative_frame_time(2.0), 1.0)
        self.assertEqual(anime_clip_tag.compute_representative_frame_time(0.0005), 0.0)

    def test_build_success_record_contains_expected_fields(self) -> None:
        pending = anime_clip_tag.PendingClip(
            clip_path=Path("/tmp/clip.mp4"),
            clip_relpath="nested/clip.mp4",
            duration=2.5,
            frame_time=1.25,
            image=object(),
        )

        record = anime_clip_tag.build_success_record(
            pending=pending,
            result=FakeTagResult(),
            model_repo="SmilingWolf/wd-swinv2-tagger-v3",
        )

        self.assertEqual(record["status"], "tagged")
        self.assertEqual(record["clip_path"], "/tmp/clip.mp4")
        self.assertEqual(record["clip_relpath"], "nested/clip.mp4")
        self.assertEqual(record["rating"], "general")
        self.assertEqual(record["general_tags"], ["running", "sky"])
        self.assertEqual(record["character_tags"], ["character_a"])
        self.assertEqual(record["all_tags"], ["general", "character_a", "running", "sky"])

    def test_flush_pending_batch_falls_back_to_individual_tagging(self) -> None:
        pending_batch = [
            anime_clip_tag.PendingClip(
                clip_path=Path("/tmp/a.mp4"),
                clip_relpath="a.mp4",
                duration=1.0,
                frame_time=0.5,
                image=Image.new("RGB", (2, 2), "red"),
            ),
            anime_clip_tag.PendingClip(
                clip_path=Path("/tmp/b.mp4"),
                clip_relpath="b.mp4",
                duration=1.0,
                frame_time=0.5,
                image=Image.new("RGB", (2, 2), "blue"),
            ),
        ]
        fake_tagger = FakeTagger(
            [
                RuntimeError("batch failure"),
                FakeTagResult(general_tag_data={"smile": 0.9}),
                RuntimeError("single failure"),
            ]
        )

        records = anime_clip_tag.flush_pending_batch(
            tagger=fake_tagger,
            pending_batch=pending_batch,
            model_repo="SmilingWolf/wd-swinv2-tagger-v3",
            general_threshold=0.35,
            character_threshold=0.9,
        )

        self.assertEqual([call["count"] for call in fake_tagger.calls], [2, 1, 1])
        self.assertEqual(records[0]["status"], "tagged")
        self.assertEqual(records[0]["general_tags"], ["smile"])
        self.assertEqual(records[1]["status"], "failed")
        self.assertEqual(records[1]["error"], "single failure")


class AnimeClipTagRunTests(unittest.TestCase):
    def test_run_writes_manifest_and_keeps_failed_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "a.mp4"
            second = root / "b.webm"
            first.write_bytes(b"a")
            second.write_bytes(b"b")
            output_file = root / "tags.jsonl"
            args = argparse.Namespace(
                input_dir=root,
                output_file=output_file,
                model_repo="SmilingWolf/wd-swinv2-tagger-v3",
                general_threshold=0.35,
                character_threshold=0.9,
                batch_size=2,
            )
            fake_tagger = FakeTagger([[FakeTagResult(general_tag_data={"jump": 0.88})]])

            def fake_extract_frame_image(clip_path: Path, frame_time: float):
                self.assertGreaterEqual(frame_time, 0.0)
                if clip_path == second:
                    raise RuntimeError("frame boom")
                return Image.new("RGB", (4, 4), "white")

            with patch("anime_clip_tag.require_runtime"), patch.object(
                anime_clip_tag,
                "Tagger",
                side_effect=lambda model_repo: fake_tagger,
            ), patch(
                "anime_clip_tag.probe_video",
                return_value=anime_clip_tag.VideoMetadata(duration=2.0),
            ), patch(
                "anime_clip_tag.extract_frame_image",
                side_effect=fake_extract_frame_image,
            ):
                exit_code = anime_clip_tag.run(args)

            self.assertEqual(exit_code, 1)
            lines = output_file.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            first_record = json.loads(lines[0])
            second_record = json.loads(lines[1])
            self.assertEqual(first_record["status"], "failed")
            self.assertEqual(second_record["status"], "tagged")
            self.assertEqual(second_record["general_tags"], ["jump"])


if __name__ == "__main__":
    unittest.main()
