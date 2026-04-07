from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import anime_clip_caption


class FakeInputs(dict):
    def to(self, device: str) -> "FakeInputs":
        self.sent_to = device
        return self


class FakeProcessor:
    def __init__(self, decoded_text: str):
        self.decoded_text = decoded_text
        self.apply_calls: list[dict[str, object]] = []
        self.decoded_ids: list[int] | None = None

    def apply_chat_template(self, messages, **kwargs):
        self.apply_calls.append({"messages": messages, "kwargs": kwargs})
        return FakeInputs({"input_ids": [[10, 11, 12]]})

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        self.decoded_ids = list(token_ids)
        self.skip_special_tokens = skip_special_tokens
        return self.decoded_text


class FakeModel:
    def __init__(self, config: object | None = None):
        self.config = config or SimpleNamespace()
        self.device = "cuda:0"
        self.generate_calls: list[dict[str, object]] = []
        self.was_eval = False

    def to(self, device: str) -> "FakeModel":
        self.device = device
        return self

    def eval(self) -> "FakeModel":
        self.was_eval = True
        return self

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        return [[10, 11, 12, 21, 22]]


def make_record(path: Path, status: str, error: str = "") -> dict[str, object]:
    record = anime_clip_caption.make_manifest_record(path, "google/gemma-4-E4B-it")
    record["status"] = status
    record["error"] = error
    if status == "captioned":
        record["caption"] = "A short scene with detailed visuals."
    return record


class AnimeClipCaptionHelperTests(unittest.TestCase):
    def test_validate_args_sets_default_output_file(self) -> None:
        parser = anime_clip_caption.build_parser()

        with tempfile.TemporaryDirectory() as temp_dir:
            args = parser.parse_args(["--input-dir", temp_dir, "--model-repo", "google/gemma-4-E4B-it"])
            anime_clip_caption.validate_args(parser, args)

            self.assertEqual(args.output_file, Path(temp_dir) / anime_clip_caption.DEFAULT_OUTPUT_FILENAME)
            self.assertIsNone(args.tags_file)

    def test_validate_args_uses_default_tags_file_when_present(self) -> None:
        parser = anime_clip_caption.build_parser()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tags_file = root / anime_clip_caption.DEFAULT_TAGS_FILENAME
            tags_file.write_text("", encoding="utf-8")

            args = parser.parse_args(["--input-dir", temp_dir, "--model-repo", "google/gemma-4-E4B-it"])
            anime_clip_caption.validate_args(parser, args)

            self.assertEqual(args.tags_file, tags_file)

    def test_require_runtime_reports_missing_cuda(self) -> None:
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))

        with patch.object(anime_clip_caption, "torch", fake_torch), patch.object(
            anime_clip_caption, "AutoProcessor", object()
        ), patch.object(anime_clip_caption, "AutoModelForMultimodalLM", object()), patch.object(
            anime_clip_caption, "torchcodec", object()
        ), patch(
            "anime_clip_caption.shutil.which", return_value="/usr/bin/ffprobe"
        ):
            with self.assertRaises(SystemExit) as context:
                anime_clip_caption.require_runtime("cuda")

        self.assertIn("torch.cuda.is_available", str(context.exception))

    def test_probe_video_parses_duration_and_audio_presence(self) -> None:
        payload = {
            "format": {"duration": "4.25"},
            "streams": [
                {"codec_type": "video"},
                {"codec_type": "audio"},
            ],
        }
        completed = subprocess.CompletedProcess(
            args=["ffprobe"],
            returncode=0,
            stdout=json.dumps(payload),
            stderr="",
        )

        with patch("anime_clip_caption.subprocess.run", return_value=completed):
            metadata = anime_clip_caption.probe_video(Path("/tmp/test.mp4"))

        self.assertEqual(metadata, anime_clip_caption.VideoMetadata(duration=4.25, has_audio_track=True))

    def test_model_supports_audio_checks_audio_config(self) -> None:
        audio_model = FakeModel(config=SimpleNamespace(audio_config=object()))
        vision_model = FakeModel(config=SimpleNamespace(audio_config=None))

        self.assertTrue(anime_clip_caption.model_supports_audio(audio_model))
        self.assertFalse(anime_clip_caption.model_supports_audio(vision_model))

    def test_determine_audio_plan_uses_best_effort_rules(self) -> None:
        self.assertEqual(
            anime_clip_caption.determine_audio_plan(
                anime_clip_caption.VideoMetadata(duration=3.0, has_audio_track=False),
                audio_supported=True,
            ),
            (False, "clip has no audio track"),
        )
        self.assertEqual(
            anime_clip_caption.determine_audio_plan(
                anime_clip_caption.VideoMetadata(duration=3.0, has_audio_track=True),
                audio_supported=False,
            ),
            (False, "model does not support audio input"),
        )
        self.assertEqual(
            anime_clip_caption.determine_audio_plan(
                anime_clip_caption.VideoMetadata(duration=31.0, has_audio_track=True),
                audio_supported=True,
            ),
            (False, "clip audio exceeds 30s limit"),
        )
        self.assertEqual(
            anime_clip_caption.determine_audio_plan(
                anime_clip_caption.VideoMetadata(duration=5.0, has_audio_track=True),
                audio_supported=True,
            ),
            (True, None),
        )

    def test_build_prompt_vision_only_requests_single_plain_paragraph(self) -> None:
        prompt = anime_clip_caption.build_prompt(include_audio=False)

        self.assertIn("one plain paragraph", prompt)
        self.assertIn("Do not use labels like SCENE_OVERVIEW", prompt)
        self.assertIn("Only describe visual content", prompt)
        self.assertNotIn("Blend any clearly audible", prompt)

    def test_build_prompt_includes_helper_tag_hints(self) -> None:
        prompt = anime_clip_caption.build_prompt(
            include_audio=False,
            tag_hints=anime_clip_caption.TagHints(
                general_tags=("hallway", "dramatic lighting"),
                character_tags=("hero",),
            ),
        )

        self.assertIn("Helper scene hints", prompt)
        self.assertIn("Character hints: hero", prompt)
        self.assertIn("Visual hints: hallway, dramatic lighting", prompt)

    def test_load_tag_hints_matches_clip_by_relpath(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            nested_dir = root / "nested"
            nested_dir.mkdir()
            clip = nested_dir / "clip.mp4"
            clip.write_bytes(b"a")
            tags_file = root / anime_clip_caption.DEFAULT_TAGS_FILENAME
            tags_file.write_text(
                json.dumps(
                    {
                        "clip_relpath": "nested/clip.mp4",
                        "general_tags": ["fight", "smear"],
                        "character_tags": ["hero"],
                        "status": "tagged",
                    }
                )
                + "\n"
                + json.dumps(
                    {
                        "clip_relpath": "nested/ignored.mp4",
                        "general_tags": ["unused"],
                        "status": "failed",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            lookup = anime_clip_caption.load_tag_hints(tags_file, root)
            tag_hints = anime_clip_caption.find_tag_hints(lookup, clip, root)

        self.assertEqual(
            tag_hints,
            anime_clip_caption.TagHints(general_tags=("fight", "smear"), character_tags=("hero",)),
        )

    def test_normalize_caption_text_collapses_plain_multiline_output(self) -> None:
        normalized = anime_clip_caption.normalize_caption_text(
            """
An energetic action cut.

A character dashes across the frame.
""".strip()
        )

        self.assertEqual(normalized, "An energetic action cut. A character dashes across the frame.")

    def test_normalize_caption_text_strips_legacy_section_headers(self) -> None:
        normalized = anime_clip_caption.normalize_caption_text(
            """
SCENE_OVERVIEW:
An energetic action cut.

VISUAL_DETAILS:
A character dashes across the frame.

DIALOGUE:
None.

OTHER_SOUNDS:
Wind.
""".strip()
        )

        self.assertEqual(
            normalized,
            "An energetic action cut. A character dashes across the frame. None. Wind.",
        )

    def test_normalize_caption_text_strips_visual_and_audio_prefixes(self) -> None:
        normalized = anime_clip_caption.normalize_caption_text(
            """
VISUAL:
Blue dusk over a city street.

AUDIO:
Traffic hums while a distant voice calls out.
""".strip()
        )

        self.assertEqual(normalized, "Blue dusk over a city street. Traffic hums while a distant voice calls out.")

    def test_normalize_caption_text_rejects_blank_or_label_only_output(self) -> None:
        with self.assertRaises(ValueError):
            anime_clip_caption.normalize_caption_text("SCENE_OVERVIEW:\n\nVISUAL:\n")

    def test_load_captioning_components_moves_model_to_requested_device(self) -> None:
        fake_model = FakeModel(config=SimpleNamespace(audio_config=object()))
        fake_processor = object()
        fake_torch = SimpleNamespace(float16="float16", bfloat16="bfloat16", float32="float32")

        with patch.object(anime_clip_caption, "torch", fake_torch), patch.object(
            anime_clip_caption, "AutoProcessor", SimpleNamespace(from_pretrained=lambda repo: fake_processor)
        ), patch.object(
            anime_clip_caption,
            "AutoModelForMultimodalLM",
            SimpleNamespace(from_pretrained=lambda repo, torch_dtype: fake_model),
        ):
            processor, model, supports_audio = anime_clip_caption.load_captioning_components(
                model_repo="google/gemma-4-E4B-it",
                device="cuda",
                dtype_name="float16",
            )

        self.assertIs(processor, fake_processor)
        self.assertIs(model, fake_model)
        self.assertTrue(model.was_eval)
        self.assertEqual(model.device, "cuda")
        self.assertTrue(supports_audio)


class AnimeClipCaptionProcessTests(unittest.TestCase):
    def test_process_video_uses_audio_when_available(self) -> None:
        processor = FakeProcessor(
            """
SCENE_OVERVIEW:
Two characters face each other in a hallway.

VISUAL_DETAILS:
The camera starts in a medium shot and slowly pushes in while one character steps forward.

DIALOGUE:
"Let's go."

OTHER_SOUNDS:
Soft footsteps and room tone.
""".strip()
        )
        model = FakeModel(config=SimpleNamespace(audio_config=object()))

        with patch(
            "anime_clip_caption.probe_video",
            return_value=anime_clip_caption.VideoMetadata(duration=5.0, has_audio_track=True),
        ):
            record = anime_clip_caption.process_video(
                processor=processor,
                model=model,
                clip_path=Path("/tmp/clip.mp4"),
                model_repo="google/gemma-4-E4B-it",
                num_frames=12,
                max_new_tokens=768,
                audio_supported=True,
                tag_hints=anime_clip_caption.TagHints(
                    general_tags=("hallway", "dramatic lighting"),
                    character_tags=("hero", "rival"),
                ),
            )

        self.assertEqual(record["status"], "captioned")
        self.assertTrue(record["used_audio"])
        self.assertIsNone(record["audio_skip_reason"])
        self.assertEqual(processor.decoded_ids, [21, 22])
        self.assertTrue(processor.apply_calls[0]["kwargs"]["load_audio_from_video"])
        self.assertIn("Blend any clearly audible", processor.apply_calls[0]["messages"][1]["content"][1]["text"])
        self.assertIn("Character hints: hero, rival", processor.apply_calls[0]["messages"][1]["content"][1]["text"])
        self.assertIn(
            "Visual hints: hallway, dramatic lighting",
            processor.apply_calls[0]["messages"][1]["content"][1]["text"],
        )
        self.assertEqual(
            record["caption"],
            'Two characters face each other in a hallway. The camera starts in a medium shot and slowly pushes in while one character steps forward. "Let\'s go." Soft footsteps and room tone.',
        )
        self.assertIsNone(record["scene_overview"])
        self.assertIsNone(record["visual_details"])
        self.assertIsNone(record["dialogue"])
        self.assertIsNone(record["other_sounds"])

    def test_process_video_falls_back_to_vision_only_when_audio_is_unavailable(self) -> None:
        processor = FakeProcessor(
            """
A bright outdoor crowd shot.

The camera pans across the audience under warm sunset light.
""".strip()
        )
        model = FakeModel(config=SimpleNamespace(audio_config=None))

        with patch(
            "anime_clip_caption.probe_video",
            return_value=anime_clip_caption.VideoMetadata(duration=5.0, has_audio_track=True),
        ):
            record = anime_clip_caption.process_video(
                processor=processor,
                model=model,
                clip_path=Path("/tmp/clip.mp4"),
                model_repo="google/gemma-4-31B-it",
                num_frames=12,
                max_new_tokens=768,
                audio_supported=False,
            )

        self.assertEqual(record["status"], "captioned")
        self.assertFalse(record["used_audio"])
        self.assertEqual(record["audio_skip_reason"], "model does not support audio input")
        self.assertFalse(processor.apply_calls[0]["kwargs"]["load_audio_from_video"])
        self.assertIn("Only describe visual content", processor.apply_calls[0]["messages"][1]["content"][1]["text"])
        self.assertNotIn("Blend any clearly audible", processor.apply_calls[0]["messages"][1]["content"][1]["text"])
        self.assertEqual(
            record["caption"],
            "A bright outdoor crowd shot. The camera pans across the audience under warm sunset light.",
        )
        self.assertIsNone(record["scene_overview"])
        self.assertIsNone(record["visual_details"])
        self.assertIsNone(record["dialogue"])
        self.assertIsNone(record["other_sounds"])

    def test_process_video_marks_long_clips_as_failed(self) -> None:
        processor = FakeProcessor("unused")
        model = FakeModel(config=SimpleNamespace(audio_config=object()))

        with patch(
            "anime_clip_caption.probe_video",
            return_value=anime_clip_caption.VideoMetadata(duration=61.0, has_audio_track=True),
        ):
            record = anime_clip_caption.process_video(
                processor=processor,
                model=model,
                clip_path=Path("/tmp/clip.mp4"),
                model_repo="google/gemma-4-E4B-it",
                num_frames=12,
                max_new_tokens=768,
                audio_supported=True,
            )

        self.assertEqual(record["status"], "failed")
        self.assertIn("60s video limit", record["error"])
        self.assertEqual(processor.apply_calls, [])


class AnimeClipCaptionRunTests(unittest.TestCase):
    def test_run_returns_one_for_empty_input_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            args = argparse.Namespace(
                input_dir=Path(temp_dir),
                model_repo="google/gemma-4-E4B-it",
                output_file=Path(temp_dir) / "captions.jsonl",
                tags_file=None,
                device="cuda",
                dtype="auto",
                num_frames=12,
                max_new_tokens=768,
            )

            exit_code = anime_clip_caption.run(args)

        self.assertEqual(exit_code, 1)

    def test_run_writes_manifest_and_returns_one_when_any_video_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "a.mp4"
            second = root / "b.webm"
            first.write_bytes(b"a")
            second.write_bytes(b"b")
            output_file = root / "captions.jsonl"
            args = argparse.Namespace(
                input_dir=root,
                model_repo="google/gemma-4-E4B-it",
                output_file=output_file,
                tags_file=None,
                device="cuda",
                dtype="auto",
                num_frames=12,
                max_new_tokens=768,
            )

            with patch("anime_clip_caption.require_runtime"), patch(
                "anime_clip_caption.load_captioning_components",
                return_value=(object(), object(), True),
            ), patch(
                "anime_clip_caption.process_video",
                side_effect=[
                    make_record(first, "captioned"),
                    make_record(second, "failed", error="boom"),
                ],
            ):
                exit_code = anime_clip_caption.run(args)

            self.assertEqual(exit_code, 1)
            lines = output_file.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            first_record = json.loads(lines[0])
            second_record = json.loads(lines[1])
            self.assertEqual(first_record["status"], "captioned")
            self.assertEqual(second_record["status"], "failed")
            self.assertEqual(second_record["error"], "boom")

    def test_run_returns_zero_when_all_videos_are_captioned(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            clip = root / "a.mp4"
            clip.write_bytes(b"a")
            args = argparse.Namespace(
                input_dir=root,
                model_repo="google/gemma-4-E4B-it",
                output_file=root / "captions.jsonl",
                tags_file=None,
                device="cuda",
                dtype="auto",
                num_frames=12,
                max_new_tokens=768,
            )

            with patch("anime_clip_caption.require_runtime"), patch(
                "anime_clip_caption.load_captioning_components",
                return_value=(object(), object(), True),
            ), patch(
                "anime_clip_caption.process_video",
                return_value=make_record(clip, "captioned"),
            ):
                exit_code = anime_clip_caption.run(args)

        self.assertEqual(exit_code, 0)

    def test_run_autodetects_tags_file_and_passes_matching_hints(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            clip = root / "a.mp4"
            clip.write_bytes(b"a")
            (root / anime_clip_caption.DEFAULT_TAGS_FILENAME).write_text(
                json.dumps(
                    {
                        "clip_path": str(clip),
                        "general_tags": ["crowd", "sunset"],
                        "character_tags": ["hero"],
                        "status": "tagged",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            args = argparse.Namespace(
                input_dir=root,
                model_repo="google/gemma-4-E4B-it",
                output_file=root / "captions.jsonl",
                tags_file=None,
                device="cuda",
                dtype="auto",
                num_frames=12,
                max_new_tokens=768,
            )
            seen_hints: list[anime_clip_caption.TagHints | None] = []

            def fake_process_video(**kwargs):
                seen_hints.append(kwargs.get("tag_hints"))
                return make_record(clip, "captioned")

            with patch("anime_clip_caption.require_runtime"), patch(
                "anime_clip_caption.load_captioning_components",
                return_value=(object(), object(), True),
            ), patch(
                "anime_clip_caption.process_video",
                side_effect=fake_process_video,
            ):
                exit_code = anime_clip_caption.run(args)

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            seen_hints,
            [anime_clip_caption.TagHints(general_tags=("crowd", "sunset"), character_tags=("hero",))],
        )


if __name__ == "__main__":
    unittest.main()
