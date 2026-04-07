from __future__ import annotations

import argparse
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import anime_clip_extract


class FakeTensor:
    def __init__(self, values: list[float]):
        self._values = np.asarray(values, dtype=float)

    def detach(self) -> "FakeTensor":
        return self

    def cpu(self) -> "FakeTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self._values


class FakeModel:
    def __init__(self, single_frame_predictions: list[float], all_frame_predictions: list[float]):
        self.single_frame_predictions = FakeTensor(single_frame_predictions)
        self.all_frame_predictions = FakeTensor(all_frame_predictions)
        self.requested_paths: list[str] = []

    def predict_video(self, video_path: str, quiet: bool = False):
        self.requested_paths.append(video_path)
        self.quiet = quiet
        return None, self.single_frame_predictions, self.all_frame_predictions


class AnimeClipExtractHelperTests(unittest.TestCase):
    def test_combine_boundary_scores_uses_max_of_both_predictions_for_high_recall(self) -> None:
        scores = anime_clip_extract.combine_boundary_scores(
            FakeTensor([0.1, 0.9, 0.2]),
            FakeTensor([0.5, 0.3, 0.7]),
        )

        np.testing.assert_allclose(scores, np.array([0.5, 0.9, 0.7]))

    def test_build_unsafe_spans_marks_boundary_when_only_one_head_crosses_threshold(self) -> None:
        scores = anime_clip_extract.combine_boundary_scores(
            FakeTensor([0.0, 0.4, 0.0, 0.0]),
            FakeTensor([0.0, 0.0, 0.0, 0.0]),
        )

        spans = anime_clip_extract.build_unsafe_spans(
            boundary_scores=scores,
            fps=2.0,
            duration=2.0,
            threshold=0.35,
            padding=0.0,
            frame_guard=anime_clip_extract.compute_frame_guard(2.0),
        )

        self.assertEqual(spans, [(0.0, 1.5)])

    def test_build_unsafe_spans_merges_overlapping_padding(self) -> None:
        spans = anime_clip_extract.build_unsafe_spans(
            boundary_scores=np.array([0.0, 0.4, 0.4, 0.0, 0.38, 0.0]),
            fps=2.0,
            duration=4.0,
            threshold=0.35,
            padding=0.5,
        )

        self.assertEqual(spans, [(0.0, 3.0)])

    def test_build_unsafe_spans_adds_one_frame_guard_when_padding_is_zero(self) -> None:
        spans = anime_clip_extract.build_unsafe_spans(
            boundary_scores=np.array([0.0, 0.4, 0.0, 0.0]),
            fps=2.0,
            duration=2.0,
            threshold=0.35,
            padding=0.0,
            frame_guard=anime_clip_extract.compute_frame_guard(2.0),
        )

        self.assertEqual(spans, [(0.0, 1.5)])

    def test_select_safe_spans_filters_segments_shorter_than_minimum(self) -> None:
        safe_spans = anime_clip_extract.select_safe_spans(
            boundary_scores=np.array([0.0, 0.5, 0.0, 0.0, 0.5, 0.0]),
            fps=2.0,
            duration=3.0,
            threshold=0.35,
            padding=0.25,
            min_duration=0.1,
        )

        self.assertEqual(safe_spans, [(0.0, 0.25), (1.25, 1.75), (2.75, 3.0)])
        filtered = anime_clip_extract.filter_safe_spans(safe_spans, min_duration=0.8)
        self.assertEqual(filtered, [])

    def test_select_safe_spans_and_tiling_respect_one_frame_guard_around_cut(self) -> None:
        safe_spans = anime_clip_extract.select_safe_spans(
            boundary_scores=np.array([0.0, 0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 0.0]),
            fps=4.0,
            duration=2.0,
            threshold=0.35,
            padding=0.0,
            min_duration=0.5,
            frame_guard=anime_clip_extract.compute_frame_guard(4.0),
        )

        self.assertEqual(safe_spans, [(0.0, 0.5), (1.25, 2.0)])

        clips = anime_clip_extract.tile_safe_spans(
            safe_spans=safe_spans,
            min_duration=0.5,
            max_duration=0.5,
            min_gap=0.0,
            rng=anime_clip_extract.make_video_rng(0, Path("/tmp/cut.mp4")),
        )

        self.assertEqual(
            clips,
            [
                anime_clip_extract.ClipWindow(start=0.0, end=0.5),
                anime_clip_extract.ClipWindow(start=1.25, end=1.75),
            ],
        )

    def test_tile_safe_spans_is_deterministic_and_non_overlapping(self) -> None:
        rng_a = anime_clip_extract.make_video_rng(7, Path("/tmp/a.mp4"))
        rng_b = anime_clip_extract.make_video_rng(7, Path("/tmp/a.mp4"))

        clips_a = anime_clip_extract.tile_safe_spans(
            safe_spans=[(0.0, 7.0)],
            min_duration=1.0,
            max_duration=2.5,
            min_gap=0.25,
            rng=rng_a,
        )
        clips_b = anime_clip_extract.tile_safe_spans(
            safe_spans=[(0.0, 7.0)],
            min_duration=1.0,
            max_duration=2.5,
            min_gap=0.25,
            rng=rng_b,
        )

        self.assertEqual(clips_a, clips_b)
        for clip in clips_a:
            self.assertGreaterEqual(clip.duration, 1.0)
            self.assertLessEqual(clip.duration, 2.5)
            self.assertGreaterEqual(clip.start, 0.0)
            self.assertLessEqual(clip.end, 7.0)
        for previous, current in zip(clips_a, clips_a[1:]):
            self.assertGreaterEqual(current.start + anime_clip_extract.EPSILON, previous.end + 0.25)

    def test_build_ffmpeg_command_preserves_optional_audio_mapping(self) -> None:
        command = anime_clip_extract.build_ffmpeg_command(
            source_path=Path("/tmp/source.webm"),
            clip=anime_clip_extract.ClipWindow(start=1.25, end=3.0),
            output_path=Path("/tmp/out.mp4"),
        )

        self.assertIn("0:v:0", command)
        self.assertIn("0:a?", command)
        self.assertIn("libx264", command)
        self.assertIn("/tmp/out.mp4", command)

    def test_probe_video_parses_duration_and_fps(self) -> None:
        payload = {
            "format": {"duration": "12.5"},
            "streams": [
                {"codec_type": "audio"},
                {"codec_type": "video", "avg_frame_rate": "24000/1001"},
            ],
        }
        completed = subprocess.CompletedProcess(
            args=["ffprobe"],
            returncode=0,
            stdout=json.dumps(payload),
            stderr="",
        )

        with patch("anime_clip_extract.subprocess.run", return_value=completed) as run_mock:
            probe = anime_clip_extract.probe_video(Path("/tmp/test.mp4"))

        run_mock.assert_called_once()
        self.assertAlmostEqual(probe.duration, 12.5)
        self.assertAlmostEqual(probe.fps, 24000 / 1001)


class AnimeClipExtractCliTests(unittest.TestCase):
    def test_parser_uses_high_recall_boundary_defaults(self) -> None:
        parser = anime_clip_extract.build_parser()

        args = parser.parse_args(["--input-dir", "/tmp"])

        self.assertEqual(args.boundary_threshold, 0.25)
        self.assertEqual(args.boundary_padding, 0.75)

    def test_parser_rejects_negative_padding(self) -> None:
        parser = anime_clip_extract.build_parser()

        with self.assertRaises(SystemExit):
            parser.parse_args(["--input-dir", "/tmp", "--boundary-padding", "-0.1"])

    def test_validate_args_rejects_duration_range(self) -> None:
        parser = anime_clip_extract.build_parser()
        args = parser.parse_args(["--input-dir", "/tmp", "--min-duration", "5", "--max-duration", "1"])

        with self.assertRaises(SystemExit):
            anime_clip_extract.validate_args(parser, args)

    def test_require_runtime_reports_missing_system_tool(self) -> None:
        with patch("anime_clip_extract.shutil.which", side_effect=lambda name: None if name == "ffmpeg" else "/usr/bin/ffprobe"):
            with self.assertRaises(SystemExit) as context:
                anime_clip_extract.require_runtime()

        self.assertIn("ffmpeg", str(context.exception))

    def test_run_returns_one_for_empty_input_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            args = argparse.Namespace(
                input_dir=Path(temp_dir),
                output_dir=Path(temp_dir) / "clips",
                min_duration=1.0,
                max_duration=5.0,
                boundary_threshold=0.25,
                boundary_padding=0.75,
                min_gap=0.25,
                device="cpu",
                seed=0,
            )
            stderr = io.StringIO()

            with patch("sys.stderr", stderr):
                exit_code = anime_clip_extract.run(args)

        self.assertEqual(exit_code, 1)
        self.assertIn("No supported videos found", stderr.getvalue())

    def test_run_returns_one_when_any_video_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "a.mp4"
            second = root / "b.webm"
            first.write_bytes(b"a")
            second.write_bytes(b"b")
            args = argparse.Namespace(
                input_dir=root,
                output_dir=root / "clips",
                min_duration=1.0,
                max_duration=5.0,
                boundary_threshold=0.25,
                boundary_padding=0.75,
                min_gap=0.25,
                device="cpu",
                seed=0,
            )

            with patch("anime_clip_extract.require_runtime"), patch(
                "anime_clip_extract.load_model", return_value=object()
            ), patch(
                "anime_clip_extract.process_video",
                side_effect=[
                    anime_clip_extract.VideoProcessResult(video_path=first, clips_created=2),
                    RuntimeError("boom"),
                ],
            ):
                exit_code = anime_clip_extract.run(args)

        self.assertEqual(exit_code, 1)

    def test_run_returns_one_when_zero_clips_are_created(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video = root / "a.mp4"
            video.write_bytes(b"a")
            args = argparse.Namespace(
                input_dir=root,
                output_dir=root / "clips",
                min_duration=1.0,
                max_duration=5.0,
                boundary_threshold=0.25,
                boundary_padding=0.75,
                min_gap=0.25,
                device="cpu",
                seed=0,
            )

            with patch("anime_clip_extract.require_runtime"), patch(
                "anime_clip_extract.load_model", return_value=object()
            ), patch(
                "anime_clip_extract.process_video",
                return_value=anime_clip_extract.VideoProcessResult(video_path=video, clips_created=0),
            ):
                exit_code = anime_clip_extract.run(args)

        self.assertEqual(exit_code, 1)


class AnimeClipExtractProcessTests(unittest.TestCase):
    def test_process_video_writes_manifest_and_clip_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video_path = root / "episode01.webm"
            video_path.write_bytes(b"video")
            output_dir = root / "clips"
            manifest_path = output_dir / "manifest.jsonl"
            output_dir.mkdir()
            model = FakeModel(
                single_frame_predictions=[0.0, 0.0, 0.0, 0.0, 0.0],
                all_frame_predictions=[0.0, 0.0, 0.0, 0.0, 0.0],
            )
            extracted_paths: list[Path] = []

            def fake_extract_clip(source_path: Path, clip: anime_clip_extract.ClipWindow, output_path: Path) -> None:
                self.assertEqual(source_path, video_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(f"{clip.start:.3f}-{clip.end:.3f}".encode("utf-8"))
                extracted_paths.append(output_path)

            with manifest_path.open("w", encoding="utf-8") as manifest_handle, patch(
                "anime_clip_extract.probe_video",
                return_value=anime_clip_extract.VideoProbe(duration=2.3, fps=2.0),
            ), patch("anime_clip_extract.extract_clip", side_effect=fake_extract_clip):
                result = anime_clip_extract.process_video(
                    model=model,
                    video_path=video_path,
                    output_dir=output_dir,
                    min_duration=1.0,
                    max_duration=1.0,
                    threshold=0.25,
                    padding=0.75,
                    min_gap=0.25,
                    seed=11,
                    manifest_handle=manifest_handle,
                )

            self.assertEqual(result.clips_created, 2)
            self.assertEqual([path.name for path in extracted_paths], ["clip_1_0_1000.mp4", "clip_2_1250_2250.mp4"])

            lines = manifest_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            first_record = json.loads(lines[0])
            self.assertEqual(first_record["source_path"], str(video_path))
            self.assertEqual(first_record["clip_path"], str(extracted_paths[0]))
            self.assertEqual(first_record["boundary_threshold"], 0.25)
            self.assertEqual(first_record["boundary_padding"], 0.75)
            self.assertEqual(first_record["boundary_mode"], "high_recall_any")
            self.assertEqual(first_record["seed"], 11)


if __name__ == "__main__":
    unittest.main()
