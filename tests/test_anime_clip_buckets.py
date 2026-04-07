from __future__ import annotations

import argparse
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import anime_clip_buckets


class AnimeClipBucketsHelperTests(unittest.TestCase):
    def test_parser_uses_default_max_bucket_count(self) -> None:
        parser = anime_clip_buckets.build_parser()

        args = parser.parse_args(["--input-dir", "/tmp", "--resolution", "512"])

        self.assertEqual(args.max_buckets, anime_clip_buckets.DEFAULT_MAX_BUCKETS)

    def test_parser_rejects_non_positive_resolution(self) -> None:
        parser = anime_clip_buckets.build_parser()

        with self.assertRaises(SystemExit):
            parser.parse_args(["--input-dir", "/tmp", "--resolution", "0"])

    def test_validate_args_rejects_missing_input_directory(self) -> None:
        parser = anime_clip_buckets.build_parser()
        args = parser.parse_args(["--input-dir", "/definitely/missing", "--resolution", "512"])

        with self.assertRaises(SystemExit):
            anime_clip_buckets.validate_args(parser, args)

    def test_require_runtime_reports_missing_ffprobe(self) -> None:
        with patch("anime_clip_buckets.shutil.which", return_value=None):
            with self.assertRaises(SystemExit) as context:
                anime_clip_buckets.require_runtime()

        self.assertIn("ffprobe", str(context.exception))

    def test_probe_video_parses_width_height_fps_and_nb_frames(self) -> None:
        payload = {
            "format": {"duration": "1.5"},
            "streams": [
                {"codec_type": "audio"},
                {
                    "codec_type": "video",
                    "width": 1280,
                    "height": 720,
                    "avg_frame_rate": "24000/1001",
                    "nb_frames": "37",
                },
            ],
        }
        completed = subprocess.CompletedProcess(
            args=["ffprobe"],
            returncode=0,
            stdout=json.dumps(payload),
            stderr="",
        )

        with patch("anime_clip_buckets.subprocess.run", return_value=completed) as run_mock:
            metadata = anime_clip_buckets.probe_video(Path("/tmp/test.mp4"))

        run_mock.assert_called_once()
        self.assertEqual(metadata.width, 1280)
        self.assertEqual(metadata.height, 720)
        self.assertAlmostEqual(metadata.fps, 24000 / 1001)
        self.assertEqual(metadata.frame_count, 37)

    def test_probe_video_falls_back_to_duration_times_fps_for_frames(self) -> None:
        payload = {
            "format": {"duration": "2.0"},
            "streams": [
                {
                    "codec_type": "video",
                    "width": "640",
                    "height": "480",
                    "avg_frame_rate": "12/1",
                }
            ],
        }
        completed = subprocess.CompletedProcess(
            args=["ffprobe"],
            returncode=0,
            stdout=json.dumps(payload),
            stderr="",
        )

        with patch("anime_clip_buckets.subprocess.run", return_value=completed):
            metadata = anime_clip_buckets.probe_video(Path("/tmp/test.mp4"))

        self.assertEqual(metadata.frame_count, 24)

    def test_quantize_frame_count_uses_legal_one_plus_multiples_of_eight(self) -> None:
        self.assertEqual(anime_clip_buckets.quantize_frame_count(1), 1)
        self.assertEqual(anime_clip_buckets.quantize_frame_count(12), 9)
        self.assertEqual(anime_clip_buckets.quantize_frame_count(16), 17)
        self.assertEqual(anime_clip_buckets.quantize_frame_count(25), 25)

    def test_project_bucket_dimensions_returns_multiples_of_thirty_two(self) -> None:
        width, height = anime_clip_buckets.project_bucket_dimensions(16 / 9, 512)

        self.assertEqual(width % anime_clip_buckets.DIMENSION_MULTIPLE, 0)
        self.assertEqual(height % anime_clip_buckets.DIMENSION_MULTIPLE, 0)
        self.assertGreater(width, height)

    def test_project_bucket_dimensions_uses_nearest_area_when_aspect_is_equal(self) -> None:
        width, height = anime_clip_buckets.project_bucket_dimensions(1.0, 100)

        self.assertEqual((width, height), (96, 96))

    def test_greedy_select_buckets_is_deterministic_for_synthetic_features(self) -> None:
        features = [
            anime_clip_buckets.FeaturePoint(width=512, height=288, frame_count=9, weight=10),
            anime_clip_buckets.FeaturePoint(width=512, height=512, frame_count=33, weight=10),
        ]

        buckets = anime_clip_buckets.greedy_select_buckets(features, resolution=512, max_buckets=2)

        self.assertEqual(
            [bucket.to_text() for bucket in buckets],
            ["512x512x33", "512x288x9"],
        )

    def test_format_buckets_joins_with_semicolons(self) -> None:
        buckets = [
            anime_clip_buckets.Bucket(width=96, height=96, frames=9),
            anime_clip_buckets.Bucket(width=192, height=96, frames=17),
        ]

        self.assertEqual(anime_clip_buckets.format_buckets(buckets), "96x96x9;192x96x17")


class AnimeClipBucketsRunTests(unittest.TestCase):
    def test_run_returns_one_for_empty_input_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            args = argparse.Namespace(
                input_dir=Path(temp_dir),
                resolution=512,
                max_buckets=30,
            )
            stderr = io.StringIO()

            with patch("sys.stderr", stderr):
                exit_code = anime_clip_buckets.run(args)

        self.assertEqual(exit_code, 1)
        self.assertIn("No supported videos found", stderr.getvalue())

    def test_run_returns_one_when_probe_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            clip_path = root / "a.mp4"
            clip_path.write_bytes(b"a")
            args = argparse.Namespace(
                input_dir=root,
                resolution=512,
                max_buckets=30,
            )
            stderr = io.StringIO()

            with patch("anime_clip_buckets.require_runtime"), patch(
                "anime_clip_buckets.probe_video",
                side_effect=RuntimeError("boom"),
            ), patch("sys.stderr", stderr):
                exit_code = anime_clip_buckets.run(args)

        self.assertEqual(exit_code, 1)
        self.assertIn("failed:", stderr.getvalue())
        self.assertIn("boom", stderr.getvalue())

    def test_run_prints_bucket_string_and_honors_max_buckets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "a.mp4"
            second = root / "b.webm"
            third = root / "nested" / "c.mov"
            third.parent.mkdir()
            first.write_bytes(b"a")
            second.write_bytes(b"b")
            third.write_bytes(b"c")
            args = argparse.Namespace(
                input_dir=root,
                resolution=512,
                max_buckets=2,
            )
            stdout = io.StringIO()

            metadata_by_path = {
                first: anime_clip_buckets.ClipMetadata(width=512, height=288, duration=1.0, fps=9.0, frame_count=9),
                second: anime_clip_buckets.ClipMetadata(width=512, height=512, duration=3.0, fps=11.0, frame_count=33),
                third: anime_clip_buckets.ClipMetadata(width=1024, height=1024, duration=3.0, fps=11.0, frame_count=33),
            }

            with patch("anime_clip_buckets.require_runtime"), patch(
                "anime_clip_buckets.probe_video",
                side_effect=lambda path: metadata_by_path[path],
            ), patch("sys.stdout", stdout):
                exit_code = anime_clip_buckets.run(args)

        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout.getvalue().strip(), "512x512x33;512x288x9")


if __name__ == "__main__":
    unittest.main()
