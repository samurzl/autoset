from __future__ import annotations

import argparse
import io
import json
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import anime_video_review


def make_gif_bytes() -> bytes:
    first = Image.new("RGB", (8, 8), "black")
    second = Image.new("RGB", (8, 8), "white")
    payload = io.BytesIO()
    first.save(payload, format="GIF", save_all=True, append_images=[second], duration=100, loop=0)
    return payload.getvalue()


class AnimeVideoReviewCliTests(unittest.TestCase):
    def test_validate_args_sets_default_paths(self) -> None:
        parser = anime_video_review.build_parser()

        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "downloads"
            input_dir.mkdir()
            args = parser.parse_args(["--input-dir", str(input_dir)])

            anime_video_review.validate_args(parser, args)

            self.assertEqual(
                args.keep_dir,
                input_dir.resolve().with_name(f"{input_dir.name}{anime_video_review.DEFAULT_KEEP_SUFFIX}"),
            )
            self.assertEqual(
                args.reject_dir,
                input_dir.resolve().with_name(f"{input_dir.name}{anime_video_review.DEFAULT_REJECT_SUFFIX}"),
            )
            self.assertEqual(
                args.review_log,
                input_dir.resolve().with_name(f"{input_dir.name}{anime_video_review.DEFAULT_REVIEW_LOG_SUFFIX}"),
            )
            self.assertEqual(args.preview_duration, anime_video_review.DEFAULT_PREVIEW_DURATION)

    def test_compute_preview_window_clamps_short_videos(self) -> None:
        full_window = anime_video_review.compute_preview_window(
            4.0,
            preview_duration=anime_video_review.DEFAULT_PREVIEW_DURATION,
        )
        short_window = anime_video_review.compute_preview_window(
            0.6,
            preview_duration=anime_video_review.DEFAULT_PREVIEW_DURATION,
        )

        self.assertEqual(
            full_window,
            anime_video_review.PreviewWindow(start=1.0, duration=anime_video_review.DEFAULT_PREVIEW_DURATION),
        )
        self.assertEqual(short_window, anime_video_review.PreviewWindow(start=0.0, duration=0.6))

    def test_collect_video_files_ignores_partials_and_excluded_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "downloads"
            keep_dir = input_dir / "kept"
            input_dir.mkdir()
            keep_dir.mkdir()
            (input_dir / "a.mp4").write_bytes(b"a")
            (input_dir / "b.mp4.part").write_bytes(b"b")
            (keep_dir / "c.mp4").write_bytes(b"c")

            files = anime_video_review.collect_video_files(input_dir, exclude_dirs=(keep_dir,))

        self.assertEqual(files, [(input_dir / "a.mp4").resolve()])


class AnimeVideoReviewSessionTests(unittest.TestCase):
    def make_session(self, root: Path) -> anime_video_review.ReviewSession:
        input_dir = root / "downloads"
        keep_dir = root / "downloads_keep"
        reject_dir = root / "downloads_reject"
        input_dir.mkdir(exist_ok=True)
        return anime_video_review.ReviewSession(
            input_dir=input_dir,
            keep_dir=keep_dir,
            reject_dir=reject_dir,
            review_log=root / "downloads_review.jsonl",
        )

    def test_scan_input_dir_picks_up_newly_completed_video(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            session = self.make_session(root)
            input_dir = session.input_dir

            self.assertEqual(session.scan_input_dir(), [])
            (input_dir / "101.mp4.part").write_bytes(b"partial")
            self.assertEqual(session.scan_input_dir(), [])

            completed = input_dir / "101.mp4"
            completed.write_bytes(b"video")
            discovered = session.scan_input_dir()

            self.assertEqual([entry.video_id for entry in discovered], ["101"])
            self.assertIn("101", session.entries)

    def test_default_accept_commits_only_when_leaving_current_video(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            session = self.make_session(root)
            for name in ("a.mp4", "b.mp4", "c.mp4"):
                (session.input_dir / name).write_bytes(name.encode("utf-8"))

            controller = anime_video_review.ReviewController(session)
            controller.refresh_queue()

            self.assertEqual(controller.current_video_id, "a")
            self.assertEqual(session.entries["a"].status, anime_video_review.STATUS_PENDING)
            self.assertEqual(session.entries["b"].status, anime_video_review.STATUS_PENDING)
            self.assertIsNone(session.entries["b"].seen_order)
            self.assertIsNone(session.entries["c"].seen_order)
            self.assertEqual(controller.get_visual_status(), anime_video_review.STATUS_ACCEPTED)

            controller.navigate(1)

            self.assertEqual(session.entries["a"].status, anime_video_review.STATUS_ACCEPTED)
            self.assertEqual(session.entries["a"].current_path, (session.keep_dir / "a.mp4").resolve())
            self.assertEqual(controller.current_video_id, "b")
            self.assertEqual(session.entries["b"].status, anime_video_review.STATUS_PENDING)
            self.assertIsNotNone(session.entries["b"].seen_order)
            self.assertIsNone(session.entries["c"].seen_order)
            self.assertEqual(session.entries["c"].status, anime_video_review.STATUS_PENDING)

    def test_accept_and_reject_move_files_immediately(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            session = self.make_session(root)
            for name in ("a.mp4", "b.mp4"):
                (session.input_dir / name).write_bytes(name.encode("utf-8"))

            controller = anime_video_review.ReviewController(session)
            controller.refresh_queue()

            controller.reject_current()
            controller.accept_current()

            self.assertTrue((session.reject_dir / "a.mp4").exists())
            self.assertTrue((session.keep_dir / "b.mp4").exists())
            self.assertFalse((session.input_dir / "a.mp4").exists())
            self.assertFalse((session.input_dir / "b.mp4").exists())

            rows = [
                json.loads(line)
                for line in session.review_log.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertIn(
                ("a", anime_video_review.STATUS_REJECTED),
                {(row["video_id"], row["status"]) for row in rows},
            )
            self.assertIn(
                ("b", anime_video_review.STATUS_ACCEPTED),
                {(row["video_id"], row["status"]) for row in rows},
            )

    def test_reversing_decision_moves_video_between_folders(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            session = self.make_session(root)
            video_path = session.input_dir / "a.mp4"
            video_path.write_bytes(b"a")

            session.scan_input_dir()
            session.reject("a")
            session.accept("a")

            self.assertFalse((session.reject_dir / "a.mp4").exists())
            self.assertTrue((session.keep_dir / "a.mp4").exists())
            self.assertEqual(session.entries["a"].status, anime_video_review.STATUS_ACCEPTED)

            last_row = json.loads(session.review_log.read_text(encoding="utf-8").splitlines()[-1])
            self.assertEqual(last_row["status"], anime_video_review.STATUS_ACCEPTED)
            self.assertEqual(Path(last_row["current_path"]), (session.keep_dir / "a.mp4").resolve())

    def test_load_restores_latest_state_and_history(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            session = self.make_session(root)
            for name in ("a.mp4", "b.mp4"):
                (session.input_dir / name).write_bytes(name.encode("utf-8"))

            controller = anime_video_review.ReviewController(session)
            controller.refresh_queue()
            controller.navigate(1)

            restored = self.make_session(root)
            restored.load()

            self.assertEqual(restored.entries["a"].status, anime_video_review.STATUS_ACCEPTED)
            self.assertEqual(restored.entries["a"].current_path, (restored.keep_dir / "a.mp4").resolve())
            self.assertEqual(restored.entries["a"].seen_order, 1)
            self.assertEqual(restored.entries["b"].status, anime_video_review.STATUS_PENDING)
            self.assertEqual(restored.entries["b"].seen_order, 2)
            self.assertEqual(restored.choose_start_video_id(), "b")
            self.assertEqual(restored.ordered_video_ids(), ["a", "b"])

    def test_ordered_video_ids_uses_motion_score_for_unseen_items(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            session = self.make_session(root)
            for name in ("a.mp4", "b.mp4", "c.mp4"):
                entry = session.add_discovered_video(session.input_dir / name)
                entry.motion_score = None
            session.entries["a"].motion_score = 0.9
            session.entries["b"].motion_score = 0.2
            session.entries["c"].motion_score = None

            self.assertEqual(session.ordered_video_ids(), ["b", "a", "c"])


class AnimeVideoReviewPreviewTests(unittest.TestCase):
    def test_extract_preview_uses_midpoint_window(self) -> None:
        completed = subprocess.CompletedProcess(args=["ffmpeg"], returncode=0, stdout=make_gif_bytes(), stderr=b"")

        with patch(
            "anime_video_review.probe_video",
            return_value=anime_video_review.VideoMetadata(duration=4.0),
        ), patch(
            "anime_video_review.subprocess.run",
            return_value=completed,
        ) as run_mock:
            preview = anime_video_review.extract_preview(Path("/tmp/example.mp4"))

        self.assertEqual(preview.preview_start, 1.0)
        self.assertEqual(preview.preview_duration, anime_video_review.DEFAULT_PREVIEW_DURATION)
        command = run_mock.call_args.args[0]
        self.assertEqual(command[command.index("-ss") + 1], "1.000000")
        self.assertEqual(command[command.index("-t") + 1], "2.000000")
        self.assertGreaterEqual(preview.motion_score, 0.0)

    def test_extract_preview_clamps_short_video_duration(self) -> None:
        completed = subprocess.CompletedProcess(args=["ffmpeg"], returncode=0, stdout=make_gif_bytes(), stderr=b"")

        with patch(
            "anime_video_review.probe_video",
            return_value=anime_video_review.VideoMetadata(duration=0.6),
        ), patch(
            "anime_video_review.subprocess.run",
            return_value=completed,
        ) as run_mock:
            preview = anime_video_review.extract_preview(Path("/tmp/example.mp4"))

        self.assertEqual(preview.preview_start, 0.0)
        self.assertEqual(preview.preview_duration, 0.6)
        command = run_mock.call_args.args[0]
        self.assertEqual(command[command.index("-ss") + 1], "0.000000")
        self.assertEqual(command[command.index("-t") + 1], "0.600000")

    def test_preview_manager_loads_in_background(self) -> None:
        release = threading.Event()

        def loader(path: Path) -> anime_video_review.PreviewBundle:
            release.wait(timeout=2.0)
            return anime_video_review.PreviewBundle(
                frames=[Image.new("RGB", (4, 4), "white")],
                preview_start=0.0,
                preview_duration=anime_video_review.DEFAULT_PREVIEW_DURATION,
                motion_score=0.0,
                frame_delay_ms=100,
            )

        manager = anime_video_review.PreviewManager(loader=loader, max_workers=1)
        try:
            manager.request("video", Path("/tmp/video.mp4"))
            self.assertIn("video", manager.inflight)
            self.assertEqual(manager.drain_finished(), [])

            release.set()
            deadline = time.time() + 2.0
            resolved: list[anime_video_review.ResolvedPreview] = []
            while time.time() < deadline and not resolved:
                resolved = manager.drain_finished()
                if not resolved:
                    time.sleep(0.01)

            self.assertEqual(len(resolved), 1)
            self.assertIsNotNone(resolved[0].bundle)
            self.assertIsNone(resolved[0].error)
        finally:
            manager.close()

    def test_preview_manager_reorders_pending_preloads(self) -> None:
        release = threading.Event()

        def loader(path: Path) -> anime_video_review.PreviewBundle:
            if path.stem == "a":
                release.wait(timeout=2.0)
            return anime_video_review.PreviewBundle(
                frames=[Image.new("RGB", (4, 4), "white")],
                preview_start=0.0,
                preview_duration=anime_video_review.DEFAULT_PREVIEW_DURATION,
                motion_score=0.0,
                frame_delay_ms=100,
            )

        manager = anime_video_review.PreviewManager(loader=loader, max_workers=1)
        try:
            manager.request_many(
                [
                    ("a", Path("/tmp/a.mp4")),
                    ("b", Path("/tmp/b.mp4")),
                    ("c", Path("/tmp/c.mp4")),
                ]
            )
            self.assertEqual(list(manager.inflight), ["a"])
            self.assertEqual(list(manager.pending), ["b", "c"])

            manager.request_many(
                [
                    ("a", Path("/tmp/a.mp4")),
                    ("c", Path("/tmp/c.mp4")),
                    ("b", Path("/tmp/b.mp4")),
                ]
            )
            self.assertEqual(list(manager.pending), ["c", "b"])

            release.set()
            deadline = time.time() + 2.0
            while time.time() < deadline and "c" not in manager.inflight:
                manager.drain_finished()
                time.sleep(0.01)

            self.assertIn("c", manager.inflight)
        finally:
            release.set()
            manager.close()


if __name__ == "__main__":
    unittest.main()
