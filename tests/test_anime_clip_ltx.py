from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

import anime_clip_ltx


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    payload = "\n".join(json.dumps(row) for row in rows)
    if payload:
        payload += "\n"
    path.write_text(payload, encoding="utf-8")


def make_caption_record(clip_path: str, caption: str, status: str = "captioned") -> dict[str, object]:
    return {
        "clip_path": clip_path,
        "caption": caption,
        "status": status,
        "error": "" if status == "captioned" else "boom",
    }


class AnimeClipLtxHelperTests(unittest.TestCase):
    def test_validate_args_sets_defaults(self) -> None:
        parser = anime_clip_ltx.build_parser()

        with tempfile.TemporaryDirectory() as temp_dir:
            args = parser.parse_args(["--input-dir", temp_dir])
            anime_clip_ltx.validate_args(parser, args)

        self.assertEqual(args.captions_filename, anime_clip_ltx.DEFAULT_CAPTIONS_FILENAME)
        self.assertEqual(args.output_filename, anime_clip_ltx.DEFAULT_OUTPUT_FILENAME)

    def test_resolve_media_relpath_accepts_absolute_split_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            split_dir = Path(temp_dir) / anime_clip_ltx.TRAIN_SPLIT_NAME
            split_dir.mkdir()
            clip_path = split_dir / "a.mp4"
            clip_path.write_bytes(b"a")

            relpath = anime_clip_ltx.resolve_media_relpath(
                {"clip_path": str(clip_path)},
                split_dir=split_dir,
                split_name=anime_clip_ltx.TRAIN_SPLIT_NAME,
                known_relpaths={"a.mp4"},
            )

        self.assertEqual(relpath, "a.mp4")

    def test_resolve_media_relpath_accepts_split_prefixed_relative_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            split_dir = root / anime_clip_ltx.TRAIN_SPLIT_NAME
            split_dir.mkdir()
            clip_path = split_dir / "a.mp4"
            clip_path.write_bytes(b"a")

            relpath = anime_clip_ltx.resolve_media_relpath(
                {"clip_path": "train/a.mp4"},
                split_dir=split_dir,
                split_name=anime_clip_ltx.TRAIN_SPLIT_NAME,
                known_relpaths={"a.mp4"},
            )

        self.assertEqual(relpath, "a.mp4")


class AnimeClipLtxRunTests(unittest.TestCase):
    def test_run_writes_split_local_ltx_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            train_dir = root / anime_clip_ltx.TRAIN_SPLIT_NAME
            val_dir = root / anime_clip_ltx.VALIDATION_SPLIT_NAME
            train_nested = train_dir / "nested"
            train_nested.mkdir(parents=True)
            val_dir.mkdir()

            train_first = train_nested / "a.mp4"
            train_second = train_dir / "z.mp4"
            val_clip = val_dir / "b.mp4"
            train_first.write_bytes(b"a")
            train_second.write_bytes(b"z")
            val_clip.write_bytes(b"b")

            write_jsonl(
                train_dir / anime_clip_ltx.DEFAULT_CAPTIONS_FILENAME,
                [
                    make_caption_record(str(train_second), "Train Z"),
                    make_caption_record("train/nested/a.mp4", "Train A with visible lines"),
                ],
            )
            write_jsonl(
                val_dir / anime_clip_ltx.DEFAULT_CAPTIONS_FILENAME,
                [make_caption_record(str(val_clip), "Val B")],
            )

            args = argparse.Namespace(
                input_dir=root,
                captions_filename=anime_clip_ltx.DEFAULT_CAPTIONS_FILENAME,
                output_filename=anime_clip_ltx.DEFAULT_OUTPUT_FILENAME,
            )

            exit_code = anime_clip_ltx.run(args)

            self.assertEqual(exit_code, 0)
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
            [
                {
                    "caption": "Train A with visible lines",
                    "media_path": "nested/a.mp4",
                },
                {
                    "caption": "Train Z",
                    "media_path": "z.mp4",
                },
            ],
        )
        self.assertEqual(val_rows, [{"caption": "Val B", "media_path": "b.mp4"}])
        self.assertEqual(set(train_rows[0]), {"caption", "media_path"})

    def test_run_rejects_missing_caption_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            train_dir = root / anime_clip_ltx.TRAIN_SPLIT_NAME
            val_dir = root / anime_clip_ltx.VALIDATION_SPLIT_NAME
            train_dir.mkdir()
            val_dir.mkdir()
            (train_dir / "a.mp4").write_bytes(b"a")
            (val_dir / "b.mp4").write_bytes(b"b")
            write_jsonl(
                val_dir / anime_clip_ltx.DEFAULT_CAPTIONS_FILENAME,
                [make_caption_record(str(val_dir / "b.mp4"), "caption")],
            )

            args = argparse.Namespace(
                input_dir=root,
                captions_filename=anime_clip_ltx.DEFAULT_CAPTIONS_FILENAME,
                output_filename=anime_clip_ltx.DEFAULT_OUTPUT_FILENAME,
            )

            exit_code = anime_clip_ltx.run(args)

            self.assertEqual(exit_code, 1)
            self.assertFalse((train_dir / anime_clip_ltx.DEFAULT_OUTPUT_FILENAME).exists())
            self.assertFalse((val_dir / anime_clip_ltx.DEFAULT_OUTPUT_FILENAME).exists())

    def test_run_rejects_missing_caption_row_for_clip(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            train_dir = root / anime_clip_ltx.TRAIN_SPLIT_NAME
            val_dir = root / anime_clip_ltx.VALIDATION_SPLIT_NAME
            train_dir.mkdir()
            val_dir.mkdir()
            (train_dir / "a.mp4").write_bytes(b"a")
            (train_dir / "b.mp4").write_bytes(b"b")
            (val_dir / "c.mp4").write_bytes(b"c")
            write_jsonl(
                train_dir / anime_clip_ltx.DEFAULT_CAPTIONS_FILENAME,
                [make_caption_record(str(train_dir / "a.mp4"), "caption a")],
            )
            write_jsonl(
                val_dir / anime_clip_ltx.DEFAULT_CAPTIONS_FILENAME,
                [make_caption_record(str(val_dir / "c.mp4"), "caption c")],
            )

            args = argparse.Namespace(
                input_dir=root,
                captions_filename=anime_clip_ltx.DEFAULT_CAPTIONS_FILENAME,
                output_filename=anime_clip_ltx.DEFAULT_OUTPUT_FILENAME,
            )

            exit_code = anime_clip_ltx.run(args)

        self.assertEqual(exit_code, 1)
        self.assertFalse((train_dir / anime_clip_ltx.DEFAULT_OUTPUT_FILENAME).exists())
        self.assertFalse((val_dir / anime_clip_ltx.DEFAULT_OUTPUT_FILENAME).exists())

    def test_run_rejects_duplicate_caption_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            train_dir = root / anime_clip_ltx.TRAIN_SPLIT_NAME
            val_dir = root / anime_clip_ltx.VALIDATION_SPLIT_NAME
            train_dir.mkdir()
            val_dir.mkdir()
            train_clip = train_dir / "a.mp4"
            val_clip = val_dir / "b.mp4"
            train_clip.write_bytes(b"a")
            val_clip.write_bytes(b"b")
            write_jsonl(
                train_dir / anime_clip_ltx.DEFAULT_CAPTIONS_FILENAME,
                [
                    make_caption_record(str(train_clip), "caption a"),
                    make_caption_record("train/a.mp4", "caption a duplicate"),
                ],
            )
            write_jsonl(
                val_dir / anime_clip_ltx.DEFAULT_CAPTIONS_FILENAME,
                [make_caption_record(str(val_clip), "caption b")],
            )

            args = argparse.Namespace(
                input_dir=root,
                captions_filename=anime_clip_ltx.DEFAULT_CAPTIONS_FILENAME,
                output_filename=anime_clip_ltx.DEFAULT_OUTPUT_FILENAME,
            )

            exit_code = anime_clip_ltx.run(args)

        self.assertEqual(exit_code, 1)
        self.assertFalse((train_dir / anime_clip_ltx.DEFAULT_OUTPUT_FILENAME).exists())
        self.assertFalse((val_dir / anime_clip_ltx.DEFAULT_OUTPUT_FILENAME).exists())

    def test_run_rejects_non_captioned_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            train_dir = root / anime_clip_ltx.TRAIN_SPLIT_NAME
            val_dir = root / anime_clip_ltx.VALIDATION_SPLIT_NAME
            train_dir.mkdir()
            val_dir.mkdir()
            train_clip = train_dir / "a.mp4"
            val_clip = val_dir / "b.mp4"
            train_clip.write_bytes(b"a")
            val_clip.write_bytes(b"b")
            write_jsonl(
                train_dir / anime_clip_ltx.DEFAULT_CAPTIONS_FILENAME,
                [make_caption_record(str(train_clip), "caption a", status="failed")],
            )
            write_jsonl(
                val_dir / anime_clip_ltx.DEFAULT_CAPTIONS_FILENAME,
                [make_caption_record(str(val_clip), "caption b")],
            )

            args = argparse.Namespace(
                input_dir=root,
                captions_filename=anime_clip_ltx.DEFAULT_CAPTIONS_FILENAME,
                output_filename=anime_clip_ltx.DEFAULT_OUTPUT_FILENAME,
            )

            exit_code = anime_clip_ltx.run(args)

        self.assertEqual(exit_code, 1)
        self.assertFalse((train_dir / anime_clip_ltx.DEFAULT_OUTPUT_FILENAME).exists())
        self.assertFalse((val_dir / anime_clip_ltx.DEFAULT_OUTPUT_FILENAME).exists())


if __name__ == "__main__":
    unittest.main()
