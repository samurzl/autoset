from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import anime_dataset_word_stats


def write_jsonl(path: Path, rows: list[object]) -> None:
    payload = "\n".join(json.dumps(row) for row in rows)
    if payload:
        payload += "\n"
    path.write_text(payload, encoding="utf-8")


def run_main(argv: list[str]) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        exit_code = anime_dataset_word_stats.main(argv)
    return exit_code, stdout.getvalue(), stderr.getvalue()


class AnimeDatasetWordStatsHelperTests(unittest.TestCase):
    def test_normalize_caption_blocks_accepts_plain_caption_paragraph(self) -> None:
        blocks = anime_dataset_word_stats.normalize_caption_blocks("Hero runs fast under blue sky.")

        self.assertEqual(blocks, (("hero", "runs", "fast", "under", "blue", "sky"),))

    def test_normalize_caption_blocks_strips_labels_and_none_placeholders(self) -> None:
        caption = """
SCENE_OVERVIEW:
Hero runs fast.

VISUAL_DETAILS:
Blue sky.
None.

DIALOGUE: None.
OTHER_SOUNDS:
Wind rises.
""".strip()

        blocks = anime_dataset_word_stats.normalize_caption_blocks(caption)

        self.assertEqual(
            blocks,
            (
                ("hero", "runs", "fast"),
                ("blue", "sky"),
                ("wind", "rises"),
            ),
        )


class AnimeDatasetWordStatsRunTests(unittest.TestCase):
    def test_main_aggregates_multiple_manifests_and_limits_top_n(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "train.jsonl"
            second = root / "val.jsonl"
            write_jsonl(
                first,
                [
                    {"caption": "Hero runs fast under blue sky"},
                    {"caption": "SCENE_OVERVIEW:\nHero runs again\n\nVISUAL_DETAILS:\nNone."},
                ],
            )
            write_jsonl(
                second,
                [
                    {"caption": "SCENE_OVERVIEW:\nHero leaps high\n\nVISUAL_DETAILS:\nBlue sky"},
                ],
            )

            exit_code, stdout, stderr = run_main(["--top-n", "2", str(first), str(second)])

        self.assertEqual(exit_code, 0)
        self.assertEqual(stderr, "")
        self.assertEqual(
            stdout.strip().splitlines(),
            [
                "manifests=2 caption_rows=3 unique_words=9 unique_bigrams=8",
                "",
                "Top 2 words",
                "1. hero (3)",
                "2. blue (2)",
                "",
                "Top 2 bigrams",
                "1. blue sky (2)",
                "2. hero runs (2)",
            ],
        )

    def test_collect_statistics_keeps_bigram_boundaries_inside_each_line_and_row(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = Path(temp_dir) / "dataset.jsonl"
            write_jsonl(
                manifest,
                [
                    {"caption": "SCENE_OVERVIEW:\nalpha beta\nVISUAL_DETAILS:\ngamma delta"},
                    {"caption": "SCENE_OVERVIEW:\nepsilon zeta"},
                ],
            )

            _, word_counts, bigram_counts = anime_dataset_word_stats.collect_statistics([manifest])

        self.assertEqual(word_counts["alpha"], 1)
        self.assertEqual(bigram_counts["alpha beta"], 1)
        self.assertEqual(bigram_counts["gamma delta"], 1)
        self.assertEqual(bigram_counts["epsilon zeta"], 1)
        self.assertEqual(bigram_counts["beta gamma"], 0)
        self.assertEqual(bigram_counts["delta epsilon"], 0)

    def test_main_rejects_malformed_json_with_file_and_line(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = Path(temp_dir) / "broken.jsonl"
            manifest.write_text('{"caption": "ok"}\nnot-json\n', encoding="utf-8")

            exit_code, stdout, stderr = run_main([str(manifest)])

        self.assertEqual(exit_code, 1)
        self.assertEqual(stdout, "")
        self.assertIn(f"{manifest}:2 contains invalid JSON", stderr)

    def test_main_rejects_missing_or_blank_caption(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cases = [
                ("missing.jsonl", [{"media_path": "a.mp4"}]),
                ("blank.jsonl", [{"caption": "   "}]),
            ]

            for filename, rows in cases:
                manifest = root / filename
                write_jsonl(manifest, rows)

                with self.subTest(filename=filename):
                    exit_code, stdout, stderr = run_main([str(manifest)])
                    self.assertEqual(exit_code, 1)
                    self.assertEqual(stdout, "")
                    self.assertIn(f"{manifest}:1 is missing a non-empty caption", stderr)

    def test_main_rejects_nonexistent_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = Path(temp_dir) / "missing.jsonl"

            exit_code, stdout, stderr = run_main([str(manifest)])

        self.assertEqual(exit_code, 1)
        self.assertEqual(stdout, "")
        self.assertIn(f"{manifest} must exist and be a file", stderr)

    def test_main_rejects_dataset_with_zero_usable_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = Path(temp_dir) / "empty.jsonl"
            write_jsonl(
                manifest,
                [
                    {"caption": "SCENE_OVERVIEW:\nNone.\n\nVISUAL_DETAILS:\nNone."},
                ],
            )

            exit_code, stdout, stderr = run_main([str(manifest)])

        self.assertEqual(exit_code, 1)
        self.assertEqual(stdout, "")
        self.assertIn("zero usable tokens after normalization", stderr)


if __name__ == "__main__":
    unittest.main()
