from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import sakuga_download


def make_posts_xml(posts: list[dict[str, str | int]]) -> str:
    lines = ['<?xml version="1.0" encoding="UTF-8"?>', '<posts count="0" offset="0">']
    for post in posts:
        attrs = {
            "id": str(post["id"]),
            "file_ext": str(post["file_ext"]),
            "file_url": str(post["file_url"]),
            "file_size": str(post["file_size"]),
            "tags": str(post.get("tags", "")),
        }
        joined = " ".join(f'{key}="{value}"' for key, value in attrs.items())
        lines.append(f"  <post {joined}/>")
    lines.append("</posts>")
    return "\n".join(lines)


class FakeResponse:
    def __init__(
        self,
        text: str = "",
        status_code: int = 200,
        chunks: list[bytes] | None = None,
        chunk_error: Exception | None = None,
    ):
        self.text = text
        self.status_code = status_code
        self._chunks = chunks or []
        self._chunk_error = chunk_error

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"status {self.status_code}")

    async def aiter_bytes(self, chunk_size: int = 0):
        del chunk_size
        for chunk in self._chunks:
            yield chunk
        if self._chunk_error is not None:
            raise self._chunk_error


class FakeStreamContext:
    def __init__(self, response: FakeResponse):
        self.response = response

    async def __aenter__(self) -> FakeResponse:
        return self.response

    async def __aexit__(self, exc_type, exc, tb) -> None:
        del exc_type, exc, tb
        return None


class MetadataClient:
    def __init__(self, pages: dict[int, str]):
        self.pages = pages
        self.requested_pages: list[int] = []

    async def get(self, url: str, params: dict[str, object]) -> FakeResponse:
        del url
        page = int(params["page"])
        self.requested_pages.append(page)
        return FakeResponse(text=self.pages[page])


class DownloadClient:
    def __init__(self, response: FakeResponse):
        self.response = response
        self.request_headers: list[dict[str, str]] = []

    def stream(self, method: str, url: str, headers: dict[str, str] | None = None) -> FakeStreamContext:
        del method, url
        self.request_headers.append(headers or {})
        return FakeStreamContext(self.response)


class SequentialDownloadClient:
    def __init__(self, responses: list[FakeResponse]):
        self.responses = responses
        self.request_headers: list[dict[str, str]] = []

    def stream(self, method: str, url: str, headers: dict[str, str] | None = None) -> FakeStreamContext:
        del method, url
        self.request_headers.append(headers or {})
        return FakeStreamContext(self.responses.pop(0))


class NoNetworkClient:
    def stream(self, method: str, url: str, headers: dict[str, str] | None = None) -> FakeStreamContext:
        del method, url, headers
        raise AssertionError("network should not be used when file is already complete")


class SakugaDownloadTests(unittest.IsolatedAsyncioTestCase):
    async def test_fetch_matching_posts_filters_and_stops_when_enough_found(self) -> None:
        client = MetadataClient(
            {
                1: make_posts_xml(
                    [
                        {"id": 101, "file_ext": "jpg", "file_url": "https://example/101.jpg", "file_size": 10},
                        {"id": 102, "file_ext": "mp4", "file_url": "https://example/102.mp4", "file_size": 20},
                    ]
                ),
                2: make_posts_xml(
                    [
                        {"id": 201, "file_ext": "webm", "file_url": "https://example/201.webm", "file_size": 30},
                        {"id": 202, "file_ext": "png", "file_url": "https://example/202.png", "file_size": 40},
                    ]
                ),
                3: make_posts_xml(
                    [
                        {"id": 301, "file_ext": "mp4", "file_url": "https://example/301.mp4", "file_size": 50},
                    ]
                ),
            }
        )

        posts = await sakuga_download.fetch_matching_posts(client, "dragon", count=2, page_size=2)

        self.assertEqual([post.post_id for post in posts], [102, 201])
        self.assertEqual(client.requested_pages, [1, 2])

    async def test_download_post_skips_existing_complete_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            target = output_dir / "123.mp4"
            target.write_bytes(b"abcd")
            post = sakuga_download.PostMetadata(
                post_id=123,
                file_ext="mp4",
                file_url="https://example/123.mp4",
                file_size=4,
                tags="dragon",
            )

            result = await sakuga_download.download_post(NoNetworkClient(), post, output_dir)

            self.assertEqual(result.status, "skipped")
            self.assertEqual(target.read_bytes(), b"abcd")

    async def test_download_post_resumes_partial_download(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            part_path = output_dir / "321.webm.part"
            part_path.write_bytes(b"abc")
            post = sakuga_download.PostMetadata(
                post_id=321,
                file_ext="webm",
                file_url="https://example/321.webm",
                file_size=6,
                tags="dragon",
            )
            client = DownloadClient(FakeResponse(status_code=206, chunks=[b"def"]))

            result = await sakuga_download.download_post(client, post, output_dir)

            self.assertEqual(result.status, "downloaded")
            self.assertEqual(result.detail, "resumed")
            self.assertEqual(client.request_headers, [{"Range": "bytes=3-"}])
            self.assertEqual((output_dir / "321.webm").read_bytes(), b"abcdef")
            self.assertFalse(part_path.exists())

    @unittest.skipIf(sakuga_download.httpx is None, "httpx not installed in this interpreter")
    async def test_download_post_retries_from_updated_partial_size(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            part_path = output_dir / "654.mp4.part"
            part_path.write_bytes(b"abc")
            post = sakuga_download.PostMetadata(
                post_id=654,
                file_ext="mp4",
                file_url="https://example/654.mp4",
                file_size=6,
                tags="dragon",
            )
            client = SequentialDownloadClient(
                [
                    FakeResponse(
                        status_code=206,
                        chunks=[b"d"],
                        chunk_error=sakuga_download.httpx.ReadError("boom"),
                    ),
                    FakeResponse(status_code=206, chunks=[b"ef"]),
                ]
            )

            result = await sakuga_download.download_post(client, post, output_dir)

            self.assertEqual(result.status, "downloaded")
            self.assertEqual(client.request_headers, [{"Range": "bytes=3-"}, {"Range": "bytes=4-"}])
            self.assertEqual((output_dir / "654.mp4").read_bytes(), b"abcdef")
            self.assertFalse(part_path.exists())


class SakugaDownloadUnitTests(unittest.TestCase):
    def test_normalize_query_preserves_explicit_order(self) -> None:
        self.assertEqual(
            sakuga_download.normalize_query("dragon order:score"),
            "dragon order:score",
        )
        self.assertEqual(
            sakuga_download.normalize_query("dragon fire"),
            "dragon fire order:id_desc",
        )

    def test_parser_rejects_non_positive_count(self) -> None:
        parser = sakuga_download.build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["--query", "dragon", "--count", "0"])


if __name__ == "__main__":
    unittest.main()
