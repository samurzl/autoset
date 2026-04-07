#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable
from urllib.parse import urlencode
import xml.etree.ElementTree as ET

try:
    import httpx
except ModuleNotFoundError:
    httpx = None

BASE_URL = "https://www.sakugabooru.com"
POST_INDEX_PATH = "/post/index.xml"
VIDEO_EXTENSIONS = {"mp4", "webm"}
DEFAULT_PAGE_SIZE = 200
DEFAULT_MAX_CONCURRENCY = 8
DEFAULT_RETRY_ATTEMPTS = 4
RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}
CHUNK_SIZE = 1024 * 256
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; sakuga-download/1.0)",
    "Accept": "application/xml,text/xml;q=0.9,*/*;q=0.8",
}


@dataclass(frozen=True)
class PostMetadata:
    post_id: int
    file_ext: str
    file_url: str
    file_size: int
    tags: str

    @property
    def filename(self) -> str:
        return f"{self.post_id}.{self.file_ext}"


@dataclass(frozen=True)
class DownloadResult:
    post: PostMetadata
    status: str
    path: Path
    detail: str = ""


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download Sakugabooru videos for a query as fast as possible."
    )
    parser.add_argument("--query", required=True, help="Raw Sakugabooru tag query.")
    parser.add_argument(
        "--count",
        required=True,
        type=positive_int,
        help="Maximum number of videos to download.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Base output directory. Defaults to downloads/<query-slug>/",
    )
    parser.add_argument(
        "--concurrency",
        type=positive_int,
        help="Concurrent download count. Defaults to min(count, 8).",
    )
    return parser


def slugify_query(query: str) -> str:
    normalized = unicodedata.normalize("NFKD", query).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", normalized).strip("-").lower()
    return slug or "sakugabooru"


def normalize_query(query: str) -> str:
    tokens = query.split()
    if any(token.startswith("order:") for token in tokens):
        return query
    return f"{query} order:id_desc".strip()


def default_output_dir(query: str) -> Path:
    return Path("downloads") / slugify_query(query)


def build_feed_url(query: str, page: int, limit: int = DEFAULT_PAGE_SIZE) -> str:
    params = urlencode(
        {
            "tags": normalize_query(query),
            "limit": limit,
            "page": page,
        }
    )
    return f"{BASE_URL}{POST_INDEX_PATH}?{params}"


def parse_posts(xml_text: str) -> list[PostMetadata]:
    root = ET.fromstring(xml_text)
    posts: list[PostMetadata] = []
    for node in root.findall("post"):
        file_ext = (node.attrib.get("file_ext") or "").lower()
        file_url = node.attrib.get("file_url")
        post_id = node.attrib.get("id")
        file_size = node.attrib.get("file_size") or "0"
        if not file_ext or not file_url or not post_id:
            continue
        posts.append(
            PostMetadata(
                post_id=int(post_id),
                file_ext=file_ext,
                file_url=file_url,
                file_size=int(file_size),
                tags=node.attrib.get("tags", ""),
            )
        )
    return posts


def is_retryable_exception(exc: Exception) -> bool:
    if httpx is None:
        return False
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in RETRYABLE_STATUS_CODES
    return False


async def retry_async(
    operation_name: str,
    operation: Callable[[], Awaitable[Any]],
    attempts: int = DEFAULT_RETRY_ATTEMPTS,
) -> Any:
    for attempt in range(1, attempts + 1):
        try:
            return await operation()
        except Exception as exc:
            if attempt >= attempts or not is_retryable_exception(exc):
                raise
            delay = 0.5 * (2 ** (attempt - 1))
            print(
                f"Retrying {operation_name} after error: {exc}",
                file=sys.stderr,
            )
            await asyncio.sleep(delay)
    raise RuntimeError(f"{operation_name} failed unexpectedly")


async def fetch_feed_page(client: Any, query: str, page: int, limit: int) -> str:
    async def operation() -> str:
        response = await client.get(
            f"{BASE_URL}{POST_INDEX_PATH}",
            params={
                "tags": normalize_query(query),
                "limit": limit,
                "page": page,
            },
        )
        response.raise_for_status()
        return response.text

    return await retry_async(f"metadata page {page}", operation)


async def fetch_matching_posts(
    client: Any,
    query: str,
    count: int,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> list[PostMetadata]:
    matches: list[PostMetadata] = []
    page = 1

    while len(matches) < count:
        xml_text = await fetch_feed_page(client, query, page=page, limit=page_size)
        posts = parse_posts(xml_text)
        if not posts:
            break

        for post in posts:
            if post.file_ext in VIDEO_EXTENSIONS:
                matches.append(post)
                if len(matches) >= count:
                    break

        if len(posts) < page_size:
            break
        page += 1

    return matches


def classify_existing_download(target_path: Path, part_path: Path, expected_size: int) -> str | None:
    if expected_size > 0 and target_path.exists() and target_path.stat().st_size == expected_size:
        return "complete"
    if expected_size > 0 and part_path.exists() and part_path.stat().st_size == expected_size:
        return "part-complete"
    return None


async def stream_download_to_disk(client: Any, post: PostMetadata, target_path: Path) -> DownloadResult:
    part_path = target_path.with_name(f"{target_path.name}.part")
    existing_state = classify_existing_download(target_path, part_path, post.file_size)
    if existing_state == "complete":
        return DownloadResult(post=post, status="skipped", path=target_path, detail="already complete")
    if existing_state == "part-complete":
        part_path.replace(target_path)
        return DownloadResult(post=post, status="skipped", path=target_path, detail="restored from partial")

    if part_path.exists() and post.file_size > 0 and part_path.stat().st_size > post.file_size:
        part_path.unlink()

    async def operation() -> DownloadResult:
        current_size = part_path.stat().st_size if part_path.exists() else 0
        request_headers: dict[str, str] = {}
        file_mode = "wb"
        local_resume_from = current_size

        if 0 < local_resume_from < post.file_size:
            request_headers["Range"] = f"bytes={local_resume_from}-"
            file_mode = "ab"

        async with client.stream("GET", post.file_url, headers=request_headers) as response:
            if response.status_code == 416 and local_resume_from >= post.file_size > 0:
                part_path.replace(target_path)
                return DownloadResult(
                    post=post,
                    status="skipped",
                    path=target_path,
                    detail="range already complete",
                )

            if response.status_code == 200 and file_mode == "ab":
                file_mode = "wb"
                local_resume_from = 0

            response.raise_for_status()

            with part_path.open(file_mode) as handle:
                async for chunk in response.aiter_bytes(chunk_size=CHUNK_SIZE):
                    if chunk:
                        handle.write(chunk)

        final_size = part_path.stat().st_size
        if post.file_size > 0 and final_size != post.file_size:
            raise OSError(
                f"expected {post.file_size} bytes for post {post.post_id}, got {final_size}"
            )

        part_path.replace(target_path)
        detail = "resumed" if local_resume_from else "downloaded"
        return DownloadResult(post=post, status="downloaded", path=target_path, detail=detail)

    return await retry_async(f"download {post.post_id}", operation)


async def download_post(client: Any, post: PostMetadata, output_dir: Path) -> DownloadResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    target_path = output_dir / post.filename

    try:
        return await stream_download_to_disk(client, post, target_path)
    except Exception as exc:
        return DownloadResult(post=post, status="failed", path=target_path, detail=str(exc))


async def download_posts(
    client: Any,
    posts: list[PostMetadata],
    output_dir: Path,
    concurrency: int,
) -> list[DownloadResult]:
    semaphore = asyncio.Semaphore(concurrency)

    async def worker(post: PostMetadata) -> DownloadResult:
        async with semaphore:
            result = await download_post(client, post, output_dir)
            print(f"{result.status}: {post.filename} ({result.detail})")
            return result

    return await asyncio.gather(*(worker(post) for post in posts))


def summarize_results(
    requested: int,
    matched: int,
    results: list[DownloadResult],
    output_dir: Path,
) -> str:
    counts = {"downloaded": 0, "skipped": 0, "failed": 0}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1

    summary = [
        f"requested={requested}",
        f"matched={matched}",
        f"downloaded={counts['downloaded']}",
        f"skipped={counts['skipped']}",
        f"failed={counts['failed']}",
        f"output={output_dir}",
    ]
    return " ".join(summary)


def require_httpx() -> None:
    if httpx is not None:
        return
    raise SystemExit(
        "Missing dependency 'httpx'. Install it with: python -m pip install -r requirements.txt"
    )


def build_http_client(concurrency: int) -> Any:
    require_httpx()
    limits = httpx.Limits(
        max_connections=max(concurrency * 2, 20),
        max_keepalive_connections=max(concurrency, 10),
    )
    timeout = httpx.Timeout(connect=10.0, read=120.0, write=120.0, pool=30.0)
    return httpx.AsyncClient(
        headers=DEFAULT_HEADERS,
        follow_redirects=True,
        http2=True,
        limits=limits,
        timeout=timeout,
    )


async def async_main(args: argparse.Namespace) -> int:
    concurrency = args.concurrency or min(args.count, DEFAULT_MAX_CONCURRENCY)
    output_dir = args.output_dir or default_output_dir(args.query)

    async with build_http_client(concurrency) as client:
        posts = await fetch_matching_posts(client, args.query, args.count)
        if not posts:
            print(f"No matching videos found for query: {normalize_query(args.query)}")
            print(summarize_results(args.count, 0, [], output_dir))
            return 0

        if len(posts) < args.count:
            print(
                f"Only found {len(posts)} matching videos for requested count {args.count}.",
                file=sys.stderr,
            )

        results = await download_posts(client, posts, output_dir, concurrency)

    print(summarize_results(args.count, len(posts), results, output_dir))
    return 1 if any(result.status == "failed" for result in results) else 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
