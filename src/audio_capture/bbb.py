from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.parse import quote, unquote, urlsplit, urlunsplit
from urllib.request import Request, urlopen

from audio_capture import __version__


class BBBError(RuntimeError):
    """Raised when a BigBlueButton recording cannot be downloaded."""


@dataclass(frozen=True)
class BBBRecording:
    recording_id: str
    media_url: str


ProgressCallback = Callable[[int, int | None], None]


def recording_id_from_url(playback_url: str) -> tuple[str, str]:
    parsed = urlsplit(playback_url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise BBBError("BBB playback URL must use HTTP or HTTPS.")

    parts = [unquote(part) for part in parsed.path.split("/") if part]
    for index in range(len(parts) - 3):
        if parts[index : index + 2] != ["playback", "presentation"]:
            continue
        if index + 4 != len(parts):
            break
        recording_id = parts[index + 3]
        if not recording_id or "/" in recording_id or "\\" in recording_id:
            break
        origin = urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))
        return recording_id, origin

    raise BBBError(
        "Expected a BBB presentation URL ending in "
        "'/playback/presentation/<version>/<recording-id>'."
    )


def media_candidates(playback_url: str) -> tuple[str, list[str]]:
    recording_id, origin = recording_id_from_url(playback_url)
    base = f"{origin}/presentation/{quote(recording_id, safe='-._~')}"
    return recording_id, [
        f"{base}/video/webcams.mp4",
        f"{base}/video/webcams.webm",
        f"{base}/audio/audio.webm",
    ]


def _request(url: str, *, method: str = "GET") -> Request:
    return Request(
        url,
        method=method,
        headers={"User-Agent": f"audio-capture/{__version__}"},
    )


def _probe_media(url: str, timeout: float) -> bool:
    try:
        with urlopen(_request(url, method="HEAD"), timeout=timeout):
            return True
    except HTTPError as exc:
        if exc.code in (404, 410):
            return False
        if exc.code not in (405, 501):
            raise

    ranged_request = _request(url)
    ranged_request.add_header("Range", "bytes=0-0")
    try:
        with urlopen(ranged_request, timeout=timeout):
            return True
    except HTTPError as exc:
        if exc.code in (404, 410):
            return False
        raise


def resolve_bbb_recording(
    playback_url: str,
    *,
    timeout: float = 20,
    probe: Callable[[str, float], bool] = _probe_media,
) -> BBBRecording:
    recording_id, candidates = media_candidates(playback_url)
    try:
        for candidate in candidates:
            if probe(candidate, timeout):
                return BBBRecording(recording_id, candidate)
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise BBBError(f"Could not inspect the BBB recording: {exc}") from exc

    raise BBBError(
        "The BBB recording has no downloadable webcam or audio media. "
        "It may be unpublished or still processing."
    )


def download_media(
    url: str,
    output_path: Path,
    *,
    timeout: float = 30,
    progress: ProgressCallback | None = None,
) -> None:
    if output_path.exists():
        raise BBBError(f"Output file already exists: {output_path}")

    partial_path = output_path.with_name(f"{output_path.name}.part")
    try:
        with urlopen(_request(url), timeout=timeout) as response:
            length = response.headers.get("Content-Length")
            total = int(length) if length and length.isdigit() else None
            downloaded = 0
            with partial_path.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
                    downloaded += len(chunk)
                    if progress is not None:
                        progress(downloaded, total)
        partial_path.replace(output_path)
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        partial_path.unlink(missing_ok=True)
        raise BBBError(f"Could not download the BBB recording: {exc}") from exc
