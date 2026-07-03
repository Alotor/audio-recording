from io import BytesIO
from pathlib import Path

import pytest

from audio_capture.bbb import (
    BBBError,
    download_media,
    media_candidates,
    recording_id_from_url,
    resolve_bbb_recording,
)


PLAYBACK_URL = (
    "https://video.example.com/playback/presentation/2.3/recording-123"
)


def test_parse_playback_url() -> None:
    assert recording_id_from_url(f"{PLAYBACK_URL}/?ignored=yes") == (
        "recording-123",
        "https://video.example.com",
    )


@pytest.mark.parametrize(
    "url",
    [
        "not-a-url",
        "ftp://video.example.com/playback/presentation/2.3/id",
        "https://video.example.com/watch/id",
    ],
)
def test_reject_invalid_playback_url(url: str) -> None:
    with pytest.raises(BBBError):
        recording_id_from_url(url)


def test_resolve_webm_after_missing_mp4() -> None:
    _, candidates = media_candidates(PLAYBACK_URL)
    probes = []

    def probe(url: str, timeout: float) -> bool:
        probes.append((url, timeout))
        return url.endswith(".webm")

    recording = resolve_bbb_recording(PLAYBACK_URL, probe=probe)

    assert recording.recording_id == "recording-123"
    assert recording.media_url == candidates[1]
    assert probes == [(candidates[0], 20), (candidates[1], 20)]


def test_download_media_atomically(monkeypatch, tmp_path: Path) -> None:
    class Response(BytesIO):
        headers = {"Content-Length": "7"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    monkeypatch.setattr(
        "audio_capture.bbb.urlopen",
        lambda request, timeout: Response(b"content"),
    )
    progress = []
    output = tmp_path / "recording.webm"

    download_media(
        "https://video.example.com/recording.webm",
        output,
        progress=lambda downloaded, total: progress.append((downloaded, total)),
    )

    assert output.read_bytes() == b"content"
    assert progress == [(7, 7)]
    assert not (tmp_path / "recording.webm.part").exists()


def test_download_media_removes_partial_file_on_failure(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class FailingResponse:
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self, size: int) -> bytes:
            raise OSError("connection lost")

    monkeypatch.setattr(
        "audio_capture.bbb.urlopen",
        lambda request, timeout: FailingResponse(),
    )
    output = tmp_path / "recording.webm"

    with pytest.raises(BBBError, match="connection lost"):
        download_media("https://video.example.com/recording.webm", output)

    assert not output.exists()
    assert not (tmp_path / "recording.webm.part").exists()
