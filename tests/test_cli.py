import argparse
from pathlib import Path

import pytest

from audio_capture.bbb import BBBRecording
from audio_capture.cli import build_parser, handle_bbb, positive_float


def test_record_defaults() -> None:
    args = build_parser().parse_args(["record", "--duration", "3"])

    assert args.duration == 3
    assert args.model == "medium"
    assert args.device == "cuda"
    assert args.compute_type is None
    assert args.no_transcribe is False


def test_bbb_defaults() -> None:
    args = build_parser().parse_args(
        [
            "bbb",
            "https://video.example.com/playback/presentation/2.3/recording-123",
        ]
    )

    assert args.model == "medium"
    assert args.device == "cuda"
    assert args.compute_type is None
    assert args.no_transcribe is False


def test_transcribe_defaults() -> None:
    args = build_parser().parse_args(["transcribe", "recording.wav"])

    assert args.model == "medium"
    assert args.device == "cuda"
    assert args.compute_type is None


def test_positive_float() -> None:
    assert positive_float("1.25") == 1.25


@pytest.mark.parametrize(
    "value",
    ["not-a-number", "nan", "inf", "-inf", "0", "-1"],
)
def test_positive_float_rejects_non_positive_or_non_finite(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        positive_float(value)


def test_bbb_downloads_and_transcribes(monkeypatch, tmp_path: Path) -> None:
    output_dir = tmp_path / "bbb-session"
    calls = []
    recording = BBBRecording(
        "recording-123",
        "https://video.example.com/presentation/recording-123/video/webcams.webm",
    )

    monkeypatch.setattr(
        "audio_capture.cli.resolve_bbb_recording", lambda url: recording
    )
    monkeypatch.setattr(
        "audio_capture.cli.download_media",
        lambda url, path, progress: calls.append(("download", url, path)),
    )
    monkeypatch.setattr(
        "audio_capture.cli.transcribe",
        lambda audio, text, **kwargs: calls.append(
            ("transcribe", audio, text, kwargs)
        ),
    )
    args = build_parser().parse_args(
        ["bbb", "https://example.invalid/playback", "-o", str(output_dir)]
    )

    assert handle_bbb(args) == 0
    transcription_call = calls[1]
    assert callable(transcription_call[3].pop("progress"))
    assert calls == [
        ("download", recording.media_url, output_dir / "recording.webm"),
        (
            "transcribe",
            output_dir / "recording.webm",
            output_dir / "transcript.txt",
            {
                "model_name": "medium",
                "language": None,
                "device": "cuda",
                "compute_type": "int8_float16",
            },
        ),
    ]


def test_bbb_reuses_downloaded_recording(monkeypatch, tmp_path: Path) -> None:
    output_dir = tmp_path / "bbb-session"
    output_dir.mkdir()
    video_path = output_dir / "recording.webm"
    video_path.write_bytes(b"existing recording")
    recording = BBBRecording(
        "recording-123",
        "https://video.example.com/presentation/recording-123/video/webcams.webm",
    )
    calls = []

    monkeypatch.setattr(
        "audio_capture.cli.resolve_bbb_recording", lambda url: recording
    )
    monkeypatch.setattr(
        "audio_capture.cli.download_media",
        lambda *args, **kwargs: calls.append("download"),
    )
    monkeypatch.setattr(
        "audio_capture.cli.transcribe",
        lambda audio, text, **kwargs: calls.append(
            ("transcribe", audio, text, kwargs)
        ),
    )
    args = build_parser().parse_args(
        ["bbb", "https://example.invalid/playback", "-o", str(output_dir)]
    )

    assert handle_bbb(args) == 0
    assert callable(calls[0][3].pop("progress"))
    assert calls == [
        (
            "transcribe",
            video_path,
            output_dir / "transcript.txt",
            {
                "model_name": "medium",
                "language": None,
                "device": "cuda",
                "compute_type": "int8_float16",
            },
        )
    ]
