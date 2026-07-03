import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from audio_capture.transcribe import (
    TranscriptionError,
    cuda_environment,
    transcribe,
    transcribe_with_model,
)


def test_medium_model_uses_cuda_by_default(monkeypatch, tmp_path: Path) -> None:
    calls = {}

    class FakeModel:
        def __init__(self, model_name, *, device, compute_type):
            calls["init"] = (model_name, device, compute_type)

        def transcribe(self, audio_path, *, language, vad_filter):
            calls["transcribe"] = (
                audio_path,
                language,
                vad_filter,
            )
            return iter([SimpleNamespace(text=" hello ")]), None

    monkeypatch.setitem(
        sys.modules,
        "faster_whisper",
        SimpleNamespace(WhisperModel=FakeModel),
    )
    output_path = tmp_path / "transcript.txt"

    transcribe(Path("audio.wav"), output_path)

    assert calls["init"] == ("medium", "cuda", "default")
    assert calls["transcribe"] == ("audio.wav", None, True)
    assert output_path.read_text() == "hello\n"


def test_transcribe_preserves_specific_transcription_error(
    monkeypatch,
    tmp_path: Path,
) -> None:
    error = TranscriptionError("specific setup problem")
    monkeypatch.setattr(
        "audio_capture.transcribe.create_model",
        lambda *args: (_ for _ in ()).throw(error),
    )

    with pytest.raises(TranscriptionError, match="^specific setup problem$"):
        transcribe(Path("audio.wav"), tmp_path / "transcript.txt")


def test_transcription_reports_audio_percentage() -> None:
    class FakeModel:
        def transcribe(self, audio_path, **kwargs):
            return iter(
                [
                    SimpleNamespace(text=" first ", end=2.5),
                    SimpleNamespace(text=" second ", end=8.0),
                ]
            ), SimpleNamespace(duration=10.0)

    percentages = []

    lines = transcribe_with_model(
        FakeModel(),
        Path("audio.wav"),
        progress=percentages.append,
    )

    assert lines == ["first", "second"]
    assert percentages == [0, 25, 80, 100]


def test_cuda_environment_adds_library_paths(monkeypatch) -> None:
    monkeypatch.setattr(
        "audio_capture.transcribe.cuda_library_paths",
        lambda: ["/cuda/cublas", "/cuda/cudnn"],
    )
    monkeypatch.setenv("LD_LIBRARY_PATH", "/existing")

    environment = cuda_environment()

    assert environment is not None
    assert environment["LD_LIBRARY_PATH"] == (
        "/cuda/cublas:/cuda/cudnn:/existing"
    )
    assert environment["AUDIO_CAPTURE_CUDA_PATH_READY"] == "1"
