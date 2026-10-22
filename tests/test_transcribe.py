import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from audio_capture.transcribe import (
    BilingualWhisperBackend,
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
            self.model = SimpleNamespace()

        def transcribe(self, audio_path, **kwargs):
            assert isinstance(self.model, BilingualWhisperBackend)
            calls["audio_path"] = audio_path
            calls["transcribe"] = kwargs
            return iter([SimpleNamespace(text=" hello ")]), None

    monkeypatch.setitem(
        sys.modules,
        "faster_whisper",
        SimpleNamespace(WhisperModel=FakeModel),
    )
    output_path = tmp_path / "transcript.txt"

    transcribe(Path("audio.wav"), output_path)

    assert calls["init"] == ("medium", "cuda", "int8_float16")
    assert calls["audio_path"] == "audio.wav"
    assert calls["transcribe"] == {
        "language": None,
        "vad_filter": True,
        "beam_size": 5,
        "best_of": 5,
        "condition_on_previous_text": False,
        "multilingual": True,
        "word_timestamps": True,
        "hallucination_silence_threshold": 2,
    }
    assert output_path.read_text() == "hello\n"


def test_cpu_uses_int8_by_default(monkeypatch, tmp_path: Path) -> None:
    calls = {}

    class FakeModel:
        def __init__(self, model_name, *, device, compute_type):
            calls["init"] = (model_name, device, compute_type)
            self.model = SimpleNamespace()

        def transcribe(self, audio_path, **kwargs):
            calls["transcribe"] = (kwargs["beam_size"], kwargs["best_of"])
            return iter([]), None

    monkeypatch.setitem(
        sys.modules,
        "faster_whisper",
        SimpleNamespace(WhisperModel=FakeModel),
    )

    transcribe(Path("audio.wav"), tmp_path / "transcript.txt", device="cpu")

    assert calls["init"] == ("medium", "cpu", "int8")
    assert calls["transcribe"] == (5, 5)


def test_bilingual_detection_handles_switches_and_ignores_other_languages() -> None:
    features = object()

    class Backend:
        is_multilingual = True

        def detect_language(self, received_features):
            assert received_features is features
            return [
                [("<|fr|>", 0.6), ("<|en|>", 0.3), ("<|es|>", 0.1)],
                [("<|pt|>", 0.5), ("<|es|>", 0.4), ("<|en|>", 0.1)],
            ]

        def generate(self, received_features, prompts, **kwargs):
            return received_features, prompts, kwargs

    model = BilingualWhisperBackend(Backend())

    assert model.detect_language(features) == [
        [("<|en|>", 0.3), ("<|es|>", 0.1)],
        [("<|es|>", 0.4), ("<|en|>", 0.1)],
    ]
    assert model.is_multilingual is True
    assert model.generate(features, [[42]], beam_size=5) == (
        features, [[42]], {"beam_size": 5},
    )


def test_bilingual_detection_rejects_missing_candidates() -> None:
    model = BilingualWhisperBackend(
        SimpleNamespace(detect_language=lambda features: [[("<|fr|>", 1.0)]])
    )

    with pytest.raises(TranscriptionError, match="no English or Spanish candidates"):
        model.detect_language(object())


@pytest.mark.parametrize("language", [None, "en", "es"])
def test_language_override_disables_per_segment_detection(language) -> None:
    calls = {}

    class FakeModel:
        def transcribe(self, audio_path, **kwargs):
            calls.update(kwargs)
            return iter([]), None

    transcribe_with_model(FakeModel(), Path("audio.wav"), language=language)

    assert calls["language"] == language
    assert calls["multilingual"] is (language is None)


@pytest.mark.parametrize("existing_transcript", [None, "Previous useful transcript\n"])
def test_repetition_loop_does_not_write_or_replace_transcript(
    monkeypatch, tmp_path: Path, existing_transcript,
) -> None:
    class FakeModel:
        def transcribe(self, audio_path, **kwargs):
            return iter(
                [SimpleNamespace(text="Opening conversation")]
                + [SimpleNamespace(text=text) for text in ["y", " Y "] * 300]
            ), None

    monkeypatch.setattr("audio_capture.transcribe.create_model", lambda *args: FakeModel())
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"original audio")
    output_path = tmp_path / "transcript.txt"
    if existing_transcript is not None:
        output_path.write_text(existing_transcript)

    with pytest.raises(TranscriptionError, match="repetition loop"):
        transcribe(audio_path, output_path)

    assert audio_path.read_bytes() == b"original audio"
    if existing_transcript is None:
        assert not output_path.exists()
    else:
        assert output_path.read_text() == existing_transcript


def test_ordinary_repetition_is_preserved() -> None:
    texts = ["Yes."] * 3 + ["Next question."] + ["Yes."] * 3

    class FakeModel:
        def transcribe(self, audio_path, **kwargs):
            return iter(SimpleNamespace(text=text) for text in texts), None

    assert transcribe_with_model(FakeModel(), Path("audio.wav")) == texts


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
