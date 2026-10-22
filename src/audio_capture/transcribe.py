from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Callable


class TranscriptionError(RuntimeError):
    """Raised when local transcription cannot be completed."""


DEFAULT_MODEL = "medium"
DEFAULT_DEVICE = "cuda"
DEFAULT_CUDA_COMPUTE_TYPE = "int8_float16"
DEFAULT_BEAM_SIZE = 5
MAX_REPEATED_SEGMENTS = 10
DEFAULT_LANGUAGE_TOKENS = frozenset({"<|en|>", "<|es|>"})


ProgressCallback = Callable[[float], None]


class BilingualWhisperBackend:
    """Restrict both initial and per-window language detection to English/Spanish."""

    def __init__(self, backend):
        self._backend = backend

    def __getattr__(self, name):
        return getattr(self._backend, name)

    def detect_language(self, features):
        results = self._backend.detect_language(features)
        filtered_results = []
        for candidates in results:
            # CTranslate2 returns candidates in descending probability order.
            allowed = [pair for pair in candidates if pair[0] in DEFAULT_LANGUAGE_TOKENS]
            if not allowed:
                raise TranscriptionError(
                    "Language detection returned no English or Spanish candidates."
                )
            filtered_results.append(allowed)
        return filtered_results


def cuda_library_paths() -> list[str]:
    paths = []
    for module_name in ("nvidia.cublas.lib", "nvidia.cudnn.lib"):
        try:
            spec = importlib.util.find_spec(module_name)
        except ModuleNotFoundError:
            spec = None
        if spec is not None and spec.submodule_search_locations:
            paths.extend(str(path) for path in spec.submodule_search_locations)
    return paths


def cuda_environment() -> dict[str, str] | None:
    paths = cuda_library_paths()
    if not paths:
        return None

    current_paths = [
        path
        for path in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
        if path
    ]
    missing_paths = [path for path in paths if path not in current_paths]
    if not missing_paths:
        return None

    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(
        missing_paths + current_paths
    )
    environment["AUDIO_CAPTURE_CUDA_PATH_READY"] = "1"
    return environment


def create_model(
    model_name: str,
    device: str,
    compute_type: str | None,
):
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise TranscriptionError(
            "Transcription support is not installed. Run: "
            "python -m pip install -e '.[transcribe]'"
        ) from exc

    if compute_type is None:
        if device == "cpu":
            compute_type = "int8"
        elif device == "cuda":
            compute_type = DEFAULT_CUDA_COMPUTE_TYPE
        else:
            compute_type = "default"
    model = WhisperModel(
        model_name,
        device=device,
        compute_type=compute_type,
    )
    # faster-whisper uses this backend for both detection paths. Filtering only
    # WhisperModel.detect_language would miss switches during transcription.
    model.model = BilingualWhisperBackend(model.model)
    return model


def transcribe_with_model(
    model,
    audio_path: Path,
    language: str | None = None,
    progress: ProgressCallback | None = None,
) -> list[str]:
    segments, info = model.transcribe(
        str(audio_path),
        language=language,
        vad_filter=True,
        beam_size=DEFAULT_BEAM_SIZE,
        best_of=DEFAULT_BEAM_SIZE,
        # Earlier errors can otherwise feed a repetition loop in later windows.
        condition_on_previous_text=False,
        multilingual=language is None,
        word_timestamps=True,
        hallucination_silence_threshold=2,
    )
    duration = getattr(info, "duration", 0) or 0
    if progress is not None:
        progress(0)

    lines = []
    previous_text = None
    repeated_segments = 0
    for segment in segments:
        text = segment.text.strip()
        if text:
            normalized = " ".join(text.casefold().split())
            repeated_segments = repeated_segments + 1 if normalized == previous_text else 1
            previous_text = normalized
            if repeated_segments >= MAX_REPEATED_SEGMENTS:
                raise TranscriptionError(
                    "Transcription appears stuck in a repetition loop "
                    f"({MAX_REPEATED_SEGMENTS} identical segments). "
                    "No transcript was written; the audio is retained. "
                    "Retry with a larger model or --language if the language is known."
                )
            lines.append(text)
        if progress is not None and duration:
            progress(segment.end / duration * 100)

    if progress is not None:
        progress(100)
    return lines


def transcribe(
    audio_path: Path,
    text_path: Path,
    model_name: str = DEFAULT_MODEL,
    language: str | None = None,
    device: str = DEFAULT_DEVICE,
    compute_type: str | None = None,
    progress: ProgressCallback | None = None,
) -> None:
    try:
        model = create_model(model_name, device, compute_type)
        lines = transcribe_with_model(
            model,
            audio_path,
            language,
            progress=progress,
        )
        with text_path.open("w", encoding="utf-8") as output:
            for line in lines:
                output.write(line)
                output.write("\n")
    except TranscriptionError:
        raise
    except Exception as exc:
        if device == "cuda" and "libcublas.so.12" in str(exc):
            raise TranscriptionError(
                "CUDA 12 cuBLAS is not visible to the process. Install "
                "nvidia-cublas-cu12 and nvidia-cudnn-cu12 in the active "
                "environment."
            ) from exc
        if device == "cuda" and "out of memory" in str(exc).lower():
            raise TranscriptionError(
                "CUDA ran out of memory while transcribing. Close other GPU "
                "workloads, try --model small, or run with --device cpu."
            ) from exc
        raise TranscriptionError(f"Transcription failed: {exc}") from exc
