from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Callable, Protocol

from audio_capture.audio import AudioDevices, discover_devices, mix, record
from audio_capture.transcribe import (
    DEFAULT_DEVICE,
    DEFAULT_MODEL,
    ProgressCallback,
    transcribe,
)


@dataclass(frozen=True)
class RecordingConfig:
    output_dir: Path
    microphone_name: str | None = None
    output_source_name: str | None = None
    duration: float | None = None
    transcribe: bool = True
    model_name: str = DEFAULT_MODEL
    language: str | None = None
    device: str = DEFAULT_DEVICE
    compute_type: str | None = None
    transcription_progress: ProgressCallback | None = None


@dataclass(frozen=True)
class RecordingPaths:
    output_dir: Path
    microphone: Path
    system_output: Path
    combined: Path
    transcript: Path

    @classmethod
    def in_directory(cls, output_dir: Path) -> RecordingPaths:
        return cls(
            output_dir=output_dir,
            microphone=output_dir / "microphone.wav",
            system_output=output_dir / "system-output.wav",
            combined=output_dir / "combined.wav",
            transcript=output_dir / "transcript.txt",
        )


@dataclass(frozen=True)
class RecordingContext:
    config: RecordingConfig
    paths: RecordingPaths
    devices: AudioDevices


class RecordingPhase(Enum):
    READY = auto()
    RECORDING = auto()
    MIXING = auto()
    TRANSCRIBING = auto()


RecordingObserver = Callable[[RecordingPhase, RecordingContext], None]


class CaptureAudio(Protocol):
    def __call__(
        self,
        devices: AudioDevices,
        microphone_path: Path,
        system_path: Path,
        duration: float | None = None,
    ) -> None: ...


class TranscribeAudio(Protocol):
    def __call__(
        self,
        audio_path: Path,
        text_path: Path,
        model_name: str = DEFAULT_MODEL,
        language: str | None = None,
        device: str = DEFAULT_DEVICE,
        compute_type: str | None = None,
        progress: ProgressCallback | None = None,
    ) -> None: ...


@dataclass(frozen=True)
class RecordingDependencies:
    discover: Callable[[str | None, str | None], AudioDevices] = discover_devices
    capture: CaptureAudio = record
    mixer: Callable[[Path, Path, Path], None] = mix
    transcriber: TranscribeAudio = transcribe


def run_recording_session(
    config: RecordingConfig,
    *,
    dependencies: RecordingDependencies | None = None,
    observer: RecordingObserver | None = None,
) -> RecordingContext:
    dependencies = dependencies or RecordingDependencies()
    devices = dependencies.discover(
        config.microphone_name,
        config.output_source_name,
    )
    paths = RecordingPaths.in_directory(config.output_dir)
    config.output_dir.mkdir(parents=True, exist_ok=False)
    context = RecordingContext(config=config, paths=paths, devices=devices)

    _notify(observer, RecordingPhase.READY, context)
    _notify(observer, RecordingPhase.RECORDING, context)
    dependencies.capture(
        devices,
        paths.microphone,
        paths.system_output,
        duration=config.duration,
    )

    _notify(observer, RecordingPhase.MIXING, context)
    dependencies.mixer(paths.microphone, paths.system_output, paths.combined)

    if config.transcribe:
        _notify(observer, RecordingPhase.TRANSCRIBING, context)
        dependencies.transcriber(
            paths.combined,
            paths.transcript,
            model_name=config.model_name,
            language=config.language,
            device=config.device,
            compute_type=config.compute_type,
            progress=config.transcription_progress,
        )

    return context


def _notify(
    observer: RecordingObserver | None,
    phase: RecordingPhase,
    context: RecordingContext,
) -> None:
    if observer is not None:
        observer(phase, context)
