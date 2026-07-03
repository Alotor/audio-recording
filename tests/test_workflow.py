from pathlib import Path

from audio_capture.audio import AudioDevices, AudioSource
from audio_capture.workflow import (
    RecordingConfig,
    RecordingDependencies,
    RecordingPhase,
    run_recording_session,
)


def test_recording_session_runs_capture_mix_and_transcription(tmp_path: Path) -> None:
    calls = []
    phases = []
    devices = AudioDevices(
        microphone=AudioSource("mic", "Microphone", False),
        output_monitor=AudioSource("monitor", "Monitor", True),
    )
    output_dir = tmp_path / "session"

    dependencies = RecordingDependencies(
        discover=lambda mic, output: calls.append(("discover", mic, output))
        or devices,
        capture=lambda *args, **kwargs: calls.append(("capture", args, kwargs)),
        mixer=lambda *args: calls.append(("mix", args)),
        transcriber=lambda *args, **kwargs: calls.append(
            ("transcribe", args, kwargs)
        ),
    )
    config = RecordingConfig(
        output_dir=output_dir,
        microphone_name="mic",
        output_source_name="monitor",
        duration=3,
        language="es",
    )

    context = run_recording_session(
        config,
        dependencies=dependencies,
        observer=lambda phase, _: phases.append(phase),
    )

    assert output_dir.is_dir()
    assert context.devices == devices
    assert phases == [
        RecordingPhase.READY,
        RecordingPhase.RECORDING,
        RecordingPhase.MIXING,
        RecordingPhase.TRANSCRIBING,
    ]
    assert calls == [
        ("discover", "mic", "monitor"),
        (
            "capture",
            (devices, output_dir / "microphone.wav", output_dir / "system-output.wav"),
            {"duration": 3},
        ),
        (
            "mix",
            (
                output_dir / "microphone.wav",
                output_dir / "system-output.wav",
                output_dir / "combined.wav",
            ),
        ),
        (
            "transcribe",
            (output_dir / "combined.wav", output_dir / "transcript.txt"),
            {
                "model_name": "medium",
                "language": "es",
                "device": "cuda",
                "compute_type": None,
                "progress": None,
            },
        ),
    ]


def test_recording_session_can_skip_transcription(tmp_path: Path) -> None:
    calls = []
    devices = AudioDevices(
        microphone=AudioSource("mic", "Microphone", False),
        output_monitor=AudioSource("monitor", "Monitor", True),
    )
    dependencies = RecordingDependencies(
        discover=lambda *_: devices,
        capture=lambda *args, **kwargs: calls.append("capture"),
        mixer=lambda *args: calls.append("mix"),
        transcriber=lambda *args, **kwargs: calls.append("transcribe"),
    )

    run_recording_session(
        RecordingConfig(output_dir=tmp_path / "session", transcribe=False),
        dependencies=dependencies,
    )

    assert calls == ["capture", "mix"]
