import json
import signal
import subprocess
from pathlib import Path

import pytest

from audio_capture.audio import (
    AudioCaptureError,
    AudioDevices,
    AudioSource,
    build_record_command,
    mix,
    parse_sources,
    record,
    resolve_devices,
)


SOURCES = [
    AudioSource(
        name="alsa_input.usb-mic",
        description="USB microphone",
        is_monitor=False,
    ),
    AudioSource(
        name="alsa_output.headphones.monitor",
        description="Headphones monitor",
        is_monitor=True,
        monitored_sink="alsa_output.headphones",
    ),
]


def test_parse_sources() -> None:
    payload = json.dumps(
        [
            {
                "name": "alsa_input.usb-mic",
                "description": "Fallback description",
                "monitor_of_sink": 4294967295,
                "properties": {"device.description": "USB microphone"},
            },
            {
                "name": "alsa_output.headphones.monitor",
                "monitor_of_sink": 12,
                "monitor_of_sink_name": "alsa_output.headphones",
                "properties": {"device.description": "Headphones monitor"},
            },
        ]
    )

    assert parse_sources(payload) == SOURCES


@pytest.mark.parametrize(
    "payload, message",
    [
        ("{}", "unexpected format"),
        ("[1]", "entry 0"),
        ('[{"name": 123}]', "invalid name"),
        ('[{"name": "mic", "properties": []}]', "invalid properties"),
    ],
)
def test_parse_sources_rejects_invalid_schema(payload: str, message: str) -> None:
    with pytest.raises(AudioCaptureError, match=message):
        parse_sources(payload)


def test_resolve_default_devices() -> None:
    devices = resolve_devices(
        SOURCES,
        default_source="alsa_input.usb-mic",
        default_sink="alsa_output.headphones",
    )

    assert devices.microphone == SOURCES[0]
    assert devices.output_monitor == SOURCES[1]


def test_reject_monitor_as_microphone() -> None:
    with pytest.raises(AudioCaptureError, match="selected microphone"):
        resolve_devices(
            SOURCES,
            default_source="alsa_input.usb-mic",
            default_sink="alsa_output.headphones",
            microphone_name="alsa_output.headphones.monitor",
        )


def test_build_record_command() -> None:
    command = build_record_command(
        AudioDevices(SOURCES[0], SOURCES[1]),
        Path("mic.wav"),
        Path("system.wav"),
        duration=5.5,
    )

    assert command.count("-i") == 2
    assert "alsa_input.usb-mic" in command
    assert "alsa_output.headphones.monitor" in command
    assert command.count("-t") == 2
    assert all(
        command[index + 1] == "5.5"
        for index, item in enumerate(command)
        if item == "-t"
    )
    assert command[-1] == "system.wav"


def test_record_reports_failed_process(monkeypatch, tmp_path: Path) -> None:
    class FailedProcess:
        def wait(self) -> int:
            return 2

    monkeypatch.setattr("audio_capture.audio.require_command", lambda _: None)
    monkeypatch.setattr(
        "audio_capture.audio.subprocess.Popen",
        lambda *args, **kwargs: FailedProcess(),
    )

    with pytest.raises(AudioCaptureError, match="exit code 2"):
        record(
            AudioDevices(SOURCES[0], SOURCES[1]),
            tmp_path / "mic.wav",
            tmp_path / "system.wav",
        )


def test_record_accepts_interrupt_when_outputs_exist(
    monkeypatch,
    tmp_path: Path,
) -> None:
    microphone_path = tmp_path / "mic.wav"
    system_path = tmp_path / "system.wav"
    signals = []

    class InterruptedProcess:
        calls = 0

        def wait(self) -> int:
            self.calls += 1
            if self.calls == 1:
                microphone_path.touch()
                system_path.touch()
                raise KeyboardInterrupt
            return 255

        def send_signal(self, sent_signal: signal.Signals) -> None:
            signals.append(sent_signal)

    monkeypatch.setattr("audio_capture.audio.require_command", lambda _: None)
    monkeypatch.setattr(
        "audio_capture.audio.subprocess.Popen",
        lambda *args, **kwargs: InterruptedProcess(),
    )

    record(
        AudioDevices(SOURCES[0], SOURCES[1]),
        microphone_path,
        system_path,
    )

    assert signals == [signal.SIGINT]


def test_mix_converts_process_failure(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("audio_capture.audio.require_command", lambda _: None)

    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "ffmpeg")

    monkeypatch.setattr("audio_capture.audio.subprocess.run", fail)

    with pytest.raises(AudioCaptureError, match="Could not mix"):
        mix(
            tmp_path / "mic.wav",
            tmp_path / "system.wav",
            tmp_path / "combined.wav",
        )
