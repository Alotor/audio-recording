from __future__ import annotations

import json
import shutil
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


class AudioCaptureError(RuntimeError):
    """Raised when audio devices or external recording tools fail."""


CAPTURE_SAMPLE_RATE = 48_000
PCM_CODEC = "pcm_s16le"
MIX_FILTER = (
    "[0:a]aresample=async=1:first_pts=0[mic];"
    "[1:a]aresample=async=1:first_pts=0[system];"
    "[mic][system]amix=inputs=2:duration=longest:"
    "dropout_transition=0:normalize=0,alimiter=limit=0.95[mixed]"
)


@dataclass(frozen=True)
class AudioSource:
    name: str
    description: str
    is_monitor: bool
    monitored_sink: str | None = None


@dataclass(frozen=True)
class AudioDevices:
    microphone: AudioSource
    output_monitor: AudioSource


def require_command(command: str) -> None:
    if shutil.which(command) is None:
        raise AudioCaptureError(
            f"Required command '{command}' was not found in PATH."
        )


def run_text(command: Sequence[str]) -> str:
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise AudioCaptureError(
            f"Required command '{command[0]}' was not found in PATH."
        ) from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        message = f"Command failed: {' '.join(command)}"
        if detail:
            message += f"\n{detail}"
        raise AudioCaptureError(message) from exc
    return result.stdout.strip()


def parse_sources(payload: str) -> list[AudioSource]:
    try:
        raw_sources: Any = json.loads(payload)
    except (json.JSONDecodeError, TypeError) as exc:
        raise AudioCaptureError("Could not parse the audio device list.") from exc
    if not isinstance(raw_sources, list):
        raise AudioCaptureError("The audio device list has an unexpected format.")

    sources: list[AudioSource] = []
    for index, raw in enumerate(raw_sources):
        if not isinstance(raw, dict):
            raise AudioCaptureError(
                f"Audio device entry {index} has an unexpected format."
            )
        raw_name = raw.get("name")
        if raw_name in (None, ""):
            continue
        if not isinstance(raw_name, str):
            raise AudioCaptureError(f"Audio device entry {index} has an invalid name.")
        name = raw_name

        properties = raw.get("properties")
        if properties is None:
            properties = {}
        elif not isinstance(properties, dict):
            raise AudioCaptureError(
                f"Audio device entry {index} has invalid properties."
            )
        description = (
            properties.get("device.description")
            or properties.get("node.description")
            or raw.get("description")
            or name
        )
        is_monitor = (
            raw.get("monitor_of_sink") not in (None, "4294967295", 4294967295)
            or bool(raw.get("monitor_of_sink_name"))
            or name.endswith(".monitor")
            or properties.get("device.class") == "monitor"
        )
        sources.append(
            AudioSource(
                name=name,
                description=str(description),
                is_monitor=is_monitor,
                monitored_sink=(
                    str(raw.get("monitor_of_sink_name"))
                    if raw.get("monitor_of_sink_name")
                    else None
                ),
            )
        )
    return sources


def list_sources() -> list[AudioSource]:
    require_command("pactl")
    return parse_sources(run_text(["pactl", "--format=json", "list", "sources"]))


def resolve_devices(
    sources: Sequence[AudioSource],
    default_source: str,
    default_sink: str,
    microphone_name: str | None = None,
    output_source_name: str | None = None,
) -> AudioDevices:
    by_name = {source.name: source for source in sources}

    if microphone_name:
        microphone = by_name.get(microphone_name)
        if microphone is None:
            raise AudioCaptureError(f"Microphone source not found: {microphone_name}")
    else:
        microphone = by_name.get(default_source)
        if microphone is None or microphone.is_monitor:
            microphone = next(
                (source for source in sources if not source.is_monitor),
                None,
            )
        if microphone is None:
            raise AudioCaptureError("No microphone source was found.")

    if output_source_name:
        output_monitor = by_name.get(output_source_name)
        if output_monitor is None:
            raise AudioCaptureError(
                f"Output monitor source not found: {output_source_name}"
            )
    else:
        monitor_name = f"{default_sink}.monitor"
        output_monitor = by_name.get(monitor_name)
        if output_monitor is None:
            output_monitor = next(
                (
                    source
                    for source in sources
                    if source.is_monitor and source.monitored_sink == default_sink
                ),
                None,
            )
        if output_monitor is None:
            output_monitor = next(
                (source for source in sources if source.is_monitor),
                None,
            )
        if output_monitor is None:
            raise AudioCaptureError(
                "No system-output monitor source was found. "
                "Check that PipeWire/PulseAudio exposes a monitor source."
            )

    if microphone.is_monitor:
        raise AudioCaptureError(
            f"The selected microphone is an output monitor: {microphone.name}"
        )
    if not output_monitor.is_monitor:
        raise AudioCaptureError(
            f"The selected output source is not a monitor: {output_monitor.name}"
        )

    return AudioDevices(microphone=microphone, output_monitor=output_monitor)


def discover_devices(
    microphone_name: str | None = None,
    output_source_name: str | None = None,
) -> AudioDevices:
    sources = list_sources()
    default_source = run_text(["pactl", "get-default-source"])
    default_sink = run_text(["pactl", "get-default-sink"])
    return resolve_devices(
        sources,
        default_source,
        default_sink,
        microphone_name,
        output_source_name,
    )


def build_record_command(
    devices: AudioDevices,
    microphone_path: Path,
    system_path: Path,
    duration: float | None,
) -> list[str]:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-y",
        "-thread_queue_size",
        "1024",
        "-f",
        "pulse",
        "-i",
        devices.microphone.name,
        "-thread_queue_size",
        "1024",
        "-f",
        "pulse",
        "-i",
        devices.output_monitor.name,
    ]
    command.extend(_pcm_output_args("0:a", 1, microphone_path, duration))
    command.extend(_pcm_output_args("1:a", 2, system_path, duration))
    return command


def _pcm_output_args(
    input_stream: str,
    channels: int,
    output_path: Path,
    duration: float | None,
) -> list[str]:
    output = []
    if duration is not None:
        output.extend(["-t", str(duration)])
    output.extend(
        [
            "-map",
            input_stream,
            "-ac",
            str(channels),
            "-ar",
            str(CAPTURE_SAMPLE_RATE),
            "-c:a",
            PCM_CODEC,
            str(output_path),
        ]
    )
    return output


def record(
    devices: AudioDevices,
    microphone_path: Path,
    system_path: Path,
    duration: float | None = None,
) -> None:
    require_command("ffmpeg")
    command = build_record_command(
        devices,
        microphone_path,
        system_path,
        duration,
    )
    process = subprocess.Popen(command, start_new_session=True)
    interrupted = False
    try:
        return_code = process.wait()
    except KeyboardInterrupt:
        interrupted = True
        process.send_signal(signal.SIGINT)
        return_code = process.wait()

    outputs_exist = microphone_path.is_file() and system_path.is_file()
    if return_code != 0 and not (interrupted and outputs_exist):
        raise AudioCaptureError(
            f"ffmpeg recording failed with exit code {return_code}."
        )


def mix(microphone_path: Path, system_path: Path, output_path: Path) -> None:
    require_command("ffmpeg")
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-y",
        "-i",
        str(microphone_path),
        "-i",
        str(system_path),
        "-filter_complex",
        MIX_FILTER,
        "-map",
        "[mixed]",
        "-ac",
        "2",
        "-ar",
        str(CAPTURE_SAMPLE_RATE),
        "-c:a",
        PCM_CODEC,
        str(output_path),
    ]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        raise AudioCaptureError("Could not mix the recorded audio streams.") from exc
