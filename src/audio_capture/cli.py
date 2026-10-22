from __future__ import annotations

import argparse
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

from audio_capture.audio import (
    AudioCaptureError,
    list_sources,
)
from audio_capture.bbb import (
    BBBError,
    download_media,
    resolve_bbb_recording,
)
from audio_capture.progress import DownloadProgress, TranscriptionProgress
from audio_capture.transcribe import (
    DEFAULT_CUDA_COMPUTE_TYPE,
    DEFAULT_DEVICE,
    DEFAULT_MODEL,
    TranscriptionError,
    cuda_environment,
    transcribe,
)
from audio_capture.workflow import (
    RecordingConfig,
    RecordingContext,
    RecordingPhase,
    run_recording_session,
)


BBB_DEFAULT_COMPUTE_TYPE = DEFAULT_CUDA_COMPUTE_TYPE


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="audio-capture",
        description=(
            "Capture local audio or download a BigBlueButton recording, then "
            "transcribe it locally."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    devices_parser = subparsers.add_parser(
        "devices", help="List microphone and output-monitor sources."
    )
    devices_parser.set_defaults(handler=handle_devices)

    record_parser = subparsers.add_parser(
        "record", help="Record both audio sources and optionally transcribe them."
    )
    record_parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        help="Session directory (default: recordings/YYYY-MM-DD_HH-MM-SS).",
    )
    record_parser.add_argument("--mic", help="PulseAudio microphone source name.")
    record_parser.add_argument(
        "--output-source", help="PulseAudio monitor source name."
    )
    record_parser.add_argument(
        "--duration",
        type=positive_float,
        help="Stop after this many seconds; otherwise press Ctrl+C.",
    )
    record_parser.add_argument(
        "--no-transcribe",
        action="store_true",
        help="Record and mix audio without running Whisper.",
    )
    record_parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(
            "faster-whisper model name or local model path "
            f"(default: {DEFAULT_MODEL})."
        ),
    )
    record_parser.add_argument(
        "--language",
        help="Pin one spoken language (default: detect English/Spanish switches).",
    )
    add_inference_arguments(record_parser)
    record_parser.set_defaults(handler=handle_record)

    transcribe_parser = subparsers.add_parser(
        "transcribe", help="Transcribe an existing audio file."
    )
    transcribe_parser.add_argument("audio_file", type=Path)
    transcribe_parser.add_argument(
        "-o", "--output", type=Path, help="Output text file."
    )
    transcribe_parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(
            "faster-whisper model name or local model path "
            f"(default: {DEFAULT_MODEL})."
        ),
    )
    transcribe_parser.add_argument(
        "--language",
        help="Pin one spoken language (default: detect English/Spanish switches).",
    )
    add_inference_arguments(transcribe_parser)
    transcribe_parser.set_defaults(handler=handle_transcribe)

    bbb_parser = subparsers.add_parser(
        "bbb", help="Download and transcribe a BigBlueButton recording."
    )
    bbb_parser.add_argument("url", help="BBB presentation playback URL.")
    bbb_parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        help="Output directory (default: recordings/<recording-id>).",
    )
    bbb_parser.add_argument(
        "--no-transcribe",
        action="store_true",
        help="Download the recording without running Whisper.",
    )
    bbb_parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(
            "faster-whisper model name or local model path "
            f"(default: {DEFAULT_MODEL})."
        ),
    )
    bbb_parser.add_argument(
        "--language",
        help="Pin one spoken language (default: detect English/Spanish switches).",
    )
    add_inference_arguments(
        bbb_parser,
        default_device=DEFAULT_DEVICE,
        default_compute_type=BBB_DEFAULT_COMPUTE_TYPE,
    )
    bbb_parser.set_defaults(handler=handle_bbb)
    return parser


def add_inference_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_device: str = DEFAULT_DEVICE,
    default_compute_type: str | None = None,
) -> None:
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "auto"),
        default=default_device,
        help=f"Whisper inference device (default: {default_device}).",
    )
    compute_type_default = default_compute_type or (
        "int8"
        if default_device == "cpu"
        else DEFAULT_CUDA_COMPUTE_TYPE
        if default_device == "cuda"
        else "automatic"
    )
    parser.add_argument(
        "--compute-type",
        help=(
            "CTranslate2 compute type "
            f"(default: {compute_type_default})."
        ),
    )


def positive_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError(
            "must be a finite number greater than zero"
        )
    return number


def handle_devices(_: argparse.Namespace) -> int:
    sources = list_sources()
    if not sources:
        print("No PulseAudio/PipeWire sources found.")
        return 1

    for source in sources:
        source_type = "output monitor" if source.is_monitor else "microphone"
        print(f"{source_type:14} {source.name}")
        print(f"{'':14} {source.description}")
    return 0


def handle_record(args: argparse.Namespace) -> int:
    output_dir = args.output_dir
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = Path("recordings") / timestamp

    progress = TranscriptionProgress()
    config = RecordingConfig(
        output_dir=output_dir,
        microphone_name=args.mic,
        output_source_name=args.output_source,
        duration=args.duration,
        transcribe=not args.no_transcribe,
        model_name=args.model,
        language=args.language,
        device=args.device,
        compute_type=args.compute_type,
        transcription_progress=progress,
    )
    try:
        context = run_recording_session(config, observer=_report_recording_phase)
    finally:
        progress.finish()

    if config.transcribe:
        print(f"Transcript:   {context.paths.transcript.resolve()}")
    print(f"Recording:    {context.paths.combined.resolve()}")
    return 0


def _report_recording_phase(
    phase: RecordingPhase,
    context: RecordingContext,
) -> None:
    if phase is RecordingPhase.READY:
        print(f"Microphone:   {context.devices.microphone.description}")
        print(f"System audio: {context.devices.output_monitor.description}")
        print(f"Saving to:    {context.paths.output_dir.resolve()}")
    elif phase is RecordingPhase.RECORDING:
        if context.config.duration is None:
            print("Recording. Press Ctrl+C to stop.")
        else:
            print(f"Recording for {context.config.duration:g} seconds.")
    elif phase is RecordingPhase.MIXING:
        print("Mixing audio streams.")
    elif phase is RecordingPhase.TRANSCRIBING:
        print(f"Transcribing with Whisper model '{context.config.model_name}'.")


def handle_transcribe(args: argparse.Namespace) -> int:
    if not args.audio_file.is_file():
        raise TranscriptionError(f"Audio file not found: {args.audio_file}")
    output_path = args.output or args.audio_file.with_suffix(".txt")
    progress = TranscriptionProgress()
    try:
        transcribe(
            args.audio_file,
            output_path,
            model_name=args.model,
            language=args.language,
            device=args.device,
            compute_type=args.compute_type,
            progress=progress,
        )
    finally:
        progress.finish()
    print(f"Transcript: {output_path.resolve()}")
    return 0


def handle_bbb(args: argparse.Namespace) -> int:
    print("Resolving BBB recording.")
    recording = resolve_bbb_recording(args.url)
    output_dir = args.output_dir or Path("recordings") / recording.recording_id
    output_dir.mkdir(parents=True, exist_ok=True)

    extension = Path(urlsplit(recording.media_url).path).suffix or ".webm"
    video_path = output_dir / f"recording{extension}"
    transcript_path = output_dir / "transcript.txt"
    if video_path.is_file():
        print(f"Recording already downloaded: {video_path.resolve()}")
    else:
        progress = DownloadProgress()
        try:
            download_media(recording.media_url, video_path, progress=progress)
        finally:
            progress.finish()

    print(f"Recording:  {video_path.resolve()}")
    if args.no_transcribe:
        return 0

    print(f"Transcribing with Whisper model '{args.model}'.")
    compute_type = args.compute_type
    if compute_type is None and args.device == "cuda":
        compute_type = BBB_DEFAULT_COMPUTE_TYPE
    transcription_progress = TranscriptionProgress()
    try:
        transcribe(
            video_path,
            transcript_path,
            model_name=args.model,
            language=args.language,
            device=args.device,
            compute_type=compute_type,
            progress=transcription_progress,
        )
    finally:
        transcription_progress.finish()
    print(f"Transcript: {transcript_path.resolve()}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if (
        getattr(args, "device", None) == "cuda"
        and os.environ.get("AUDIO_CAPTURE_CUDA_PATH_READY") != "1"
    ):
        environment = cuda_environment()
        if environment is not None:
            command_args = list(sys.argv[1:] if argv is None else argv)
            os.execvpe(
                sys.executable,
                [sys.executable, "-m", "audio_capture", *command_args],
                environment,
            )
    try:
        return args.handler(args)
    except (
        AudioCaptureError,
        BBBError,
        TranscriptionError,
        FileExistsError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
