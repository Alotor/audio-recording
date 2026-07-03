# Audio Capture

Linux command-line program that records local audio or downloads a
BigBlueButton recording, then transcribes it locally.

For local capture it records:

- your default microphone;
- the audio currently playing through the default headphones or speakers;
- a combined WAV file;
- a local Whisper transcript.

It supports PipeWire and PulseAudio through `pactl` and FFmpeg's PulseAudio
input.

## Requirements

- `uv`
- Python 3.10 or newer
- `ffmpeg`
- `pactl` (provided by PulseAudio or `pipewire-pulse`)

On Arch Linux:

```bash
sudo pacman -S ffmpeg libpulse
```

## Install

Install the project with NVIDIA GPU transcription:

```bash
uv sync --extra cuda
```

`faster-whisper` downloads the selected model the first time it transcribes.
All transcription commands use the `medium` model on CUDA by default.

For CPU transcription, install the transcription dependencies without the
CUDA runtime and pass `--device cpu` when running a command:

```bash
uv sync --extra transcribe
uv run audio-capture record --device cpu
```

To install recording support without transcription:

```bash
uv sync
```

Run development tests with:

```bash
uv run pytest
```

## Use

Commands are run through `uv`, so activating `.venv` is unnecessary.

List available microphone and system-output monitor sources:

```bash
uv run audio-capture devices
```

Start recording, press `Ctrl+C` to stop, then transcribe:

```bash
uv run audio-capture record
```

The command records the microphone and system output into separate files,
mixes them into `combined.wav`, and then transcribes the complete recording.
For a slower, more accurate transcript, select a larger model:

```bash
uv run audio-capture record --model large-v3
```

The first run downloads the selected model.

Record for a fixed duration:

```bash
uv run audio-capture record --duration 60
```

Set a language or override the model:

```bash
uv run audio-capture record --language es --model large-v3
```

Override the default CUDA compute type when needed:

```bash
uv run audio-capture record --compute-type float16
```

Choose devices explicitly when the defaults are wrong:

```bash
uv run audio-capture record \
  --mic 'alsa_input.usb-Your_Microphone.analog-stereo' \
  --output-source 'alsa_output.pci-Your_Headphones.analog-stereo.monitor'
```

Record without transcription:

```bash
uv run audio-capture record --no-transcribe
```

Transcribe an existing file:

```bash
uv run audio-capture transcribe recordings/session/combined.wav
```

Download the audio-bearing video from a BigBlueButton presentation recording
and transcribe it:

```bash
uv run audio-capture bbb \
  'https://video.example.com/playback/presentation/2.3/RECORDING_ID'
```

The video and `transcript.txt` are saved under `recordings/RECORDING_ID` by
default. The BBB screen-share is a separate visual-only stream; transcription
uses the webcam video because it contains the complete meeting audio. BBB
recordings use the Whisper `medium` model on CUDA with `int8_float16` compute
by default. The usual inference options are supported, including `--language
es`, `--model small` for a faster run, or `--device cpu` when no NVIDIA GPU is
available. To download the video without transcribing it, add
`--no-transcribe`. Re-running the command with the same output directory reuses
an existing recording and continues with transcription instead of downloading
it again.

Each recording session contains:

```text
microphone.wav
system-output.wav
combined.wav
transcript.txt
```

The separate source files are retained so levels can be adjusted or the
recording can be remixed later.
