"""Capture microphone and system audio, then transcribe it."""

from importlib.metadata import PackageNotFoundError, version


try:
    __version__ = version("audio-capture")
except PackageNotFoundError:
    __version__ = "0+unknown"
