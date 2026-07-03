from __future__ import annotations


def format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


class DownloadProgress:
    def __init__(self) -> None:
        self._last_width = 0

    def __call__(self, downloaded: int, total: int | None) -> None:
        if total:
            percentage = min(downloaded / total * 100, 100)
            message = (
                f"Downloading: {percentage:5.1f}% "
                f"({format_bytes(downloaded)} / {format_bytes(total)})"
            )
        else:
            message = f"Downloading: {format_bytes(downloaded)}"
        padding = " " * max(self._last_width - len(message), 0)
        print(f"\r{message}{padding}", end="", flush=True)
        self._last_width = len(message)

    def finish(self) -> None:
        if self._last_width:
            print()


class TranscriptionProgress:
    def __init__(self) -> None:
        self._started = False

    def __call__(self, percentage: float) -> None:
        percentage = min(max(percentage, 0), 100)
        print(f"\rTranscribing: {percentage:5.1f}%", end="", flush=True)
        self._started = True

    def finish(self) -> None:
        if self._started:
            print()
