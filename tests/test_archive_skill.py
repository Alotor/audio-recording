"""Exercise archive deletion safeguards using isolated, synthetic recordings."""

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "skills/process-recordings/scripts/archive_recordings.py"
spec = importlib.util.spec_from_file_location("archive_recordings", SCRIPT)
assert spec is not None and spec.loader is not None
archiver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(archiver)


@pytest.fixture
def prepared(tmp_path: Path):
    recordings = tmp_path / "recordings"
    archive = recordings / "archive"
    source = recordings / "2026-10-14_12-00-02"
    source.mkdir(parents=True)
    (source / "transcript.txt").write_text("Acordamos probar el cambio mañana.\n", encoding="utf-8")
    (source / "transcript-live.txt").write_text("Acordamos probar…\n", encoding="utf-8")
    for name in ("combined.wav", "microphone.wav", "system-output.wav"):
        (source / name).write_bytes(b"synthetic audio")
    summary = tmp_path / "summary.md"
    summary.write_text("# Prueba\n\nSe acordó probar el cambio mañana.\n\n[Source](transcript.txt)\n", encoding="utf-8")
    metadata = {"title": "Daily — prueba", "team_or_scope": ["Example team"],
                "topic_tags": ["testing"], "language": "es"}
    return recordings, archive, source, summary, metadata


def process(prepared):
    recordings, archive, source, summary, metadata = prepared
    return archiver.archive_session(recordings, archive, source.name, summary, metadata)


def test_archive_preserves_text_updates_all_indices_and_removes_wavs(prepared):
    recordings, archive, source, summary, _ = prepared
    original = (source / "transcript.txt").read_bytes()
    assert archiver.scan(recordings, archive)[0]["status"] == "ready"

    result = process(prepared)

    destination = archive / result["path"]
    assert (destination / "transcript.txt").read_bytes() == original
    assert (destination / "summary.md").read_bytes() == summary.read_bytes()
    assert (destination / "transcript-live.txt").is_file()
    assert not source.exists()
    catalog = archiver.load_catalog(archive)
    entry = catalog["sessions"][0]
    assert entry["date"] == "2026-10-14T12:00:02+02:00"
    assert {m["name"] for m in entry["removed_media"]} == set(result["deleted_wavs"])
    assert entry["generated_files"][0]["source"] == "transcript.txt"
    for name in ("README.md", "BY_TEAM.md", "BY_TOPIC.md"):
        index = (archive / name).read_text()
        assert result["path"] + "/transcript.txt" in index
        assert result["path"] + "/summary.md" in index
    archiver.check_catalog(archive, catalog)
    archiver.check_indices(archive)
    before = (archive / "catalog.json").read_bytes()
    assert process(prepared)["status"] == "already archived"
    assert (archive / "catalog.json").read_bytes() == before


@pytest.mark.parametrize("problem", ["missing transcript", "empty transcript", "empty summary", "conflicting archive", "symlink"])
def test_invalid_inputs_leave_recordings_intact(prepared, problem):
    _, archive, source, summary, _ = prepared
    if problem == "missing transcript":
        (source / "transcript.txt").unlink()
    elif problem == "empty transcript":
        (source / "transcript.txt").write_text("\n")
    elif problem == "empty summary":
        summary.write_text("\n")
    elif problem == "conflicting archive":
        destination = archive / "2026/10" / source.name
        destination.mkdir(parents=True)
        (destination / "transcript.txt").write_text("different session")
    else:
        (source / "combined.wav").unlink()
        (source / "combined.wav").symlink_to(summary)
    before = {p.name: p.read_bytes() for p in source.iterdir()}

    with pytest.raises(ValueError):
        process(prepared)

    assert {p.name: p.read_bytes() for p in source.iterdir()} == before


def test_index_write_failure_can_be_retried_without_duplicate_entries(prepared, monkeypatch):
    _, archive, source, _, _ = prepared
    original_write = archiver.atomic_write

    def interrupted_write(path, data):
        if path.name == "BY_TOPIC.md":
            raise OSError("simulated disk failure")
        original_write(path, data)

    with monkeypatch.context() as patch:
        patch.setattr(archiver, "atomic_write", interrupted_write)
        with pytest.raises(OSError, match="disk failure"):
            process(prepared)
    assert (source / "combined.wav").is_file()
    assert (source / "transcript.txt").is_file()

    process(prepared)

    assert len(archiver.load_catalog(archive)["sessions"]) == 1
    assert not source.exists()
    archiver.check_indices(archive)


def test_cleanup_retry_works_after_transcript_was_removed(prepared, monkeypatch):
    _, archive, source, _, _ = prepared
    remaining = source / "z-notes.txt"
    remaining.write_text("Supporting notes")
    original_unlink = Path.unlink

    def interrupted_unlink(path, *args, **kwargs):
        if path == remaining:
            raise OSError("simulated interruption during cleanup")
        return original_unlink(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", interrupted_unlink)
        with pytest.raises(OSError, match="interruption"):
            process(prepared)
    assert not (source / "transcript.txt").exists()
    assert remaining.exists()

    process(prepared)

    assert not source.exists()
    assert len(archiver.load_catalog(archive)["sessions"]) == 1


def test_source_change_during_archiving_prevents_deletion(prepared, monkeypatch):
    _, _, source, _, _ = prepared
    original_render = archiver.render_indices

    def recording_still_active(catalog):
        (source / "combined.wav").write_bytes(b"audio recording continued")
        return original_render(catalog)

    monkeypatch.setattr(archiver, "render_indices", recording_still_active)
    with pytest.raises(ValueError, match="Source changed"):
        process(prepared)
    assert (source / "combined.wav").read_bytes() == b"audio recording continued"
    assert (source / "transcript.txt").is_file()


def test_original_summary_and_other_media_are_preserved(prepared):
    _, archive, source, _, _ = prepared
    (source / "summary.md").write_text("Old summary")
    (source / "recording.webm").write_bytes(b"video")
    (source / "unknown.bin").write_bytes(b"unknown")

    result = process(prepared)

    destination = archive / result["path"]
    assert (destination / "summary.md").read_text() == "Old summary"
    assert (destination / "meeting-summary.md").is_file()
    assert result["retained_in_source"] == ["recording.webm", "unknown.bin"]
    assert archiver.load_catalog(archive)["sessions"][0]["summary"] == "meeting-summary.md"


@pytest.mark.parametrize("name", ["summary.md", "summary.txt"])
def test_existing_summary_can_be_reused_verbatim(prepared, name):
    _, archive, source, summary, _ = prepared
    existing = source / name
    existing.write_bytes(summary.read_bytes())

    result = process(prepared)

    entry = archiver.load_catalog(archive)["sessions"][0]
    assert entry["summary"] == name
    assert entry["summary_origin"] == "existing document"
    assert entry["generated_files"] == []
    assert (archive / result["path"] / name).read_bytes() == summary.read_bytes()


def test_existing_catalog_metadata_and_sessions_are_preserved(prepared):
    recordings, archive, source, summary, metadata = prepared
    process(prepared)
    catalog = archiver.load_catalog(archive)
    first = catalog["sessions"][0]
    catalog["custom_metadata"] = {"keep": "this"}
    (archive / "catalog.json").write_text(json.dumps(catalog))
    second = recordings / "2026-10-15_12-00-02"
    second.mkdir()
    (second / "combined.txt").write_text("Second transcript")
    summary.write_text("Second summary [Source](combined.txt)")

    archiver.archive_session(recordings, archive, second.name, summary, metadata)

    updated = archiver.load_catalog(archive)
    assert updated["sessions"][0] == first
    assert updated["custom_metadata"] == {"keep": "this"}
    assert len(updated["sessions"]) == 2
    index = (archive / "README.md").read_text()
    assert index.index(second.name) < index.index(source.name)


def test_bbb_date_uses_madrid_timezone():
    date, origin = archiver.recording_date("0123456789abcdef0123456789abcdef01234567-1704067200000", {})
    assert date.isoformat() == "2024-01-01T01:00:00+01:00"
    assert origin == "BBB recording ID timestamp"


def test_archive_check_detects_corrupt_copy_and_broken_index(prepared):
    _, archive, _, _, _ = prepared
    result = process(prepared)
    transcript = archive / result["path"] / "transcript.txt"
    original = transcript.read_bytes()
    transcript.write_text("corrupted")
    with pytest.raises(ValueError, match="does not match catalog"):
        archiver.check_catalog(archive, archiver.load_catalog(archive))
    transcript.write_bytes(original)
    with (archive / "BY_TOPIC.md").open("a") as stream:
        stream.write("\n[Missing](missing.txt)\n")
    with pytest.raises(ValueError, match="Broken index link"):
        archiver.check_indices(archive)
