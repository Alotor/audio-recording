#!/usr/bin/env python3
"""Verified text archiving and index maintenance; summaries are supplied by Codex."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit
from zoneinfo import ZoneInfo

TEXT_SUFFIXES = {".txt", ".md", ".log", ".json"}
TZ = ZoneInfo("Europe/Madrid")
NAV = "[By date](README.md) · [By team / scope](BY_TEAM.md) · [By topic](BY_TOPIC.md)"


def fail(message: str) -> None:
    raise ValueError(message)


def regular(path: Path) -> Path:
    if path.is_symlink() or not path.is_file():
        fail(f"Expected a regular file, not a symlink: {path}")
    return path


def inside(base: Path, relative: str) -> Path:
    rel = Path(relative)
    if rel.is_absolute() or not rel.parts or ".." in rel.parts:
        fail(f"Unsafe relative path: {relative}")
    path = base
    for part in rel.parts:
        path = path / part
        if path.is_symlink():
            fail(f"Symlinks are not supported: {path}")
    return path


def fingerprint(path: Path) -> dict:
    regular(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"name": path.name, "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def atomic_write(path: Path, data: bytes) -> None:
    if path.is_symlink():
        fail(f"Refusing to overwrite symlink: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def load_catalog(archive: Path) -> dict:
    path = inside(archive, "catalog.json")
    if not path.exists():
        now = datetime.now(TZ).isoformat()
        return {"created_at": now, "timezone": "Europe/Madrid", "sessions": []}
    catalog = json.loads(regular(path).read_text(encoding="utf-8"))
    if not isinstance(catalog, dict) or not isinstance(catalog.get("sessions"), list):
        fail("Expected an archive catalog object with a sessions list")
    ids = [s["id"] for s in catalog["sessions"]]
    if len(set(ids)) != len(ids):
        fail("Duplicate session IDs in catalog")
    return catalog


def recording_date(session: str, metadata: dict) -> tuple[datetime, str]:
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", session):
        date = datetime.strptime(session, "%Y-%m-%d_%H-%M-%S").replace(tzinfo=TZ)
        return date, "local folder name"
    bbb = re.fullmatch(r"[0-9a-fA-F]+-(\d{13})", session)
    if bbb:
        return datetime.fromtimestamp(int(bbb[1]) / 1000, TZ), "BBB recording ID timestamp"
    date = datetime.fromisoformat(metadata.get("date", ""))
    if date.tzinfo is None or not metadata.get("date_source"):
        fail("Unrecognized session date; supply date with offset and date_source")
    return date.astimezone(TZ), metadata["date_source"]


def markdown(value: str) -> str:
    return value.replace("\n", " ").replace("|", "\\|").replace("[", "\\[").replace("]", "\\]")


def anchor(label: str) -> str:
    return re.sub(r"[^\w -]", "", label.lower()).replace(" ", "-")


def link(label: str, path: str) -> str:
    return f"[{markdown(label)}]({quote(path, safe='/#')})"


def render_indices(catalog: dict) -> dict[str, bytes]:
    sessions = sorted(catalog["sessions"], key=lambda s: datetime.fromisoformat(s["date"]), reverse=True)
    rows = ["# Recording archive", "", NAV, "",
            f"{len(sessions)} sessions, each with a transcript and a meeting summary.", "",
            "Text archive organized by year and month, newest first. Dates use Europe/Madrid.", "",
            "Team/scope and topic labels describe the content for browsing; they do not establish formal ownership.", ""]
    month = None
    for s in sessions:
        date = datetime.fromisoformat(s["date"]).astimezone(TZ)
        if date.strftime("%Y-%m") != month:
            if month is not None:
                rows += [""]
            month = date.strftime("%Y-%m")
            rows += [f"## {month}", "",
                     "| Date and time | Meeting topic | Team / scope | Topics | Transcript | Summary | Other text |",
                     "| --- | --- | --- | --- | --- | --- | --- |"]
        path = s["path"]
        other = sorted({f["name"] for f in s.get("preserved_files", []) + s.get("generated_files", [])}
                       - {s["transcript"], s["summary"]})
        cells = [date.strftime("%Y-%m-%d %H:%M"), link(s["title"], path + "/"),
                 ", ".join(link(t, "BY_TEAM.md#" + anchor(t)) for t in s.get("team_or_scope", [])) or "—",
                 ", ".join(link(t, "BY_TOPIC.md#" + anchor(t)) for t in s.get("topic_tags", [])) or "—",
                 link("Open", path + "/" + s["transcript"]), link("Open", path + "/" + s["summary"]),
                 ", ".join(link(f, path + "/" + f) for f in other) or "—"]
        rows += ["| " + " | ".join(cells) + " |"]
    output = {"README.md": ("\n".join(rows) + "\n").encode()}
    for filename, field, heading in [("BY_TEAM.md", "team_or_scope", "team / scope"),
                                     ("BY_TOPIC.md", "topic_tags", "topic")]:
        labels = sorted({label for s in sessions for label in s.get(field, [])}, key=str.casefold)
        rows = [f"# Archive by {heading}", "", NAV, "",
                "Labels group recorded content for browsing; ownership may be inferred from summaries.", "",
                "Jump to: " + " · ".join(link(t, "#" + anchor(t)) for t in labels), ""]
        for label in labels:
            group = [s for s in sessions if label in s.get(field, [])]
            rows += [f"## {markdown(label)}", "", f"{len(group)} sessions.", ""]
            for s in group:
                date = datetime.fromisoformat(s["date"]).astimezone(TZ).strftime("%Y-%m-%d %H:%M")
                rows += [f"- **{date}** — {markdown(s['title'])} · "
                         + link("Transcript", s["path"] + "/" + s["transcript"]) + " · "
                         + link("Summary", s["path"] + "/" + s["summary"])]
            rows += [""]
        output[filename] = ("\n".join(rows) + "\n").encode()
    return output


def check_catalog(archive: Path, catalog: dict) -> None:
    for session in catalog["sessions"]:
        folder = inside(archive, session["path"])
        for key in ("transcript", "summary"):
            path = regular(inside(folder, session[key]))
            if not path.read_text(encoding="utf-8").strip():
                fail(f"Empty archived {key}: {path}")
        for record in session.get("preserved_files", []) + session.get("generated_files", []):
            actual = fingerprint(inside(folder, record["name"]))
            if any(actual[k] != record[k] for k in ("bytes", "sha256")):
                fail(f"Archived file does not match catalog: {folder / record['name']}")


def check_indices(archive: Path) -> None:
    for name in ("README.md", "BY_TEAM.md", "BY_TOPIC.md"):
        path = regular(inside(archive, name))
        for href in re.findall(r"\[[^\]]*\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
            url = urlsplit(href)
            if url.scheme or url.netloc:
                continue
            target = inside(archive, unquote(url.path)) if url.path else path
            if not target.exists():
                fail(f"Broken index link in {path}: {href}")
            if url.fragment:
                headings = re.findall(r"^#+ (.+)$", regular(target).read_text(encoding="utf-8"), re.M)
                if unquote(url.fragment) not in {anchor(h) for h in headings}:
                    fail(f"Broken index anchor in {path}: {href}")


def scan(recordings: Path, archive: Path) -> list[dict]:
    archived = {s["id"] for s in load_catalog(archive)["sessions"]}
    pending = []
    for folder in sorted(recordings.iterdir()):
        if folder == archive or not folder.is_dir():
            continue
        if folder.is_symlink():
            pending.append({"session": folder.name, "status": "symlink; skipped"})
            continue
        transcript = next((p for n in ("transcript.txt", "combined.txt")
                           if (p := folder / n).is_file() and not p.is_symlink()
                           and p.read_text(encoding="utf-8").strip()), None)
        status = "ready" if transcript else "no usable transcript; skipped"
        if folder.name in archived:
            cleanup = any(p.suffix.lower() in TEXT_SUFFIXES | {".wav"} for p in folder.iterdir())
            status = "unfinished cleanup" if cleanup else "retained non-WAV files; skipped"
        pending.append({"session": folder.name, "status": status,
                        "transcript": transcript.name if transcript else None,
                        "files": sorted(p.name for p in folder.iterdir())})
    return pending


def archive_session(recordings: Path, archive: Path, session: str, summary: Path, metadata: dict) -> dict:
    if not isinstance(metadata, dict):
        fail("Metadata must be a JSON object")
    if not session or Path(session).name != session or session in {".", "..", "archive"}:
        fail("--session must be a recording folder name, not a path")
    catalog = load_catalog(archive)
    check_catalog(archive, catalog)
    source = inside(recordings, session)
    old = next((s for s in catalog["sessions"] if s["id"] == session), None)
    if not source.exists():
        if old:
            check_indices(archive)
            return {"session": session, "status": "already archived", "path": old["path"]}
        fail(f"Session not found: {source}")
    if not source.is_dir():
        fail(f"Not a session directory: {source}")
    files = sorted(source.iterdir())
    for path in files:
        regular(path)
    transcript = next((p for n in ("transcript.txt", "combined.txt")
                       if (p := source / n).is_file() and p.read_text(encoding="utf-8").strip()), None)
    if transcript is None and old:
        transcript = inside(inside(archive, old["path"]), old["transcript"])
    if transcript is None:
        fail(f"No usable transcript: {source}")
    summary_data = regular(summary).read_bytes()
    if not summary_data.decode("utf-8").strip():
        fail("Summary is empty")
    for field in ("title", "language"):
        if not isinstance(metadata.get(field), str) or not metadata[field].strip():
            fail(f"Metadata requires a nonempty {field}")
    for field in ("team_or_scope", "topic_tags"):
        if not isinstance(metadata.get(field), list) or any(not isinstance(v, str) or not v.strip() for v in metadata[field]):
            fail(f"Metadata requires a list of labels for {field}")
    if not metadata["team_or_scope"]:
        fail("Use Unclassified when no team/scope is supported")
    date, date_source = recording_date(session, metadata)
    relative = f"{date:%Y/%m}/{session}"
    destination = inside(archive, relative)
    wavs = [p for p in files if p.suffix.lower() == ".wav"]
    snapshots = {p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in files}
    retained = [p for p in files if p.suffix.lower() in TEXT_SUFFIXES]
    records = [fingerprint(p) for p in retained]
    if old and (old["path"] != relative or old["title"] != metadata["title"]
                or old.get("team_or_scope") != metadata["team_or_scope"]
                or old.get("topic_tags") != metadata["topic_tags"]):
        fail("Session metadata conflicts with existing catalog entry")
    contents = {p.name: p.read_bytes() for p in retained}
    reused = next((name for name in ("summary.md", "summary.txt")
                   if contents.get(name) == summary_data), None)
    summary_name = old["summary"] if old else reused or "summary.md"
    if not old and summary_name in contents and contents[summary_name] != summary_data:
        summary_name = "meeting-summary.md"
    if summary_name in contents and contents[summary_name] != summary_data:
        fail("Supplied summary conflicts with the session's existing summary")
    contents[summary_name] = summary_data
    if old:
        known = {r["name"]: r for r in old.get("preserved_files", []) + old.get("generated_files", [])}
        if any(r["name"] not in known or any(r[k] != known[r["name"]][k] for k in r) for r in records):
            fail("Source text differs from the cataloged session")
        known_wavs = {r["name"]: r for r in old.get("removed_media", [])}
        for wav in wavs:
            record = known_wavs.get(wav.name)
            if record is None or (record["bytes"], record["mtime_ns"]) != snapshots[wav.name]:
                fail(f"WAV changed since archiving: {wav}")
    destination.mkdir(parents=True, exist_ok=True)
    for name, data in contents.items():
        target = inside(destination, name)
        if target.exists():
            if regular(target).read_bytes() != data:
                fail(f"Conflicting archive file: {target}")
        else:
            atomic_write(target, data)
    for record in records:
        if fingerprint(destination / record["name"]) != record:
            fail(f"Archive copy failed verification: {record['name']}")
    summary_record = fingerprint(destination / summary_name)
    now = datetime.now(TZ).isoformat()
    if not old:
        generated = [] if summary_name in {p.name for p in retained} else [
            {**summary_record, "created_at": now, "source": transcript.name, "language": metadata["language"]}]
        entry = {"id": session, "date": date.isoformat(), "date_source": date_source,
                 "path": relative, "title": metadata["title"], "transcript": transcript.name,
                 "summary": summary_name, "status": "summary available", "preserved_files": records,
                 "removed_media": [{"name": p.name, "bytes": snapshots[p.name][0],
                                    "mtime_ns": snapshots[p.name][1]} for p in wavs],
                 "team_or_scope": metadata["team_or_scope"], "topic_tags": metadata["topic_tags"],
                 "recording_kind": "meeting", "summary_origin": "existing document" if not generated else "generated from transcript",
                 "generated_files": generated}
        catalog["sessions"].append(entry)
        catalog["updated_at"] = now
    check_catalog(archive, catalog)
    indices = render_indices(catalog)
    for name in ("catalog.json", *indices):
        inside(archive, name)
    atomic_write(archive / "catalog.json", (json.dumps(catalog, ensure_ascii=False, indent=2) + "\n").encode())
    for name, data in indices.items():
        atomic_write(archive / name, data)
    check_indices(archive)
    # Revalidate source files and archived copies before the first deletion.
    if sorted(p.name for p in source.iterdir()) != sorted(snapshots):
        fail("Session files changed during archiving; originals retained")
    for path in files:
        regular(path)
        if (path.stat().st_size, path.stat().st_mtime_ns) != snapshots[path.name]:
            fail(f"Source changed during archiving; originals retained: {path}")
    for record in records:
        if fingerprint(source / record["name"]) != record or fingerprint(destination / record["name"]) != record:
            fail("Source or archive text changed; originals retained")
    if (destination / summary_name).read_bytes() != summary_data:
        fail("Archived summary changed; originals retained")
    for path in wavs + retained:
        path.unlink()
    remaining = sorted(p.name for p in source.iterdir())
    if not remaining:
        source.rmdir()
    return {"session": session, "status": "archived", "path": relative,
            "deleted_wavs": [p.name for p in wavs], "retained_in_source": remaining}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("scan", help="List pending sessions without modifying files")
    commands.add_parser("check", help="Verify archive links and catalog checksums")
    process = commands.add_parser("archive", help="Archive one prepared session and delete its WAV files")
    process.add_argument("--session", required=True)
    process.add_argument("--summary", type=Path, required=True)
    process.add_argument("--metadata", type=Path, required=True)
    args = parser.parse_args()
    try:
        recordings = inside(args.root.resolve(), "recordings")
        archive = inside(recordings, "archive")
        if not recordings.is_dir():
            fail(f"Recordings directory not found: {recordings}")
        if args.command == "scan":
            result = scan(recordings, archive)
        elif args.command == "check":
            catalog = load_catalog(archive)
            check_catalog(archive, catalog)
            check_indices(archive)
            result = {"status": "verified", "sessions": len(catalog["sessions"])}
        else:
            lock_path = inside(recordings, ".process-recordings.lock")
            with lock_path.open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                metadata = json.loads(regular(args.metadata).read_text(encoding="utf-8"))
                result = archive_session(recordings, archive, args.session, args.summary, metadata)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
