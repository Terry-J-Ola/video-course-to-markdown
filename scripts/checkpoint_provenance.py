import json
import pathlib
import uuid
from typing import Any


CHECKPOINT_VERSION = 1


def file_identity(path: pathlib.Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def checkpoint_matches(path: pathlib.Path, fingerprint: dict[str, Any]) -> bool:
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return False
    return stored == {"version": CHECKPOINT_VERSION, "fingerprint": fingerprint}


def write_checkpoint(path: pathlib.Path, fingerprint: dict[str, Any]) -> None:
    write_json_atomic(
        path,
        {"version": CHECKPOINT_VERSION, "fingerprint": fingerprint},
    )


def write_json_atomic(path: pathlib.Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_checkpoint_fingerprint(path: pathlib.Path) -> dict[str, Any] | None:
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    if stored.get("version") != CHECKPOINT_VERSION:
        return None
    fingerprint = stored.get("fingerprint")
    return fingerprint if isinstance(fingerprint, dict) else None
