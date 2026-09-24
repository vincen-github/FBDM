from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

SCHEMA_NAME = "imagenet-packed-hdf5"
SCHEMA_VERSION = 1
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def content_digest(shard_records: Iterable[dict[str, Any]], class_to_idx: dict[str, int]) -> str:
    """Digest the logical ordered dataset without hashing a manifest recursively."""
    digest = hashlib.sha256()
    digest.update(json.dumps(class_to_idx, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    for record in shard_records:
        digest.update(str(record["index"]).encode("ascii"))
        digest.update(str(record["num_samples"]).encode("ascii"))
        digest.update(str(record["image_bytes"]).encode("ascii"))
        digest.update(record["sha256"].encode("ascii"))
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".part", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def load_wnids(path: Path) -> list[str]:
    wnids: list[str] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            for wnid in line.split():
                if wnid in seen:
                    raise ValueError(f"duplicate WNID {wnid!r} at {path}:{line_number}")
                seen.add(wnid)
                wnids.append(wnid)
    if not wnids:
        raise ValueError(f"WNID list is empty: {path}")
    return wnids


def load_class_to_idx(path: Path) -> dict[str, int]:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if isinstance(raw, dict) and "class_to_idx" in raw:
        raw = raw["class_to_idx"]
    if not isinstance(raw, dict):
        raise ValueError("class mapping must be a JSON object or contain class_to_idx")
    mapping = {str(key): int(value) for key, value in raw.items()}
    expected = set(range(len(mapping)))
    if set(mapping.values()) != expected:
        raise ValueError("class indices must be unique and contiguous from zero")
    return mapping


def normalize_member_name(name: str) -> str:
    normalized = name.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized.lstrip("/")


def is_image_name(name: str) -> bool:
    return Path(name).suffix.lower() in IMAGE_EXTENSIONS


def ensure_new_output_dir(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise FileExistsError(f"output exists and is not a directory: {path}")
        if any(path.iterdir()):
            raise FileExistsError(f"refusing to write into non-empty output directory: {path}")
    else:
        path.mkdir(parents=True)
