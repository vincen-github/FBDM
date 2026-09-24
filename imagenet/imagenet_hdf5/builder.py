from __future__ import annotations

import argparse
import csv
import io
import json
import os
import tarfile
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, Iterator

import h5py
import numpy as np

from .format import (
    SCHEMA_NAME,
    SCHEMA_VERSION,
    atomic_write_json,
    content_digest,
    ensure_new_output_dir,
    is_image_name,
    load_class_to_idx,
    load_wnids,
    normalize_member_name,
    sha256_file,
)


@dataclass(frozen=True)
class Sample:
    sample_id: str
    wnid: str
    image: bytes


def _parse_label_manifest(path: Path) -> dict[str, str]:
    """Read {member: wnid}, JSON records, CSV, or TSV label manifests."""
    def add_label(result: dict[str, str], member: object, wnid: object, context: str) -> None:
        normalized = normalize_member_name(str(member))
        if normalized in result:
            raise ValueError(f"duplicate label-manifest member {normalized!r} at {context}")
        result[normalized] = str(wnid)

    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if isinstance(value, dict) and "samples" in value:
            value = value["samples"]
        if isinstance(value, dict):
            result = {}
            for key, label in value.items():
                add_label(result, key, label, str(path))
        elif isinstance(value, list):
            result = {}
            for index, record in enumerate(value):
                if not isinstance(record, dict):
                    raise ValueError(f"manifest record {index} is not an object")
                member = record.get("member", record.get("path", record.get("sample_id")))
                wnid = record.get("wnid", record.get("label"))
                if member is None or wnid is None:
                    raise ValueError(f"manifest record {index} needs member/path and wnid")
                add_label(result, member, wnid, f"{path}:record {index}")
        else:
            raise ValueError("JSON label manifest must be an object or list")
    else:
        with path.open("r", encoding="utf-8", newline="") as handle:
            preview = handle.read(4096)
            handle.seek(0)
            delimiter = "\t" if "\t" in preview else ","
            rows = list(csv.reader(handle, delimiter=delimiter))
        result = {}
        for row_number, row in enumerate(rows, start=1):
            if not row or not any(field.strip() for field in row):
                continue
            if len(row) < 2:
                raise ValueError(f"label manifest row {row_number} needs at least two columns")
            member, wnid = row[0].strip(), row[1].strip()
            if row_number == 1 and member.lower() in {"member", "path", "sample_id", "filename"}:
                continue
            add_label(result, member, wnid, f"{path}:{row_number}")
    if not result:
        raise ValueError(f"label manifest is empty: {path}")
    return result


def _discover_nested_tar_wnids(path: Path) -> list[str]:
    wnids: set[str] = set()
    with tarfile.open(path, mode="r:*") as archive:
        for member in archive:
            base = Path(normalize_member_name(member.name)).name
            if member.isfile() and base.lower().endswith(".tar"):
                wnids.add(base[:-4])
    if not wnids:
        raise ValueError(f"no per-class tar members found in {path}")
    return sorted(wnids)


def _discover_directory_wnids(path: Path) -> list[str]:
    wnids = sorted(child.name for child in path.iterdir() if child.is_dir())
    if not wnids:
        raise ValueError(f"no class directories found in {path}")
    return wnids


def _resolve_class_mapping(
    source: Path,
    split: str,
    source_kind: str,
    class_to_idx_path: Path | None,
    wnids_path: Path | None,
    labels: dict[str, str] | None,
) -> dict[str, int]:
    supplied = load_class_to_idx(class_to_idx_path) if class_to_idx_path else None
    if wnids_path:
        selected = load_wnids(wnids_path)
        if supplied:
            missing = [wnid for wnid in selected if wnid not in supplied]
            if missing:
                raise ValueError(f"WNIDs missing from class mapping: {missing[:8]}")
        return {wnid: index for index, wnid in enumerate(selected)}
    if supplied:
        return supplied
    if labels:
        discovered = sorted(set(labels.values()))
    elif source_kind == "directory":
        discovered = _discover_directory_wnids(source)
    elif split == "train":
        discovered = _discover_nested_tar_wnids(source)
    else:
        raise ValueError("cannot infer validation classes without --labels or --class-to-idx")
    return {wnid: index for index, wnid in enumerate(discovered)}


def _iter_train_nested_tar(source: Path, allowed: set[str]) -> Iterator[Sample]:
    with tarfile.open(source, mode="r|*") as outer:
        for class_member in outer:
            base = Path(normalize_member_name(class_member.name)).name
            if not class_member.isfile() or not base.lower().endswith(".tar"):
                continue
            wnid = base[:-4]
            nested_stream = outer.extractfile(class_member)
            if nested_stream is None:
                raise OSError(f"cannot read nested tar member {class_member.name}")
            if wnid not in allowed:
                while nested_stream.read(8 * 1024 * 1024):
                    pass
                continue
            with nested_stream:
                with tarfile.open(fileobj=nested_stream, mode="r|*") as inner:
                    for member in inner:
                        name = normalize_member_name(member.name)
                        if not member.isfile() or not is_image_name(name):
                            continue
                        image_stream = inner.extractfile(member)
                        if image_stream is None:
                            raise OSError(f"cannot read image member {member.name} in {class_member.name}")
                        with image_stream:
                            payload = image_stream.read()
                        if not payload:
                            raise ValueError(f"empty image: {class_member.name}:{member.name}")
                        yield Sample(f"{wnid}/{Path(name).name}", wnid, payload)


def _iter_directory(
    source: Path,
    allowed: set[str],
    labels: dict[str, str] | None,
) -> Iterator[Sample]:
    if labels:
        for relative_name, wnid in sorted(labels.items()):
            if wnid not in allowed:
                continue
            path = source / Path(relative_name)
            if not path.is_file():
                raise FileNotFoundError(f"manifest member is missing: {path}")
            payload = path.read_bytes()
            if not payload:
                raise ValueError(f"empty image: {path}")
            yield Sample(relative_name, wnid, payload)
        return
    for wnid in sorted(allowed):
        class_dir = source / wnid
        if not class_dir.is_dir():
            continue
        paths = sorted(path for path in class_dir.rglob("*") if path.is_file() and is_image_name(path.name))
        for path in paths:
            relative = normalize_member_name(path.relative_to(source).as_posix())
            payload = path.read_bytes()
            if not payload:
                raise ValueError(f"empty image: {path}")
            yield Sample(relative, wnid, payload)


def _iter_val_tar(source: Path, allowed: set[str], labels: dict[str, str]) -> Iterator[Sample]:
    observed: set[str] = set()
    with tarfile.open(source, mode="r|*") as archive:
        for member in archive:
            name = normalize_member_name(member.name)
            if not member.isfile() or not is_image_name(name):
                continue
            if name not in labels:
                raise KeyError(f"tar image has no explicit label: {name}")
            observed.add(name)
            wnid = labels[name]
            if wnid not in allowed:
                continue
            image_stream = archive.extractfile(member)
            if image_stream is None:
                raise OSError(f"cannot read tar member {name}")
            with image_stream:
                payload = image_stream.read()
            if not payload:
                raise ValueError(f"empty image: {name}")
            yield Sample(name, wnid, payload)
    missing = sorted(set(labels) - observed)
    if missing:
        raise ValueError(f"label manifest contains {len(missing)} members absent from tar; first={missing[0]}")


class PackedShardWriter:
    def __init__(self, output_dir: Path, split: str, shard_index: int, buffer_samples: int = 256):
        self.output_dir = output_dir
        self.split = split
        self.shard_index = shard_index
        self.final_name = f"{split}-{shard_index:05d}.h5"
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.final_name}.", suffix=".part", dir=str(output_dir)
        )
        os.close(descriptor)
        self.temporary_path = Path(temporary_name)
        self.final_path = output_dir / self.final_name
        self.handle = h5py.File(self.temporary_path, "w", libver="latest")
        self.handle.attrs["schema_name"] = SCHEMA_NAME
        self.handle.attrs["schema_version"] = SCHEMA_VERSION
        self.handle.attrs["split"] = split
        self.handle.attrs["shard_index"] = shard_index
        images = self.handle.create_group("images")
        sample_ids = self.handle.create_group("sample_ids")
        self.image_data = images.create_dataset(
            "data", shape=(0,), maxshape=(None,), chunks=(4 * 1024 * 1024,), dtype="u1"
        )
        self.image_offsets = images.create_dataset(
            "offsets", data=np.array([0], dtype=np.uint64), maxshape=(None,), chunks=(8192,)
        )
        self.labels = self.handle.create_dataset(
            "labels", shape=(0,), maxshape=(None,), chunks=(8192,), dtype="i4"
        )
        self.sample_data = sample_ids.create_dataset(
            "data", shape=(0,), maxshape=(None,), chunks=(1024 * 1024,), dtype="u1"
        )
        self.sample_offsets = sample_ids.create_dataset(
            "offsets", data=np.array([0], dtype=np.uint64), maxshape=(None,), chunks=(8192,)
        )
        self.buffer_samples = buffer_samples
        self._images: list[bytes] = []
        self._sample_ids: list[bytes] = []
        self._labels: list[int] = []
        self.num_samples = 0
        self.image_bytes = 0
        self.sample_id_bytes = 0
        self.closed = False

    def append(self, image: bytes, label: int, sample_id: str) -> None:
        encoded_id = sample_id.encode("utf-8")
        self._images.append(image)
        self._sample_ids.append(encoded_id)
        self._labels.append(label)
        if len(self._labels) >= self.buffer_samples:
            self._flush_buffer()

    @staticmethod
    def _append_packed(data_set: h5py.Dataset, offset_set: h5py.Dataset, values: list[bytes]) -> int:
        if not values:
            return 0
        old_bytes = int(data_set.shape[0])
        lengths = np.fromiter((len(value) for value in values), dtype=np.uint64, count=len(values))
        total = int(lengths.sum())
        data_set.resize((old_bytes + total,))
        packed = b"".join(values)
        data_set[old_bytes : old_bytes + total] = np.frombuffer(packed, dtype=np.uint8)
        old_offsets = int(offset_set.shape[0])
        offset_set.resize((old_offsets + len(values),))
        offset_set[old_offsets:] = old_bytes + np.cumsum(lengths, dtype=np.uint64)
        return total

    def _flush_buffer(self) -> None:
        if not self._labels:
            return
        self.image_bytes += self._append_packed(self.image_data, self.image_offsets, self._images)
        self.sample_id_bytes += self._append_packed(
            self.sample_data, self.sample_offsets, self._sample_ids
        )
        old_labels = int(self.labels.shape[0])
        self.labels.resize((old_labels + len(self._labels),))
        self.labels[old_labels:] = np.asarray(self._labels, dtype=np.int32)
        self.num_samples += len(self._labels)
        self._images.clear()
        self._sample_ids.clear()
        self._labels.clear()

    @property
    def pending_samples(self) -> int:
        return len(self._labels)

    @property
    def projected_image_bytes(self) -> int:
        return self.image_bytes + sum(len(value) for value in self._images)

    def finish(self) -> dict[str, int | str]:
        if self.closed:
            raise RuntimeError("shard writer already closed")
        self._flush_buffer()
        self.handle.attrs["num_samples"] = self.num_samples
        self.handle.attrs["image_bytes"] = self.image_bytes
        self.handle.attrs["sample_id_bytes"] = self.sample_id_bytes
        self.handle.flush()
        self.handle.close()
        self.closed = True
        with self.temporary_path.open("rb+") as file_handle:
            os.fsync(file_handle.fileno())
        if self.final_path.exists():
            raise FileExistsError(f"refusing to overwrite shard: {self.final_path}")
        os.replace(self.temporary_path, self.final_path)
        return {
            "index": self.shard_index,
            "file": self.final_name,
            "num_samples": self.num_samples,
            "image_bytes": self.image_bytes,
            "sample_id_bytes": self.sample_id_bytes,
            "file_bytes": self.final_path.stat().st_size,
            "sha256": sha256_file(self.final_path),
        }

    def abort(self) -> None:
        if not self.closed:
            self.handle.close()
            self.closed = True
        self.temporary_path.unlink(missing_ok=True)


def build_dataset(args: argparse.Namespace) -> Path:
    source = args.input.resolve()
    output = args.output.resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    source_kind = "directory" if source.is_dir() else "tar"
    if source_kind == "tar" and not tarfile.is_tarfile(source):
        raise ValueError(f"input is neither a directory nor a tar archive: {source}")
    labels = _parse_label_manifest(args.labels.resolve()) if args.labels else None
    if args.split == "val" and source_kind == "tar" and labels is None:
        raise ValueError("validation tar input requires --labels with an explicit member-to-WNID map")
    mapping = _resolve_class_mapping(
        source,
        args.split,
        source_kind,
        args.class_to_idx.resolve() if args.class_to_idx else None,
        args.wnids.resolve() if args.wnids else None,
        labels,
    )
    allowed = set(mapping)
    if source_kind == "directory":
        samples = _iter_directory(source, allowed, labels)
    elif args.split == "train":
        samples = _iter_train_nested_tar(source, allowed)
    else:
        assert labels is not None
        samples = _iter_val_tar(source, allowed, labels)

    ensure_new_output_dir(output)
    class_mapping_path = output / "class_to_idx.json"
    atomic_write_json(
        class_mapping_path,
        {"schema_version": SCHEMA_VERSION, "class_to_idx": mapping},
    )
    shard_records: list[dict[str, int | str]] = []
    class_counts: Counter[str] = Counter()
    total_samples = 0
    total_image_bytes = 0
    writer: PackedShardWriter | None = None
    try:
        for sample in samples:
            if sample.wnid not in mapping:
                continue
            if writer is None:
                writer = PackedShardWriter(output, args.split, len(shard_records), args.buffer_samples)
            current_count = writer.num_samples + writer.pending_samples
            projected_bytes = writer.projected_image_bytes + len(sample.image)
            if current_count and (
                current_count >= args.samples_per_shard
                or (args.max_shard_bytes > 0 and projected_bytes > args.max_shard_bytes)
            ):
                record = writer.finish()
                record["sample_start"] = total_samples
                record["sample_end_exclusive"] = total_samples + int(record["num_samples"])
                total_samples += int(record["num_samples"])
                total_image_bytes += int(record["image_bytes"])
                shard_records.append(record)
                writer = PackedShardWriter(output, args.split, len(shard_records), args.buffer_samples)
            writer.append(sample.image, mapping[sample.wnid], sample.sample_id)
            class_counts[sample.wnid] += 1
        if writer is not None and writer.num_samples + writer.pending_samples:
            record = writer.finish()
            record["sample_start"] = total_samples
            record["sample_end_exclusive"] = total_samples + int(record["num_samples"])
            total_samples += int(record["num_samples"])
            total_image_bytes += int(record["image_bytes"])
            shard_records.append(record)
            writer = None
    except BaseException:
        if writer is not None:
            writer.abort()
        close_samples = getattr(samples, "close", None)
        if close_samples is not None:
            close_samples()
        raise
    if not shard_records:
        raise ValueError("no samples matched the selected classes")
    absent = [wnid for wnid in mapping if class_counts[wnid] == 0]
    if absent and not args.allow_empty_classes:
        raise ValueError(f"selected classes have no samples: {absent[:8]}")
    manifest = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "split": args.split,
        "source": {
            "kind": source_kind,
            "name": source.name,
            "size_bytes": source.stat().st_size if source.is_file() else None,
            "checksum": args.source_checksum,
        },
        "class_mapping": {
            "file": class_mapping_path.name,
            "sha256": sha256_file(class_mapping_path),
        },
        "class_to_idx": mapping,
        "num_classes": len(mapping),
        "num_samples": total_samples,
        "image_bytes": total_image_bytes,
        "class_counts": {wnid: class_counts[wnid] for wnid in mapping},
        "shards": shard_records,
        "content_sha256": content_digest(shard_records, mapping),
    }
    manifest_path = output / f"{args.split}.manifest.json"
    atomic_write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "event": "imagenet_hdf5_build_complete",
                "manifest": str(manifest_path),
                "num_samples": total_samples,
                "num_classes": len(mapping),
                "num_shards": len(shard_records),
                "image_bytes": total_image_bytes,
                "content_sha256": manifest["content_sha256"],
            },
            sort_keys=True,
        )
    )
    return manifest_path


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stream ImageNet tar/directory data into atomic packed HDF5 shards."
    )
    parser.add_argument("--split", required=True, choices=("train", "val"))
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--labels",
        type=Path,
        help="Explicit JSON/CSV/TSV member-to-WNID labels (required for a flat val tar/directory).",
    )
    parser.add_argument("--class-to-idx", type=Path, help="Reuse a sealed train class_to_idx.json.")
    parser.add_argument(
        "--source-checksum",
        help="Optional preverified provenance string such as md5:<hex> or sha256:<hex>.",
    )
    parser.add_argument(
        "--wnids",
        type=Path,
        help="Fixed ordered WNID list; filters samples and remaps labels to this exact order.",
    )
    parser.add_argument("--samples-per-shard", type=int, default=10_000)
    parser.add_argument(
        "--max-shard-bytes",
        type=int,
        default=4 * 1024**3,
        help="Rotate before this many packed JPEG bytes; zero disables byte rotation.",
    )
    parser.add_argument("--buffer-samples", type=int, default=256)
    parser.add_argument("--allow-empty-classes", action="store_true")
    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    if args.samples_per_shard <= 0 or args.buffer_samples <= 0 or args.max_shard_bytes < 0:
        parser.error("shard and buffer sizes must be positive (max shard bytes may be zero)")
    build_dataset(args)


if __name__ == "__main__":
    main()
