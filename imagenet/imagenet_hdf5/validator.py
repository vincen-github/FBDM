from __future__ import annotations

import argparse
import io
import json
import random
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from .format import SCHEMA_NAME, SCHEMA_VERSION, content_digest, sha256_file


def _assert_offsets(offsets: np.ndarray, data_length: int, expected_samples: int, name: str) -> None:
    if offsets.dtype.kind != "u":
        raise ValueError(f"{name} offsets must have an unsigned integer dtype")
    if offsets.shape != (expected_samples + 1,):
        raise ValueError(f"{name} offsets have shape {offsets.shape}, expected {(expected_samples + 1,)}")
    if int(offsets[0]) != 0 or int(offsets[-1]) != data_length:
        raise ValueError(f"{name} offsets do not span [0, data_length]")
    if np.any(offsets[1:] < offsets[:-1]):
        raise ValueError(f"{name} offsets are not monotonic")
    if expected_samples and np.any(offsets[1:] == offsets[:-1]):
        raise ValueError(f"{name} contains an empty packed value")


def _decode_one(handle: h5py.File, local_index: int) -> None:
    offsets = handle["images/offsets"]
    start, end = (int(value) for value in offsets[local_index : local_index + 2])
    payload = bytes(np.asarray(handle["images/data"][start:end], dtype=np.uint8))
    with Image.open(io.BytesIO(payload)) as image:
        image.load()
        if image.width <= 0 or image.height <= 0:
            raise ValueError("decoded image has invalid dimensions")
    sample_offsets = handle["sample_ids/offsets"]
    id_start, id_end = (int(value) for value in sample_offsets[local_index : local_index + 2])
    bytes(np.asarray(handle["sample_ids/data"][id_start:id_end], dtype=np.uint8)).decode("utf-8")


def validate(manifest_path: Path, decode_samples: int, decode_all: bool) -> dict:
    manifest_path = manifest_path.resolve()
    root = manifest_path.parent
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_name") != SCHEMA_NAME:
        raise ValueError("manifest schema_name mismatch")
    if int(manifest.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("manifest schema_version mismatch")
    mapping = {str(key): int(value) for key, value in manifest["class_to_idx"].items()}
    if set(mapping.values()) != set(range(len(mapping))):
        raise ValueError("class_to_idx must be contiguous and unique")
    idx_to_wnid = {value: key for key, value in mapping.items()}
    class_mapping_record = manifest.get("class_mapping")
    if not isinstance(class_mapping_record, dict):
        raise ValueError("manifest has no sealed class_mapping record")
    class_mapping_path = root / class_mapping_record["file"]
    if sha256_file(class_mapping_path) != class_mapping_record["sha256"]:
        raise ValueError("class_to_idx.json SHA256 mismatch")
    with class_mapping_path.open("r", encoding="utf-8") as handle:
        class_mapping_value = json.load(handle)
    if class_mapping_value.get("class_to_idx") != manifest["class_to_idx"]:
        raise ValueError("class_to_idx.json differs from manifest")
    expected_total = int(manifest["num_samples"])
    requested_decode: set[int] = set()
    if not decode_all:
        sample_count = min(max(decode_samples, 0), expected_total)
        requested_decode = set(random.Random(0).sample(range(expected_total), sample_count))
    observed_counts: Counter[str] = Counter()
    observed_total = 0
    observed_bytes = 0
    decoded = 0
    rebuilt_records: list[dict] = []
    for shard_position, record in enumerate(manifest["shards"]):
        if int(record["index"]) != shard_position:
            raise ValueError("shard indices are not contiguous and ordered")
        if int(record["sample_start"]) != observed_total:
            raise ValueError("shard sample_start is not contiguous")
        path = root / record["file"]
        if path.stat().st_size != int(record["file_bytes"]):
            raise ValueError(f"file size differs from manifest: {path}")
        actual_hash = sha256_file(path)
        if actual_hash != record["sha256"]:
            raise ValueError(f"SHA256 mismatch: {path}")
        expected_samples = int(record["num_samples"])
        if int(record["sample_end_exclusive"]) != observed_total + expected_samples:
            raise ValueError("shard sample_end_exclusive is inconsistent")
        with h5py.File(path, "r", libver="latest", swmr=True) as handle:
            if handle.attrs.get("schema_name") != SCHEMA_NAME:
                raise ValueError(f"shard schema mismatch: {path}")
            if int(handle.attrs.get("schema_version", -1)) != SCHEMA_VERSION:
                raise ValueError(f"shard schema version mismatch: {path}")
            if int(handle.attrs.get("num_samples", -1)) != expected_samples:
                raise ValueError(f"shard num_samples attribute mismatch: {path}")
            if int(handle.attrs.get("shard_index", -1)) != shard_position:
                raise ValueError(f"shard_index attribute mismatch: {path}")
            required = {"images/data", "images/offsets", "labels", "sample_ids/data", "sample_ids/offsets"}
            missing = sorted(name for name in required if name not in handle)
            if missing:
                raise ValueError(f"shard is missing datasets {missing}: {path}")
            image_data = handle["images/data"]
            image_offsets = np.asarray(handle["images/offsets"][:])
            labels = np.asarray(handle["labels"][:])
            sample_data = handle["sample_ids/data"]
            sample_offsets = np.asarray(handle["sample_ids/offsets"][:])
            if int(image_data.shape[0]) != int(record["image_bytes"]):
                raise ValueError(f"packed image byte count differs from manifest: {path}")
            if int(sample_data.shape[0]) != int(record["sample_id_bytes"]):
                raise ValueError(f"packed sample-ID byte count differs from manifest: {path}")
            if int(handle.attrs.get("image_bytes", -1)) != int(image_data.shape[0]):
                raise ValueError(f"image_bytes attribute mismatch: {path}")
            if int(handle.attrs.get("sample_id_bytes", -1)) != int(sample_data.shape[0]):
                raise ValueError(f"sample_id_bytes attribute mismatch: {path}")
            if image_data.dtype != np.dtype("uint8") or sample_data.dtype != np.dtype("uint8"):
                raise ValueError(f"packed data arrays must be uint8: {path}")
            if labels.dtype != np.dtype("int32") or labels.shape != (expected_samples,):
                raise ValueError(f"labels must be int32[{expected_samples}]: {path}")
            _assert_offsets(image_offsets, int(image_data.shape[0]), expected_samples, "image")
            _assert_offsets(sample_offsets, int(sample_data.shape[0]), expected_samples, "sample ID")
            if np.any(labels < 0) or np.any(labels >= len(mapping)):
                raise ValueError(f"label outside class mapping: {path}")
            unique, counts = np.unique(labels, return_counts=True)
            for label, count in zip(unique.tolist(), counts.tolist()):
                observed_counts[idx_to_wnid[int(label)]] += int(count)
            shard_start = observed_total
            local_decode = (
                range(expected_samples)
                if decode_all
                else (
                    index - shard_start
                    for index in sorted(requested_decode)
                    if shard_start <= index < shard_start + expected_samples
                )
            )
            for local_index in local_decode:
                _decode_one(handle, local_index)
                decoded += 1
            observed_total += expected_samples
            observed_bytes += int(image_data.shape[0])
        rebuilt_records.append(
            {
                "index": shard_position,
                "num_samples": expected_samples,
                "image_bytes": int(record["image_bytes"]),
                "sha256": actual_hash,
            }
        )
    if observed_total != expected_total:
        raise ValueError(f"sample total mismatch: observed={observed_total}, manifest={expected_total}")
    if observed_bytes != int(manifest["image_bytes"]):
        raise ValueError(f"image byte total mismatch: observed={observed_bytes}")
    expected_counts = {str(key): int(value) for key, value in manifest["class_counts"].items()}
    if {wnid: observed_counts[wnid] for wnid in mapping} != expected_counts:
        raise ValueError("class distribution differs from manifest")
    rebuilt_digest = content_digest(rebuilt_records, mapping)
    if rebuilt_digest != manifest["content_sha256"]:
        raise ValueError("manifest content_sha256 mismatch")
    result = {
        "event": "imagenet_hdf5_validation",
        "status": "ok",
        "manifest": str(manifest_path),
        "num_samples": observed_total,
        "num_classes": len(mapping),
        "num_shards": len(manifest["shards"]),
        "image_bytes": observed_bytes,
        "decoded_images": decoded,
        "content_sha256": rebuilt_digest,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate packed ImageNet HDF5 shards and hashes.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--decode-samples", type=int, default=128)
    parser.add_argument("--decode-all", action="store_true")
    args = parser.parse_args()
    validate(args.manifest, args.decode_samples, args.decode_all)


if __name__ == "__main__":
    main()
