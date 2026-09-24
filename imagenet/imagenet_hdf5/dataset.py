from __future__ import annotations

import bisect
import io
import json
import os
from pathlib import Path
from typing import Callable, Sequence

import h5py
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .format import SCHEMA_NAME, SCHEMA_VERSION, load_wnids


class ImageNetHDF5(Dataset):
    """PyTorch dataset for packed, sharded ImageNet HDF5 files.

    Shards are opened lazily inside each worker process. Passing ``wnids`` (or a
    text file through ``wnids_file``) creates a deterministic subset and remaps
    labels to the exact WNID-list order, which is suitable for a sealed
    ImageNet-100 protocol.
    """

    def __init__(
        self,
        manifest: str | Path,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        wnids: Sequence[str] | None = None,
        wnids_file: str | Path | None = None,
        return_sample_id: bool = False,
    ) -> None:
        self.manifest_path = Path(manifest).resolve()
        self.root = self.manifest_path.parent
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if metadata.get("schema_name") != SCHEMA_NAME:
            raise ValueError(f"unsupported schema: {metadata.get('schema_name')!r}")
        if int(metadata.get("schema_version", -1)) != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema version: {metadata.get('schema_version')!r}")
        self.transform = transform
        self.target_transform = target_transform
        self.return_sample_id = return_sample_id
        original_mapping = {str(key): int(value) for key, value in metadata["class_to_idx"].items()}
        self._original_idx_to_wnid = {value: key for key, value in original_mapping.items()}
        self.shards = [self.root / record["file"] for record in metadata["shards"]]
        self._counts = [int(record["num_samples"]) for record in metadata["shards"]]
        self._cumulative = np.cumsum(self._counts, dtype=np.int64).tolist()
        expected_total = int(metadata["num_samples"])
        if (self._cumulative[-1] if self._cumulative else 0) != expected_total:
            raise ValueError("manifest shard counts do not sum to num_samples")
        if wnids is not None and wnids_file is not None:
            raise ValueError("pass either wnids or wnids_file, not both")
        selected = list(wnids) if wnids is not None else (
            load_wnids(Path(wnids_file)) if wnids_file is not None else None
        )
        self._indices: np.ndarray | None = None
        self._new_label_by_old: dict[int, int] | None = None
        if selected is None:
            self.class_to_idx = original_mapping
        else:
            if len(selected) != len(set(selected)):
                raise ValueError("selected WNID list contains duplicates")
            missing = [wnid for wnid in selected if wnid not in original_mapping]
            if missing:
                raise ValueError(f"selected WNIDs absent from manifest: {missing[:8]}")
            self.class_to_idx = {wnid: index for index, wnid in enumerate(selected)}
            self._new_label_by_old = {
                original_mapping[wnid]: new_index for new_index, wnid in enumerate(selected)
            }
            selected_old = np.asarray(sorted(self._new_label_by_old), dtype=np.int32)
            global_chunks: list[np.ndarray] = []
            start = 0
            for shard, count in zip(self.shards, self._counts):
                with h5py.File(shard, "r", libver="latest", swmr=True) as handle:
                    labels = np.asarray(handle["labels"][:], dtype=np.int32)
                if len(labels) != count:
                    raise ValueError(f"label count differs from manifest: {shard}")
                local = np.flatnonzero(np.isin(labels, selected_old)).astype(np.int64)
                if len(local):
                    global_chunks.append(local + start)
                start += count
            self._indices = (
                np.concatenate(global_chunks) if global_chunks else np.empty((0,), dtype=np.int64)
            )
        self.classes = [wnid for wnid, _ in sorted(self.class_to_idx.items(), key=lambda item: item[1])]
        self._handles: dict[int, h5py.File] = {}
        self._handle_pid: int | None = None

    def __len__(self) -> int:
        return int(len(self._indices)) if self._indices is not None else (
            self._cumulative[-1] if self._cumulative else 0
        )

    def _global_index(self, index: int) -> int:
        length = len(self)
        if index < 0:
            index += length
        if index < 0 or index >= length:
            raise IndexError(index)
        return int(self._indices[index]) if self._indices is not None else index

    def _locate(self, global_index: int) -> tuple[int, int]:
        shard_index = bisect.bisect_right(self._cumulative, global_index)
        previous = self._cumulative[shard_index - 1] if shard_index else 0
        return shard_index, global_index - previous

    def _ensure_process_local_handles(self) -> None:
        pid = os.getpid()
        if self._handle_pid != pid:
            self.close()
            self._handle_pid = pid

    def _handle(self, shard_index: int) -> h5py.File:
        self._ensure_process_local_handles()
        handle = self._handles.get(shard_index)
        if handle is None:
            handle = h5py.File(self.shards[shard_index], "r", libver="latest", swmr=True)
            self._handles[shard_index] = handle
        return handle

    @staticmethod
    def _read_packed(group: h5py.Group, local_index: int) -> bytes:
        offsets = group["offsets"]
        start, end = (int(value) for value in offsets[local_index : local_index + 2])
        return bytes(np.asarray(group["data"][start:end], dtype=np.uint8))

    def __getitem__(self, index: int):
        global_index = self._global_index(index)
        shard_index, local_index = self._locate(global_index)
        handle = self._handle(shard_index)
        image_bytes = self._read_packed(handle["images"], local_index)
        sample_id = (self._read_packed(handle["sample_ids"], local_index).decode("utf-8")
                     if self.return_sample_id else None)
        old_label = int(handle["labels"][local_index])
        label = self._new_label_by_old[old_label] if self._new_label_by_old is not None else old_label
        with Image.open(io.BytesIO(image_bytes)) as encoded:
            image = encoded.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)
        if self.return_sample_id:
            return image, label, sample_id
        return image, label

    def close(self) -> None:
        for handle in self._handles.values():
            try:
                handle.close()
            except Exception:
                pass
        self._handles = {}

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handles"] = {}
        state["_handle_pid"] = None
        return state

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
