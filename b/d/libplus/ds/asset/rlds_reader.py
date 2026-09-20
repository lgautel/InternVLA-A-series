#!/usr/bin/env python3
"""Dependency-free RLDS / TFRecord reader for /B/Dta/opvla_libero.

TensorFlow is not installed on this host, so `tfds.load` is unavailable. The
TFRecord container and `tf.train.Example` payload are both simple enough to
decode directly:

    TFRecord record := uint64 length | uint32 masked_crc(length)
                     | bytes  data   | uint32 masked_crc(data)

    Example := Features features = 1
    Features := map<string, Feature> feature = 1
    Feature  := BytesList bytes_list = 1 | FloatList float_list = 2
                | Int64List int64_list = 3

RLDS flattens an episode into one Example: every `steps/...` key holds the
whole trajectory (a `BytesList` with T entries for images/strings, or a single
packed `FloatList` of T*dim floats for tensors).

`crc32c` is not installed either, so record CRCs are skipped; integrity is
instead checked against the `shardLengths` declared in `dataset_info.json`
(see `verify_shards`).
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Iterator

import numpy as np

DATA_ROOT = Path("/B/Dta/opvla_libero")

# Directory name -> tfrecord file stem. `libero_10` ships as `liber_o10`:
# a typo baked into the published OpenVLA builder name.
SUBSETS: dict[str, str] = {
    "libero_spatial": "libero_spatial",
    "libero_object": "libero_object",
    "libero_goal": "libero_goal",
    "libero_10": "liber_o10",
}

STEP_TENSOR_DIMS = {
    "steps/action": 7,
    "steps/observation/state": 8,
    "steps/observation/joint_state": 7,
}


# --------------------------------------------------------------------------
# protobuf wire format
# --------------------------------------------------------------------------
def _read_varint(buf: bytes, pos: int) -> tuple[int, int]:
    result = 0
    shift = 0
    while True:
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
        shift += 7


def _split_fields(buf: bytes) -> dict[int, list]:
    """Return {field_number: [raw_value, ...]} for one protobuf message."""
    out: dict[int, list] = {}
    pos = 0
    end = len(buf)
    while pos < end:
        key, pos = _read_varint(buf, pos)
        field_no, wire_type = key >> 3, key & 7
        if wire_type == 0:
            value, pos = _read_varint(buf, pos)
        elif wire_type == 1:
            value, pos = buf[pos : pos + 8], pos + 8
        elif wire_type == 2:
            length, pos = _read_varint(buf, pos)
            value, pos = buf[pos : pos + length], pos + length
        elif wire_type == 5:
            value, pos = buf[pos : pos + 4], pos + 4
        else:
            raise ValueError(f"unsupported wire type {wire_type} at offset {pos}")
        out.setdefault(field_no, []).append(value)
    return out


def _parse_example(payload: bytes) -> dict[str, bytes]:
    """Example bytes -> {feature_name: serialized Feature}."""
    example = _split_fields(payload)
    features = _split_fields(example[1][0])
    out: dict[str, bytes] = {}
    for entry in features.get(1, []):
        kv = _split_fields(entry)
        name = kv[1][0].decode()
        out[name] = kv[2][0] if 2 in kv else b""
    return out


def _as_bytes_list(feature: bytes) -> list[bytes]:
    holder = _split_fields(feature).get(1)
    if not holder:
        return []
    return _split_fields(holder[0]).get(1, [])


def _as_float_array(feature: bytes) -> np.ndarray:
    """FloatList -> float32 array. TFDS always writes one packed run."""
    holder = _split_fields(feature).get(2)
    if not holder:
        return np.zeros(0, dtype=np.float32)
    chunks = _split_fields(holder[0]).get(1, [])
    if len(chunks) == 1:
        return np.frombuffer(chunks[0], dtype="<f4")
    return np.concatenate([np.frombuffer(c, dtype="<f4") for c in chunks])


def _as_int64_array(feature: bytes) -> np.ndarray:
    holder = _split_fields(feature).get(3)
    if not holder:
        return np.zeros(0, dtype=np.int64)
    values: list[int] = []
    for chunk in _split_fields(holder[0]).get(1, []):
        pos = 0
        while pos < len(chunk):
            value, pos = _read_varint(chunk, pos)
            values.append(value)
    return np.asarray(values, dtype=np.int64)


# --------------------------------------------------------------------------
# TFRecord container
# --------------------------------------------------------------------------
def iter_tfrecords(path: Path) -> Iterator[bytes]:
    """Yield raw record payloads. CRC footers are read but not verified."""
    with open(path, "rb") as fh:
        while True:
            header = fh.read(8)
            if len(header) < 8:
                return
            length = struct.unpack("<Q", header)[0]
            fh.read(4)
            payload = fh.read(length)
            if len(payload) < length:
                raise EOFError(f"truncated record in {path}")
            fh.read(4)
            yield payload


# --------------------------------------------------------------------------
# episode-level API
# --------------------------------------------------------------------------
class Episode:
    """One RLDS episode. Tensor fields are decoded lazily."""

    __slots__ = ("_raw", "subset", "shard", "index")

    def __init__(self, raw: dict[str, bytes], subset: str, shard: str, index: int):
        self._raw = raw
        self.subset = subset
        self.shard = shard
        self.index = index

    @property
    def file_path(self) -> str:
        return _as_bytes_list(self._raw["episode_metadata/file_path"])[0].decode()

    @property
    def task_name(self) -> str:
        """Original demo stem, e.g. `KITCHEN_SCENE6_put_..._and_close_it`."""
        return Path(self.file_path).name.removesuffix("_demo.hdf5")

    @property
    def language_instructions(self) -> list[str]:
        return [b.decode() for b in _as_bytes_list(self._raw["steps/language_instruction"])]

    @property
    def instruction(self) -> str:
        return self.language_instructions[0]

    @property
    def num_steps(self) -> int:
        return len(_as_bytes_list(self._raw["steps/observation/image"]))

    def tensor(self, key: str) -> np.ndarray:
        """`steps/...` float tensor reshaped to [T, dim]."""
        dim = STEP_TENSOR_DIMS[key]
        return _as_float_array(self._raw[key]).reshape(-1, dim)

    @property
    def action(self) -> np.ndarray:
        return self.tensor("steps/action")

    @property
    def state(self) -> np.ndarray:
        return self.tensor("steps/observation/state")

    @property
    def joint_state(self) -> np.ndarray:
        return self.tensor("steps/observation/joint_state")

    @property
    def reward(self) -> np.ndarray:
        return _as_float_array(self._raw["steps/reward"])

    @property
    def discount(self) -> np.ndarray:
        return _as_float_array(self._raw["steps/discount"])

    def flags(self, name: str) -> np.ndarray:
        """`is_first` / `is_last` / `is_terminal` as a bool array."""
        return _as_int64_array(self._raw[f"steps/{name}"]).astype(bool)

    def jpeg(self, camera: str = "image", t: int = 0) -> bytes:
        """Raw JPEG bytes for `image` (agentview) or `wrist_image`."""
        return _as_bytes_list(self._raw[f"steps/observation/{camera}"])[t]

    def jpeg_sizes(self, camera: str = "image") -> np.ndarray:
        return np.asarray([len(b) for b in _as_bytes_list(self._raw[f"steps/observation/{camera}"])])

    def qpos9(self) -> np.ndarray:
        """[T, 9] MuJoCo qpos: 7 arm joints + 2 gripper fingers.

        Matches `_build_qpos` in util_scripts/generate_libero_keypoints.py,
        which takes the fingers from `observation.state[6:8]`.
        """
        return np.concatenate([self.joint_state, self.state[:, 6:8]], axis=1).astype(np.float64)


def shard_paths(subset: str) -> list[Path]:
    stem = SUBSETS[subset]
    version_dir = DATA_ROOT / f"{subset}_no_noops" / "1.0.0"
    return sorted(version_dir.glob(f"{stem}-train.tfrecord-*"))


def declared_shard_lengths(subset: str) -> list[int]:
    info = json.loads((DATA_ROOT / f"{subset}_no_noops" / "1.0.0" / "dataset_info.json").read_text())
    split = next(s for s in info["splits"] if s["name"] == "train")
    return [int(n) for n in split["shardLengths"]]


def iter_episodes(subset: str, max_episodes: int | None = None) -> Iterator[Episode]:
    seen = 0
    for path in shard_paths(subset):
        for index, payload in enumerate(iter_tfrecords(path)):
            yield Episode(_parse_example(payload), subset, path.name, index)
            seen += 1
            if max_episodes is not None and seen >= max_episodes:
                return


def verify_shards(subset: str) -> dict:
    """Record counts per shard vs `dataset_info.json`. Replaces the CRC check."""
    declared = declared_shard_lengths(subset)
    paths = shard_paths(subset)
    observed = [sum(1 for _ in iter_tfrecords(p)) for p in paths]
    return {
        "subset": subset,
        "num_shards": len(paths),
        "declared_total": sum(declared),
        "observed_total": sum(observed),
        "per_shard_match": observed == declared,
        "mismatches": [
            {"shard": p.name, "declared": d, "observed": o}
            for p, d, o in zip(paths, declared, observed, strict=True)
            if d != o
        ],
    }


if __name__ == "__main__":
    for name in SUBSETS:
        report = verify_shards(name)
        status = "OK" if report["per_shard_match"] else "MISMATCH"
        print(
            f"[{status}] {name:15s} shards={report['num_shards']:3d} "
            f"declared={report['declared_total']:5d} observed={report['observed_total']:5d}"
        )
        for bad in report["mismatches"]:
            print(f"        {bad}")
