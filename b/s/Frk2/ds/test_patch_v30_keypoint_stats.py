"""Unit tests + mini-dataset end-to-end tests for keypoint stats patching.

Default (no flags): synthetic tests only, no writes to the real v30 dataset.

    source /B/VENV/itnvla15rbt20/bin/activate
    python b/s/Frk2/ds/test_patch_v30_keypoint_stats.py

After patching the real dataset, also run live acceptance:

    python b/s/Frk2/ds/test_patch_v30_keypoint_stats.py --live \
        --dataset /home/a26113/b/Dta/plug_into_socket_franka3_15hz_lerobot_4d_v30
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import traceback
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

_DS_DIR = Path(__file__).resolve().parent
if str(_DS_DIR) not in sys.path:
    sys.path.insert(0, str(_DS_DIR))

from kpt_stats import (  # noqa: E402
    DEFAULT_DATASET,
    KEYPOINT_COLUMN,
    NUM_KEYPOINTS,
    TOTAL_DIM,
    KeypointStatsError,
    compute_keypoint_stats,
    compute_vector_stats,
    file_sha256,
    insert_after,
    patch_dataset,
    stack_list_column,
    verify_dataset,
)


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=4) + "\n")


def make_mini_v30(root: Path, *, n_ep: int = 3, dim: int = TOTAL_DIM, seed: int = 0) -> dict:
    """Create a tiny v3.0-like dataset with keypoint_3d but missing kpt stats."""
    rng = np.random.default_rng(seed)
    lengths = [5, 7, 4][:n_ep]
    rows = []
    ep_ids = []
    kpts = []
    for ep_i, length in enumerate(lengths):
        block = rng.normal(size=(length, dim)).astype(np.float32)
        # hemisphere-ish: force last-of-7 qw >= 0 for each keypoint
        block = block.reshape(length, NUM_KEYPOINTS, 7)
        q = block[:, :, 3:7]
        q = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12)
        neg = q[:, :, 3] < 0
        q[neg] *= -1
        block[:, :, 3:7] = q
        block[:, :, :3] *= 0.3
        block = block.reshape(length, dim).astype(np.float32)
        for t in range(length):
            rows.append(
                {
                    KEYPOINT_COLUMN: block[t].tolist(),
                    "observation.state": (rng.normal(size=15)).astype(np.float32).tolist(),
                    "action": (rng.normal(size=8)).astype(np.float32).tolist(),
                    "episode_index": ep_i,
                    "index": len(rows),
                    "timestamp": float(t) / 15.0,
                }
            )
            ep_ids.append(ep_i)
            kpts.append(block[t])
    kpts_arr = np.stack(kpts).astype(np.float64)

    data_dir = root / "data" / "chunk-000"
    data_dir.mkdir(parents=True)
    data_table = pa.table(
        {
            KEYPOINT_COLUMN: pa.array([r[KEYPOINT_COLUMN] for r in rows], type=pa.list_(pa.float32())),
            "observation.state": pa.array(
                [r["observation.state"] for r in rows], type=pa.list_(pa.float32())
            ),
            "action": pa.array([r["action"] for r in rows], type=pa.list_(pa.float32())),
            "episode_index": pa.array([r["episode_index"] for r in rows], type=pa.int64()),
            "index": pa.array([r["index"] for r in rows], type=pa.int64()),
            "timestamp": pa.array([r["timestamp"] for r in rows], type=pa.float32()),
        }
    )
    pq.write_table(data_table, data_dir / "file-000.parquet")

    n_frames = len(rows)
    info = {
        "codebase_version": "v3.0",
        "robot_type": "franka3",
        "total_episodes": n_ep,
        "total_frames": n_frames,
        "fps": 15,
        "features": {
            "observation.images.global": {"dtype": "video", "shape": [480, 640, 3]},
            "observation.state": {"dtype": "float32", "shape": [15]},
            "action": {"dtype": "float32", "shape": [8]},
            KEYPOINT_COLUMN: {"dtype": "float32", "shape": [dim]},
            "timestamp": {"dtype": "float32", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "index": {"dtype": "int64", "shape": [1]},
        },
    }
    _write_json(root / "meta" / "info.json", info)

    # stats.json WITHOUT keypoint_3d (the bug we patch)
    state_arr = np.stack([r["observation.state"] for r in rows])
    act_arr = np.stack([r["action"] for r in rows])
    ts_arr = np.array([r["timestamp"] for r in rows], dtype=np.float64)
    stats = {
        "timestamp": compute_vector_stats(ts_arr),
        "action": compute_vector_stats(act_arr),
        "observation.state": compute_vector_stats(state_arr),
        "observation.images.global": {
            "min": [[[0.0]], [[0.0]], [[0.0]]],
            "max": [[[1.0]], [[1.0]], [[1.0]]],
            "mean": [[[0.4]], [[0.4]], [[0.4]]],
            "std": [[[0.1]], [[0.1]], [[0.1]]],
            "count": [100],
        },
    }
    _write_json(root / "meta" / "stats.json", stats)

    jsonl_lines = []
    ep_records = []
    offset = 0
    for ep_i, length in enumerate(lengths):
        sl = slice(offset, offset + length)
        ep_stat = {
            "observation.state": compute_vector_stats(state_arr[sl]),
            "action": compute_vector_stats(act_arr[sl]),
            "timestamp": compute_vector_stats(ts_arr[sl]),
            "observation.images.global": {
                "min": [[[0.01]], [[0.02]], [[0.0]]],
                "max": [[[0.9]], [[0.8]], [[0.7]]],
                "mean": [[[0.41]], [[0.42]], [[0.43]]],
                "std": [[[0.11]], [[0.12]], [[0.13]]],
                "count": [100],
            },
        }
        jsonl_lines.append(json.dumps({"episode_index": ep_i, "stats": ep_stat}))
        rec = {
            "episode_index": ep_i,
            "length": length,
            "dataset_from_index": offset,
            "dataset_to_index": offset + length,
            "stats/observation.images.global/min": [[[0.01]], [[0.02]], [[0.0]]],
            "stats/action/min": ep_stat["action"]["min"],
            "stats/action/max": ep_stat["action"]["max"],
            "stats/action/mean": ep_stat["action"]["mean"],
            "stats/action/std": ep_stat["action"]["std"],
            "stats/action/count": ep_stat["action"]["count"],
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
        }
        ep_records.append(rec)
        offset += length
    (root / "meta" / "episodes_stats.jsonl").write_text("\n".join(jsonl_lines) + "\n")

    nested_list = pa.list_(pa.list_(pa.list_(pa.float64())))
    hf_features = {
        "episode_index": {"dtype": "int64", "_type": "Value"},
        "length": {"dtype": "int64", "_type": "Value"},
        "stats/observation.images.global/min": {
            "feature": {
                "feature": {"feature": {"dtype": "float64", "_type": "Value"}, "_type": "List"},
                "_type": "List",
            },
            "_type": "List",
        },
        "stats/action/min": {"feature": {"dtype": "float64", "_type": "Value"}, "_type": "List"},
        "meta/episodes/chunk_index": {"dtype": "int64", "_type": "Value"},
        "meta/episodes/file_index": {"dtype": "int64", "_type": "Value"},
    }
    hf_meta = {b"huggingface": json.dumps({"info": {"features": hf_features}}, separators=(",", ":")).encode()}
    ep_table = pa.table(
        {
            "episode_index": pa.array([r["episode_index"] for r in ep_records], type=pa.int64()),
            "length": pa.array([r["length"] for r in ep_records], type=pa.int64()),
            "dataset_from_index": pa.array([r["dataset_from_index"] for r in ep_records], type=pa.int64()),
            "dataset_to_index": pa.array([r["dataset_to_index"] for r in ep_records], type=pa.int64()),
            "stats/observation.images.global/min": pa.array(
                [r["stats/observation.images.global/min"] for r in ep_records],
                type=nested_list,
            ),
            "stats/action/min": pa.array(
                [r["stats/action/min"] for r in ep_records], type=pa.list_(pa.float64())
            ),
            "stats/action/max": pa.array(
                [r["stats/action/max"] for r in ep_records], type=pa.list_(pa.float64())
            ),
            "stats/action/mean": pa.array(
                [r["stats/action/mean"] for r in ep_records], type=pa.list_(pa.float64())
            ),
            "stats/action/std": pa.array(
                [r["stats/action/std"] for r in ep_records], type=pa.list_(pa.float64())
            ),
            "stats/action/count": pa.array(
                [r["stats/action/count"] for r in ep_records], type=pa.list_(pa.int64())
            ),
            "meta/episodes/chunk_index": pa.array([0] * n_ep, type=pa.int64()),
            "meta/episodes/file_index": pa.array([0] * n_ep, type=pa.int64()),
        },
        metadata=hf_meta,
    )
    ep_dir = root / "meta" / "episodes" / "chunk-000"
    ep_dir.mkdir(parents=True)
    pq.write_table(ep_table, ep_dir / "file-000.parquet")

    return {"kpts": kpts_arr, "lengths": lengths, "n_frames": n_frames, "n_ep": n_ep}


class TestComputeVectorStats(unittest.TestCase):
    def test_known_array(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
        s = compute_vector_stats(arr)
        self.assertEqual(s["count"], [3])
        np.testing.assert_allclose(s["min"], [1.0, 2.0])
        np.testing.assert_allclose(s["max"], [5.0, 6.0])
        np.testing.assert_allclose(s["mean"], [3.0, 4.0])
        # population std of [1,3,5] = sqrt(((1-3)^2+(3-3)^2+(5-3)^2)/3) = sqrt(8/3)
        np.testing.assert_allclose(s["std"], [np.sqrt(8.0 / 3.0), np.sqrt(8.0 / 3.0)])

    def test_constant_column_std_zero(self):
        arr = np.array([[0.0, 1.0], [0.0, 3.0]], dtype=np.float64)
        s = compute_vector_stats(arr)
        self.assertEqual(s["std"][0], 0.0)
        self.assertGreater(s["std"][1], 0.0)

    def test_1d_promoted(self):
        s = compute_vector_stats(np.array([2.0, 4.0, 6.0]))
        self.assertEqual(len(s["mean"]), 1)
        np.testing.assert_allclose(s["mean"], [4.0])

    def test_empty_raises(self):
        with self.assertRaises(KeypointStatsError):
            compute_vector_stats(np.zeros((0, 56)))


class TestInsertAfter(unittest.TestCase):
    def test_insert_middle(self):
        d = {"a": 1, "b": 2, "c": 3}
        out = insert_after(d, "b", "x", 9)
        self.assertEqual(list(out.keys()), ["a", "b", "x", "c"])
        self.assertEqual(out["x"], 9)

    def test_replace_existing_keeps_others(self):
        d = {"a": 1, "x": 0, "c": 3}
        out = insert_after(d, "a", "x", 9)
        self.assertEqual(out["x"], 9)
        self.assertEqual(out["c"], 3)


class TestStackListColumn(unittest.TestCase):
    def test_lists_and_arrays(self):
        vals = [[1, 2], np.array([3, 4], dtype=np.float32)]
        arr = stack_list_column(vals)
        self.assertEqual(arr.shape, (2, 2))
        self.assertEqual(arr.dtype, np.float64)


class TestMiniDatasetPatch(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="kpt_stats_test_"))
        self.meta = make_mini_v30(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_patch_then_verify_all_pass(self):
        data_pq = next((self.tmp / "data").rglob("*.parquet"))
        sha_before = file_sha256(data_pq)
        result = patch_dataset(self.tmp, backup=True, dry_run=False)
        self.assertFalse(result.dry_run)
        self.assertEqual(result.n_frames, self.meta["n_frames"])
        self.assertEqual(result.n_episodes, self.meta["n_ep"])
        self.assertEqual(file_sha256(data_pq), sha_before)
        self.assertIsNotNone(result.backup_dir)
        self.assertTrue((result.backup_dir / "stats.json").exists())

        checks = verify_dataset(self.tmp)
        failed = [c for c in checks if not c.ok]
        self.assertFalse(failed, msg="\n".join(f"{c.name}: {c.detail}" for c in failed))

        stats = json.loads((self.tmp / "meta" / "stats.json").read_text())
        self.assertIn(KEYPOINT_COLUMN, stats)
        rec, _ = compute_keypoint_stats(self.meta["kpts"], np.repeat(
            np.arange(self.meta["n_ep"]), self.meta["lengths"]
        ))
        np.testing.assert_allclose(stats[KEYPOINT_COLUMN]["mean"], rec["mean"], rtol=1e-9)

        # pre-existing action stats untouched numerically
        np.testing.assert_allclose(
            stats["action"]["mean"],
            json.loads((result.backup_dir / "stats.json").read_text())["action"]["mean"],
        )

    def test_nested_image_stats_type_preserved(self):
        patch_dataset(self.tmp, backup=False)
        et = pq.read_table(self.tmp / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
        t = et.schema.field("stats/observation.images.global/min").type
        self.assertEqual(str(t), "list<element: list<element: list<element: double>>>")
        # column still sits before meta/episodes/chunk_index
        names = et.column_names
        self.assertLess(
            names.index(f"stats/{KEYPOINT_COLUMN}/mean"),
            names.index("meta/episodes/chunk_index"),
        )

    def test_idempotent_second_patch(self):
        patch_dataset(self.tmp, backup=False)
        first = json.loads((self.tmp / "meta" / "stats.json").read_text())[KEYPOINT_COLUMN]
        patch_dataset(self.tmp, backup=False)
        second = json.loads((self.tmp / "meta" / "stats.json").read_text())[KEYPOINT_COLUMN]
        np.testing.assert_allclose(first["mean"], second["mean"])
        checks = verify_dataset(self.tmp)
        self.assertTrue(all(c.ok for c in checks), [c for c in checks if not c.ok])

    def test_skip_if_present_does_not_rewrite(self):
        patch_dataset(self.tmp, backup=False)
        mtime = (self.tmp / "meta" / "stats.json").stat().st_mtime
        result = patch_dataset(self.tmp, backup=False, skip_if_present=True)
        self.assertTrue(result.dry_run)
        self.assertEqual((self.tmp / "meta" / "stats.json").stat().st_mtime, mtime)

    def test_dry_run_writes_nothing(self):
        stats_before = (self.tmp / "meta" / "stats.json").read_text()
        jsonl_before = (self.tmp / "meta" / "episodes_stats.jsonl").read_text()
        result = patch_dataset(self.tmp, dry_run=True, backup=False)
        self.assertTrue(result.dry_run)
        self.assertEqual((self.tmp / "meta" / "stats.json").read_text(), stats_before)
        self.assertEqual((self.tmp / "meta" / "episodes_stats.jsonl").read_text(), jsonl_before)
        self.assertNotIn(KEYPOINT_COLUMN, json.loads(stats_before))

    def test_wrong_shape_raises(self):
        info_path = self.tmp / "meta" / "info.json"
        info = json.loads(info_path.read_text())
        info["features"][KEYPOINT_COLUMN]["shape"] = [55]
        _write_json(info_path, info)
        with self.assertRaises(KeypointStatsError):
            patch_dataset(self.tmp, backup=False)

    def test_jsonl_episode_coverage(self):
        patch_dataset(self.tmp, backup=False)
        n = 0
        with open(self.tmp / "meta" / "episodes_stats.jsonl") as f:
            for line in f:
                obj = json.loads(line)
                n += 1
                k = obj["stats"][KEYPOINT_COLUMN]
                self.assertEqual(len(k["mean"]), TOTAL_DIM)
                self.assertEqual(k["count"][0], self.meta["lengths"][obj["episode_index"]])
        self.assertEqual(n, self.meta["n_ep"])

    def test_backup_does_not_contain_kpt_in_original_stats(self):
        result = patch_dataset(self.tmp, backup=True)
        bak = json.loads((result.backup_dir / "stats.json").read_text())
        self.assertNotIn(KEYPOINT_COLUMN, bak)


class TestVerifyFailsWhenMissing(unittest.TestCase):
    def test_unpatched_mini_fails_stats_check(self):
        tmp = Path(tempfile.mkdtemp(prefix="kpt_stats_unpatched_"))
        try:
            make_mini_v30(tmp)
            checks = verify_dataset(tmp)
            by_name = {c.name: c for c in checks}
            self.assertFalse(by_name["stats.json contains observation.keypoint_3d"].ok)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestLiveDataset(unittest.TestCase):
    """Optional: only collected when --live is passed (see load_tests)."""

    dataset: Path = DEFAULT_DATASET

    def test_live_acceptance(self):
        if LIVE_DATASET is None:
            self.skipTest("pass --live to verify the real v30 dataset")
        self.assertTrue(self.dataset.exists(), f"missing {self.dataset}")
        checks = verify_dataset(self.dataset)
        failed = [c for c in checks if not c.ok]
        for c in checks:
            print(f"    [{'PASS' if c.ok else 'FAIL'}] {c.name}: {c.detail}")
        self.assertFalse(failed, msg="\n".join(f"{c.name}: {c.detail}" for c in failed))


LIVE_DATASET: Path | None = None


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    for case in (
        TestComputeVectorStats,
        TestInsertAfter,
        TestStackListColumn,
        TestMiniDatasetPatch,
        TestVerifyFailsWhenMissing,
    ):
        suite.addTests(loader.loadTestsFromTestCase(case))
    if LIVE_DATASET is not None:
        TestLiveDataset.dataset = LIVE_DATASET
        suite.addTests(loader.loadTestsFromTestCase(TestLiveDataset))
    return suite


def main() -> int:
    global LIVE_DATASET
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Also verify the real v30 dataset")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("-v", "--verbose", action="store_true")
    args, rest = parser.parse_known_args()
    if args.live:
        LIVE_DATASET = args.dataset
    verbosity = 2 if args.verbose else 2
    runner = unittest.TextTestRunner(verbosity=verbosity)
    # Re-load so load_tests sees LIVE_DATASET
    loader = unittest.TestLoader()
    suite = load_tests(loader, None, None)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(2)
