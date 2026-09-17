"""Reusable train/eval checks that are not image-orientation (see orientation_contract.py).

Used by:
  - evaluation/LIBERO2/test_preflight_f1f2.py  (A23–A26, B5', B8, T8, T9)
  - tests/test_keypoint_utils.py

Checks:
  U8  keypoint history excludes the current frame at request time
  U9  std evaluate_policy does not import libero (parent EGL)
  T8  parquet gripper sign vs libero_native
  T9  fps label vs timestamps / episode length (skip if no HDF5)
"""
from __future__ import annotations

import ast
import os
import re
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = Path("/B/Dta/opvla_libero_merged_kpt")
EEF_CANDIDATES = ("gripper0_eef", "gripper0_right_eef")


# ---------------------------------------------------------------------------
# U8: history schedule (no MuJoCo)
# ---------------------------------------------------------------------------

def simulate_eval_history_loop(
    num_wait: int = 10,
    num_action: int = 5,
) -> list[tuple[list[int], int]]:
    """Replay the evaluate_task push-before-step schedule with integer pose ids.

    Each env state has a unique id. ``push`` records the current id as *past*;
    ``env.step`` advances the id. A "request" must see history that does **not**
    contain the current id (training ``his_kpts`` = offsets ``[-H, ..., -1]``).

    Returns a list of ``(history_ids, current_id)`` at each policy request.
    """
    history: list[int] = []
    current = 0
    requests: list[tuple[list[int], int]] = []
    for t in range(num_wait + num_action):
        if t < num_wait:
            history.append(current)
            current += 1
            continue
        requests.append((list(history), current))
        history.append(current)
        current += 1
    return requests


def history_excludes_current(requests: list[tuple[list[int], int]] | None = None) -> bool:
    if requests is None:
        requests = simulate_eval_history_loop()
    if not requests:
        return False
    return all(current not in hist for hist, current in requests)


def first_request_his_len(num_wait: int = 10) -> int:
    reqs = simulate_eval_history_loop(num_wait=num_wait, num_action=1)
    return len(reqs[0][0]) if reqs else 0


# ---------------------------------------------------------------------------
# Static source contracts
# ---------------------------------------------------------------------------

def _fn_node(src: str, name: str) -> ast.AST | None:
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def function_imports_libero(src: str, fn_name: str) -> bool:
    node = _fn_node(src, fn_name)
    if node is None:
        return False
    for n in ast.walk(node):
        if isinstance(n, ast.ImportFrom) and n.module and n.module.startswith("libero"):
            return True
        if isinstance(n, ast.Import):
            for alias in n.names:
                if alias.name == "libero" or alias.name.startswith("libero."):
                    return True
    return False


def evaluate_task_pushes_before_env_step(src: str) -> tuple[bool, str]:
    """True when wait and action loops call push_keypoint before env.step."""
    fn = _fn_node(src, "evaluate_task")
    if fn is None:
        return False, "evaluate_task not found"
    try:
        body = ast.get_source_segment(src, fn) or ""
    except Exception:
        body = src
    wait_ok = False
    action_ok = False
    lines = body.splitlines()
    for i, line in enumerate(lines):
        if "num_steps_wait" in line and i + 3 < len(lines):
            window = "\n".join(lines[i : i + 8])
            push_i = window.find("push_keypoint")
            dummy_i = window.find("LIBERO_DUMMY_ACTION")
            wait_ok = 0 <= push_i < dummy_i
        if "client.step(" in line:
            window = "\n".join(lines[i : i + 10])
            push_i = window.find("push_keypoint")
            step_i = window.find("env.step(action")
            action_ok = 0 <= push_i < step_i
    if wait_ok and action_ok:
        return True, ""
    missing = []
    if not wait_ok:
        missing.append("wait: push before dummy step")
    if not action_ok:
        missing.append("action: push before env.step(action)")
    return False, "; ".join(missing)


def extract_eef_candidates(src: str) -> tuple[str, ...] | None:
    m = re.search(
        r"EEF_BODY_CANDIDATES(?:\s*:\s*[^=]+)?\s*=\s*\(([^)]*)\)",
        src,
        re.DOTALL,
    )
    if not m:
        return None
    names = re.findall(r'"([^"]+)"', m.group(1))
    return tuple(names) if names else None


# ---------------------------------------------------------------------------
# T8 gripper sign
# ---------------------------------------------------------------------------

def gripper_opening(state):
    """Left-finger opening \(g_L\) = ``observation.state[..., 6]`` (ANALYSIS.md)."""
    import numpy as np

    arr = np.asarray(state, dtype=np.float64)
    if arr.ndim == 1:
        return arr[6:7]
    return arr[..., 6]


def libero_native_corr_ok(corr: float) -> bool:
    """libero_native: action[6] > 0 closes → corr(action[6], next g_L) < 0."""
    return bool(corr == corr and corr < 0.0)  # corr==corr rejects NaN


def t8_gripper_sign_from_parquet(
    dataset_root: str | Path | None = None,
    max_episodes: int = 40,
) -> dict[str, Any]:
    root = Path(dataset_root or os.environ.get("LIBERO_KPT_DATASET", DEFAULT_DATASET_ROOT))
    parquet = root / "data" / "chunk-000" / "file-000.parquet"
    out: dict[str, Any] = {
        "skipped": False,
        "passed": False,
        "parquet": str(parquet),
        "corr_next": None,
        "n_episodes": 0,
        "detail": "",
    }
    if not parquet.exists():
        out["skipped"] = True
        out["detail"] = f"missing {parquet}"
        return out
    try:
        import numpy as np
        import pyarrow.parquet as pq
    except ImportError:
        try:
            import numpy as np
            import pandas as pd
        except ImportError:
            out["skipped"] = True
            out["detail"] = "need pyarrow or pandas to read parquet"
            return out
        df = pd.read_parquet(parquet, columns=["action", "observation.state", "episode_index"])
        act = np.stack(df["action"].to_list()).astype(np.float64)
        state = np.stack(df["observation.state"].to_list()).astype(np.float64)
        ep = np.asarray(df["episode_index"].to_list()).reshape(-1)
    else:
        table = pq.read_table(parquet, columns=["action", "observation.state", "episode_index"])
        act = np.stack(table.column("action").to_pylist()).astype(np.float64)
        state = np.stack(table.column("observation.state").to_pylist()).astype(np.float64)
        ep = np.asarray(table.column("episode_index").to_pylist()).reshape(-1)
    g = gripper_opening(state)
    corrs: list[float] = []
    for e in np.unique(ep)[:max_episodes]:
        m = ep == e
        a = act[m, 6]
        gg = g[m]
        if a.size < 4:
            continue
        c = float(np.corrcoef(a[:-1], gg[1:])[0, 1])
        if np.isfinite(c):
            corrs.append(c)
    if not corrs:
        out["skipped"] = True
        out["detail"] = "no episodes long enough"
        return out
    corr = float(np.mean(corrs))
    out["corr_next"] = corr
    out["n_episodes"] = len(corrs)
    out["passed"] = libero_native_corr_ok(corr)
    out["detail"] = (
        f"mean corr(action[6], next g_L)={corr:.4f} over {len(corrs)} episodes; "
        "libero_native expects < 0"
    )
    return out


# ---------------------------------------------------------------------------
# T9 fps / episode length
# ---------------------------------------------------------------------------

def t9_fps_contract(
    dataset_root: str | Path | None = None,
    hdf5_root: str | Path | None = None,
) -> dict[str, Any]:
    """Check fps label vs timestamps; optionally vs original HDF5 lengths.

    Pass if:
      - median timestamp Δt ≈ 1/fps
      - mean episode length is not ~half of typical 20 Hz LIBERO demos
        (that would mean true 2× subsampling). Typical no_noops lengths are 80–250.
      - if an HDF5 demo file is found, mean length ratio ∈ [0.7, 1.4]
    """
    root = Path(dataset_root or os.environ.get("LIBERO_KPT_DATASET", DEFAULT_DATASET_ROOT))
    out: dict[str, Any] = {
        "skipped": False,
        "passed": False,
        "fps": None,
        "median_dt": None,
        "mean_ep_len": None,
        "hdf5_ratio": None,
        "detail": "",
    }
    info_path = root / "meta" / "info.json"
    parquet = root / "data" / "chunk-000" / "file-000.parquet"
    if not info_path.exists() or not parquet.exists():
        out["skipped"] = True
        out["detail"] = "dataset info/parquet missing"
        return out
    try:
        import json

        import numpy as np
        import pyarrow.parquet as pq
    except ImportError:
        try:
            import json

            import numpy as np
            import pandas as pd
        except ImportError:
            out["skipped"] = True
            out["detail"] = "need pyarrow or pandas to read parquet"
            return out
        info = json.loads(info_path.read_text())
        fps = float(info.get("fps") or 0)
        out["fps"] = fps
        df = pd.read_parquet(parquet, columns=["timestamp", "episode_index"])
        ts = np.asarray(df["timestamp"].to_list(), dtype=np.float64).reshape(-1)
        ep = np.asarray(df["episode_index"].to_list()).reshape(-1)
    else:
        info = json.loads(info_path.read_text())
        fps = float(info.get("fps") or 0)
        out["fps"] = fps
        table = pq.read_table(parquet, columns=["timestamp", "episode_index"])
        ts = np.asarray(table.column("timestamp").to_pylist(), dtype=np.float64).reshape(-1)
        ep = np.asarray(table.column("episode_index").to_pylist()).reshape(-1)
    dts = []
    lengths = []
    for e in np.unique(ep)[:80]:
        m = ep == e
        tse = ts[m]
        lengths.append(int(m.sum()))
        if tse.size >= 2:
            dts.append(np.median(np.diff(tse)))
    median_dt = float(np.median(dts)) if dts else float("nan")
    mean_len = float(np.mean(lengths)) if lengths else float("nan")
    out["median_dt"] = median_dt
    out["mean_ep_len"] = mean_len

    dt_ok = bool(np.isfinite(median_dt) and fps > 0 and abs(median_dt - 1.0 / fps) < 0.02)
    # 2× downsample of ~120-step 20 Hz demos would yield ~60 frames.
    not_half_rate = bool(np.isfinite(mean_len) and mean_len >= 80.0)

    hdf5_ratio = None
    hdf5_path = Path(hdf5_root) if hdf5_root else None
    if hdf5_path is None:
        env_p = os.environ.get("LIBERO_HDF5_DEMO")
        if env_p:
            hdf5_path = Path(env_p)
    if hdf5_path is not None and hdf5_path.exists():
        try:
            import h5py
            import numpy as np

            with h5py.File(hdf5_path, "r") as f:
                data = f["data"] if "data" in f else f
                demo_keys = [k for k in data.keys() if str(k).startswith("demo_")]
                hdf_lens = []
                for k in demo_keys[:40]:
                    grp = data[k]
                    if "actions" in grp:
                        hdf_lens.append(int(np.asarray(grp["actions"]).shape[0]))
                if hdf_lens and mean_len:
                    hdf5_ratio = mean_len / float(np.mean(hdf_lens))
        except Exception as exc:  # pragma: no cover - optional
            out["detail"] = f"hdf5 read failed: {exc}"
    out["hdf5_ratio"] = hdf5_ratio
    hdf_ok = hdf5_ratio is None or (0.7 <= float(hdf5_ratio) <= 1.4)
    out["passed"] = dt_ok and not_half_rate and hdf_ok
    bits = [
        f"fps={fps}",
        f"median_dt={median_dt:.4f}",
        f"mean_ep_len={mean_len:.1f}",
        f"dt_ok={dt_ok}",
        f"not_half_rate={not_half_rate}",
    ]
    if hdf5_ratio is not None:
        bits.append(f"hdf5_ratio={hdf5_ratio:.3f}")
    else:
        bits.append("hdf5 skipped")
    out["detail"] = ", ".join(bits)
    return out


def load_keypoints_meta_eef(dataset_root: str | Path | None = None) -> str | None:
    root = Path(dataset_root or DEFAULT_DATASET_ROOT)
    meta = root / "meta" / "keypoints_meta.json"
    if not meta.exists():
        return None
    import json

    bodies = json.loads(meta.read_text()).get("keypoint_bodies") or []
    return str(bodies[-1]) if bodies else None
