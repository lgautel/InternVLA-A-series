#!/usr/bin/env python3
"""Decide the stored orientation of the RLDS agentview frames, geometrically.

Earlier conclusions in b/d/libplus/ about image orientation came from pixel
diffs against a live simulator. There is no usable GL backend on this host, so
this script settles it from geometry instead, using only the archive itself:

  1. `observation.state[0:3]` is the end-effector position in the live arena
     world frame (robosuite `robot0_eef_pos`).
  2. LIBERO fixes the `agentview` camera pose per scene, in the problem class
     (e.g. LIBERO/libero/libero/envs/problems/libero_tabletop_manipulation.py:187),
     with no `fovy` override, so MuJoCo's default 45 deg applies.
  3. Projecting (1) through (2) predicts where the gripper must appear.
  4. The gripper is the dominant moving object early in an episode, so the
     motion-energy centroid of consecutive frames locates it observationally.

MuJoCo's render buffer has row 0 at the bottom. Scoring all four dihedral
variants of that buffer against the projection tells us which one the JPEGs
actually store.

Runs on the base python3 (numpy + PIL); no mujoco, no GL.
"""

from __future__ import annotations

import io
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from rlds_reader import SUBSETS, iter_episodes

HERE = Path(__file__).parent
OUT_JSON = HERE / "orientation_report.json"
OUT_NPZ = HERE / "orientation_overlay.npz"

IMG_HW = 256
DEFAULT_FOVY_DEG = 45.0  # MuJoCo default; LIBERO never overrides it

# agentview pose per scene, wxyz world quaternion, from the LIBERO problem
# classes' `_setup_camera`. LIBERO-plus uses the same values as its unperturbed
# baseline (LIBERO-plus/.../problems/libero_tabletop_manipulation.py:306).
CAMERAS: dict[str, dict] = {
    "table": {
        "pos": [0.6586131746834771, 0.0, 1.6103500240372423],
        "quat": [0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349],
        "source": "problems/libero_tabletop_manipulation.py:187",
    },
    "kitchen_table": {
        "pos": [0.6586131746834771, 0.0, 1.6103500240372423],
        "quat": [0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349],
        "source": "problems/libero_kitchen_tabletop_manipulation.py:190",
    },
    "study_table": {
        "pos": [0.4586131746834771, 0.0, 1.6103500240372423],
        "quat": [0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349],
        "source": "problems/libero_study_tabletop_manipulation.py:193",
    },
    "living_room_table": {
        "pos": [0.6065773716836134, 0.0, 0.96],
        "quat": [0.6182166934013367, 0.3432307541370392, 0.3432314395904541, 0.6182177066802979],
        "source": "problems/libero_living_room_tabletop_manipulation.py:190",
    },
    "floor": {
        "pos": [0.8965773716836134, 5.216182733499864e-07, 0.65],
        "quat": [0.6182166934013367, 0.3432307541370392, 0.3432314395904541, 0.6182177066802979],
        "source": "problems/libero_floor_manipulation.py:184",
    },
}

# Dihedral variants of the MuJoCo render buffer.
VARIANTS = {
    "identity (MuJoCo raw, row 0 = bottom)": (False, False),
    "vflip (upright, row 0 = top)": (True, False),
    "hflip": (False, True),
    "rot180": (True, True),
}


def quat_wxyz_to_mat(q) -> np.ndarray:
    w, x, y, z = q
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def project(points_world: np.ndarray, cam: dict, hw: int = IMG_HW) -> np.ndarray:
    """World points -> (row, col) in the raw MuJoCo buffer (row 0 = bottom).

    MuJoCo cameras look down -z with +x right and +y up in the image plane.
    """
    rot = quat_wxyz_to_mat(cam["quat"])
    rel = np.asarray(points_world, dtype=np.float64) - np.asarray(cam["pos"], dtype=np.float64)
    p_cam = rel @ rot  # == rot.T @ rel per row
    depth = -p_cam[:, 2]
    focal = (hw / 2.0) / np.tan(np.radians(DEFAULT_FOVY_DEG) / 2.0)
    col = hw / 2.0 + focal * p_cam[:, 0] / depth
    row = hw / 2.0 + focal * p_cam[:, 1] / depth
    return np.stack([row, col], axis=1)


def apply_variant(rowcol: np.ndarray, flip_row: bool, flip_col: bool, hw: int = IMG_HW) -> np.ndarray:
    out = rowcol.copy()
    if flip_row:
        out[:, 0] = hw - 1 - out[:, 0]
    if flip_col:
        out[:, 1] = hw - 1 - out[:, 1]
    return out


def motion_centroids(frames: np.ndarray, stride: int) -> tuple[np.ndarray, np.ndarray]:
    """Energy-weighted centroid of |I[t+stride] - I[t]|, and its total energy."""
    diff = np.abs(frames[stride:].astype(np.float32) - frames[:-stride].astype(np.float32))
    energy = diff.sum(axis=3) if diff.ndim == 4 else diff
    rows = np.arange(energy.shape[1], dtype=np.float32)
    cols = np.arange(energy.shape[2], dtype=np.float32)
    centroids = np.empty((len(energy), 2), dtype=np.float32)
    totals = np.empty(len(energy), dtype=np.float32)
    for t, e in enumerate(energy):
        # keep only the strongest 2% of pixels: suppresses JPEG ringing and
        # leaves the dominant moving structure
        thresh = np.quantile(e, 0.98)
        masked = np.where(e >= thresh, e, 0.0)
        total = masked.sum()
        totals[t] = total
        if total <= 0:
            centroids[t] = np.nan
            continue
        centroids[t, 0] = (masked.sum(axis=1) * rows).sum() / total
        centroids[t, 1] = (masked.sum(axis=0) * cols).sum() / total
    return centroids, totals


def decode(ep, indices: list[int]) -> np.ndarray:
    return np.stack(
        [np.asarray(Image.open(io.BytesIO(ep.jpeg("image", t))).convert("RGB")) for t in indices]
    )


def main(episodes_per_suite: int = 12, stride: int = 4) -> int:
    from panda_fk import arena_of

    scores: dict[str, list[float]] = defaultdict(list)
    per_arena: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    # Correlating the trajectory shape is the primary test: the four variants
    # differ only by negating the row and/or column series, so the sign of each
    # correlation reads off the answer directly and is immune to the constant
    # offset between a point projection and the centroid of an extended arm.
    corr_row: list[float] = []
    corr_col: list[float] = []
    overlay = None
    n_used = 0

    for subset in SUBSETS:
        for ep in iter_episodes(subset, max_episodes=episodes_per_suite):
            arena = arena_of(subset, ep.task_name)
            cam = CAMERAS[arena]
            T = ep.num_steps
            # reaching phase: the arm dominates the motion energy here
            window = list(range(0, max(stride + 1, int(0.45 * T))))
            frames = decode(ep, window)
            centroids, totals = motion_centroids(frames, stride)

            eef = ep.state[window, :3].astype(np.float64)
            midpoints = (eef[stride:] + eef[:-stride]) / 2.0
            predicted_raw = project(midpoints, cam)

            keep = np.isfinite(centroids[:, 0]) & (totals > np.quantile(totals, 0.25))
            if keep.sum() < 5:
                continue
            n_used += 1

            for name, (fr, fc) in VARIANTS.items():
                pred = apply_variant(predicted_raw, fr, fc)
                dist = np.linalg.norm(pred[keep] - centroids[keep], axis=1)
                scores[name].append(float(np.median(dist)))
                per_arena[arena][name].append(float(np.median(dist)))

            for axis, bucket in ((0, corr_row), (1, corr_col)):
                p, o = predicted_raw[keep, axis], centroids[keep, axis]
                if p.std() > 1e-6 and o.std() > 1e-6:
                    bucket.append(float(np.corrcoef(p, o)[0, 1]))

            if overlay is None:
                overlay = {
                    "frame": frames[len(frames) // 2],
                    "centroid": centroids[len(centroids) // 2],
                    "predictions": {
                        name: apply_variant(predicted_raw, fr, fc)[len(predicted_raw) // 2]
                        for name, (fr, fc) in VARIANTS.items()
                    },
                    "task": ep.task_name,
                    "arena": arena,
                }

    summary = {
        name: {
            "median_pixel_error": round(float(np.median(v)), 2),
            "mean_pixel_error": round(float(np.mean(v)), 2),
        }
        for name, v in scores.items()
    }
    ranked = sorted(summary.items(), key=lambda kv: kv[1]["median_pixel_error"])
    best, runner_up = ranked[0], ranked[1]

    row_corr, col_corr = float(np.median(corr_row)), float(np.median(corr_col))
    correlation_verdict = next(
        name
        for name, (fr, fc) in VARIANTS.items()
        if fr == (row_corr < 0) and fc == (col_corr < 0)
    )

    report = {
        "method": (
            "project observation.state[0:3] through the scene's fixed agentview camera "
            "and compare against the motion-energy centroid of the stored JPEGs"
        ),
        "image_hw": IMG_HW,
        "fovy_deg": DEFAULT_FOVY_DEG,
        "stride_frames": stride,
        "episodes_scored": n_used,
        "cameras": CAMERAS,
        "variant_scores_px": summary,
        "trajectory_correlation": {
            "note": (
                "correlation of the identity projection against the observed centroid; "
                "a negative sign means that axis is stored flipped"
            ),
            "row_median": round(row_corr, 4),
            "col_median": round(col_corr, 4),
            "row_negative_fraction": round(float(np.mean(np.asarray(corr_row) < 0)), 4),
            "col_negative_fraction": round(float(np.mean(np.asarray(corr_col) < 0)), 4),
            "verdict": correlation_verdict,
        },
        "verdict": {
            "stored_as": best[0],
            "agrees_with_correlation_test": correlation_verdict == best[0],
            "median_pixel_error": best[1]["median_pixel_error"],
            "runner_up": runner_up[0],
            "runner_up_error": runner_up[1]["median_pixel_error"],
            "margin_ratio": round(
                runner_up[1]["median_pixel_error"] / max(best[1]["median_pixel_error"], 1e-6), 2
            ),
            "residual_error_note": (
                "the residual tens of pixels is expected: a point projection is compared "
                "against the centroid of an extended moving arm, which biases the estimate "
                "but not the sign of the correlation"
            ),
        },
        "per_arena_median_px": {
            arena: {name: round(float(np.median(v)), 2) for name, v in d.items()}
            for arena, d in per_arena.items()
        },
    }
    OUT_JSON.write_text(json.dumps(report, indent=2))

    if overlay is not None:
        # numeric arrays only: the figure script runs under a different numpy
        # major version and cannot unpickle object arrays
        report["overlay"] = {
            "variant_order": list(overlay["predictions"]),
            "task": overlay["task"],
            "arena": overlay["arena"],
        }
        OUT_JSON.write_text(json.dumps(report, indent=2))
        np.savez_compressed(
            OUT_NPZ,
            frame=overlay["frame"],
            centroid=overlay["centroid"],
            predictions=np.stack(list(overlay["predictions"].values())),
        )

    print(f"episodes scored: {n_used}")
    for name, s in ranked:
        print(f"  {name:38s} median {s['median_pixel_error']:7.2f} px   mean {s['mean_pixel_error']:7.2f} px")
    print(f"\ntrajectory correlation vs identity projection: row={row_corr:+.4f} col={col_corr:+.4f}")
    print(f"  -> correlation test says: {correlation_verdict}")
    print(f"\nVERDICT: RLDS agentview frames are stored as '{best[0]}'")
    print(f"         {report['verdict']['margin_ratio']}x better than the runner-up")
    print(f"         correlation test agrees: {correlation_verdict == best[0]}")
    print(f"wrote {OUT_JSON}")
    return 0 if correlation_verdict == best[0] else 1


if __name__ == "__main__":
    raise SystemExit(main())
