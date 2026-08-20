"""Offline pinocchio IK+FK 3D keypoint generation for LIBERO LeRobot datasets (Franka, J=8).

LIBERO LeRobot exports store ``observation.state`` as an **EEF pose**
``[x, y, z, ax, ay, az, grip_l, grip_r]`` (OpenVLA / PI style), not joint angles.
This script therefore:

1. Fits a per-episode robot-base translation so that the world-frame EEF becomes
   reachable in the Panda URDF base frame (table / mount layout differs by suite).
2. Solves position-prioritized numerical IK (pinocchio) for the 7 arm joints.
3. Runs FK for ``panda_link1..7`` and pins the EEF keypoint to the dataset EEF
   (→ 8 keypoints, GeoPredict single-arm default).
4. Writes ``observation.keypoint_3d`` as flattened ``[24]`` float32 (world frame)
   and declares it in ``meta/info.json``.

Usage:
    conda activate internvla_a1_5
    python util_scripts/generate_libero_keypoints.py \\
        --source_root /home/a26160/DATA/LIBERO/libero \\
        --dest_root   /home/a26160/DATA/LIBERO/libero_kpt
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pinocchio as pin

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_URDF = (
    "/home/a26160/SRC/phantom/submodules/phantom-robosuite/robosuite/models/assets/"
    "bullet_data/panda_description/urdf/panda_arm_hand.urdf"
)

# GeoPredict / v3.1 RoboCasa naming (J=8).
KEYPOINT_LINKS: list[str] = [
    "robot0_link1",
    "robot0_link2",
    "robot0_link3",
    "robot0_link4",
    "robot0_link5",
    "robot0_link6",
    "robot0_link7",
    "gripper0_right_eef",
]
URDF_LINK_FRAMES: list[str] = [f"panda_link{i}" for i in range(1, 8)]
EE_FRAME = "panda_hand"
NUM_KEYPOINTS = 8
KEYPOINT_FEATURE_NAMES: list[str] = [
    f"{link}_{axis}" for link in KEYPOINT_LINKS for axis in ("x", "y", "z")
]

_Q_READY7 = np.array([0.0, -0.785398, 0.0, -2.356194, 0.0, 1.570796, 0.785398], dtype=np.float64)


class FrankaPinocchioKeypoints:
    """pinocchio FK/IK wrapper for LIBERO Franka EEF-state → 8 world-frame keypoints."""

    def __init__(self, urdf_path: str = DEFAULT_URDF):
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.ee_fid = self.model.getFrameId(EE_FRAME)
        self.link_fids = [self.model.getFrameId(n) for n in URDF_LINK_FRAMES]
        if any(fid >= self.model.nframes for fid in [self.ee_fid, *self.link_fids]):
            raise RuntimeError(f"Failed to resolve required frames in {urdf_path}")
        self.urdf_path = urdf_path

    def _neutral_q(self) -> np.ndarray:
        q = pin.neutral(self.model)
        q[:7] = _Q_READY7
        q[7:] = 0.0
        return q

    def ik_position(
        self,
        target_base: np.ndarray,
        q_init: np.ndarray | None = None,
        max_iters: int = 60,
        tol: float = 1e-3,
    ) -> tuple[np.ndarray, dict]:
        q = self._neutral_q() if q_init is None else q_init.copy()
        target = np.asarray(target_base, dtype=np.float64).reshape(3)
        info = {"success": False, "pos_err": np.inf, "iters": 0}
        for it in range(max_iters):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            err = target - self.data.oMf[self.ee_fid].translation
            pos_err = float(np.linalg.norm(err))
            info.update(iters=it + 1, pos_err=pos_err)
            if pos_err < tol:
                info["success"] = True
                break
            J = pin.computeFrameJacobian(
                self.model, self.data, q, self.ee_fid, pin.LOCAL_WORLD_ALIGNED
            )[:3, :7]
            dq = np.zeros(self.model.nv, dtype=np.float64)
            dq[:7] = np.linalg.lstsq(J, err, rcond=None)[0]
            step = float(np.linalg.norm(dq[:7]))
            if step > 0.5:
                dq[:7] *= 0.5 / step
            q = pin.integrate(self.model, q, dq)
            q[7:] = 0.0
        return q, info

    def fit_base(
        self,
        states: np.ndarray,
        subsample: int = 5,
        prior: np.ndarray | None = None,
    ) -> np.ndarray:
        """Fast coarse+refine search for a fixed base translation for one episode.

        Uses the first frame for the coarse grid (cheap), then re-scores a short
        subsample during local refinement. ``prior`` (e.g. previous episode base)
        is tried first and usually wins within a suite.
        """
        states = np.asarray(states, dtype=np.float64)
        p0 = states[0, :3]
        idx = np.linspace(0, len(states) - 1, num=min(subsample, len(states)), dtype=int)
        sample = states[idx, :3]

        def score_one(base: np.ndarray, p: np.ndarray, q0: np.ndarray | None = None) -> float:
            _, info = self.ik_position(p - base, q_init=q0, max_iters=35, tol=1e-3)
            return float(info["pos_err"])

        def score_sample(base: np.ndarray) -> float:
            q = self._neutral_q()
            errs = []
            for p in sample:
                q, info = self.ik_position(p - base, q_init=q, max_iters=35, tol=1e-3)
                errs.append(info["pos_err"])
            return float(np.mean(errs))

        best_base = np.array([-0.55, 0.0, 0.775], dtype=np.float64) if prior is None else np.asarray(prior, dtype=np.float64).copy()
        best_score = score_sample(best_base)
        if best_score < 2e-3:
            return best_base

        # Also try origin (some LIBERO exports look already base-relative).
        for cand in (
            np.zeros(3, dtype=np.float64),
            np.array([0.0, 0.0, 0.0], dtype=np.float64),
            np.array([-0.55, 0.0, 0.775], dtype=np.float64),
            np.array([-0.6, 0.0, 0.0], dtype=np.float64),
        ):
            sc = score_sample(cand)
            if sc < best_score:
                best_score, best_base = sc, cand.copy()
        if best_score < 2e-3:
            return best_base

        # Coarse grid on the first frame only. Cover both world-mounted (~z=0.8)
        # and already-base-relative (~z=0) LIBERO exports.
        for bz in np.linspace(0.0, 0.95, 10):
            for bx in np.linspace(-0.80, 0.20, 15):
                for by in np.linspace(-0.30, 0.30, 7):
                    base = np.array([bx, by, bz], dtype=np.float64)
                    nrm = float(np.linalg.norm(p0 - base))
                    if not (0.12 < nrm < 1.05):
                        continue
                    sc = score_one(base, p0)
                    if sc < best_score:
                        best_score, best_base = sc, base

        # Local refine scored on the subsample.
        center = best_base.copy()
        best_score = score_sample(best_base)
        for scale in (0.05, 0.02):
            improved = False
            for dx in (-scale, 0.0, scale):
                for dy in (-scale, 0.0, scale):
                    for dz in (-scale, 0.0, scale):
                        base = center + np.array([dx, dy, dz])
                        sc = score_sample(base)
                        if sc < best_score:
                            best_score, best_base = sc, base
                            improved = True
            center = best_base.copy()
            if best_score < 2e-3 and not improved:
                break
        return best_base

    def keypoints_episode(
        self, states: np.ndarray, base_prior: np.ndarray | None = None
    ) -> tuple[np.ndarray, dict]:
        """``states [T,8]`` → world-frame keypoints ``[T,8,3]`` + report dict."""
        states = np.asarray(states, dtype=np.float64)
        base = self.fit_base(states, prior=base_prior)
        t = states.shape[0]
        kpts = np.empty((t, NUM_KEYPOINTS, 3), dtype=np.float32)
        q = self._neutral_q()
        pos_errs: list[float] = []
        n_ok = 0
        for i in range(t):
            target = states[i, :3] - base
            q, info = self.ik_position(target, q_init=q, max_iters=60, tol=1e-3)
            if states.shape[1] >= 8:
                q[7] = float(np.clip(abs(states[i, 6]), 0.0, 0.04))
                q[8] = float(np.clip(abs(states[i, 7]), 0.0, 0.04))
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            for j, fid in enumerate(self.link_fids):
                kpts[i, j] = (self.data.oMf[fid].translation + base).astype(np.float32)
            kpts[i, 7] = states[i, :3].astype(np.float32)
            pos_errs.append(info["pos_err"])
            n_ok += int(info["success"])
        report = {
            "base": base.tolist(),
            "ik_success_rate": n_ok / max(t, 1),
            "mean_ik_pos_err": float(np.mean(pos_errs)),
            "max_ik_pos_err": float(np.max(pos_errs)),
        }
        return kpts, report


def _copy_dataset(source: Path, dest: Path, force: bool) -> None:
    """Copy only ``data/`` + ``meta/``; symlink ``videos/`` to the source.

    LIBERO suites are video-heavy (~95%+ of bytes). Full rsync on this filesystem
    is prohibitively slow; videos are read-only for keypoint generation, so a
    symlink is sufficient and keeps training loaders happy.
    """
    if dest.exists():
        if force:
            logger.info("Removing existing destination %s (--force)", dest)
            shutil.rmtree(dest)
        else:
            raise FileExistsError(
                f"Destination {dest} already exists. Pass --force to overwrite, or pick a new --dest."
            )
    dest.mkdir(parents=True, exist_ok=True)
    for sub in ("data", "meta"):
        src_sub = source / sub
        if not src_sub.exists():
            raise FileNotFoundError(f"Missing {src_sub}")
        logger.info("Copying %s -> %s ...", src_sub, dest / sub)
        subprocess.run(["rsync", "-a", f"{src_sub}/", f"{dest / sub}/"], check=True)
    videos_src = source / "videos"
    videos_dst = dest / "videos"
    if videos_src.exists():
        if videos_dst.exists() or videos_dst.is_symlink():
            videos_dst.unlink()
        videos_dst.symlink_to(videos_src.resolve())
        logger.info("Symlinked videos -> %s", videos_src.resolve())
    # Optional top-level files (e.g. README) if present.
    for extra in source.iterdir():
        if extra.name in {"data", "meta", "videos"}:
            continue
        if extra.is_file():
            shutil.copy2(extra, dest / extra.name)


def _add_keypoint_column(dest: Path, extractor: FrankaPinocchioKeypoints) -> dict:
    parquet_files = sorted((dest / "data").rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {dest / 'data'}")

    all_kpts: list[np.ndarray] = []
    ep_reports: list[dict] = []
    total_rows = 0
    t0 = time.perf_counter()
    base_prior: np.ndarray | None = None

    for ep_i, pq_path in enumerate(parquet_files):
        df = pd.read_parquet(pq_path)
        states = np.stack(df["observation.state"].to_numpy()).astype(np.float64)
        kpts, report = extractor.keypoints_episode(states, base_prior=base_prior)
        base_prior = np.asarray(report["base"], dtype=np.float64)
        df["observation.keypoint_3d"] = [row.reshape(-1) for row in kpts]
        df.to_parquet(pq_path)
        all_kpts.append(kpts)
        total_rows += len(df)
        report["episode"] = pq_path.stem
        ep_reports.append(report)
        if (ep_i + 1) % 20 == 0 or ep_i == 0 or ep_i + 1 == len(parquet_files):
            logger.info(
                "[%d/%d] %s rows=%d base=%s ik_ok=%.1f%% mean_err=%.4f (%.1fs)",
                ep_i + 1,
                len(parquet_files),
                pq_path.name,
                total_rows,
                np.round(report["base"], 3).tolist(),
                100.0 * report["ik_success_rate"],
                report["mean_ik_pos_err"],
                time.perf_counter() - t0,
            )

    all_kpts_cat = np.concatenate(all_kpts, axis=0)
    mean_ok = float(np.mean([r["ik_success_rate"] for r in ep_reports]))
    mean_err = float(np.mean([r["mean_ik_pos_err"] for r in ep_reports]))
    max_err = float(np.max([r["max_ik_pos_err"] for r in ep_reports]))
    return {
        "total_rows": total_rows,
        "total_episodes": len(parquet_files),
        "ik_success_rate": mean_ok,
        "mean_ik_pos_err": mean_err,
        "max_ik_pos_err": max_err,
        "min_xyz": all_kpts_cat.reshape(-1, 3).min(axis=0).tolist(),
        "max_xyz": all_kpts_cat.reshape(-1, 3).max(axis=0).tolist(),
        "mean_xyz": all_kpts_cat.reshape(-1, 3).mean(axis=0).tolist(),
        "per_link_mean_norm": {
            link: float(np.linalg.norm(all_kpts_cat[:, i], axis=-1).mean())
            for i, link in enumerate(KEYPOINT_LINKS)
        },
        "episodes": ep_reports,
    }


def _update_info_json(dest: Path, urdf_path: str) -> None:
    info_path = dest / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)
    info["features"]["observation.keypoint_3d"] = {
        "dtype": "float32",
        "shape": [NUM_KEYPOINTS * 3],
        "names": KEYPOINT_FEATURE_NAMES,
        "fps": info.get("fps", 20),
    }
    info["keypoint_coord_mode"] = "world"
    info["keypoint_generation"] = {
        "method": "pinocchio_ik_fk_per_episode_base",
        "urdf": urdf_path,
        "num_keypoints": NUM_KEYPOINTS,
        "links": KEYPOINT_LINKS,
        "urdf_frames": URDF_LINK_FRAMES + [EE_FRAME],
        "note": (
            "LIBERO state is EEF pose; per-episode base translation is fitted, "
            "arm joints via position IK, link1-7 from FK, EEF keypoint pinned to state[:3]."
        ),
    }
    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)
    logger.info("Updated %s", info_path)


def _write_report(dest: Path, stats: dict) -> None:
    # Keep episode-level detail but also a short summary file.
    full = dest / "meta" / "keypoint_generation_report.json"
    summary = {k: v for k, v in stats.items() if k != "episodes"}
    with open(full, "w") as f:
        json.dump(stats, f)
    with open(dest / "meta" / "keypoint_generation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(
        "Summary: rows=%d eps=%d ik_ok=%.1f%% mean_err=%.4fm max_err=%.4fm",
        summary["total_rows"],
        summary["total_episodes"],
        100.0 * summary["ik_success_rate"],
        summary["mean_ik_pos_err"],
        summary["max_ik_pos_err"],
    )


def process_one(source: Path, dest: Path, force: bool, skip_copy: bool, urdf: str) -> dict:
    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")
    if not skip_copy:
        _copy_dataset(source, dest, force=force)
    else:
        if not dest.exists():
            raise FileNotFoundError(f"--skip-copy given but {dest} does not exist")
        logger.info("--skip-copy: reusing %s", dest)

    extractor = FrankaPinocchioKeypoints(urdf)
    stats = _add_keypoint_column(dest, extractor)
    _update_info_json(dest, urdf)
    _write_report(dest, stats)
    return {k: v for k, v in stats.items() if k != "episodes"}


def _discover_suites(source_root: Path) -> list[Path]:
    return sorted(
        d
        for d in source_root.iterdir()
        if d.is_dir() and d.name.startswith("libero_") and (d / "meta" / "info.json").exists()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=str, default=None)
    parser.add_argument("--dest", type=str, default=None)
    parser.add_argument("--source_root", type=str, default="/home/a26160/DATA/LIBERO/libero")
    parser.add_argument("--dest_root", type=str, default="/home/a26160/DATA/LIBERO/libero_kpt")
    parser.add_argument("--suites", type=str, nargs="*", default=None)
    parser.add_argument("--urdf", type=str, default=DEFAULT_URDF)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-copy", action="store_true")
    args = parser.parse_args()

    if args.source or args.dest:
        if not (args.source and args.dest):
            raise SystemExit("Provide both --source and --dest for single-suite mode.")
        stats = process_one(Path(args.source), Path(args.dest), args.force, args.skip_copy, args.urdf)
        logger.info("Done. %s", json.dumps(stats, indent=2))
        return

    source_root = Path(args.source_root)
    dest_root = Path(args.dest_root)
    suites = _discover_suites(source_root)
    if args.suites:
        wanted = set(args.suites)
        suites = [s for s in suites if s.name in wanted]
    if not suites:
        raise FileNotFoundError(f"No suites under {source_root}")

    summary = {}
    for src in suites:
        dst = dest_root / src.name
        logger.info("=" * 72)
        logger.info("Suite %s -> %s", src.name, dst)
        summary[src.name] = process_one(src, dst, args.force, args.skip_copy, args.urdf)

    dest_root.mkdir(parents=True, exist_ok=True)
    out = dest_root / "keypoint_generation_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
