#!/usr/bin/env python3
"""Goal-specific live image-orientation gate (the Goal version of evaluation/LIBERO2 T1).

Why not reuse ``evaluation/LIBERO2/test_live_orientation.py`` as is:
  * it renders the *libero_spatial* BDDL and compares against the first frames of the first
    video file of a dataset -> for the Goal dataset that is a different scene, so the MSE
    ratio means nothing;
  * ``load_train_bank`` caches ``/tmp/kptimg/train_<cam>_bank.npy`` and silently reuses a bank built
    from a previous dataset.
This script keeps the judge (``orientation_contract.score_camera``) and replaces the two inputs:
the bank is built from frames of the *same* Goal task (unperturbed rendering, perturbation
"language"), and the live frame is rendered from that task's own BDDL.

Two venvs are involved, hence three modes:

  # 1) server venv (torch + lerobot): decode first frames of one task into an .npz. No simulator.
  python live_orientation_goal.py bank  --dataset D --task put_the_bowl_on_the_plate --out /tmp/goal_bank.npz

  # 2) client venv (mujoco + LIBERO, no torch): render the same task live and judge.
  LIBERO_HOME=... LIBERO_CONFIG_PATH=... \
  python live_orientation_goal.py live  --bank /tmp/goal_bank.npz --task put_the_bowl_on_the_plate [--render_backend osmesa]

  # 3) any venv, no simulator: test the judge itself on dataset frames (positive + negative control)
  python live_orientation_goal.py selftest --dataset D --task put_the_bowl_on_the_plate

Exit 0 = both cameras match the dataset as RAW (do not rotate at eval).
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[3]
for p in (str(_ROOT), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

CAMS = (
    {"name": "agentview", "obs_key": "agentview_image", "train_key": "observation.images.image"},
    {"name": "wrist", "obs_key": "robot0_eye_in_hand_image", "train_key": "observation.images.image2"},
)
UNPERTURBED = "language"  # the perturbation directory whose frames are rendered without any visual change


def build_bank(dataset: Path, task: str, video_backend: str, n_max: int = 64) -> dict[str, np.ndarray]:
    """First frame of every unperturbed episode of ``task``: {camera: uint8 [n,H,W,3]}."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    rows = [json.loads(l) for l in open(dataset / "meta" / "goal_episodes.jsonl")]
    eps = [r["episode_index"] for r in rows if r["task_name"] == task and r["perturbation"] == UNPERTURBED]
    if len(eps) < 2:
        raise SystemExit(f"need >=2 '{UNPERTURBED}' episodes of task {task!r}, found {len(eps)}")
    ds = LeRobotDataset("goal4d_orient_bank", root=dataset, video_backend=video_backend)
    out: dict[str, list[np.ndarray]] = {c["name"]: [] for c in CAMS}
    for k in eps[:n_max]:
        lo = int(ds.meta.episodes[k]["dataset_from_index"])
        sample = ds[lo]
        for c in CAMS:
            img = sample[c["train_key"]]  # CHW float [0,1]
            out[c["name"]].append((img.permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8))
    return {k: np.stack(v) for k, v in out.items()}


def judge(live: dict[str, np.ndarray], bank: dict[str, np.ndarray], min_ratio: float) -> dict:
    """Same rule as evaluation/LIBERO2 T1: live-as-is must beat live-rot180 by >= min_ratio on every
    bank frame's vote and in best MSE."""
    from evaluation.LIBERO2.orientation_contract import score_camera

    report, ok = {}, True
    for c in CAMS:
        s = score_camera(np.asarray(live[c["name"]], dtype=np.uint8), bank[c["name"]].astype(np.float32), min_ratio)
        for k in ("raw", "rot", "best_train"):
            s.pop(k)
        report[c["name"]] = s
        ok &= bool(s["matches_raw"])
    return {"passed": ok, "cameras": report, "min_ratio": min_ratio}


def render_live(task: str, seed: int = 7, dummy_wait: int = 10) -> dict[str, np.ndarray]:
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bddl = Path(get_libero_path("bddl_files")) / "libero_goal" / f"{task}.bddl"
    if not bddl.exists():
        raise FileNotFoundError(f"unperturbed Goal BDDL missing: {bddl}")
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    env.seed(seed)
    obs = env.reset()
    for _ in range(dummy_wait):
        obs, _, _, _ = env.step([0.0] * 6 + [-1.0])
    return {c["name"]: np.ascontiguousarray(np.asarray(obs[c["obs_key"]], dtype=np.uint8)) for c in CAMS}


def _print(report: dict) -> None:
    for name, s in report["cameras"].items():
        print(f"[{name}] raw={s['mse_live_raw_vs_train']} rot180={s['mse_live_rot180_vs_train']} "
              f"ratio={s['ratio_rot_over_raw']} vote={s['majority_vote']} matches_raw={s['matches_raw']}")
    print("PASS: dataset is RAW like the live env" if report["passed"] else "FAIL")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    b = sub.add_parser("bank")
    b.add_argument("--dataset", type=Path, required=True)
    b.add_argument("--task", required=True)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--video-backend", default="pyav")
    lv = sub.add_parser("live")
    lv.add_argument("--bank", type=Path, required=True)
    lv.add_argument("--task", required=True)
    lv.add_argument("--min_ratio", type=float, default=5.0)
    lv.add_argument("--seed", type=int, default=7)
    lv.add_argument("--json_out", type=Path)
    from evaluation.LIBERO2.render_backend import add_render_backend_cli  # numpy-only import

    add_render_backend_cli(lv)
    st = sub.add_parser("selftest")
    st.add_argument("--dataset", type=Path, required=True)
    st.add_argument("--task", required=True)
    st.add_argument("--video-backend", default="pyav")
    st.add_argument("--min_ratio", type=float, default=5.0)
    args = ap.parse_args()

    if args.mode == "bank":
        bank = build_bank(args.dataset, args.task, args.video_backend)
        np.savez_compressed(args.out, **bank)
        print(f"wrote {args.out}: " + ", ".join(f"{k}{v.shape}" for k, v in bank.items()))
        return 0

    if args.mode == "live":
        from evaluation.LIBERO2.orientation_contract import setup_client_render_env

        backend = setup_client_render_env(backend=args.render_backend)
        bank = dict(np.load(args.bank))
        report = judge(render_live(args.task, seed=args.seed), bank, args.min_ratio)
        report["render_backend"] = backend
        _print(report)
        if args.json_out:
            args.json_out.write_text(json.dumps(report, indent=2))
        sys.stdout.flush()
        import os

        os._exit(0 if report["passed"] else 1)  # MuJoCo EGL teardown can SIGABRT (U7)

    # selftest: hold one episode out as the pseudo-live frame
    bank = build_bank(args.dataset, args.task, args.video_backend)
    n = len(next(iter(bank.values())))
    live = {k: v[0] for k, v in bank.items()}
    rest = {k: v[1:] for k, v in bank.items()}
    pos = judge(live, rest, args.min_ratio)
    neg = judge({k: np.ascontiguousarray(v[::-1, ::-1]) for k, v in live.items()}, rest, args.min_ratio)
    print(f"held-out episode vs {n - 1} bank frames")
    _print(pos)
    print("negative control (pseudo-live rotated 180 deg) must FAIL:")
    _print(neg)
    ok = pos["passed"] and not neg["passed"]
    print("SELFTEST OK" if ok else "SELFTEST FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
