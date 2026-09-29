#!/usr/bin/env python3
"""Acceptance checks for a generated Goal 4D LeRobot dataset.

    LD_LIBRARY_PATH=... /B/VENV/itnvla15rbt20/bin/python verify_goal_dataset.py \
        --dataset /B/Dta/LIBERO/libero_plus_goal_lrb3_4D [--full] [--video-backend torchcodec]

``--full`` additionally asserts the published release size: 4,243 episodes,
512,604 frames, 428 unique (joint_state, action) trajectories.

Each check prints PASS/FAIL with a one-line reason. Exit status is non-zero on any
FAIL. Nothing here needs the LIBERO simulator.

Check ids map to incidents in 3d4d_gen2.markdown section 2.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
_ROOT = _HERE.parents[3]
for extra in (_ROOT, _ROOT / "src"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from contract import (  # noqa: E402
    CHUNK_SIZE,
    CONTRACT_FILENAME,
    EPISODE_EEF_TOL_M,
    FPS,
    GOAL_BASE_XPOS,
    GOAL_MJCF,
    HISTORY_MAX_LEN,
    IMAGE_ORIENTATION,
    KPT_4D_MODE,
    KIN_XML,
    ROBOT_TYPE,
)
from fk import GoalTableFK, qpos9_from_state  # noqa: E402
from orientation_probe import (  # noqa: E402
    WRIST_POS_IN_HAND,
    WRIST_QUAT_WXYZ,
    agentview_orientation,
    quat_to_mat,
    to_gray,
    wrist_orientation,
)
from rlds_goal import DEFAULT_ROOT, iter_goal_episodes  # noqa: E402

FULL_EPISODES = 4243
FULL_FRAMES = 512604
FULL_UNIQUE = 428


class Report:
    def __init__(self):
        self.rows: list[tuple[str, bool, str]] = []

    def add(self, cid: str, ok: bool, detail: str) -> bool:
        self.rows.append((cid, bool(ok), detail))
        print(f"[{'PASS' if ok else 'FAIL'}] {cid}: {detail}", flush=True)
        return bool(ok)

    @property
    def ok(self) -> bool:
        return all(r[1] for r in self.rows)


def _load_frames_table(dest: Path):
    import pandas as pd

    files = sorted((dest / "data").glob("chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no data parquet under {dest}/data")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def check_info_and_contract(rep: Report, dest: Path, info: dict, contract: dict):
    rep.add("V01.info", info.get("codebase_version") == "v3.0" and info.get("fps") == FPS
            and info.get("robot_type") == ROBOT_TYPE,
            f"codebase={info.get('codebase_version')} fps={info.get('fps')} robot_type={info.get('robot_type')}")
    feats = info["features"]
    shapes_ok = (
        tuple(feats["observation.state"]["shape"]) == (8,)
        and tuple(feats["observation.keypoint_3d"]["shape"]) == (56,)
        and tuple(feats["action"]["shape"]) == (7,)
        and "observation.images.image" in feats
        and "observation.images.image2" in feats
    )
    rep.add("V02.features", shapes_ok, "state[8] keypoint_3d[56] action[7] image image2")
    vid_ok = contract.get("has_video") is False or all(
        feats[k]["dtype"] == "video" for k in ("observation.images.image", "observation.images.image2")
    )
    rep.add("V03.video_dtype", vid_ok, "both cameras stored as video" if vid_ok else "images are not video")

    kin_md5 = hashlib.md5(KIN_XML.read_bytes()).hexdigest()
    goal_md5 = hashlib.md5(GOAL_MJCF.read_bytes()).hexdigest()
    want = {
        "robot_type": ROBOT_TYPE,
        "stats_key": "panda",
        "image_orientation": IMAGE_ORIENTATION,
        "kpt_4d_mode": KPT_4D_MODE,
        "keypoint_history_max_len": HISTORY_MAX_LEN,
        "chunk_size": CHUNK_SIZE,
        "fps": FPS,
        "eval_mjcf_md5": goal_md5,
        "kin_mjcf_md5": kin_md5,
        "history_includes_current_frame": False,
        "wait_steps_committed_to_history": False,
    }
    bad = {k: (contract.get(k), v) for k, v in want.items() if contract.get(k) != v}
    rep.add("V04.contract", not bad and contract.get("r_pad", 0) > 0 and list(contract["base_xpos_m"]) == GOAL_BASE_XPOS.tolist(),
            "contract file matches code constants" if not bad else f"mismatch {bad}")


def check_arrays(rep: Report, dest: Path, df, sidecar: list[dict], contract: dict, full: bool):
    n = len(df)
    lens = sum(r["length"] for r in sidecar)
    ok = n == lens == json.loads((dest / "meta/info.json").read_text())["total_frames"]
    if full:
        ok = ok and n == FULL_FRAMES and len(sidecar) == FULL_EPISODES
    rep.add("V05.counts", ok, f"frames={n} sidecar_frames={lens} episodes={len(sidecar)}")

    kp = np.stack(df["observation.keypoint_3d"].to_numpy()).astype(np.float64).reshape(n, 8, 7)
    st = np.stack(df["observation.state"].to_numpy()).astype(np.float64)
    jp = np.stack(df["observation.state.joint_position"].to_numpy()).astype(np.float64)
    act = np.stack(df["action"].to_numpy()).astype(np.float64)
    finite = all(np.isfinite(a).all() for a in (kp, st, jp, act))
    rep.add("V06.finite", finite, "no NaN/Inf in state, joints, action, keypoints")

    qn = np.linalg.norm(kp[:, :, 3:], axis=-1)
    rep.add("V07.quat", np.abs(qn - 1).max() < 1e-5 and (kp[:, :, 6] >= -1e-7).all(),
            f"max|norm-1|={np.abs(qn-1).max():.2e} min qw={kp[:,:,6].min():.3g}")

    r_pad = float(contract["r_pad"])
    pos_max = np.abs(kp[:, :, :3]).max()
    bound = 1.0 / (1.0 + 0.15) + 1e-4
    rep.add("V08.r_pad_range", pos_max <= bound + 1e-3 if full else pos_max <= 1.0,
            f"max|pos/R_pad|={pos_max:.4f} (full-run bound {bound:.4f}); r_pad={r_pad:.6f}")

    fk = GoalTableFK()
    idx = np.unique(np.linspace(0, n - 1, min(n, 4000)).astype(int))
    worst_k = 0.0
    worst_e = 0.0
    for i in idx:
        q9 = qpos9_from_state(jp[i], st[i])
        got = fk.keypoints(q9, r_pad)
        worst_k = max(worst_k, float(np.abs(got - kp[i]).max()))
        worst_e = max(worst_e, float(np.linalg.norm(fk.world_poses(q9)[7, :3] - st[i, :3])))
    rep.add("V09.fk_recompute", worst_k < 1e-5, f"stored keypoints == FK(joint_position, fingers): max diff {worst_k:.2e}")
    rep.add("V10.fk_vs_state", worst_e < EPISODE_EEF_TOL_M, f"|FK eef - state[0:3]| max {worst_e*1e3:.2f} mm (tol {EPISODE_EEF_TOL_M*1e3:.1f} mm)")

    # eval FK (StandaloneFK on the derived MJCF) must equal the generator FK
    from evaluation.LIBERO2.keypoint_utils import StandaloneFK

    sfk = StandaloneFK(mjcf_path=GOAL_MJCF, r_pad=r_pad)
    worst_s = 0.0
    for i in idx[:: max(1, len(idx) // 300)]:
        q9 = qpos9_from_state(jp[i], st[i])
        worst_s = max(worst_s, float(np.abs(sfk.extract(q9) - kp[i]).max()))
    rep.add("V11.eval_fk_equals_train_fk", worst_s < 1e-5, f"StandaloneFK(panda_goal_table.xml) vs stored: max diff {worst_s:.2e}")

    # gripper semantics (LIBERO native: +1 close, -1 open  => finger opening decreases after +1)
    ep = df["episode_index"].to_numpy()
    same = ep[1:] == ep[:-1]
    d_finger = (st[1:, 6] - st[:-1, 6])[same]
    corr = float(np.corrcoef(act[:-1, 6][same], d_finger)[0, 1])
    rep.add("V12.gripper_libero_native", corr < -0.2 and act[:, 6].min() >= -1 - 1e-6 and act[:, 6].max() <= 1 + 1e-6,
            f"corr(action[6], d finger_l)={corr:+.3f} (negative => +1 closes) range=[{act[:,6].min():.2f},{act[:,6].max():.2f}]")

    stats = json.loads((dest / "meta/stats.json").read_text())
    bad = []
    for key, dim in (("observation.state", 8), ("action", 7)):
        s = stats.get(key)
        if not s or len(s["mean"]) != dim or min(s["std"]) < 1e-6:
            bad.append(key)
    rep.add("V13.stats", not bad, "meta/stats.json has state[8], action[7] with std>1e-6" if not bad else f"bad stats: {bad}")

    hashes = {r["trajectory_hash"] for r in sidecar}
    rep.add("V14.unique_trajectories", (len(hashes) == FULL_UNIQUE) if full else len(hashes) >= 1,
            f"{len(hashes)} unique (joint_state, action) trajectories over {len(sidecar)} episodes")
    return kp, st, jp


def _decode_episode(ds, ep_idx: int, key: str) -> np.ndarray:
    meta = ds.meta.episodes[ep_idx]
    lo, hi = int(meta["dataset_from_index"]), int(meta["dataset_to_index"])
    frames = []
    for i in range(lo, hi):
        img = ds[i][key]  # CHW float [0,1]
        frames.append((img.permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8))
    return np.stack(frames)


def check_orientation_and_window(rep: Report, dest: Path, sidecar: list[dict], contract: dict, video_backend: str, rlds_root: Path, n_probe: int):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if not contract.get("has_video", True):
        rep.add("V15.orientation", False, "dataset stored without video; orientation probe needs the video path")
        return
    H, C = HISTORY_MAX_LEN, CHUNK_SIZE
    ds = LeRobotDataset("goal4d_verify", root=dest, video_backend=video_backend)

    # ---- pixel identity against RLDS (orientation + codec loss) ----
    probe_eps = [r["episode_index"] for r in sidecar][:2]
    mse = {}
    it = iter_goal_episodes(rlds_root, max_episodes=max(probe_eps) + 1)
    src_eps = list(it)
    from PIL import Image

    for cam_key, jpeg_attr, name in (("observation.images.image", "image_jpeg", "agentview"), ("observation.images.image2", "wrist_jpeg", "wrist")):
        errs = {"as_rlds": [], "raw(rot180)": []}
        for k in probe_eps:
            src = src_eps[k]
            dec = _decode_episode(ds, k, cam_key)
            for t in np.linspace(0, src.num_steps - 1, 5).astype(int):
                a = np.asarray(Image.open(io.BytesIO(getattr(src, jpeg_attr)[t])).convert("RGB")).astype(np.float32)
                b = dec[t].astype(np.float32)
                errs["as_rlds"].append(float(np.mean((a - b) ** 2)))
                errs["raw(rot180)"].append(float(np.mean((a[::-1, ::-1] - b) ** 2)))
        m_rlds, m_raw = float(np.mean(errs["as_rlds"])), float(np.mean(errs["raw(rot180)"]))
        expect_raw = contract["image_orientation"] == "raw"
        good = (m_raw < m_rlds / 10) if expect_raw else (m_rlds < m_raw / 10)
        rep.add(f"V15.pixels_{name}", good, f"mse vs RLDS as-is={m_rlds:.1f}  vs rot180={m_raw:.1f}  (contract={contract['image_orientation']})")
        mse[name] = (m_rlds, m_raw)

    # ---- geometry: which orientation do the decoded frames have? ----
    lang = [r["episode_index"] for r in sidecar if r["perturbation"] == "language"][:n_probe]
    if len(lang) >= 2:
        fk = GoalTableFK()
        eef, gray_a, gray_w, camp, camr = [], [], [], [], []
        df = _load_frames_table(dest)
        for k in lang:
            lo, hi = int(ds.meta.episodes[k]["dataset_from_index"]), int(ds.meta.episodes[k]["dataset_to_index"])
            sub = df.iloc[lo:hi]
            st = np.stack(sub["observation.state"].to_numpy()).astype(np.float64)
            jp = np.stack(sub["observation.state.joint_position"].to_numpy()).astype(np.float64)
            eef.append(st[:, :3])
            gray_a.append(to_gray(_decode_episode(ds, k, "observation.images.image")))
            gray_w.append(to_gray(_decode_episode(ds, k, "observation.images.image2")))
            pos, rot = [], []
            for t in range(len(st)):
                hp, hq = fk.hand_pose(qpos9_from_state(jp[t], st[t]))
                rh = quat_to_mat(hq)
                pos.append(hp + rh @ WRIST_POS_IN_HAND)
                rot.append(rh @ quat_to_mat(WRIST_QUAT_WXYZ))
            camp.append(np.asarray(pos))
            camr.append(rot)
        want = "flipV_RAW" if contract["image_orientation"] == "raw" else "flipH_RLDS"
        try:
            a = agentview_orientation(eef, gray_a)
            rep.add("V16.geometry_agentview", a["best"] == want and a["scores"][want]["min_corr"] > 0.5,
                    f"best={a['best']} (want {want}) corr_col={a['scores'][want]['corr_col']:+.3f} corr_row={a['scores'][want]['corr_row']:+.3f} n={a['samples']}")
        except RuntimeError as e:
            rep.add("V16.geometry_agentview", False, str(e))
        try:
            w = wrist_orientation(camp, camr, gray_w)
            rep.add("V17.geometry_wrist", w["best"] == want and w["scores"][want]["min_corr"] > 0.3,
                    f"best={w['best']} (want {want}) corr_col={w['scores'][want]['corr_col']:+.3f} corr_row={w['scores'][want]['corr_row']:+.3f} n={w['samples']}")
        except RuntimeError as e:
            rep.add("V17.geometry_wrist", False, str(e))
    else:
        rep.add("V16.geometry_agentview", False, "need >=2 language-perturbation episodes in the probe set")

    # ---- training window through the real dataset path ----
    import torch
    from lerobot.policies.internvla_a1_5.transform_internvla_a1_5 import Extract3DKeypointTransformFn

    delta = {"observation.keypoint_3d": [i / FPS for i in range(-H, C + 1)]}
    ds2 = LeRobotDataset("goal4d_verify", root=dest, delta_timestamps=delta, video_backend=video_backend)
    tf = Extract3DKeypointTransformFn(num_joints=8, history_max_len=H, chunk_size=C, keypoint_dim=7)
    df = _load_frames_table(dest)
    kp_all = np.stack(df["observation.keypoint_3d"].to_numpy()).astype(np.float32).reshape(-1, 8, 7)
    bad = []
    e0 = sidecar[0]["episode_index"]
    lo, hi = int(ds2.meta.episodes[e0]["dataset_from_index"]), int(ds2.meta.episodes[e0]["dataset_to_index"])
    length = hi - lo
    for t in (0, 1, 7, length // 2, length - 1):
        sample = ds2[lo + t]
        raw = {
            "observation.keypoint_3d": sample["observation.keypoint_3d"],
            "observation.keypoint_3d_is_pad": sample["observation.keypoint_3d_is_pad"],
        }
        out = tf(raw)
        his_len = int(out["observation.his_len"])
        want_len = min(t, H)
        his = out["observation.his_kpts"].numpy().reshape(H, 8, 7)
        ok = his_len == want_len
        if ok and want_len:
            ok = np.allclose(his[:want_len], kp_all[lo + t - want_len: lo + t], atol=1e-5)
        ok = ok and np.allclose(out["observation.kpt_t"].numpy().reshape(8, 7), kp_all[lo + t], atol=1e-5)
        ok = ok and (not want_len or np.all(his[want_len:] == 0))
        fut = out["observation.kpt_future"].numpy().reshape(C, 8, 7)
        exp0 = kp_all[lo + min(t + 1, length - 1)]
        ok = ok and np.allclose(fut[0], exp0, atol=1e-5)
        if not ok:
            bad.append((t, his_len, want_len))
    rep.add("V18.training_window", not bad, "his_len=min(t,H), his_kpts = past frames, oldest first, zero padded; kpt_t/kpt_future correct"
            if not bad else f"mismatch at (t,got,want)={bad}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--rlds-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--video-backend", default="pyav", choices=["pyav", "torchcodec"])
    parser.add_argument("--n-probe", type=int, default=6, help="language-perturbation episodes for geometry probes")
    parser.add_argument("--skip-decode", action="store_true", help="skip checks that decode video (fast structural pass)")
    args = parser.parse_args()

    dest = args.dataset
    rep = Report()
    info = json.loads((dest / "meta/info.json").read_text())
    contract = json.loads((dest / "meta" / CONTRACT_FILENAME).read_text())
    sidecar = [json.loads(line) for line in (dest / "meta/goal_episodes.jsonl").read_text().splitlines() if line]
    check_info_and_contract(rep, dest, info, contract)
    df = _load_frames_table(dest)
    check_arrays(rep, dest, df, sidecar, contract, args.full)
    if not args.skip_decode:
        check_orientation_and_window(rep, dest, sidecar, contract, args.video_backend, args.rlds_root, args.n_probe)
    print(f"\n{sum(r[1] for r in rep.rows)}/{len(rep.rows)} passed")
    return 0 if rep.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
