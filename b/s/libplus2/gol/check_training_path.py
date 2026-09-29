#!/usr/bin/env python3
"""Push the generated dataset through the REAL training data path, no model, no GPU.

Uses the same arguments as launch/libplus_sft_launch.sh for policy/dataset (H=200,
pos_rot, panda schema, external stats = dataset meta/stats.json) and calls
``lerobot.datasets.factory.make_dataset``. Then reads a few samples and checks what
the network would receive.

    LD_LIBRARY_PATH=<as in the launch script> \
    /B/VENV/itnvla15rbt20/bin/python check_training_path.py --dataset /tmp/goal4d_smoke

Exit status 0 when every assertion holds.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[3]
for p in (str(_HERE), str(_ROOT), str(_ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from contract import CHUNK_SIZE, HISTORY_MAX_LEN  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--video-backend", default="torchcodec")
    ap.add_argument("--n", type=int, default=6)
    args = ap.parse_args()

    home = Path(tempfile.mkdtemp(prefix="lerobot_home_"))
    repo_id = "goal4d_verify"
    (home / repo_id).symlink_to(args.dataset.resolve())
    os.environ["HF_LEROBOT_HOME"] = str(home)

    import draccus

    import lerobot.policies.factory  # noqa: F401  populates the policy/dataset choice registries
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.datasets.factory import make_dataset

    base = Path(os.environ.get("HF_HOME", "/B/VENV/hf_home")) / "ckpts" / "InternVLA-A1.5-base"
    cli = [
        "--output_dir", str(home / "out"), "--job_name", "check", "--batch_size", "2", "--steps", "1",
        "--policy.type=internvla_a1_5", "--policy.push_to_hub=false", f"--policy.pretrained_path={base}",
        "--policy.dtype=bfloat16",
        "--policy.enable_vqa_loss=true", "--policy.tokenize_state=true",
        "--policy.enable_keypoint_predictor=true", "--policy.num_keypoint_joints=8",
        "--policy.kpt_4d_mode=pos_rot", f"--policy.keypoint_history_max_len={HISTORY_MAX_LEN}",
        f"--dataset.type=internvla_a1_5", f"--dataset.repo_id={repo_id}",
        "--dataset.enable_keypoint_predictor=true", "--dataset.num_keypoint_joints=8",
        "--dataset.kpt_4d_mode=pos_rot", f"--dataset.keypoint_history_max_len={HISTORY_MAX_LEN}",
        "--dataset.action_mode=abs", "--dataset.use_external_stats=true",
        f"--dataset.external_stats_path={args.dataset}/meta/stats.json",
        "--dataset.dist_loading=false", "--dataset.tokenize_state=true",
        "--dataset.use_fast_action_tokens=true", f"--dataset.video_backend={args.video_backend}",
        "--wandb.enable=false",
    ]
    cfg = draccus.parse(TrainPipelineConfig, args=cli)
    dataset, stats = make_dataset(cfg)

    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")

    check("robot_type key in stats", "panda" in stats, f"keys={list(stats)}")
    n = len(dataset)
    check("dataset length > 0", n > 0, f"len={n}")
    idxs = np.unique(np.linspace(0, n - 1, args.n).astype(int))
    for i in idxs:
        s = dataset[int(i)]
        his = s["observation.his_kpts"]
        hl = int(s["observation.his_len"])
        check(f"sample {i}: his_kpts shape", tuple(his.shape) in {(HISTORY_MAX_LEN, 56), (HISTORY_MAX_LEN, 8, 7)}, str(tuple(his.shape)))
        check(f"sample {i}: 0<=his_len<=H", 0 <= hl <= HISTORY_MAX_LEN, f"his_len={hl}")
        h = his.reshape(HISTORY_MAX_LEN, -1)
        check(f"sample {i}: zeros after his_len", bool((h[hl:] == 0).all()))
        check(f"sample {i}: filled before his_len", hl == 0 or bool((h[:hl].abs().sum(-1) > 0).all()))
        kf = s["observation.kpt_future"]
        check(f"sample {i}: kpt_future has C steps", kf.shape[0] == CHUNK_SIZE, str(tuple(kf.shape)))
        act = s["action"]
        check(f"sample {i}: action chunk (C x padded-dim)", act.shape[0] == CHUNK_SIZE and act.shape[-1] >= 7, str(tuple(act.shape)))
        check(f"sample {i}: action dims >=7 are zero padding", bool((act[:, 7:] == 0).all()))
        check(f"sample {i}: action[:, :7] finite and non-degenerate", bool(np.isfinite(act[:, :7].numpy()).all()) and float(act[:, :7].abs().sum()) > 0)
        grid = s["observation.image_grid_thw"]
        check(f"sample {i}: two camera images tokenised", grid.shape[0] == 2, f"image_grid_thw={tuple(grid.shape)}")
        check(f"sample {i}: pixel_values present", s["observation.pixel_values"].numel() > 0)
        st = s["observation.state"]
        check(f"sample {i}: state finite", bool(np.isfinite(st.numpy()).all()), str(tuple(st.shape)))
    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
