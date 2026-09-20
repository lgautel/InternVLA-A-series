#!/usr/bin/env python3
"""Compare a raw RLDS JPEG against the matching merged_kpt training video frame.

check_orientation.py proves geometrically that the RLDS JPEGs are a 180 deg
rotation of MuJoCo's render buffer. The eval contract
(evaluation/LIBERO2/train_eval_contract.json) says the *training* frames are
robosuite raw and that eval must therefore send `rotate_images=false`. Both
cannot describe the same pixels, so the RLDS -> LeRobot conversion must rotate
a second time. This script checks that directly.

`opvla_libero_merged_kpt` keeps its frames in mp4, so we stream the first video
shard out of the tarball, decode frame 0 with ffmpeg, and diff it against
episode 0 frame 0 of libero_spatial under both hypotheses.

Episode 0 lines up across the two datasets: the merged `meta/episodes` parquet
reports length 110 with the same instruction as RLDS libero_spatial episode 0.
"""

from __future__ import annotations

import io
import json
import subprocess
import tarfile
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from rlds_reader import iter_episodes

HERE = Path(__file__).parent
TARBALL = Path("/B/Dta/opvla_libero_merged_kpt.tar.gz")
FFMPEG = Path("/B/VENV/itnvla15rbt20/bin/ffmpeg")
WANTED = "videos/observation.images.image/chunk-000/file-000.mp4"
OUT_JSON = HERE / "chain_orientation_report.json"
OUT_NPZ = HERE / "chain_orientation_frames.npz"


def extract_video(dest: Path) -> bool:
    with tarfile.open(TARBALL, mode="r|gz") as tf:
        for member in tf:
            if member.isfile() and member.name.split("/", 1)[-1] == WANTED:
                dest.write_bytes(tf.extractfile(member).read())
                return True
    return False


def first_frame(video: Path) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmp:
        png = Path(tmp) / "f0.png"
        subprocess.run(
            [str(FFMPEG), "-v", "error", "-i", str(video), "-frames:v", "1", "-y", str(png)],
            check=True,
        )
        return np.asarray(Image.open(png).convert("RGB"))


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean())


def main() -> int:
    if not FFMPEG.exists():
        print(f"ffmpeg not found at {FFMPEG}")
        return 1

    episode = next(iter_episodes("libero_spatial", max_episodes=1))
    rlds = np.asarray(Image.open(io.BytesIO(episode.jpeg("image", 0))).convert("RGB"))

    with tempfile.TemporaryDirectory() as tmp:
        video = Path(tmp) / "file-000.mp4"
        print(f"streaming {WANTED} out of {TARBALL.name} ...")
        if not extract_video(video):
            print("video member not found")
            return 1
        merged = first_frame(video)

    if merged.shape != rlds.shape:
        print(f"shape mismatch: merged {merged.shape} vs rlds {rlds.shape}")
        return 1

    candidates = {
        "as_stored": rlds,
        "rot180": rlds[::-1, ::-1],
        "vflip": rlds[::-1, :],
        "hflip": rlds[:, ::-1],
    }
    scores = {name: round(mse(img, merged), 2) for name, img in candidates.items()}
    ranked = sorted(scores.items(), key=lambda kv: kv[1])
    best, runner_up = ranked[0], ranked[1]

    report = {
        "rlds_episode": {
            "subset": "libero_spatial",
            "task": episode.task_name,
            "num_steps": episode.num_steps,
            "instruction": episode.instruction,
        },
        "merged_video": WANTED,
        "mse_rlds_vs_merged": scores,
        "verdict": {
            "transform_from_rlds_to_merged": best[0],
            "mse": best[1],
            "runner_up": runner_up[0],
            "runner_up_mse": runner_up[1],
            "margin_ratio": round(runner_up[1] / max(best[1], 1e-6), 2),
        },
        "interpretation": (
            "combined with check_orientation.py (RLDS = rot180 of the MuJoCo buffer), "
            "a rot180 here means the merged training frames are back in robosuite raw "
            "orientation, which is what evaluation/LIBERO2/train_eval_contract.json assumes"
        ),
    }
    OUT_JSON.write_text(json.dumps(report, indent=2))
    np.savez_compressed(OUT_NPZ, rlds=rlds, merged=merged)

    for name, value in ranked:
        print(f"  MSE(rlds.{name:9s}, merged_mp4_frame0) = {value:10.2f}")
    print(f"\nVERDICT: merged_kpt = rlds.{best[0]}  ({report['verdict']['margin_ratio']}x margin)")
    print(f"wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
