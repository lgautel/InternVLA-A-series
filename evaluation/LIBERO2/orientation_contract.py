"""Train/eval image-orientation contract for InternVLA-A1.5 on LIBERO.

Training videos in `opvla_libero_merged_kpt` are robosuite **raw** (unrotated).
Eval must send the same orientation: `rotate_images=False` (default).
Both `agentview_image` and `robot0_eye_in_hand_image` share this contract.
The on-disk contract is `train_eval_contract.json` next to this file.

This module is the reusable core behind:
  - evaluation/LIBERO2/test_live_orientation.py   (acceptance T1)
  - b/d/libplus/asset/eval3_optim_wrist_t1.py     (figures)

Do not compare against LIBERO-plus *perturbed* BDDLs (e.g. `*_table_1`):
wood vs stone textures flatten pixel MSE. Use the unperturbed spatial BDDL.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = Path(__file__).resolve().parent / "train_eval_contract.json"
DEFAULT_TRAIN_ROOT = Path("/B/Dta/opvla_libero_merged_kpt/videos")
DEFAULT_BANK_DIR = Path("/tmp/kptimg")
RECORDED_T1_JSON = REPO_ROOT / "b/d/libplus/asset/eval3_optim_wrist_t1.json"
RAW_ORIENTATIONS = frozenset({"raw", "unrotated", "none", "identity"})
ROT180_ORIENTATIONS = frozenset({"rot180", "rotated", "180", "flip"})
UNPERTURBED_BDDL = (
    "libero_spatial/"
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.bddl"
)

CAMS = (
    {
        "name": "agentview",
        "obs_key": "agentview_image",
        "train_key": "observation.images.image",
        "min_ratio": 5.0,
    },
    {
        "name": "wrist",
        "obs_key": "robot0_eye_in_hand_image",
        "train_key": "observation.images.image2",
        "min_ratio": 5.0,
    },
)


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))


def best_mse(frame: np.ndarray, bank: np.ndarray) -> tuple[float, int]:
    dists = [mse(frame, b) for b in bank]
    i = int(np.argmin(dists))
    return dists[i], i


def majority_vote(live_raw: np.ndarray, live_rot: np.ndarray, bank: np.ndarray) -> dict[str, int]:
    n_raw = n_rot = 0
    for b in bank:
        if mse(live_raw, b) <= mse(live_rot, b):
            n_raw += 1
        else:
            n_rot += 1
    return {"prefer_raw": n_raw, "prefer_rot180": n_rot, "n": int(len(bank))}


def load_train_bank(
    name: str,
    rel: str,
    n: int = 80,
    train_root: Path = DEFAULT_TRAIN_ROOT,
    bank_dir: Path = DEFAULT_BANK_DIR,
) -> np.ndarray:
    npy = bank_dir / f"train_{name}_bank.npy"
    if npy.exists():
        return np.load(npy)[:n].astype(np.float32)

    path = train_root / rel / "chunk-000" / "file-000.mp4"
    if not path.exists():
        matches = sorted((train_root / rel).glob("**/*.mp4")) if (train_root / rel).exists() else []
        if not matches:
            raise FileNotFoundError(f"no training video for {rel} under {train_root}")
        path = matches[0]
    try:
        from torchvision.io import read_video

        vid, _, _ = read_video(str(path), start_pts=0, end_pts=10.0, pts_unit="sec", output_format="THWC")
        arr = vid[:n].numpy().astype(np.float32)
    except Exception:
        import av

        images = []
        container = av.open(str(path))
        try:
            for frame in container.decode(video=0):
                images.append(frame.to_ndarray(format="rgb24"))
                if len(images) >= n:
                    break
        finally:
            container.close()
        if not images:
            raise RuntimeError(f"decoded 0 frames from {path}")
        arr = np.stack(images, axis=0).astype(np.float32)
    bank_dir.mkdir(parents=True, exist_ok=True)
    np.save(npy, arr.astype(np.uint8))
    return arr


def render_live_unperturbed(seed: int = 7, dummy_wait: int = 10) -> dict[str, np.ndarray]:
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bddl = Path(get_libero_path("bddl_files")) / UNPERTURBED_BDDL
    if not bddl.exists():
        raise FileNotFoundError(f"unperturbed BDDL missing: {bddl}")
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    env.seed(seed)
    obs = env.reset()
    dummy = [0.0] * 6 + [-1.0]
    for _ in range(dummy_wait):
        obs, _, _, _ = env.step(dummy)
    out = {
        c["name"]: np.ascontiguousarray(np.asarray(obs[c["obs_key"]], dtype=np.uint8))
        for c in CAMS
    }
    try:
        env.close = lambda: None
    except Exception:
        pass
    return out


def score_camera(raw: np.ndarray, bank: np.ndarray, min_ratio: float) -> dict[str, Any]:
    rot = np.ascontiguousarray(raw[::-1, ::-1])
    e_raw, i_raw = best_mse(raw, bank)
    e_rot, _ = best_mse(rot, bank)
    e_self = best_mse(bank[0], bank[1:])[0] if len(bank) > 1 else 0.0
    e_self_flip = best_mse(np.ascontiguousarray(bank[0][::-1, ::-1]), bank[1:])[0] if len(bank) > 1 else 0.0
    vote = majority_vote(raw, rot, bank)
    ratio = (e_rot / e_raw) if e_raw > 0 else float("inf")
    matches_raw = e_raw < e_rot and ratio >= min_ratio and vote["prefer_raw"] == vote["n"]
    return {
        "mse_live_raw_vs_train": round(e_raw, 1),
        "mse_live_rot180_vs_train": round(e_rot, 1),
        "ratio_rot_over_raw": round(float(ratio), 3),
        "control_train_vs_train": round(e_self, 1),
        "control_train_rot180_vs_train": round(e_self_flip, 1),
        "majority_vote": vote,
        "best_train_index": int(i_raw),
        "matches_raw": bool(matches_raw),
        "min_ratio": min_ratio,
        "raw": raw,
        "rot": rot,
        "best_train": bank[i_raw],
        "verdict": (
            "train matches LIVE RAW (do not rotate)"
            if e_raw < e_rot
            else "train matches LIVE ROT180 (keep rotate)"
        ),
    }


def evaluate_orientation_contract(
    *,
    n_train: int = 80,
    seed: int = 7,
    min_ratio: float | None = None,
    live: dict[str, np.ndarray] | None = None,
    train_root: Path = DEFAULT_TRAIN_ROOT,
) -> dict[str, Any]:
    """Return a JSON-serializable report plus pass/fail. Drops image arrays from cameras."""
    banks = {
        c["name"]: load_train_bank(c["name"], c["train_key"], n=n_train, train_root=train_root)
        for c in CAMS
    }
    if live is None:
        live = render_live_unperturbed(seed=seed)
    cameras: dict[str, Any] = {}
    vis: dict[str, Any] = {}
    all_ok = True
    for c in CAMS:
        thresh = min_ratio if min_ratio is not None else c["min_ratio"]
        scored = score_camera(live[c["name"]], banks[c["name"]], thresh)
        vis[c["name"]] = {
            "raw": scored.pop("raw"),
            "rot": scored.pop("rot"),
            "best_train": scored.pop("best_train"),
        }
        cameras[c["name"]] = scored
        all_ok = all_ok and bool(scored["matches_raw"])
    return {
        "passed": all_ok,
        "task": "libero_spatial original (unperturbed BDDL)",
        "seed": seed,
        "cameras": cameras,
        "_vis": vis,
    }


def camera_matches_raw(cam: dict[str, Any], min_ratio: float = 5.0) -> bool:
    """Same pass rule as `score_camera`, usable on live reports or recorded JSON."""
    e_raw = float(cam["mse_live_raw_vs_train"])
    e_rot = float(cam["mse_live_rot180_vs_train"])
    ratio = float(cam.get("ratio_rot_over_raw", (e_rot / e_raw) if e_raw > 0 else float("inf")))
    vote = cam.get("majority_vote") or {}
    n = int(vote.get("n", 0))
    prefer_raw = int(vote.get("prefer_raw", 0))
    return e_raw < e_rot and ratio >= min_ratio and n > 0 and prefer_raw == n


def check_recorded_t1(
    path: Path = RECORDED_T1_JSON,
    min_ratio: float = 5.0,
    required: tuple[str, ...] = ("agentview", "wrist"),
) -> dict[str, Any]:
    """Offline gate: previously measured T1 numbers still say both cameras are raw."""
    data = json.loads(path.read_text())
    cameras = data.get("cameras") or {}
    missing = [n for n in required if n not in cameras]
    if missing:
        raise KeyError(f"{path} missing cameras {missing}")
    per: dict[str, Any] = {}
    all_ok = True
    for name in required:
        cam = cameras[name]
        ok = camera_matches_raw(cam, min_ratio=min_ratio)
        per[name] = {
            "matches_raw": ok,
            "mse_live_raw_vs_train": cam["mse_live_raw_vs_train"],
            "mse_live_rot180_vs_train": cam["mse_live_rot180_vs_train"],
            "ratio_rot_over_raw": cam.get("ratio_rot_over_raw"),
            "majority_vote": cam.get("majority_vote"),
            "verdict": cam.get("verdict"),
        }
        all_ok = all_ok and ok
    return {"passed": all_ok, "path": str(path), "cameras": per}


def maybe_rotate(image: np.ndarray, rotate: bool) -> np.ndarray:
    """Same pixel op as `LiberoModelClient._maybe_rotate` (no websocket needed)."""
    arr = np.asarray(image)
    if rotate:
        arr = arr[::-1, ::-1]
    return np.ascontiguousarray(arr)


def load_train_eval_contract(path: Path | None = None) -> dict[str, Any]:
    """Load the dataset-dependent train/eval image contract (not server metadata)."""
    p = path or CONTRACT_PATH
    data = json.loads(p.read_text())
    if "image_orientation" not in data:
        raise KeyError(f"{p} missing image_orientation")
    return data


def enforce_rotate_against_contract(
    rotate_images: bool,
    contract: dict[str, Any] | None = None,
) -> None:
    """Raise if client rotation disagrees with train_eval_contract.json."""
    c = contract if contract is not None else load_train_eval_contract()
    orient = str(c.get("image_orientation", "raw")).strip().lower()
    if orient in RAW_ORIENTATIONS and rotate_images:
        raise RuntimeError(
            f"train_eval_contract image_orientation={orient!r} requires rotate_images=False "
            "(training frames are robosuite raw). Pass --no-rotate_images / omit --rotate_images."
        )
    if orient in ROT180_ORIENTATIONS and not rotate_images:
        raise RuntimeError(
            f"train_eval_contract image_orientation={orient!r} requires rotate_images=True. "
            "Pass --rotate_images."
        )


def add_rotate_images_cli(parser: Any) -> None:
    """`--rotate_images/--no-rotate_images` default False; keep `--no_rotate_images` alias."""
    import argparse

    parser.add_argument(
        "--rotate_images",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="180 deg flip of agentview + wrist. Default false (raw training frames).",
    )
    parser.add_argument(
        "--no_rotate_images",
        action="store_true",
        help="Deprecated alias forcing rotate_images=False.",
    )


def rotate_images_from_args(args: Any) -> bool:
    if getattr(args, "no_rotate_images", False):
        return False
    return bool(getattr(args, "rotate_images", False))


def setup_client_render_env(
    backend: str | None = None,
    libero_home: str | None = None,
    libero_config: str | None = None,
    device: str = "0",
    n_workers: int | None = None,
) -> str:
    """Configure LIBERO paths + the MuJoCo render backend (`egl` or `osmesa`).

    Returns the resolved backend name. `backend=None` follows `RENDER_BACKEND`
    then `MUJOCO_GL`, else prefers EGL and falls back to OSMesa.
    """
    from evaluation.LIBERO2.render_backend import setup_libero_paths, setup_render_env

    setup_libero_paths(libero_home=libero_home, libero_config=libero_config)
    return setup_render_env(backend, device=device, n_workers=n_workers, set_cuda_visible=True)


def setup_client_egl_env(
    libero_home: str | None = None,
    libero_config: str | None = None,
    egl_device: str = "0",
) -> None:
    """Back-compat wrapper. Honours RENDER_BACKEND/MUJOCO_GL=osmesa when set."""
    setup_client_render_env(
        backend=None,
        libero_home=libero_home,
        libero_config=libero_config,
        device=egl_device,
    )
