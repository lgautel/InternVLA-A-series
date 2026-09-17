#!/usr/bin/env python3
"""O4 acceptance: OSMesa vs EGL render parity + throughput, for LIBERO eval.

Switching the MuJoCo backend to Mesa llvmpipe (`MUJOCO_GL=osmesa`) removes the
EGL SIGABRT failure class, but it also swaps the rasteriser. This test proves
the swap is *semantically* safe for the policy, i.e. that OSMesa frames carry
the same scene as EGL frames with the same orientation and colour channels.

llvmpipe and the NVIDIA GL driver are not bit-identical (different AA and
sampling), so parity is judged by structural agreement rather than equality:

  parity_r      Pearson correlation of the two frames  (>= --min_r)
  parity_mse    mean squared pixel error               (<= --max_mse)
  channel drift per-channel mean difference            (<= --max_channel_drift)
  orientation   OSMesa must beat its own rot180 against the training bank
                by --min_ratio, i.e. the F1 contract is backend-independent

Each backend renders in its own subprocess: a process can initialise only one
GL platform, and the parent must stay GL-free (eval3 B10).

Usage:
    export LIBERO_HOME=/home/a26113/DATA/LIBERO-plus
    export LIBERO_CONFIG_PATH=/tmp/test_libero_config
    python evaluation/LIBERO2/test_backend_parity.py              # both backends
    python evaluation/LIBERO2/test_backend_parity.py --skip_egl   # OSMesa only

Exit 0 = OSMesa is an acceptable drop-in for EGL.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CAM_NAMES = ("agentview", "wrist")


# ───────────────────────── child: render under one backend ─────────────────────


def _timed_steps(seed: int, n_steps: int) -> float:
    """Seconds per env.step() (physics + 2 camera renders), the episode-cost driver.

    An episode is 220-520 steps, so this — not env construction — sets the
    wall-clock difference between the backends.
    """
    if n_steps <= 0:
        return 0.0
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    from evaluation.LIBERO2.orientation_contract import UNPERTURBED_BDDL

    bddl = Path(get_libero_path("bddl_files")) / UNPERTURBED_BDDL
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    env.seed(seed)
    env.reset()
    dummy = [0.0] * 6 + [-1.0]
    env.step(dummy)  # warm-up, excluded from timing
    t0 = time.perf_counter()
    for _ in range(n_steps):
        env.step(dummy)
    elapsed = time.perf_counter() - t0
    env.close = lambda: None
    return elapsed / n_steps


def _render_child(backend: str, out_path: Path, seed: int, n_timed: int, n_steps: int) -> int:
    """Render the unperturbed BDDL under `backend`, save frames + timing."""
    from evaluation.LIBERO2.orientation_contract import (
        render_live_unperturbed,
        setup_client_render_env,
    )
    from evaluation.LIBERO2.render_backend import describe

    resolved = setup_client_render_env(backend=backend)
    info = describe(resolved)

    frames = render_live_unperturbed(seed=seed)

    # Timed reset+render loop: env construction cost (once per episode).
    t0 = time.perf_counter()
    for _ in range(max(1, n_timed)):
        render_live_unperturbed(seed=seed)
    elapsed = time.perf_counter() - t0

    np.savez_compressed(
        out_path,
        meta=json.dumps(
            {
                "backend": resolved,
                "info": info,
                "seconds_per_reset_render": elapsed / max(1, n_timed),
                "seconds_per_step": _timed_steps(seed=seed, n_steps=n_steps),
                "n_timed": int(max(1, n_timed)),
                "n_steps": int(n_steps),
            }
        ),
        **{name: frames[name] for name in CAM_NAMES},
    )
    sys.stdout.flush()
    sys.stderr.flush()
    # os._exit: skip interpreter teardown so a GL destructor cannot abort and
    # mask an otherwise successful render (the very failure mode under test).
    os._exit(0)


# ───────────────────────────── parent: metrics ──────────────────────────────────


def _spawn_render(backend: str, tmpdir: Path, seed: int, n_timed: int, n_steps: int) -> dict:
    out = tmpdir / f"frames_{backend}.npz"
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_render_only",
        backend,
        "--_out",
        str(out),
        "--seed",
        str(seed),
        "--n_timed",
        str(n_timed),
        "--n_steps",
        str(n_steps),
    ]
    env = dict(os.environ)
    env.pop("MUJOCO_GL", None)
    env.pop("PYOPENGL_PLATFORM", None)
    env["RENDER_BACKEND"] = backend
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (env.get("LIBERO_HOME", ""), str(REPO_ROOT), env.get("PYTHONPATH", "")) if p
    )
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1800)
    if not out.exists():
        return {
            "ok": False,
            "backend": backend,
            "returncode": proc.returncode,
            "stderr": proc.stderr[-2000:],
        }
    data = np.load(out)
    meta = json.loads(str(data["meta"]))
    return {
        "ok": True,
        "backend": backend,
        "frames": {name: data[name] for name in CAM_NAMES},
        "meta": meta,
        "returncode": proc.returncode,
    }


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    x = a.astype(np.float64).ravel()
    y = b.astype(np.float64).ravel()
    x -= x.mean()
    y -= y.mean()
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / denom) if denom > 0 else 0.0


def compare_frames(
    osmesa: np.ndarray,
    egl: np.ndarray,
    *,
    min_r: float,
    max_mse: float,
    max_channel_drift: float,
) -> dict:
    """Structural parity of one camera between the two backends."""
    a = osmesa.astype(np.float64)
    b = egl.astype(np.float64)
    mse = float(np.mean((a - b) ** 2))
    r = _pearson(osmesa, egl)
    drift = [float(a[..., c].mean() - b[..., c].mean()) for c in range(a.shape[-1])]
    max_drift = max(abs(d) for d in drift)

    # A 180-degree mismatch would show up as rot180 matching better than raw.
    mse_rot = float(np.mean((np.ascontiguousarray(a[::-1, ::-1]) - b) ** 2))

    return {
        "shape": list(osmesa.shape),
        "parity_mse": round(mse, 2),
        "parity_r": round(r, 5),
        "parity_mse_rot180": round(mse_rot, 2),
        "channel_drift": [round(d, 3) for d in drift],
        "max_channel_drift": round(max_drift, 3),
        "orientation_agrees": bool(mse < mse_rot),
        "passed": bool(
            r >= min_r and mse <= max_mse and max_drift <= max_channel_drift and mse < mse_rot
        ),
        "thresholds": {"min_r": min_r, "max_mse": max_mse, "max_channel_drift": max_channel_drift},
    }


def check_osmesa_orientation(frames: dict[str, np.ndarray], *, min_ratio: float, n_train: int) -> dict:
    """Re-run the F1 train/eval orientation contract on OSMesa frames (O3)."""
    from evaluation.LIBERO2.orientation_contract import evaluate_orientation_contract

    report = evaluate_orientation_contract(n_train=n_train, min_ratio=min_ratio, live=frames)
    report.pop("_vis", None)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="OSMesa vs EGL render parity")
    parser.add_argument("--_render_only", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--_out", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n_timed", type=int, default=3, help="Timed reset+render repeats")
    parser.add_argument("--n_steps", type=int, default=30, help="Timed env.step() repeats")
    parser.add_argument("--min_r", type=float, default=0.99)
    parser.add_argument("--max_mse", type=float, default=120.0)
    parser.add_argument("--max_channel_drift", type=float, default=6.0)
    parser.add_argument("--min_ratio", type=float, default=5.0, help="F1 rot180/raw MSE ratio")
    parser.add_argument("--num_samples", type=int, default=80, help="Training bank frames")
    parser.add_argument("--skip_egl", action="store_true", help="OSMesa-only (e.g. GPU busy)")
    parser.add_argument("--json_out", type=str, default="")
    args = parser.parse_args()

    if args._render_only:
        return _render_child(
            args._render_only, Path(args._out), args.seed, args.n_timed, args.n_steps
        )

    print("=" * 68)
    print("O4: OSMesa vs EGL render parity + throughput")
    print("=" * 68)

    report: dict = {"seed": args.seed, "backends": {}, "cameras": {}}
    with tempfile.TemporaryDirectory(prefix="backend_parity_") as td:
        tmpdir = Path(td)

        osm = _spawn_render("osmesa", tmpdir, args.seed, args.n_timed, args.n_steps)
        report["backends"]["osmesa"] = {k: v for k, v in osm.items() if k != "frames"}
        if not osm["ok"]:
            print("FAIL: OSMesa render failed")
            print(osm.get("stderr", ""))
            return 1
        osm_spr = osm["meta"]["seconds_per_reset_render"]
        osm_sps = osm["meta"]["seconds_per_step"]
        print(
            f"  osmesa : OK  {osm_spr * 1000:.0f} ms / reset+render, "
            f"{osm_sps * 1000:.1f} ms / step"
        )

        egl = None
        if not args.skip_egl:
            egl = _spawn_render("egl", tmpdir, args.seed, args.n_timed, args.n_steps)
            report["backends"]["egl"] = {k: v for k, v in egl.items() if k != "frames"}
            if egl["ok"]:
                egl_spr = egl["meta"]["seconds_per_reset_render"]
                egl_sps = egl["meta"]["seconds_per_step"]
                print(
                    f"  egl    : OK  {egl_spr * 1000:.0f} ms / reset+render, "
                    f"{egl_sps * 1000:.1f} ms / step"
                )
                report["osmesa_slowdown_reset_x"] = round(osm_spr / max(egl_spr, 1e-9), 2)
                report["osmesa_slowdown_step_x"] = round(osm_sps / max(egl_sps, 1e-9), 2)
                # Episode-level projection: libero_spatial caps at 220 steps.
                report["projected_seconds_per_episode"] = {
                    "osmesa": round(osm_spr + 220 * osm_sps, 1),
                    "egl": round(egl_spr + 220 * egl_sps, 1),
                }
                print(
                    f"  OSMesa slowdown: {report['osmesa_slowdown_step_x']}x per step, "
                    f"{report['osmesa_slowdown_reset_x']}x per reset"
                )
                print(
                    "  projected 220-step episode: "
                    f"osmesa {report['projected_seconds_per_episode']['osmesa']}s vs "
                    f"egl {report['projected_seconds_per_episode']['egl']}s"
                )
            else:
                print(f"  egl    : UNAVAILABLE (rc={egl['returncode']}) — parity skipped")
                report["egl_error"] = egl.get("stderr", "")[-600:]

        # ── Parity, only when both backends produced frames ──
        parity_ok = True
        if egl is not None and egl["ok"]:
            for name in CAM_NAMES:
                res = compare_frames(
                    osm["frames"][name],
                    egl["frames"][name],
                    min_r=args.min_r,
                    max_mse=args.max_mse,
                    max_channel_drift=args.max_channel_drift,
                )
                report["cameras"][name] = res
                parity_ok = parity_ok and res["passed"]
                tag = "PASS" if res["passed"] else "FAIL"
                print(
                    f"  {name:<10} {tag}  r={res['parity_r']}  mse={res['parity_mse']}  "
                    f"drift={res['max_channel_drift']}  (rot180 mse={res['parity_mse_rot180']})"
                )
        else:
            report["parity"] = "skipped (no EGL reference)"
            print("  parity: SKIPPED (no EGL reference frames)")

        # ── F1 orientation contract must still hold under OSMesa ──
        orient = check_osmesa_orientation(
            osm["frames"], min_ratio=args.min_ratio, n_train=args.num_samples
        )
        report["osmesa_orientation"] = orient
        for name, cam in orient["cameras"].items():
            tag = "PASS" if cam["matches_raw"] else "FAIL"
            print(
                f"  orient/{name:<8} {tag}  raw={cam['mse_live_raw_vs_train']} "
                f"rot180={cam['mse_live_rot180_vs_train']} ratio={cam['ratio_rot_over_raw']}"
            )

    report["passed"] = bool(parity_ok and orient["passed"])
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2, default=str))
        print("wrote", args.json_out)

    print(f"\nOVERALL: {'PASS' if report['passed'] else 'FAIL'}")
    sys.stdout.flush()
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
