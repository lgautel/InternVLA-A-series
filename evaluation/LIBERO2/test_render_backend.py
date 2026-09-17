#!/usr/bin/env python3
"""O1/O2 acceptance: render-backend plumbing, offline (no GL, no LIBERO import).

Guards the EGL -> OSMesa escape hatch ("plan B"): every switch point must be a
single configuration knob, and selecting `osmesa` must leave no EGL state
behind that could drag MuJoCo back onto the NVIDIA stack.

Checks:
  O1a  render_backend.py exports the expected API
  O1b  resolve_backend precedence: arg > RENDER_BACKEND > MUJOCO_GL > auto
  O1c  osmesa requested but libOSMesa missing -> actionable RuntimeError
  O2a  setup_render_env(osmesa) sets osmesa and strips EGL-only vars
  O2b  setup_render_env(egl) sets the EGL device + vendor dirs
  O2c  effective_cpu_count respects the cgroup quota, not just nproc
  O2d  lp_num_threads shrinks with worker count and stays bounded
  O3a  both eval scripts resolve the backend instead of hardcoding egl
  O3b  both shells expose RENDER_BACKEND with an osmesa branch
  O3c  orientation_contract keeps a back-compat EGL wrapper + new entry point
  O3d  live test entry points accept --render_backend
  O3e  all new/edited modules parse

Usage (any Python 3.10+, no venv needed):
    python evaluation/LIBERO2/test_render_backend.py
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

L2 = REPO_ROOT / "evaluation/LIBERO2"
LP2 = REPO_ROOT / "evaluation/LIBERO-plus2"

RENDER_ENV_KEYS = (
    "MUJOCO_GL",
    "PYOPENGL_PLATFORM",
    "MUJOCO_EGL_DEVICE_ID",
    "__EGL_VENDOR_LIBRARY_DIRS",
    "RENDER_BACKEND",
    "RENDER_N_WORKERS",
    "LP_NUM_THREADS",
    "GALLIUM_DRIVER",
    "CUDA_VISIBLE_DEVICES",
)


@contextmanager
def clean_render_env(**overrides: str):
    """Run a check with render env vars controlled, then restore the real ones."""
    saved = {k: os.environ.get(k) for k in RENDER_ENV_KEYS}
    for k in RENDER_ENV_KEYS:
        os.environ.pop(k, None)
    os.environ.update(overrides)
    try:
        yield
    finally:
        for k in RENDER_ENV_KEYS:
            os.environ.pop(k, None)
            if saved[k] is not None:
                os.environ[k] = saved[k]


def _report(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  {label}: {'PASS' if ok else 'FAIL'}{('  ' + detail) if detail else ''}")
    return ok


def test_o1a_api() -> bool:
    import evaluation.LIBERO2.render_backend as rb

    expected = [
        "EGL",
        "OSMESA",
        "AUTO",
        "BACKENDS",
        "add_render_backend_cli",
        "describe",
        "effective_cpu_count",
        "egl_available",
        "lp_num_threads",
        "osmesa_available",
        "osmesa_library",
        "resolve_backend",
        "setup_libero_paths",
        "setup_render_env",
    ]
    missing = [name for name in expected if not hasattr(rb, name)]
    ok = _report("exports complete", not missing, f"missing={missing}" if missing else "")
    ok &= _report("backends are (egl, osmesa)", rb.BACKENDS == ("egl", "osmesa"), str(rb.BACKENDS))
    return ok


def test_o1b_precedence() -> bool:
    from evaluation.LIBERO2.render_backend import osmesa_available, resolve_backend

    have_osmesa = osmesa_available()
    ok = True

    with clean_render_env(RENDER_BACKEND="egl"):
        ok &= _report("explicit arg beats env", resolve_backend("osmesa" if have_osmesa else "egl")
                      == ("osmesa" if have_osmesa else "egl"))
    with clean_render_env(RENDER_BACKEND="egl", MUJOCO_GL="osmesa"):
        ok &= _report("RENDER_BACKEND beats MUJOCO_GL", resolve_backend() == "egl")
    with clean_render_env(MUJOCO_GL="osmesa"):
        if have_osmesa:
            ok &= _report("MUJOCO_GL honoured", resolve_backend() == "osmesa")
        else:
            ok &= _report("MUJOCO_GL honoured", True, "skipped (no libOSMesa)")
    with clean_render_env():
        resolved = resolve_backend()
        ok &= _report("auto resolves to a real backend", resolved in ("egl", "osmesa"), resolved)
    with clean_render_env():
        bad = False
        try:
            resolve_backend("glx")
        except ValueError:
            bad = True
        ok &= _report("unsupported backend rejected", bad)
    return ok


def test_o1c_missing_osmesa_message() -> bool:
    """A hard `osmesa` request must fail loudly with the install command."""
    import evaluation.LIBERO2.render_backend as rb

    saved = rb.osmesa_library
    rb.osmesa_library = lambda: None
    try:
        with clean_render_env():
            try:
                rb.resolve_backend("osmesa")
            except RuntimeError as e:
                msg = str(e)
                return _report(
                    "actionable error when libOSMesa absent",
                    "libosmesa6" in msg and "apt-get" in msg,
                    msg[:70],
                )
            return _report("actionable error when libOSMesa absent", False, "no error raised")
    finally:
        rb.osmesa_library = saved


def test_o2a_osmesa_env() -> bool:
    from evaluation.LIBERO2.render_backend import osmesa_available, setup_render_env

    if not osmesa_available():
        return _report("osmesa env setup", True, "skipped (no libOSMesa installed)")

    with clean_render_env(MUJOCO_EGL_DEVICE_ID="3", __EGL_VENDOR_LIBRARY_DIRS="/stale"):
        resolved = setup_render_env("osmesa")
        ok = _report("resolves to osmesa", resolved == "osmesa", resolved)
        ok &= _report("MUJOCO_GL=osmesa", os.environ.get("MUJOCO_GL") == "osmesa")
        ok &= _report("PYOPENGL_PLATFORM=osmesa", os.environ.get("PYOPENGL_PLATFORM") == "osmesa")
        ok &= _report(
            "stale MUJOCO_EGL_DEVICE_ID stripped", "MUJOCO_EGL_DEVICE_ID" not in os.environ
        )
        ok &= _report(
            "stale __EGL_VENDOR_LIBRARY_DIRS stripped",
            "__EGL_VENDOR_LIBRARY_DIRS" not in os.environ,
        )
        threads = int(os.environ.get("LP_NUM_THREADS", "0"))
        ok &= _report("LP_NUM_THREADS bounded", 1 <= threads <= 8, f"={threads}")
        ok &= _report("GALLIUM_DRIVER=llvmpipe", os.environ.get("GALLIUM_DRIVER") == "llvmpipe")
    return ok


def test_o2b_egl_env() -> bool:
    from evaluation.LIBERO2.render_backend import egl_available, setup_render_env

    if not egl_available():
        return _report("egl env setup", True, "skipped (no NVIDIA EGL ICD)")

    with clean_render_env():
        resolved = setup_render_env("egl", device="2")
        ok = _report("resolves to egl", resolved == "egl", resolved)
        ok &= _report("MUJOCO_GL=egl", os.environ.get("MUJOCO_GL") == "egl")
        ok &= _report("MUJOCO_EGL_DEVICE_ID set", os.environ.get("MUJOCO_EGL_DEVICE_ID") == "2")
        ok &= _report(
            "vendor dirs set", "egl_vendor.d" in (os.environ.get("__EGL_VENDOR_LIBRARY_DIRS") or "")
        )
        ok &= _report("LP_NUM_THREADS untouched", "LP_NUM_THREADS" not in os.environ)
    return ok


def test_o2c_cpu_quota() -> bool:
    from evaluation.LIBERO2.render_backend import effective_cpu_count

    eff = effective_cpu_count()
    nproc = os.cpu_count() or 1
    ok = _report("effective_cpus >= 1", eff >= 1, f"eff={eff} nproc={nproc}")

    quota_file = Path("/sys/fs/cgroup/cpu.max")
    if quota_file.is_file():
        parts = quota_file.read_text().split()
        if parts and parts[0] != "max":
            quota = int(float(parts[0]) / float(parts[1]))
            ok &= _report("respects cgroup quota", eff <= max(quota, 1), f"quota={quota} eff={eff}")
        else:
            ok &= _report("respects cgroup quota", True, "no quota set")
    else:
        ok &= _report("respects cgroup quota", True, "cgroup v2 cpu.max absent")
    return ok


def test_o2d_thread_budget() -> bool:
    from evaluation.LIBERO2.render_backend import effective_cpu_count, lp_num_threads

    with clean_render_env():
        one = lp_num_threads(1)
        many = lp_num_threads(max(effective_cpu_count(), 64))
        ok = _report("1 worker gets >= 1 thread", one >= 1, f"={one}")
        ok &= _report("threads capped at 8", one <= 8, f"={one}")
        ok &= _report("many workers shrink the pool", many <= one, f"many={many} one={one}")
    with clean_render_env(RENDER_N_WORKERS="1000"):
        ok &= _report("RENDER_N_WORKERS honoured", lp_num_threads() == 1, f"={lp_num_threads()}")
    return ok


def test_o3a_eval_scripts() -> bool:
    ok = True
    for path in (L2 / "eval_libero_std.py", LP2 / "eval_libero_plus.py"):
        src = path.read_text()
        name = path.name
        ok &= _report(f"{name} calls setup_render_env", "setup_render_env(" in src)
        ok &= _report(
            f"{name} drops hardcoded egl default",
            'os.environ.setdefault("MUJOCO_GL", "egl")' not in src,
        )
        ok &= _report(f"{name} reads RENDER_BACKEND", 'RENDER_BACKEND' in src)
    return ok


def test_o3b_shells() -> bool:
    ok = True
    shells = (
        L2 / "run_eval_libero_std_venv.sh",
        LP2 / "run_eval_libero_plus_venv.sh",
    )
    for path in shells:
        src = path.read_text()
        name = path.name
        ok &= _report(
            f"{name} RENDER_BACKEND config", 'RENDER_BACKEND="${RENDER_BACKEND:-auto}"' in src
        )
        ok &= _report(f"{name} osmesa branch", 'MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa' in src)
        ok &= _report(
            f"{name} unsets EGL vars for osmesa",
            "unset MUJOCO_EGL_DEVICE_ID __EGL_VENDOR_LIBRARY_DIRS" in src,
        )
        ok &= _report(f"{name} echoes the backend", "Render backend" in src)
    # plus2 runs NUM_GPUS clients at once, so it must declare the worker count.
    plus2 = (LP2 / "run_eval_libero_plus_venv.sh").read_text()
    ok &= _report(
        "plus2 exports RENDER_N_WORKERS", 'RENDER_N_WORKERS="${NUM_GPUS}"' in plus2
    )
    return ok


def test_o3c_orientation_contract() -> bool:
    src = (L2 / "orientation_contract.py").read_text()
    ok = _report("setup_client_render_env added", "def setup_client_render_env(" in src)
    ok &= _report("setup_client_egl_env kept (back-compat)", "def setup_client_egl_env(" in src)
    ok &= _report(
        "wrapper delegates to render backend", "setup_client_render_env(" in src.split(
            "def setup_client_egl_env("
        )[-1]
    )
    from evaluation.LIBERO2.orientation_contract import setup_client_render_env

    ok &= _report("importable", callable(setup_client_render_env))
    return ok


def test_o3d_live_entrypoints() -> bool:
    ok = True
    for path, flag in (
        (L2 / "test_live_orientation.py", "add_render_backend_cli"),
        (L2 / "test_backend_parity.py", "--_render_only"),
        (L2 / "test_osmesa_soak.py", "--render_backend"),
    ):
        exists = path.exists()
        ok &= _report(f"{path.name} exists", exists)
        if exists:
            ok &= _report(f"{path.name} backend-selectable", flag in path.read_text())
    return ok


def test_o3e_syntax() -> bool:
    ok = True
    for path in (
        L2 / "render_backend.py",
        L2 / "orientation_contract.py",
        L2 / "eval_libero_std.py",
        L2 / "test_live_orientation.py",
        L2 / "test_backend_parity.py",
        L2 / "test_osmesa_soak.py",
        L2 / "test_render_backend.py",
        LP2 / "eval_libero_plus.py",
    ):
        try:
            ast.parse(path.read_text())
            ok &= _report(f"{path.name} parses", True)
        except SyntaxError as e:
            ok &= _report(f"{path.name} parses", False, str(e))
    return ok


CHECKS = (
    ("O1a exports", test_o1a_api),
    ("O1b resolve precedence", test_o1b_precedence),
    ("O1c missing-osmesa error", test_o1c_missing_osmesa_message),
    ("O2a osmesa env", test_o2a_osmesa_env),
    ("O2b egl env", test_o2b_egl_env),
    ("O2c cpu quota", test_o2c_cpu_quota),
    ("O2d thread budget", test_o2d_thread_budget),
    ("O3a eval scripts", test_o3a_eval_scripts),
    ("O3b shell scripts", test_o3b_shells),
    ("O3c orientation contract", test_o3c_orientation_contract),
    ("O3d live entry points", test_o3d_live_entrypoints),
    ("O3e syntax", test_o3e_syntax),
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline render-backend plumbing checks")
    parser.add_argument("--json_out", type=str, default="")
    args = parser.parse_args()

    print("=" * 68)
    print("O1/O2/O3: render backend plumbing (offline)")
    print("=" * 68)

    results: dict[str, bool] = {}
    for label, fn in CHECKS:
        print(f"\n[{label}]")
        try:
            results[label] = bool(fn())
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            results[label] = False

    from evaluation.LIBERO2.render_backend import describe

    print("\n[environment]")
    print(json.dumps(describe(), indent=2))

    all_pass = all(results.values())
    print("\n" + "=" * 68)
    for label, ok in results.items():
        print(f"  {label}: {'PASS' if ok else 'FAIL'}")
    print(f"\nOVERALL: {'PASS' if all_pass else 'FAIL'}")
    print("=" * 68)

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps({"passed": all_pass, "checks": results}, indent=2)
        )
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
