"""MuJoCo offscreen render backend selection for LIBERO / LIBERO-plus eval.

Two backends, selected by `RENDER_BACKEND` (env) or `--render_backend` (CLI):

  egl     NVIDIA GPU rendering. Fast, but the forked child's EGL context can
          abort (SIGABRT) after a few episodes, killing the task result.
  osmesa  Mesa llvmpipe CPU rendering. Bypasses NVIDIA EGL entirely, so the
          EGL SIGABRT class of failure cannot occur. Slower, CPU-hungry.

`auto` prefers egl and falls back to osmesa when EGL is unusable.

Probing never issues a GL call and never imports mujoco: the parent process
must stay GL-free or `eglCreateContext` breaks in the forked child (eval3 B10).
Availability is therefore decided from library/vendor-file presence only.

llvmpipe sizes its rasteriser thread pool from the *host* core count, which in
a cgroup-limited container (here: 224 visible cores, 64-core quota) oversubscribes
badly. `setup_render_env` pins `LP_NUM_THREADS` to a quota-aware value instead.
"""
from __future__ import annotations

import ctypes.util
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

EGL = "egl"
OSMESA = "osmesa"
AUTO = "auto"
BACKENDS = (EGL, OSMESA)

DEFAULT_CLIENT_VENV = Path("/B/VENV/libero_plus_client")
DEFAULT_LIBERO_HOME = "/home/a26113/DATA/LIBERO-plus"
NVIDIA_EGL_LIB = Path("/usr/local/nvidia/lib64/libEGL_nvidia.so.0")

# llvmpipe threads per rendering process. LIBERO renders 2 small (256x256)
# cameras per step, which does not scale past a handful of threads; the cap
# exists to stop N parallel workers from each spawning host-core-count threads.
DEFAULT_LP_THREADS_PER_PROC = 4
MAX_LP_THREADS = 8


def effective_cpu_count() -> int:
    """Usable CPUs: min(cgroup quota, affinity mask). `nproc` ignores quota."""
    candidates: list[int] = []
    try:
        candidates.append(len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        candidates.append(os.cpu_count() or 1)

    quota_v2 = Path("/sys/fs/cgroup/cpu.max")
    if quota_v2.is_file():
        try:
            quota_s, period_s = quota_v2.read_text().split()[:2]
            if quota_s != "max":
                candidates.append(max(1, int(float(quota_s) / float(period_s))))
        except (ValueError, OSError):
            pass
    else:
        q = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
        p = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
        if q.is_file() and p.is_file():
            try:
                quota, period = int(q.read_text()), int(p.read_text())
                if quota > 0 and period > 0:
                    candidates.append(max(1, quota // period))
            except (ValueError, OSError):
                pass

    return max(1, min(candidates))


def osmesa_library() -> str | None:
    """Path/soname of libOSMesa, or None when Mesa offscreen is not installed."""
    found = ctypes.util.find_library("OSMesa")
    if found:
        return found
    for cand in (
        "/lib/x86_64-linux-gnu/libOSMesa.so.8",
        "/usr/lib/x86_64-linux-gnu/libOSMesa.so.8",
    ):
        if Path(cand).exists():
            return cand
    return None


def osmesa_available() -> bool:
    return osmesa_library() is not None


def egl_vendor_dir(client_venv: Path | str | None = None) -> Path:
    venv = Path(client_venv or os.environ.get("CLIENT_VENV", DEFAULT_CLIENT_VENV))
    return venv / "egl_vendor.d"


def egl_available(client_venv: Path | str | None = None) -> bool:
    """NVIDIA EGL ICD present. Does not prove a context can be created."""
    vendor = egl_vendor_dir(client_venv)
    has_vendor = vendor.is_dir() and any(vendor.glob("*.json"))
    return bool(NVIDIA_EGL_LIB.exists() and has_vendor)


def resolve_backend(requested: str | None = None, *, client_venv: Path | str | None = None) -> str:
    """Normalise a backend request to `egl` or `osmesa`.

    Precedence: explicit argument > `RENDER_BACKEND` > `MUJOCO_GL` > `auto`.
    """
    raw = requested or os.environ.get("RENDER_BACKEND") or os.environ.get("MUJOCO_GL") or AUTO
    name = str(raw).strip().lower()

    if name in BACKENDS:
        if name == OSMESA and not osmesa_available():
            raise RuntimeError(
                "RENDER_BACKEND=osmesa but libOSMesa is missing. "
                "Install it with: sudo apt-get install -y libosmesa6"
            )
        return name

    if name not in (AUTO, "", "none"):
        raise ValueError(f"Unsupported render backend {raw!r}; expected one of {BACKENDS + (AUTO,)}")

    if egl_available(client_venv):
        return EGL
    if osmesa_available():
        return OSMESA
    raise RuntimeError(
        "No usable render backend: NVIDIA EGL ICD not found and libOSMesa not installed."
    )


def n_workers_default(n_workers: int | None = None) -> int:
    """Concurrent rendering processes; `RENDER_N_WORKERS` lets the shell declare it."""
    if n_workers is not None:
        return max(1, int(n_workers))
    try:
        return max(1, int(os.environ.get("RENDER_N_WORKERS", "1")))
    except ValueError:
        return 1


def lp_num_threads(n_workers: int | None = None) -> int:
    """llvmpipe threads per process, sized to the cgroup quota."""
    budget = max(1, effective_cpu_count() // n_workers_default(n_workers))
    return max(1, min(DEFAULT_LP_THREADS_PER_PROC, budget, MAX_LP_THREADS))


def setup_render_env(
    backend: str | None = None,
    *,
    device: str = "0",
    n_workers: int | None = None,
    client_venv: Path | str | None = None,
    set_cuda_visible: bool = False,
) -> str:
    """Configure process env for the chosen backend. Returns the backend name.

    Safe to call before forking: sets env vars only, issues no GL call.
    """
    name = resolve_backend(backend, client_venv=client_venv)

    os.environ["MUJOCO_GL"] = name
    os.environ["PYOPENGL_PLATFORM"] = name
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if name == EGL:
        os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", device)
        vendor = egl_vendor_dir(client_venv)
        if vendor.is_dir():
            os.environ.setdefault("__EGL_VENDOR_LIBRARY_DIRS", str(vendor))
        if set_cuda_visible:
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", device)
    else:
        # Strip EGL-only knobs so a stale MUJOCO_EGL_DEVICE_ID cannot make
        # mujoco probe EGL devices while MUJOCO_GL says osmesa.
        for key in ("MUJOCO_EGL_DEVICE_ID", "__EGL_VENDOR_LIBRARY_DIRS"):
            os.environ.pop(key, None)
        os.environ.setdefault("GALLIUM_DRIVER", "llvmpipe")
        os.environ.setdefault("LP_NUM_THREADS", str(lp_num_threads(n_workers)))

    return name


def describe(backend: str | None = None, *, n_workers: int | None = None) -> dict[str, Any]:
    """JSON-serialisable backend snapshot, for logs and healthchecks."""
    name = resolve_backend(backend)
    info: dict[str, Any] = {
        "backend": name,
        "mujoco_gl": os.environ.get("MUJOCO_GL"),
        "pyopengl_platform": os.environ.get("PYOPENGL_PLATFORM"),
        "effective_cpus": effective_cpu_count(),
        "egl_available": egl_available(),
        "osmesa_available": osmesa_available(),
        "osmesa_library": osmesa_library(),
        "gpu_free": name == OSMESA,
    }
    if name == EGL:
        info["egl_device_id"] = os.environ.get("MUJOCO_EGL_DEVICE_ID")
        info["egl_vendor_dirs"] = os.environ.get("__EGL_VENDOR_LIBRARY_DIRS")
        info["sigabrt_risk"] = "yes (NVIDIA EGL context teardown)"
    else:
        info["lp_num_threads"] = os.environ.get("LP_NUM_THREADS", str(lp_num_threads(n_workers)))
        info["gallium_driver"] = os.environ.get("GALLIUM_DRIVER")
        info["sigabrt_risk"] = "no (no EGL context)"
    return info


def add_render_backend_cli(parser: Any) -> None:
    """`--render_backend {auto,egl,osmesa}`; default follows RENDER_BACKEND env."""
    parser.add_argument(
        "--render_backend",
        type=str,
        default=os.environ.get("RENDER_BACKEND", AUTO),
        choices=list(BACKENDS + (AUTO,)),
        help="MuJoCo offscreen backend. osmesa = CPU, immune to EGL SIGABRT.",
    )


def setup_libero_paths(libero_home: str | None = None, libero_config: str | None = None) -> str:
    """LIBERO_HOME / LIBERO_CONFIG_PATH plus sys.path, independent of backend."""
    home = libero_home or os.environ.get("LIBERO_HOME", DEFAULT_LIBERO_HOME)
    os.environ.setdefault("LIBERO_HOME", home)
    if libero_config:
        os.environ["LIBERO_CONFIG_PATH"] = libero_config
    for p in (home, str(REPO_ROOT)):
        if p not in sys.path:
            sys.path.insert(0, p)
    return home


def main() -> int:
    """`python -m evaluation.LIBERO2.render_backend [--render_backend X]` -> JSON."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Report the resolved render backend")
    add_render_backend_cli(parser)
    parser.add_argument("--n_workers", type=int, default=None)
    parser.add_argument("--setup", action="store_true", help="Apply env before reporting")
    args = parser.parse_args()

    if args.setup:
        setup_render_env(args.render_backend, n_workers=args.n_workers)
    print(json.dumps(describe(args.render_backend, n_workers=args.n_workers), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
