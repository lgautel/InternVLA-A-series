"""Patch missing observation.keypoint_3d (56D) norm stats into a LeRobot v3.0 dataset.

Usage:
    source /B/VENV/itnvla15rbt20/bin/activate
    python b/s/Frk2/ds/patch_v30_keypoint_stats.py \
        --dataset /home/a26113/b/Dta/plug_into_socket_franka3_15hz_lerobot_4d_v30
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow `python b/s/Frk2/ds/patch_v30_keypoint_stats.py` from repo root.
_DS_DIR = Path(__file__).resolve().parent
if str(_DS_DIR) not in sys.path:
    sys.path.insert(0, str(_DS_DIR))

from kpt_stats import DEFAULT_DATASET, PatchResult, patch_dataset  # noqa: E402

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="LeRobot v3.0 dataset root containing meta/stats.json",
    )
    p.add_argument("--dry-run", action="store_true", help="Compute stats but do not write files")
    p.add_argument("--no-backup", action="store_true", help="Do not copy meta files before writing")
    p.add_argument(
        "--skip-if-present",
        action="store_true",
        help="Exit successfully if stats.json already has observation.keypoint_3d",
    )
    return p.parse_args()


def _print_result(result: PatchResult) -> None:
    g = result.global_stats
    logger.info(
        "patched %s: frames=%d episodes=%d kpt_count=%s mean[:3]=%s backup=%s dry_run=%s",
        result.dataset,
        result.n_frames,
        result.n_episodes,
        g["count"],
        [round(x, 6) for x in g["mean"][:3]],
        result.backup_dir,
        result.dry_run,
    )


def main() -> int:
    args = parse_args()
    result = patch_dataset(
        args.dataset,
        dry_run=args.dry_run,
        backup=not args.no_backup,
        skip_if_present=args.skip_if_present,
    )
    _print_result(result)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 — CLI surface
        logger.error("%s: %s", type(exc).__name__, exc)
        raise SystemExit(1) from exc
