#!/usr/bin/env python3
"""What LIBERO-plus perturbs, and how much of it the keypoint branch can see.

LIBERO-plus keeps the LIBERO action/observation interface but multiplies each
base task into thousands of perturbed variants. For the 4D keypoint branch the
question is narrow: keypoints are a pure function of `qpos`, so only the
perturbations that move the robot itself can reach them.

This script
  * classifies the perturbation catalogue from task_classification.json,
  * confirms from source that the robot-state variants override `init_qpos`
    and nothing else (in particular not `base_xpos_offset`),
  * runs FK on all 500 perturbed initial poses and converts the joint-space
    perturbation into keypoint displacement, in metres and in the normalized
    units the policy actually consumes,
  * compares that displacement against the spread of first-frame keypoints in
    the training set, which is the distribution the history window was fit to.

Requires mujoco:
    /B/VENV/libero_plus_client/bin/python analyze_libero_plus.py
"""

from __future__ import annotations

import ast
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from panda_fk import ARENA_BASE_XPOS, DEFAULT_R_PAD, LIFT_BASE_XPOS, PandaFK, arena_of
from rlds_reader import SUBSETS, iter_episodes

HERE = Path(__file__).parent
PLUS_ROOT = Path("/B/SRC/LIBERO-plus/libero/libero")
MOUNTED = PLUS_ROOT / "envs/robots/mounted_panda.py"
GROUND = PLUS_ROOT / "envs/robots/on_the_ground_panda.py"
CLASSIFICATION = PLUS_ROOT / "benchmark/task_classification.json"
OUT_JSON = HERE / "libero_plus_report.json"
OUT_NPZ = HERE / "libero_plus_arrays.npz"

BODY_NAMES = [f"link{i}" for i in range(1, 8)] + ["eef"]
NOMINAL_QPOS = np.array([0.0, -1.61037389e-01, 0.0, -2.44459747e00, 0.0, 2.22675220e00, np.pi / 4])
GRIPPER_OPEN = np.array([0.04, -0.04])

# new_init.py draws a unit-norm direction and scales it, in blocks of 100.
DESIGNED_TIERS = {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4, 5: 0.5}


def parse_init_qpos(path: Path, class_prefix: str) -> dict[int, np.ndarray]:
    """{variant index: init_qpos} for the generated robot subclasses."""
    text = path.read_text()
    pattern = re.compile(
        rf"^class {class_prefix}(\d+)\({class_prefix}\):.*?def init_qpos\(self\):\s*"
        r"return np\.array\((\[[^\]]*\])\)",
        re.MULTILINE | re.DOTALL,
    )
    return {int(m.group(1)): np.asarray(ast.literal_eval(m.group(2)), float) for m in pattern.finditer(text)}


def overrides_other_properties(path: Path, class_prefix: str) -> list[str]:
    """Properties the generated subclasses override besides `init_qpos`."""
    text = path.read_text()
    blocks = re.split(rf"^class {class_prefix}\d+\(", text, flags=re.MULTILINE)[1:]
    found: set[str] = set()
    for block in blocks:
        found.update(re.findall(r"def (\w+)\(self\)", block))
    return sorted(found - {"init_qpos"})


def classify_catalogue() -> dict:
    data = json.loads(CLASSIFICATION.read_text())
    per_suite = {}
    overall: Counter[str] = Counter()
    difficulty: Counter[int] = Counter()
    for suite, entries in data.items():
        counts = Counter(e["category"] for e in entries)
        per_suite[suite] = {"total_tasks": len(entries), "by_category": dict(counts)}
        overall.update(counts)
        difficulty.update(e.get("difficulty_level") for e in entries)
    return {
        "per_suite": per_suite,
        "overall_by_category": dict(overall.most_common()),
        "total_tasks": sum(v["total_tasks"] for v in per_suite.values()),
        "difficulty_levels": {str(k): v for k, v in sorted(difficulty.items(), key=lambda kv: str(kv[0]))},
    }


# Which perturbation families can reach the 4D keypoint input. Keypoints come
# from FK on the live `qpos`, so anything that only edits the scene's
# appearance is invisible to that channel while still hitting the image tokens.
CHANNEL_REACH = {
    "Background Textures": {"images": True, "keypoints": False, "mechanism": "scene XML texture swap"},
    "Camera Viewpoints": {
        "images": True,
        "keypoints": False,
        "mechanism": "mujoco_arena.set_camera on agentview only (problems/*.py `_setup_camera`)",
    },
    "Language Instructions": {"images": False, "keypoints": False, "mechanism": "BDDL language_instruction"},
    "Light Conditions": {"images": True, "keypoints": False, "mechanism": "scene XML light edits"},
    "Objects Layout": {
        "images": True,
        "keypoints": False,
        "mechanism": "object placement / init_states; robot joints untouched",
    },
    "Robot Initial States": {
        "images": True,
        "keypoints": True,
        "mechanism": "MountedPanda{i}/OnTheGroundPanda{i} override init_qpos",
    },
    "Sensor Noise": {
        "images": True,
        "keypoints": False,
        "mechanism": "agentview_image post-processing in envs/env_wrapper.py",
    },
}


def main() -> int:
    fk = PandaFK()

    mounted = parse_init_qpos(MOUNTED, "MountedPanda")
    ground = parse_init_qpos(GROUND, "OnTheGroundPanda")
    extra_mounted = overrides_other_properties(MOUNTED, "MountedPanda")
    extra_ground = overrides_other_properties(GROUND, "OnTheGroundPanda")

    nominal_kpt = fk.keypoints(
        np.concatenate([NOMINAL_QPOS, GRIPPER_OPEN]), base_xpos=LIFT_BASE_XPOS, r_pad=DEFAULT_R_PAD
    )

    rows = []
    for idx, qpos in sorted(mounted.items()):
        kpt = fk.keypoints(
            np.concatenate([qpos, GRIPPER_OPEN]), base_xpos=LIFT_BASE_XPOS, r_pad=DEFAULT_R_PAD
        )
        disp_norm = np.linalg.norm(kpt[:, :3] - nominal_kpt[:, :3], axis=1)
        rows.append(
            {
                "idx": idx,
                "dq_norm": float(np.linalg.norm(qpos - NOMINAL_QPOS)),
                "tier": DESIGNED_TIERS.get((idx - 1) // 100 + 1, np.nan),
                "eef_disp_m": float(disp_norm[7] * DEFAULT_R_PAD),
                "max_body_disp_m": float(disp_norm.max() * DEFAULT_R_PAD),
                "mean_body_disp_norm": float(disp_norm.mean()),
            }
        )

    by_tier: dict[float, list[dict]] = defaultdict(list)
    for r in rows:
        by_tier[r["tier"]].append(r)
    tier_summary = {
        f"{tier:.1f}": {
            "variants": len(v),
            "dq_norm_mean_rad": round(float(np.mean([r["dq_norm"] for r in v])), 4),
            "eef_disp_mean_m": round(float(np.mean([r["eef_disp_m"] for r in v])), 4),
            "eef_disp_max_m": round(float(np.max([r["eef_disp_m"] for r in v])), 4),
            "max_body_disp_max_m": round(float(np.max([r["max_body_disp_m"] for r in v])), 4),
            "eef_disp_mean_normalised": round(
                float(np.mean([r["eef_disp_m"] for r in v]) / DEFAULT_R_PAD), 5
            ),
        }
        for tier, v in sorted(by_tier.items())
    }

    # --- how far outside the training distribution does that land?
    first_frames = []
    for subset in SUBSETS:
        for ep in iter_episodes(subset):
            base = ARENA_BASE_XPOS[arena_of(subset, ep.task_name)]
            del base  # keypoints always use the Lift frame, never the arena base
            first_frames.append(
                fk.keypoints(ep.qpos9()[0], base_xpos=LIFT_BASE_XPOS, r_pad=DEFAULT_R_PAD)[:, :3]
            )
    first_frames = np.stack(first_frames)
    train_eef = first_frames[:, 7, :] * DEFAULT_R_PAD
    train_centre = train_eef.mean(axis=0)
    train_spread = np.linalg.norm(train_eef - train_centre, axis=1)

    all_eef_disp = np.asarray([r["eef_disp_m"] for r in rows])
    report = {
        "catalogue": classify_catalogue(),
        "channel_reach": CHANNEL_REACH,
        "robot_variants": {
            "mounted_panda_variants": len(mounted),
            "on_the_ground_panda_variants": len(ground),
            "properties_overridden_besides_init_qpos": {
                "MountedPanda*": extra_mounted,
                "OnTheGroundPanda*": extra_ground,
            },
            "base_xpos_offset_unchanged": not (
                "base_xpos_offset" in extra_mounted or "base_xpos_offset" in extra_ground
            ),
            "nominal_init_qpos": NOMINAL_QPOS.round(6).tolist(),
            "generator": "LIBERO-plus/libero/libero/envs/robots/new_init.py (unit direction * tier)",
        },
        "init_qpos_perturbation": {
            "tiers": tier_summary,
            "all_variants": {
                "eef_disp_mean_m": round(float(all_eef_disp.mean()), 4),
                "eef_disp_p95_m": round(float(np.percentile(all_eef_disp, 95)), 4),
                "eef_disp_max_m": round(float(all_eef_disp.max()), 4),
            },
        },
        "training_first_frame_distribution": {
            "episodes": int(len(first_frames)),
            "eef_centre_m": train_centre.round(4).tolist(),
            "eef_radius_mean_m": round(float(train_spread.mean()), 4),
            "eef_radius_p95_m": round(float(np.percentile(train_spread, 95)), 4),
            "eef_radius_max_m": round(float(train_spread.max()), 4),
            "note": (
                "spread of the very first keypoint frame across the 1693 training episodes; "
                "the history window at eval starts from the perturbed pose instead"
            ),
        },
        "verdict": {
            "keypoint_visible_categories": [k for k, v in CHANNEL_REACH.items() if v["keypoints"]],
            "keypoint_invisible_categories": [k for k, v in CHANNEL_REACH.items() if not v["keypoints"]],
            "fraction_of_catalogue_invisible_to_keypoints": None,  # filled below
        },
    }

    overall = report["catalogue"]["overall_by_category"]
    total = sum(overall.values())
    invisible = sum(v for k, v in overall.items() if not CHANNEL_REACH.get(k, {}).get("keypoints"))
    report["verdict"]["fraction_of_catalogue_invisible_to_keypoints"] = round(invisible / total, 4)
    report["verdict"]["tasks_invisible_to_keypoints"] = invisible
    report["verdict"]["tasks_total"] = total

    OUT_JSON.write_text(json.dumps(report, indent=2))
    np.savez_compressed(
        OUT_NPZ,
        dq_norm=np.asarray([r["dq_norm"] for r in rows]),
        eef_disp_m=all_eef_disp,
        max_body_disp_m=np.asarray([r["max_body_disp_m"] for r in rows]),
        tier=np.asarray([r["tier"] for r in rows]),
        train_eef_radius=train_spread,
    )

    print(f"mounted variants: {len(mounted)}  ground variants: {len(ground)}")
    print(f"extra overridden properties: mounted={extra_mounted} ground={extra_ground}")
    print(f"base_xpos_offset unchanged: {report['robot_variants']['base_xpos_offset_unchanged']}")
    print("\ninit_qpos tiers -> EEF displacement:")
    for tier, s in tier_summary.items():
        print(
            f"  |dq|={tier} rad ({s['variants']:3d} variants): "
            f"mean {s['eef_disp_mean_m'] * 100:5.2f} cm, max {s['eef_disp_max_m'] * 100:5.2f} cm"
        )
    t = report["training_first_frame_distribution"]
    print(f"\ntraining first-frame EEF spread: mean {t['eef_radius_mean_m'] * 100:.2f} cm, "
          f"p95 {t['eef_radius_p95_m'] * 100:.2f} cm")
    v = report["verdict"]
    print(
        f"\n{v['tasks_invisible_to_keypoints']}/{v['tasks_total']} "
        f"({100 * v['fraction_of_catalogue_invisible_to_keypoints']:.1f}%) of LIBERO-plus tasks "
        f"perturb something the keypoint channel cannot see"
    )
    print(f"wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
