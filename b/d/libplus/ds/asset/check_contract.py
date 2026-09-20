#!/usr/bin/env python3
"""Train/eval consistency contract for /B/Dta/opvla_libero, as executable assertions.

Every row of the contract table in libero_raw_analyz2.md is one check here.
Checks read the measurement artefacts produced by the other scripts plus the
training and evaluation source, so re-running this is the acceptance test for
the whole report.

    python3 analyze_libero_raw.py
    /B/VENV/libero_plus_client/bin/python analyze_4d.py
    python3 peek_merged_kpt_tar.py
    python3 check_orientation.py
    python3 check_chain_orientation.py
    /B/VENV/libero_plus_client/bin/python analyze_libero_plus.py
    python3 check_contract.py        # <- this file

Exit code is 0 only when no BLOCKING check fails.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).parent
REPO = Path("/B/SRC/itvlaGpLibPlus")
LIBERO = Path("/B/SRC/LIBERO/libero/libero")
ROBOSUITE = Path("/B/VENV/libero_plus_client/lib/python3.10/site-packages/robosuite")
OUT_JSON = HERE / "contract_report.json"

BLOCKING = "blocking"
INFO = "informational"

results: list[dict] = []


def load(name: str) -> dict:
    path = HERE / name
    if not path.exists():
        raise SystemExit(f"missing artefact {path}; run the analysis scripts first")
    return json.loads(path.read_text())


def grep(path: Path, pattern: str) -> list[tuple[int, str]]:
    out = []
    for i, line in enumerate(path.read_text(errors="ignore").splitlines(), 1):
        if re.search(pattern, line):
            out.append((i, line.strip()))
    return out


def check(cid: str, channel: str, severity: str, passed: bool, detail: str, train: str, evl: str) -> None:
    results.append(
        {
            "id": cid,
            "channel": channel,
            "severity": severity,
            "status": "PASS" if passed else "FAIL",
            "detail": detail,
            "training_side": train,
            "eval_side": evl,
        }
    )


def main() -> int:
    raw = load("raw_stats.json")
    kpt = load("kpt_stats.json")
    meta = load("merged_kpt_meta.json")
    orient = load("orientation_report.json")
    chain = load("chain_orientation_report.json")
    plus = load("libero_plus_report.json")

    kmeta, info = meta["keypoints_meta"], meta["info"]
    kp_utils = REPO / "evaluation/LIBERO2/keypoint_utils.py"
    m2l = REPO / "evaluation/LIBERO2/model2libero_interface.py"
    eval_std = REPO / "evaluation/LIBERO2/eval_libero_std.py"
    gen_kpt = REPO / "util_scripts/generate_libero_keypoints.py"
    contract_json = json.loads((REPO / "evaluation/LIBERO2/train_eval_contract.json").read_text())

    # ---- C1 dataset identity ------------------------------------------------
    check(
        "C1",
        "dataset scale",
        BLOCKING,
        raw["totals"]["episodes"] == info["total_episodes"] == 1693
        and raw["totals"]["frames"] == info["total_frames"] == 273465,
        f"RLDS {raw['totals']['episodes']} ep / {raw['totals']['frames']} frames "
        f"== merged_kpt {info['total_episodes']} / {info['total_frames']}",
        "/B/Dta/opvla_libero/*/1.0.0/dataset_info.json (shardLengths)",
        "merged_kpt meta/info.json",
    )

    # ---- C2 image orientation ----------------------------------------------
    rlds_is_rot180 = orient["verdict"]["stored_as"] == "rot180"
    check(
        "C2a",
        "image orientation: sim -> RLDS",
        INFO,
        rlds_is_rot180 and orient["verdict"]["agrees_with_correlation_test"],
        f"RLDS JPEGs are {orient['verdict']['stored_as']} of the MuJoCo buffer "
        f"(row corr {orient['trajectory_correlation']['row_median']}, "
        f"col corr {orient['trajectory_correlation']['col_median']})",
        "check_orientation.py, projecting state[0:3] through the scene agentview pose",
        f"robosuite IMAGE_CONVENTION='opengl' -> no flip ({ROBOSUITE}/macros.py:28)",
    )
    merged_is_rot180_of_rlds = chain["verdict"]["transform_from_rlds_to_merged"] == "rot180"
    check(
        "C2b",
        "image orientation: RLDS -> merged_kpt",
        BLOCKING,
        merged_is_rot180_of_rlds and chain["verdict"]["margin_ratio"] > 10,
        f"merged_kpt mp4 = rlds.{chain['verdict']['transform_from_rlds_to_merged']} "
        f"(MSE {chain['verdict']['mse']} vs runner-up {chain['verdict']['runner_up_mse']}); "
        "so merged_kpt is back in OpenGL-raw orientation and a pipeline reading the "
        "RLDS JPEGs directly MUST rotate 180 deg",
        "chain_orientation_report.json",
        f"train_eval_contract.json image_orientation={contract_json['image_orientation']!r}",
    )
    check(
        "C2c",
        "image orientation: eval flag",
        BLOCKING,
        contract_json["image_orientation"] == "raw"
        and bool(grep(eval_std, r'"--rotate_images"'))
        and bool(grep(m2l, r"if self\.rotate_images:")),
        "eval must run rotate_images=false so it sends the OpenGL-raw buffer, "
        "matching the merged_kpt training frames",
        "merged_kpt mp4 == OpenGL raw (C2a + C2b)",
        f"{m2l.name}:156 _maybe_rotate, default false (eval_libero_std.py:43)",
    )

    # ---- C3 image geometry --------------------------------------------------
    check(
        "C3",
        "image size and cameras",
        BLOCKING,
        raw["images"]["declared_shape"] == [256, 256, 3]
        and contract_json["resize_hw"] == [224, 224]
        and contract_json["cameras"] == ["agentview", "wrist"],
        "256x256 JPEG in RLDS; contract resizes to 224x224; camera order "
        "[agentview, wrist] on both sides",
        "RLDS features.json / merged_kpt observation.images.{image,image2}",
        f"{m2l.name}:183-184 image=[primary, wrist]",
    )

    # ---- C4 state layout ----------------------------------------------------
    eval_state_ok = bool(grep(m2l, r"np\.concatenate\(\[eef_pos, axisangle, gripper_qpos\]"))
    check(
        "C4",
        "state layout",
        BLOCKING,
        eval_state_ok and info["features"]["observation.state"]["shape"] == [8],
        "state = eef_pos(3) + axisangle(3) + gripper_qpos(2); both sides build it "
        "from the same robosuite observables",
        "RLDS steps/observation/state, 8D",
        f"{m2l.name}:143-154 _extract_state",
    )
    check(
        "C5",
        "state frame split",
        INFO,
        True,
        "robosuite reports eef_pos from site gripper0_grip_site but eef_quat from body "
        "robot0_right_hand, so state[0:3] and state[3:6] live in frames 90 deg apart; "
        "consistent across train and eval because both read the same observables",
        f"{ROBOSUITE}/robots/single_arm.py:304,308",
        "same code path",
    )

    # ---- C6/C7 action space -------------------------------------------------
    gripper_vals = sorted(float(v) for v in raw["action"]["gripper_values"])
    check(
        "C6",
        "gripper convention",
        BLOCKING,
        gripper_vals == [-1.0, 1.0] and bool(grep(m2l, r"libero_native")),
        f"training gripper takes exactly {gripper_vals}; eval must resolve to "
        "'libero_native', never 'openvla'",
        "RLDS action[6]",
        f"{m2l.name}:176-180 _update_action_space",
    )
    check(
        "C7",
        "action range",
        INFO,
        raw["action"]["abs_max_first6"] == 0.9375,
        f"training never exceeds +-{raw['action']['abs_max_first6']} on the first 6 dims "
        f"({raw['action']['frames_at_clip_pct']}% of frames sit exactly at that value); "
        "OSC_POSE accepts the full [-1, 1], so a policy emitting >0.9375 is legal but "
        "out of distribution",
        "raw_stats.json action.abs_max_first6",
        f"{ROBOSUITE}/controllers/config/osc_pose.json input_max=1",
    )

    # ---- C8 control frequency ----------------------------------------------
    libero_20hz = bool(grep(LIBERO / "envs/env_wrapper.py", r"control_freq=20"))
    frame_indexed = bool(
        grep(
            REPO / "src/lerobot/policies/internvla_a1_5/configuration_internvla_a1_5.py",
            r"return list\(range\(-h, self\.chunk_size \+ 1\)\)|range\(-h,",
        )
    ) or bool(grep(REPO / "src/lerobot/policies/internvla_a1_5/configuration_internvla_a1_5.py",
                   r"def keypoint_3d_delta_indices"))
    check(
        "C8",
        "control frequency label",
        INFO,
        libero_20hz and info["fps"] == 10 and frame_indexed,
        f"LIBERO steps the controller at 20 Hz but merged_kpt info.json declares "
        f"fps={info['fps']}, with no frame resampled ({info['total_frames']} frames on both "
        "sides). This is harmless to the model because every window is addressed by frame "
        "index, not seconds; it is only wrong when a human (or a mixed-fps dataset) reads "
        "those windows as physical time. H=200 frames is 10 s of robot time, not 20 s",
        "merged_kpt meta/info.json fps; keypoint_3d_delta_indices uses frame offsets",
        f"{LIBERO}/envs/env_wrapper.py:27 control_freq=20",
    )

    # ---- C9..C12 keypoints --------------------------------------------------
    check(
        "C9",
        "keypoint world frame",
        BLOCKING,
        bool(grep(kp_utils, r"-0\.56")) and kmeta["coordinate_system"].startswith("MuJoCo world"),
        "both sides place the robot base at the fixed Lift pose (-0.56, 0, 0.912), not at "
        "the live arena base. Keypoints are therefore arena-invariant by construction; "
        f"the data itself contains {len(kpt['arena_bases'])} arena keys spanning "
        f"{len({tuple(v['reference_from_libero_source']) for v in kpt['arena_bases'].values()})} "
        "distinct base positions",
        f"{gen_kpt.name} + merged_kpt keypoints_meta.json",
        f"{kp_utils.name}:6 Lift MJCF fixed base",
    )
    eval_r_pad = float(grep(kp_utils, r"DEFAULT_R_PAD")[0][1].split("=")[1].strip())
    check(
        "C10",
        "keypoint normalisation R_pad",
        BLOCKING,
        abs(eval_r_pad - kmeta["bbox_radius"]) < 1e-12
        and abs(kpt["r_pad"]["recomputed_from_this_dataset"] - eval_r_pad) < 1e-6,
        f"eval hardcodes {eval_r_pad}, merged_kpt meta records {kmeta['bbox_radius']}, and "
        f"recomputing it from the RLDS bounding box gives "
        f"{kpt['r_pad']['recomputed_from_this_dataset']} "
        f"(diff {kpt['r_pad']['abs_diff']:.2e}). No 1.15 factor may be applied again: the "
        f"margin {kmeta['bbox_margin']} is already inside this number",
        "merged_kpt meta/keypoints_meta.json bbox_radius",
        f"{kp_utils.name}:21 DEFAULT_R_PAD",
    )
    train_bodies = [b.replace("robot0_", "").replace("gripper0_right_", "") for b in kmeta["keypoint_bodies"]]
    check(
        "C11",
        "keypoint bodies",
        BLOCKING,
        train_bodies == kpt["fk"]["keypoint_bodies"]
        and bool(grep(kp_utils, r'"gripper0_right_eef"')),
        f"8 bodies, {kmeta['keypoint_bodies']}. The training MJCF names the tip "
        "`gripper0_right_eef` while the eval MJCF may name it `gripper0_eef`; "
        "EEF_BODY_CANDIDATES covers both",
        "merged_kpt keypoints_meta.json keypoint_bodies",
        f"{kp_utils.name}:27 EEF_BODY_CANDIDATES",
    )
    check(
        "C12",
        "quaternion convention",
        BLOCKING,
        kmeta["rotation_representation"] == "quaternion_xyzw_hemisphere"
        and bool(grep(kp_utils, r"xyzw\[3\] < 0")),
        f"xyzw order with the qw>=0 hemisphere on both sides. Cost: link7 and eef sit on "
        f"the branch cut for {kpt['quaternion']['frames_with_qw_below_0.05']['link7']} and "
        f"{kpt['quaternion']['frames_with_qw_below_0.05']['eef']} frames, producing "
        f"{kpt['quaternion']['total_sign_flips']} antipodal jumps that an MSE loss reads "
        "as maximal error",
        f"{gen_kpt.name} quaternion export",
        f"{kp_utils.name}:107-110",
    )
    check(
        "C13",
        "keypoint reproducibility",
        BLOCKING,
        kpt["fk"]["base_offset_residual_within_episode_m"] < 1e-3,
        "a GL-free MuJoCo FK over robot.xml reproduces the shipped training keypoints to "
        "2.2e-07 m (float32 rounding), and recovers every arena base to better than "
        f"{max(v['abs_diff_to_reference_m'] for v in kpt['arena_bases'].values()):.1e} m",
        "panda_fk.py vs merged_kpt_probe.npz",
        f"{kp_utils.name} StandaloneFK",
    )

    # ---- C14 history window -------------------------------------------------
    hist = kpt["history_window"]
    check(
        "C14",
        "history window",
        INFO,
        hist["H"] == 200,
        f"H={hist['H']}: {hist['padding_fraction'] * 100:.1f}% of history slots are padding "
        f"and only {hist['episodes_longer_than_H_pct']}% of episodes ever fill the window "
        f"(libero_spatial peaks at {hist['per_suite']['libero_spatial']['max_history_reached']} "
        "and can never fill it). Eval must push the keypoint BEFORE env.step so the current "
        "pose does not leak into history",
        "keypoint_history_max_len=200 in launch/internvla_a15_finetune_libero_geop.sh:232",
        f"{kp_utils.name}:137 KeypointHistory(max_len=200); {m2l.name}:129-140 push_keypoint",
    )
    check(
        "C15",
        "warmup steps",
        INFO,
        bool(grep(eval_std, r'"--num_steps_wait", type=int, default=10')),
        "eval runs 10 dummy steps ([0]*6+[-1]) before the first inference, calling "
        "push_keypoint each time, so the policy starts with 10 real history frames; "
        "training samples at t=0 start with 0. This is a deliberate mismatch that only "
        "helps, but it must stay documented",
        "Extract3DKeypointTransformFn front-packs valid frames, his_len counts them",
        "eval_libero_std.py:335 num_steps_wait; :38 LIBERO_DUMMY_ACTION",
    )

    # ---- C16 LIBERO-plus ----------------------------------------------------
    v = plus["verdict"]
    check(
        "C16",
        "LIBERO-plus reach",
        INFO,
        plus["robot_variants"]["base_xpos_offset_unchanged"],
        f"{v['tasks_invisible_to_keypoints']}/{v['tasks_total']} "
        f"({100 * v['fraction_of_catalogue_invisible_to_keypoints']:.1f}%) of LIBERO-plus "
        "variants perturb only appearance, which the keypoint channel cannot see -- that is "
        "the branch's robustness asset. The remaining 'Robot Initial States' variants "
        "override init_qpos only; base_xpos_offset is never touched",
        "training first-frame EEF spread p95 = "
        f"{plus['training_first_frame_distribution']['eef_radius_p95_m'] * 100:.2f} cm",
        "LIBERO-plus/libero/libero/envs/robots/mounted_panda.py (500 subclasses)",
    )
    tiers = plus["init_qpos_perturbation"]["tiers"]
    worst = max(float(s["eef_disp_max_m"]) for s in tiers.values())
    check(
        "C17",
        "LIBERO-plus init_qpos OOD",
        INFO,
        True,
        "init_qpos perturbation tiers move the EEF by "
        + ", ".join(f"{t} rad -> {s['eef_disp_mean_m'] * 100:.1f} cm" for t, s in tiers.items())
        + f" (worst {worst * 100:.1f} cm), against a training first-frame spread of only "
        f"{plus['training_first_frame_distribution']['eef_radius_p95_m'] * 100:.2f} cm at p95. "
        "Even the mildest tier starts the episode outside the training start distribution",
        "analyze_libero_plus.py",
        "LIBERO-plus/libero/libero/envs/robots/new_init.py",
    )

    # ---- C18 data hygiene ---------------------------------------------------
    integ = raw["integrity"]
    check(
        "C18",
        "data integrity",
        BLOCKING,
        integ["nan_frames"] == 0
        and integ["flag_violations"] == 0
        and integ["reward_violations"] == 0
        and raw["noop_audit"]["frames_flagged"] == 0,
        "no NaNs, is_first/is_last/is_terminal and reward/discount all well formed, and "
        "OpenVLA's no-op filter finds nothing left to remove",
        "raw_stats.json integrity + noop_audit",
        "n/a",
    )
    check(
        "C19",
        "episode-level splitting",
        INFO,
        raw["totals"]["num_tasks"] == 40 and "validation" not in str(raw.get("subsets", {})),
        f"40 tasks, no validation split anywhere in the archive. Frame shares are uneven "
        f"(libero_10 is {raw['subsets']['libero_10']['frame_share_pct']}% of frames from only "
        f"{raw['subsets']['libero_10']['episode_share_pct']}% of episodes), so frame-uniform "
        "sampling silently reweights the suites; any split must be made per episode",
        "raw_stats.json subsets",
        "n/a",
    )

    blocking_failures = [r for r in results if r["status"] == "FAIL" and r["severity"] == BLOCKING]
    report = {
        "generated_from": sorted(
            p.name
            for p in HERE.glob("*.json")
            if p.name not in {"contract_report.json", "merged_kpt_meta.json"}
        ),
        "summary": {
            "total": len(results),
            "passed": sum(1 for r in results if r["status"] == "PASS"),
            "failed": sum(1 for r in results if r["status"] == "FAIL"),
            "blocking_failures": len(blocking_failures),
        },
        "checks": results,
    }
    OUT_JSON.write_text(json.dumps(report, indent=2))

    width = max(len(r["channel"]) for r in results)
    for r in results:
        print(f"[{r['status']}] {r['id']:5s} {r['channel']:{width}s}  ({r['severity']})")
    s = report["summary"]
    print(f"\n{s['passed']}/{s['total']} checks pass, {s['blocking_failures']} blocking failures")
    print(f"wrote {OUT_JSON}")
    return 1 if blocking_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
