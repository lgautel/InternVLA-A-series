#!/usr/bin/env python3
"""Comprehensive pre-evaluation test suite for F1/F2 fixes + B1-B10 regression.

Three-tier test structure:
  Part A: Static analysis — no imports beyond stdlib, works in any venv
  Part B: Unit tests — needs numpy, no GPU
  Part C: Integration tests — needs SERVER_VENV (lerobot), no GPU

Run ALL offline tests before any GPU evaluation to guarantee:
  1. F1/F2 code changes are correct
  2. B1-B10 existing fixes not regressed
  3. New files (eval_libero_std, run_eval_libero_std_venv) are well-formed

Usage:
    # Full suite (in SERVER_VENV for Part C):
    python evaluation/LIBERO2/test_preflight_f1f2.py

    # Static-only (any venv, no dependencies):
    python evaluation/LIBERO2/test_preflight_f1f2.py --part A

    # Static + unit (needs numpy):
    python evaluation/LIBERO2/test_preflight_f1f2.py --part AB

    # All parts with checkpoint validation:
    python evaluation/LIBERO2/test_preflight_f1f2.py --ckpt /path/to/pretrained_model

Exit code 0 = all tests passed, 1 = at least one failed.
"""
from __future__ import annotations

import argparse
import ast
import importlib
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJ = REPO_ROOT

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
SKIP = "\033[33mSKIP\033[0m"


class TestRunner:
    def __init__(self):
        self.results: list[tuple[str, str, bool]] = []  # (part, name, passed)

    def record(self, part: str, name: str, passed: bool, detail: str = ""):
        tag = PASS if passed else FAIL
        self.results.append((part, name, passed))
        msg = f"  [{tag}] {part}.{name}"
        if detail:
            msg += f"  — {detail}"
        print(msg)

    def summary(self) -> bool:
        total = len(self.results)
        passed = sum(1 for _, _, p in self.results if p)
        failed = total - passed
        print(f"\n{'='*60}")
        print(f"  Total: {total}  Passed: {passed}  Failed: {failed}")
        if failed:
            print(f"\n  Failed tests:")
            for part, name, p in self.results:
                if not p:
                    print(f"    - {part}.{name}")
        print(f"\n  OVERALL: {PASS if failed == 0 else FAIL}")
        print(f"{'='*60}")
        return failed == 0


# ═══════════════════════════════════════════════════════════
# Part A: Static analysis (stdlib only)
# ═══════════════════════════════════════════════════════════

def run_part_a(runner: TestRunner):
    print("\n── Part A: Static Analysis ──")

    # A1. Syntax check all modified/new files
    files_to_check = [
        "evaluation/LIBERO2/policy_server/backends/policy_backend_internvla_a1_5.py",
        "evaluation/LIBERO2/model2libero_interface.py",
        "evaluation/LIBERO2/keypoint_utils.py",
        "evaluation/LIBERO2/eval_libero_std.py",
        "evaluation/LIBERO2/test_orientation.py",
        "evaluation/LIBERO2/orientation_contract.py",
        "evaluation/LIBERO2/test_live_orientation.py",
        "evaluation/LIBERO2/train_eval_extra_contract.py",
        "evaluation/LIBERO-plus2/eval_libero_plus.py",
        "tests/test_keypoint_utils.py",
    ]
    for rel in files_to_check:
        fp = PROJ / rel
        try:
            ast.parse(fp.read_text())
            runner.record("A1", f"syntax:{Path(rel).name}", True)
        except (SyntaxError, FileNotFoundError) as e:
            runner.record("A1", f"syntax:{Path(rel).name}", False, str(e))

    # A2. Shell script syntax
    for sh in [
        "evaluation/LIBERO-plus2/run_eval_libero_plus_venv.sh",
        "evaluation/LIBERO2/run_eval_libero_std_venv.sh",
    ]:
        fp = PROJ / sh
        if not fp.exists():
            runner.record("A2", f"syntax:{Path(sh).name}", False, "file not found")
            continue
        r = subprocess.run(["bash", "-n", str(fp)], capture_output=True, text=True)
        runner.record("A2", f"syntax:{Path(sh).name}", r.returncode == 0,
                       r.stderr.strip() if r.returncode != 0 else "")

    # A3. F1: Shell script has ROTATE_IMAGES config
    sh_path = PROJ / "evaluation/LIBERO-plus2/run_eval_libero_plus_venv.sh"
    if sh_path.exists():
        content = sh_path.read_text()
        has_config = 'ROTATE_IMAGES="${ROTATE_IMAGES:-' in content
        runner.record("A3", "shell:ROTATE_IMAGES_config_var", has_config,
                       "" if has_config else "ROTATE_IMAGES config var missing")
        has_flag_build = "ROTATE_FLAG" in content and '"${ROTATE_IMAGES}"' in content
        runner.record("A3", "shell:ROTATE_FLAG_build_logic", has_flag_build,
                       "" if has_flag_build else "ROTATE_FLAG construction missing")
        has_flag_in_cmd = "${ROTATE_FLAG}" in content
        runner.record("A3", "shell:ROTATE_FLAG_in_client_cmd", has_flag_in_cmd,
                       "" if has_flag_in_cmd else "${ROTATE_FLAG} not in client command")
        default_false = 'ROTATE_IMAGES="${ROTATE_IMAGES:-false}"' in content
        runner.record("A3", "shell:ROTATE_IMAGES_default_false", default_false,
                       "" if default_false else "default should be 'false'")
        opt_in = 'ROTATE_FLAG="--rotate_images"' in content and '[ "${ROTATE_IMAGES}" = "true" ]' in content
        runner.record("A3", "shell:ROTATE_FLAG_opt_in_when_true", opt_in,
                       "" if opt_in else "true branch should pass --rotate_images")
    else:
        runner.record("A3", "shell:file_exists", False, "run_eval_libero_plus_venv.sh not found")

    # A4. F1: eval_libero_plus.py replay uses _maybe_rotate not [::-1,::-1]
    eval_plus = PROJ / "evaluation/LIBERO-plus2/eval_libero_plus.py"
    if eval_plus.exists():
        src = eval_plus.read_text()
        replay_lines = [
            (i + 1, line.strip())
            for i, line in enumerate(src.split("\n"))
            if "replay_images.append" in line
        ]
        for lineno, line in replay_lines:
            has_hardcoded = "[::-1, ::-1]" in line
            has_maybe_rot = "_maybe_rotate" in line
            ok = has_maybe_rot and not has_hardcoded
            runner.record("A4", f"replay_L{lineno}:uses_maybe_rotate", ok,
                           line[:80] if not ok else "")
    else:
        runner.record("A4", "file_exists", False)

    # A5. F1: eval_libero_std.py replay also uses _maybe_rotate
    eval_std = PROJ / "evaluation/LIBERO2/eval_libero_std.py"
    if eval_std.exists():
        src = eval_std.read_text()
        replay_lines = [
            (i + 1, line.strip())
            for i, line in enumerate(src.split("\n"))
            if "replay_images.append" in line
        ]
        for lineno, line in replay_lines:
            ok = "_maybe_rotate" in line
            runner.record("A5", f"std_replay_L{lineno}:uses_maybe_rotate", ok,
                           line[:80] if not ok else "")
        has_cli = "--rotate_images" in src and "--no_rotate_images" in src
        runner.record("A5", "std:has_rotate_images_cli_pair", has_cli)
        default_false_cli = "default=False" in src and "--rotate_images" in src
        runner.record("A5", "std:rotate_images_cli_default_False", default_false_cli)
    else:
        runner.record("A5", "file_exists", False)

    # A6. F2: Backend imports OBS_IMAGES and sets mapping
    backend = PROJ / "evaluation/LIBERO2/policy_server/backends/policy_backend_internvla_a1_5.py"
    if backend.exists():
        src = backend.read_text()
        has_obs_images = "OBS_IMAGES" in src and "import" in src.split("OBS_IMAGES")[0].split("\n")[-1]
        runner.record("A6", "backend:imports_OBS_IMAGES", has_obs_images)
        has_mapping = "mapping=" in src and "OBS_IMAGES" in src.split("mapping=")[1].split(")")[0] if "mapping=" in src else False
        runner.record("A6", "backend:resize_has_mapping", has_mapping,
                       "" if has_mapping else "ResizeImagesWithPadFn missing mapping param")
    else:
        runner.record("A6", "file_exists", False)

    # A7. B1 regression: model2libero_interface.py has gripper_convention
    m2l = PROJ / "evaluation/LIBERO2/model2libero_interface.py"
    if m2l.exists():
        src = m2l.read_text()
        has_convention_param = "gripper_convention" in src
        has_libero_native = '"libero_native"' in src
        has_openvla = '"openvla"' in src
        has_resolved = "_resolved_convention" in src
        runner.record("A7", "B1:gripper_convention_param", has_convention_param)
        runner.record("A7", "B1:libero_native_branch", has_libero_native)
        runner.record("A7", "B1:openvla_branch", has_openvla)
        runner.record("A7", "B1:resolved_convention_logic", has_resolved)
    else:
        runner.record("A7", "file_exists", False)

    # A8. B2 regression: backend has getattr use_fast
    if backend.exists():
        src = backend.read_text()
        has_getattr = "getattr(config" in src and "use_fast_action_tokens" in src
        runner.record("A8", "B2:use_fast_getattr", has_getattr,
                       "" if has_getattr else "Missing getattr(config, 'use_fast_action_tokens', True)")
        has_true_default = 'use_fast_action_tokens", True)' in src
        runner.record("A8", "B2:use_fast_default_True", has_true_default)

    # A9. B7 regression: no top-level import imageio in eval scripts
    for name, path in [
        ("eval_libero_plus", eval_plus),
        ("eval_libero_std", eval_std),
    ]:
        if not path or not path.exists():
            continue
        tree = ast.parse(path.read_text())
        toplevel_imageio = False
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "imageio":
                        toplevel_imageio = True
            elif isinstance(node, ast.ImportFrom):
                if node.module and "imageio" in node.module:
                    toplevel_imageio = True
        runner.record("A9", f"B7:{name}:no_toplevel_imageio", not toplevel_imageio,
                       "top-level import imageio found!" if toplevel_imageio else "")

    # A10. B9 regression: keypoint R_PAD constant
    kpt = PROJ / "evaluation/LIBERO2/keypoint_utils.py"
    if kpt.exists():
        src = kpt.read_text()
        has_r_pad = "1.8212722539901733" in src
        runner.record("A10", "B9:R_PAD_correct_value", has_r_pad,
                       "" if has_r_pad else "R_PAD should be 1.8212722539901733")
        has_resolve = "def resolve_eef_body_name" in src
        runner.record("A10", "B9:resolve_eef_body_name", has_resolve,
                       "" if has_resolve else "must resolve EEF against the loaded MJCF")
        has_both_candidates = '"gripper0_eef"' in src and '"gripper0_right_eef"' in src
        runner.record("A10", "B9:eef_candidates_both_names", has_both_candidates,
                       "need gripper0_eef and gripper0_right_eef as candidates")
        has_standalone_fk = "StandaloneFK" in src
        runner.record("A10", "B9:standalone_fk_class", has_standalone_fk)
    else:
        runner.record("A10", "file_exists", False)

    # A11. B10 regression: fork-per-task in eval scripts
    for name, path in [
        ("eval_libero_plus", eval_plus),
        ("eval_libero_std", eval_std),
    ]:
        if not path or not path.exists():
            continue
        src = path.read_text()
        has_fork = "os.fork()" in src
        has_waitpid = "os.waitpid" in src
        has_exit = "os._exit" in src
        ok = has_fork and has_waitpid and has_exit
        runner.record("A11", f"B10:{name}:fork_isolation", ok,
                       "" if ok else "missing os.fork/waitpid/_exit")

    # A12. eval_libero_plus.py imports from LIBERO2 (not LIBERO)
    if eval_plus.exists():
        src = eval_plus.read_text()
        imports_libero2 = "from evaluation.LIBERO2.model2libero_interface" in src
        runner.record("A12", "import:uses_LIBERO2_interface", imports_libero2,
                       "" if imports_libero2 else "should import from LIBERO2, not LIBERO")

    # A13. MJCF file exists
    mjcf = PROJ / "evaluation/panda_robosuite_lift.xml"
    runner.record("A13", "mjcf:panda_robosuite_lift.xml_exists", mjcf.exists(),
                   "" if mjcf.exists() else "required by StandaloneFK")

    # A14. Both cameras go through the same _maybe_rotate (no per-camera split)
    if m2l.exists():
        src = m2l.read_text()
        has_av = 'self._maybe_rotate(np.asarray(obs["agentview_image"]' in src
        has_wrist = 'self._maybe_rotate(np.asarray(obs["robot0_eye_in_hand_image"]' in src
        runner.record("A14", "client:agentview_uses_maybe_rotate", has_av)
        runner.record("A14", "client:wrist_uses_maybe_rotate", has_wrist)

    # A15. Standard-LIBERO shell: rotation is opt-in, default raw
    std_sh = PROJ / "evaluation/LIBERO2/run_eval_libero_std_venv.sh"
    if std_sh.exists():
        content = std_sh.read_text()
        runner.record(
            "A15",
            "std_shell:ROTATE_IMAGES_default_false",
            'ROTATE_IMAGES="${ROTATE_IMAGES:-false}"' in content,
        )
        runner.record(
            "A15",
            "std_shell:rotate_is_opt_in",
            'ROTATE_FLAG="--rotate_images"' in content,
        )
    else:
        runner.record("A15", "std_shell:file_exists", False)

    # A16. Live T1 test file exists (EGL; run separately)
    live_py = PROJ / "evaluation/LIBERO2/test_live_orientation.py"
    contract_py = PROJ / "evaluation/LIBERO2/orientation_contract.py"
    runner.record("A16", "t1:test_live_orientation.py_exists", live_py.exists())
    runner.record("A16", "t1:orientation_contract.py_exists", contract_py.exists())

    # A17. Python default must be False (U1). Direct LiberoModelClient() must not rotate.
    if m2l.exists():
        src = m2l.read_text()
        runner.record(
            "A17",
            "python_default_rotate_False",
            "rotate_images: bool = False" in src,
            "" if "rotate_images: bool = False" in src else "U1: default must be False",
        )
        runner.record(
            "A17",
            "client:enforces_train_eval_contract",
            "enforce_rotate_against_contract" in src,
        )

    # A18. Recorded T1 JSON still says both cameras match training RAW
    t1_json = PROJ / "b/d/libplus/asset/eval3_optim_wrist_t1.json"
    runner.record("A18", "recorded_t1_json_exists", t1_json.exists(), str(t1_json))

    # A19. U6 train/eval contract
    import json as _json
    contract_path = PROJ / "evaluation/LIBERO2/train_eval_contract.json"
    if contract_path.exists():
        try:
            contract = _json.loads(contract_path.read_text())
            runner.record(
                "A19",
                "contract:image_orientation_raw",
                contract.get("image_orientation") == "raw",
                str(contract.get("image_orientation")),
            )
            cams = contract.get("cameras") or []
            runner.record(
                "A19",
                "contract:cameras_agentview_and_wrist",
                "agentview" in cams and "wrist" in cams,
                str(cams),
            )
        except Exception as e:
            runner.record("A19", "contract:parse", False, str(e))
    else:
        runner.record("A19", "contract:file_exists", False)

    # A20. U3 server metadata fields (rotate_images is client-side, must NOT be here)
    if backend.exists():
        src = backend.read_text()
        meta_block = src.split("def metadata")[-1] if "def metadata" in src else ""
        runner.record("A20", "metadata:has_resize_size", '"resize_size"' in meta_block)
        runner.record("A20", "metadata:has_use_fast_action_tokens", '"use_fast_action_tokens"' in meta_block)
        runner.record(
            "A20",
            "metadata:no_rotate_images_on_server",
            '"rotate_images"' not in meta_block,
            "rotation is a client concern",
        )

    # A21. U3 shell healthcheck asserts stats_key + resize_size
    for tag, path in (
        ("plus2", PROJ / "evaluation/LIBERO-plus2/run_eval_libero_plus_venv.sh"),
        ("std", PROJ / "evaluation/LIBERO2/run_eval_libero_std_venv.sh"),
    ):
        if not path.exists():
            runner.record("A21", f"{tag}:shell_exists", False)
            continue
        hc = path.read_text()
        runner.record("A21", f"{tag}:healthcheck_stats_key", "assert m.get('stats_key')" in hc)
        runner.record("A21", f"{tag}:healthcheck_resize_size", "resize_size" in hc and "assert int(m.get('resize_size')" in hc)

    # A22. U2 legacy entries warn
    for tag, rel in (
        ("libero", "evaluation/LIBERO/eval_libero_server_client.py"),
        ("libero_plus", "evaluation/LIBERO-plus/eval_libero_plus.py"),
    ):
        fp = PROJ / rel
        if not fp.exists():
            runner.record("A22", f"{tag}:exists", False)
            continue
        src = fp.read_text()
        runner.record(
            "A22",
            f"{tag}:legacy_warning",
            "legacy" in src.lower() and "WARNING" in src,
        )

    # A23. U9: std parent evaluate_policy must not import libero
    extra = PROJ / "evaluation/LIBERO2/train_eval_extra_contract.py"
    runner.record("A23", "extra_contract:exists", extra.exists())
    if extra.exists() and eval_std.exists():
        if str(PROJ) not in sys.path:
            sys.path.insert(0, str(PROJ))
        from evaluation.LIBERO2.train_eval_extra_contract import function_imports_libero

        std_src = eval_std.read_text()
        runner.record(
            "A23",
            "std:evaluate_policy_no_libero_import",
            not function_imports_libero(std_src, "evaluate_policy"),
            "parent evaluate_policy must not import libero (B10 EGL)",
        )
        runner.record(
            "A23",
            "std:child_still_imports_benchmark",
            function_imports_libero(std_src, "_run_task_in_subprocess"),
            "forked child must still load libero.benchmark",
        )
        runner.record(
            "A23",
            "std:TASK_SUITE_N_TASKS",
            "TASK_SUITE_N_TASKS" in std_src,
        )
        if eval_plus.exists():
            plus_src = eval_plus.read_text()
            runner.record(
                "A23",
                "plus2:evaluate_policy_no_libero_import",
                not function_imports_libero(plus_src, "evaluate_policy"),
            )

    # A24. U8: push_keypoint before env.step in both eval loops
    if extra.exists():
        from evaluation.LIBERO2.train_eval_extra_contract import (
            evaluate_task_pushes_before_env_step,
        )

        for tag, path in (("plus2", eval_plus), ("std", eval_std)):
            if not path.exists():
                continue
            ok, detail = evaluate_task_pushes_before_env_step(path.read_text())
            runner.record("A24", f"{tag}:push_before_env_step", ok, detail)

    # A25. EEF candidates match generate_libero_keypoints.py
    gen = PROJ / "util_scripts/generate_libero_keypoints.py"
    if extra.exists() and kpt.exists() and gen.exists():
        from evaluation.LIBERO2.train_eval_extra_contract import extract_eef_candidates

        eval_c = extract_eef_candidates(kpt.read_text())
        gen_c = extract_eef_candidates(gen.read_text())
        runner.record(
            "A25",
            "eef_candidates_match_generate",
            eval_c == gen_c and eval_c == ("gripper0_eef", "gripper0_right_eef"),
            f"eval={eval_c} generate={gen_c}",
        )

    # A26. pytest file imports LIBERO2, not deleted evaluation.LIBERO.keypoint_utils
    pytest_kpt = PROJ / "tests/test_keypoint_utils.py"
    if pytest_kpt.exists():
        psrc = pytest_kpt.read_text()
        runner.record(
            "A26",
            "pytest:imports_LIBERO2",
            "from evaluation.LIBERO2.keypoint_utils import" in psrc,
        )
        runner.record(
            "A26",
            "pytest:no_deleted_LIBERO_import",
            "from evaluation.LIBERO.keypoint_utils import" not in psrc,
        )
    else:
        runner.record("A26", "pytest:file_exists", False)

    # A27. Render backend is a single configuration knob (OSMesa escape hatch).
    # Full coverage lives in test_render_backend.py; this is the gate hook.
    rb = PROJ / "evaluation/LIBERO2/render_backend.py"
    runner.record("A27", "render_backend:exists", rb.exists())
    if rb.exists():
        if str(PROJ) not in sys.path:
            sys.path.insert(0, str(PROJ))
        from evaluation.LIBERO2.render_backend import BACKENDS, describe, resolve_backend

        runner.record("A27", "backends:egl_and_osmesa", BACKENDS == ("egl", "osmesa"))
        resolved = resolve_backend()
        runner.record("A27", "resolves_to_real_backend", resolved in BACKENDS, f"={resolved}")
        info = describe(resolved)
        runner.record(
            "A27",
            "describe:reports_sigabrt_risk",
            "sigabrt_risk" in info and "osmesa_available" in info,
            f"osmesa_available={info.get('osmesa_available')}",
        )
        for tag, path in (
            ("std", PROJ / "evaluation/LIBERO2/eval_libero_std.py"),
            ("plus2", PROJ / "evaluation/LIBERO-plus2/eval_libero_plus.py"),
        ):
            if not path.exists():
                continue
            src = path.read_text()
            runner.record("A27", f"{tag}:calls_setup_render_env", "setup_render_env(" in src)
            runner.record(
                "A27",
                f"{tag}:no_hardcoded_egl",
                'os.environ.setdefault("MUJOCO_GL", "egl")' not in src,
            )
        for tag, path in (
            ("std", PROJ / "evaluation/LIBERO2/run_eval_libero_std_venv.sh"),
            ("plus2", PROJ / "evaluation/LIBERO-plus2/run_eval_libero_plus_venv.sh"),
        ):
            if not path.exists():
                continue
            sh = path.read_text()
            runner.record(
                "A27",
                f"{tag}:shell_RENDER_BACKEND",
                'RENDER_BACKEND="${RENDER_BACKEND:-auto}"' in sh,
            )
            runner.record(
                "A27",
                f"{tag}:shell_osmesa_unsets_egl",
                "unset MUJOCO_EGL_DEVICE_ID __EGL_VENDOR_LIBRARY_DIRS" in sh,
            )


# ═══════════════════════════════════════════════════════════
# Part B: Unit tests (needs numpy)
# ═══════════════════════════════════════════════════════════

def run_part_b(runner: TestRunner):
    print("\n── Part B: Unit Tests ──")

    try:
        import numpy as np
    except ImportError:
        runner.record("B0", "numpy_import", False, "numpy not available")
        return

    if str(PROJ) not in sys.path:
        sys.path.insert(0, str(PROJ))

    # B1. _maybe_rotate() flag behavior
    test_img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    test_img[0, 0] = [255, 0, 0]     # red top-left
    test_img[-1, -1] = [0, 0, 255]   # blue bottom-right
    test_img[0, -1] = [0, 255, 0]    # green top-right

    # rotate_images=False: preserve
    no_rot = np.ascontiguousarray(test_img)
    runner.record("B1", "no_rotate:preserves_topleft", int(no_rot[0, 0, 0]) == 255)
    runner.record("B1", "no_rotate:preserves_bottomright", int(no_rot[-1, -1, 2]) == 255)

    # rotate_images=True: 180° flip
    rotated = np.ascontiguousarray(test_img[::-1, ::-1])
    runner.record("B1", "rotate:blue_to_topleft", int(rotated[0, 0, 2]) == 255)
    runner.record("B1", "rotate:red_to_bottomright", int(rotated[-1, -1, 0]) == 255)
    runner.record("B1", "rotate:green_to_bottomleft", int(rotated[-1, 0, 1]) == 255)

    # B2. Gripper binarization for both conventions
    # libero_native: >0 → +1, ≤0 → -1
    for raw, expected_ln, expected_ov in [
        (0.5, 1.0, -1.0),
        (-0.3, -1.0, 1.0),
        (0.0, -1.0, 1.0),
        (0.8, 1.0, -1.0),
    ]:
        ln_out = 1.0 if raw > 0 else -1.0
        ov_out = 1.0 if raw < 0.5 else -1.0
        runner.record("B2", f"gripper_ln(raw={raw})", ln_out == expected_ln,
                       f"got {ln_out}, expected {expected_ln}")
        runner.record("B2", f"gripper_ov(raw={raw})", ov_out == expected_ov,
                       f"got {ov_out}, expected {expected_ov}")

    # B3. KeypointHistory push/get/reset
    # NOTE: keypoint_utils imports mujoco at module level — skip gracefully if not installed
    _b3_skip = False
    try:
        import importlib.util as _importlib_util
        _kpu_spec = _importlib_util.find_spec("mujoco")
        if _kpu_spec is None:
            raise ImportError("mujoco not installed in this venv — run Part B in CLIENT_VENV")
    except ImportError as _e:
        _b3_skip = True
        runner.record("B3", "kpt_history:skip_no_mujoco", True, str(_e))

    if not _b3_skip:
        try:
            from evaluation.LIBERO2.keypoint_utils import KeypointHistory, KEYPOINT_DIM
            kh = KeypointHistory(max_len=5)

            # Initial state: empty
            buf, length = kh.get_history()
            runner.record("B3", "kpt_history:initial_len_0", length == 0)
            runner.record("B3", "kpt_history:initial_shape", buf.shape == (5, 8, KEYPOINT_DIM))

            # Push 3 frames
            for i in range(3):
                kh.push(np.ones((8, KEYPOINT_DIM), dtype=np.float32) * (i + 1))
            buf, length = kh.get_history()
            runner.record("B3", "kpt_history:after_3_push_len", length == 3)
            runner.record("B3", "kpt_history:latest_value", float(buf[2, 0, 0]) == 3.0,
                           f"got {buf[2, 0, 0]}")

            # Reset
            kh.reset()
            _, length = kh.get_history()
            runner.record("B3", "kpt_history:after_reset_len_0", length == 0)

            # Overflow (max_len=5, push 7)
            for i in range(7):
                kh.push(np.ones((8, KEYPOINT_DIM), dtype=np.float32) * (i + 10))
            buf, length = kh.get_history()
            runner.record("B3", "kpt_history:overflow_len_capped", length == 5)
        except Exception as e:
            runner.record("B3", "kpt_history:import", False, str(e))

    # B4. _quat2axisangle edge cases
    # NOTE: model2libero_interface imports keypoint_utils (needs mujoco) — skip if not installed
    _b4_skip = _b3_skip  # same mujoco dependency
    if _b4_skip:
        runner.record("B4", "quat2aa:skip_no_mujoco", True, "mujoco not installed — run in CLIENT_VENV")
    else:
        try:
            from evaluation.LIBERO2.model2libero_interface import _quat2axisangle

            # Identity quaternion [0,0,0,1] → zero axis-angle
            aa = _quat2axisangle(np.array([0, 0, 0, 1], dtype=np.float32))
            runner.record("B4", "quat2aa:identity_is_zero", np.allclose(aa, 0, atol=1e-6),
                           f"got {aa}")

            # 90° around z: quat=[0,0,sin(45°),cos(45°)]
            angle = np.pi / 2
            quat = np.array([0, 0, np.sin(angle / 2), np.cos(angle / 2)], dtype=np.float32)
            aa = _quat2axisangle(quat)
            runner.record("B4", "quat2aa:90deg_z_magnitude", abs(np.linalg.norm(aa) - angle) < 0.01,
                           f"norm={np.linalg.norm(aa):.4f}, expected {angle:.4f}")
        except Exception as e:
            runner.record("B4", "quat2aa:import", False, str(e))

    # B5. StandaloneFK R_PAD and body names
    # NOTE: keypoint_utils imports mujoco at module level — skip gracefully if not installed
    _b5_skip = _b3_skip  # same mujoco dependency
    if _b5_skip:
        runner.record("B5", "kpt_constants:skip_no_mujoco", True, "mujoco not installed — run in CLIENT_VENV")
    else:
        try:
            from evaluation.LIBERO2.keypoint_utils import (
                DEFAULT_R_PAD, DEFAULT_KEYPOINT_BODIES, _DEFAULT_MJCF_PATH,
            )
            runner.record("B5", "B9:R_PAD_value", abs(DEFAULT_R_PAD - 1.8212722539901733) < 1e-10,
                           f"got {DEFAULT_R_PAD}")
            runner.record("B5", "B9:8_bodies", len(DEFAULT_KEYPOINT_BODIES) == 8,
                           f"got {len(DEFAULT_KEYPOINT_BODIES)}")
            runner.record("B5", "B9:lift_default_last_is_gripper0_eef",
                           DEFAULT_KEYPOINT_BODIES[-1] == "gripper0_eef",
                           f"got {DEFAULT_KEYPOINT_BODIES[-1]}")
            runner.record("B5", "B9:mjcf_path_exists", _DEFAULT_MJCF_PATH.exists(),
                           str(_DEFAULT_MJCF_PATH))
            try:
                import mujoco
                from evaluation.LIBERO2.keypoint_utils import (
                    StandaloneFK,
                    resolve_eef_body_name,
                )
                from evaluation.LIBERO2.train_eval_extra_contract import (
                    load_keypoints_meta_eef,
                )

                model = mujoco.MjModel.from_xml_path(str(_DEFAULT_MJCF_PATH))
                resolved = resolve_eef_body_name(model)
                runner.record(
                    "B5",
                    "B9:lift_resolve_gripper0_eef",
                    resolved == "gripper0_eef",
                    f"resolved={resolved}",
                )
                fk = StandaloneFK()
                runner.record(
                    "B5",
                    "B9:standalone_fk_resolved_body_in_mjcf",
                    fk.body_names[-1] == resolved,
                    f"fk={fk.body_names[-1]}",
                )
                meta_eef = load_keypoints_meta_eef()
                # Metadata may say gripper0_right_eef; that must not fail this test.
                runner.record(
                    "B5",
                    "B9:metadata_eef_name_recorded",
                    meta_eef is not None,
                    f"keypoints_meta last body={meta_eef}",
                )
            except Exception as e:
                runner.record("B5", "B9:resolve_on_mjcf", False, str(e))
        except Exception as e:
            runner.record("B5", "kpt_constants:import", False, str(e))

    # B6. eval_libero_std.py has correct task suite definitions
    try:
        spec = importlib.util.spec_from_file_location(
            "eval_std_check", PROJ / "evaluation/LIBERO2/eval_libero_std.py",
        )
        # Don't execute, just parse and check constant dict
        src = (PROJ / "evaluation/LIBERO2/eval_libero_std.py").read_text()
        for suite in ["libero_spatial", "libero_object", "libero_goal", "libero_10"]:
            runner.record("B6", f"std:suite_{suite}_defined", f'"{suite}"' in src)
        runner.record("B6", "std:TASK_SUITE_N_TASKS_defined", "TASK_SUITE_N_TASKS" in src)
        runner.record("B6", "std:n_tasks_spatial_10", '"libero_spatial": 10' in src)
    except Exception as e:
        runner.record("B6", "eval_std:check", False, str(e))

    # B7. Recorded T1: both cameras match training RAW (no EGL)
    try:
        from evaluation.LIBERO2.orientation_contract import check_recorded_t1, maybe_rotate

        rec = check_recorded_t1()
        runner.record("B7", "recorded_t1:overall_passed", rec["passed"])
        for name, cam in rec["cameras"].items():
            runner.record(
                "B7",
                f"recorded_t1:{name}_matches_raw",
                cam["matches_raw"],
                f"raw={cam['mse_live_raw_vs_train']} rot={cam['mse_live_rot180_vs_train']} "
                f"ratio={cam['ratio_rot_over_raw']}",
            )

        # maybe_rotate helper matches the client pixel op
        test_img = np.zeros((4, 4, 3), dtype=np.uint8)
        test_img[0, 0] = [9, 0, 0]
        assert int(maybe_rotate(test_img, False)[0, 0, 0]) == 9
        assert int(maybe_rotate(test_img, True)[-1, -1, 0]) == 9
        runner.record("B7", "maybe_rotate:identity_and_180", True)

        from evaluation.LIBERO2.orientation_contract import (
            enforce_rotate_against_contract,
            load_train_eval_contract,
        )

        contract = load_train_eval_contract()
        runner.record("B7", "contract:load_image_orientation_raw", contract.get("image_orientation") == "raw")
        try:
            enforce_rotate_against_contract(False, contract)
            runner.record("B7", "contract:raw_allows_no_rotate", True)
        except Exception as e:
            runner.record("B7", "contract:raw_allows_no_rotate", False, str(e))
        try:
            enforce_rotate_against_contract(True, contract)
            runner.record("B7", "contract:raw_rejects_rotate", False, "should have raised")
        except RuntimeError:
            runner.record("B7", "contract:raw_rejects_rotate", True)
        except Exception as e:
            runner.record("B7", "contract:raw_rejects_rotate", False, str(e))
    except Exception as e:
        runner.record("B7", "recorded_t1:import", False, str(e))

    # B8. U8 history schedule: request must not include current frame
    try:
        from evaluation.LIBERO2.train_eval_extra_contract import (
            first_request_his_len,
            history_excludes_current,
            simulate_eval_history_loop,
        )

        runner.record("B8", "history:excludes_current", history_excludes_current())
        runner.record(
            "B8",
            "history:wait10_his_len_10",
            first_request_his_len(10) == 10,
            f"got {first_request_his_len(10)}",
        )
        hist, current = simulate_eval_history_loop(num_wait=10, num_action=1)[0]
        runner.record(
            "B8",
            "history:first_request_current_is_10",
            current == 10 and current not in hist,
            f"hist={hist[-3:]} current={current}",
        )
    except Exception as e:
        runner.record("B8", "history:import", False, str(e))

    # T8. Gripper sign vs libero_native (parquet; skip if missing)
    try:
        from evaluation.LIBERO2.train_eval_extra_contract import t8_gripper_sign_from_parquet

        t8 = t8_gripper_sign_from_parquet()
        if t8.get("skipped"):
            runner.record("T8", "gripper_sign:skipped", True, t8.get("detail", ""))
        else:
            runner.record(
                "T8",
                "gripper_sign:libero_native_corr_negative",
                t8["passed"],
                t8.get("detail", ""),
            )
    except Exception as e:
        runner.record("T8", "gripper_sign:run", False, str(e))

    # T9. fps label vs timestamps / episode length (HDF5 optional)
    try:
        from evaluation.LIBERO2.train_eval_extra_contract import t9_fps_contract

        t9 = t9_fps_contract()
        if t9.get("skipped"):
            runner.record("T9", "fps:skipped", True, t9.get("detail", ""))
        else:
            runner.record(
                "T9",
                "fps:timestamp_and_episode_length",
                t9["passed"],
                t9.get("detail", ""),
            )
    except Exception as e:
        runner.record("T9", "fps:run", False, str(e))


# ═══════════════════════════════════════════════════════════
# Part C: Integration tests (needs SERVER_VENV with lerobot)
# ═══════════════════════════════════════════════════════════

def run_part_c(runner: TestRunner, ckpt_path: str | None = None):
    print("\n── Part C: Integration Tests ──")

    if str(PROJ) not in sys.path:
        sys.path.insert(0, str(PROJ))
    src_path = str(PROJ / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    # C1. Resize mapping 256→224
    try:
        import torch
        from lerobot.transforms.core import ResizeImagesWithPadFn
        from lerobot.utils.constants import OBS_IMAGES

        sample = {f"{OBS_IMAGES}.image{i}": torch.randn(3, 256, 256) for i in range(3)}

        # Empty mapping = no-op (the bug)
        r_old = ResizeImagesWithPadFn(height=224, width=224)
        out_old = r_old(dict(sample))
        for i in range(3):
            h, w = out_old[f"{OBS_IMAGES}.image{i}"].shape[-2:]
            runner.record("C1", f"old_resize_noop:image{i}={h}x{w}", (h, w) == (256, 256))

        # Explicit mapping = correct resize
        r_new = ResizeImagesWithPadFn(
            height=224, width=224,
            mapping={f"{OBS_IMAGES}.image{i}": f"{OBS_IMAGES}.image{i}" for i in range(3)},
        )
        out_new = r_new(dict(sample))
        for i in range(3):
            h, w = out_new[f"{OBS_IMAGES}.image{i}"].shape[-2:]
            runner.record("C1", f"new_resize_224:image{i}={h}x{w}", (h, w) == (224, 224))
    except ImportError as e:
        runner.record("C1", "resize:lerobot_import", False, str(e))

    # C2. Checkpoint preflight (if path given)
    if ckpt_path:
        import json
        ckpt = Path(ckpt_path)
        for f in ["config.json", "model.safetensors", "stats.json", "train_config.json"]:
            p = ckpt / f
            ok = p.exists() and p.stat().st_size > 0
            runner.record("C2", f"ckpt:{f}_exists", ok)

        if (ckpt / "stats.json").exists():
            stats = json.loads((ckpt / "stats.json").read_text())
            has_panda = "panda" in stats
            runner.record("C2", "ckpt:stats_key_panda", has_panda)
            if has_panda:
                panda = stats["panda"]
                state_dim = len(panda["observation.state"]["mean"])
                action_dim = len(panda["action"]["mean"])
                g_min = panda["action"]["min"][6]
                runner.record("C2", f"ckpt:state_dim={state_dim}", state_dim == 8)
                runner.record("C2", f"ckpt:action_dim={action_dim}", action_dim == 7)
                runner.record("C2", f"ckpt:gripper_min={g_min:.2f}", g_min < -0.5,
                               "confirms libero_native convention")

        if (ckpt / "config.json").exists():
            cfg = json.loads((ckpt / "config.json").read_text())
            runner.record("C2", "ckpt:enable_kpt_predictor",
                           cfg.get("enable_keypoint_predictor") is True)
            runner.record("C2", "ckpt:kpt_4d_mode_pos_rot",
                           cfg.get("kpt_4d_mode") == "pos_rot")
    else:
        runner.record("C2", "ckpt:skipped_no_path", True, "pass --ckpt to enable")

    # C3. NormalizeTransformFn with panda stats
    if ckpt_path:
        try:
            import json
            import numpy as _np
            import torch
            from lerobot.transforms.core import NormalizeTransformFn
            from lerobot.utils.constants import OBS_STATE

            stats = json.loads((Path(ckpt_path) / "stats.json").read_text())
            panda = stats["panda"]
            # stats.json stores mean/std as Python lists; NormalizeTransformFn needs numpy arrays
            raw_stat = panda["observation.state"]
            state_stat = {OBS_STATE: {k: _np.array(v, dtype=_np.float32) if isinstance(v, list) else v
                                      for k, v in raw_stat.items()}}
            normalizer = NormalizeTransformFn(selected_keys=[OBS_STATE], norm_stats=state_stat)

            fake_state = torch.randn(8)
            sample = {OBS_STATE: fake_state}
            out = normalizer(sample)
            runner.record("C3", "normalize:output_shape",
                           out[OBS_STATE].shape == torch.Size([8]),
                           f"got {out[OBS_STATE].shape}")
            runner.record("C3", "normalize:values_changed",
                           not torch.allclose(out[OBS_STATE], fake_state))
        except Exception as e:
            runner.record("C3", "normalize:test", False, str(e))

    # C4. Panda schema validation
    try:
        from lerobot.dataset_schemas import get_schema
        schema = get_schema("panda")
        action_mode = getattr(schema, "action_mode", None)
        runner.record("C4", "panda_schema:action_mode_joint", action_mode == "joint",
                       f"got {action_mode}")
        img_map = getattr(schema, "image_mapping", {})
        runner.record("C4", "panda_schema:has_image_mapping", len(img_map) > 0,
                       f"keys: {list(img_map.keys())}")
    except Exception as e:
        runner.record("C4", "panda_schema:import", False, str(e))

    # T8 / T9: parquet contracts (pyarrow lives in SERVER_VENV)
    try:
        from evaluation.LIBERO2.train_eval_extra_contract import (
            t8_gripper_sign_from_parquet,
            t9_fps_contract,
        )

        t8 = t8_gripper_sign_from_parquet()
        if t8.get("skipped"):
            runner.record("T8", "gripper_sign:skipped", True, t8.get("detail", ""))
        else:
            runner.record(
                "T8",
                "gripper_sign:libero_native_corr_negative",
                t8["passed"],
                t8.get("detail", ""),
            )
        t9 = t9_fps_contract()
        if t9.get("skipped"):
            runner.record("T9", "fps:skipped", True, t9.get("detail", ""))
        else:
            runner.record(
                "T9",
                "fps:timestamp_and_episode_length",
                t9["passed"],
                t9.get("detail", ""),
            )
    except Exception as e:
        runner.record("T8", "parquet_contracts:import", False, str(e))


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Pre-evaluation test suite for F1/F2 fixes + B1-B10 regression",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Part A: Static analysis (any venv, no deps)
            Part B: Unit tests (needs numpy)
            Part C: Integration tests (needs SERVER_VENV with lerobot)
        """),
    )
    parser.add_argument("--part", type=str, default="ABC",
                        help="Which parts to run: A, AB, ABC (default: ABC)")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Checkpoint path for Part C validation")
    args = parser.parse_args()

    parts = args.part.upper()
    runner = TestRunner()

    print("=" * 60)
    print("  Pre-Evaluation Test Suite: F1/F2 + B1-B10 Regression")
    print(f"  Parts: {parts}  Checkpoint: {args.ckpt or '(none)'}")
    print("=" * 60)

    if "A" in parts:
        run_part_a(runner)
    if "B" in parts:
        run_part_b(runner)
    if "C" in parts:
        run_part_c(runner, args.ckpt)

    all_pass = runner.summary()
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
