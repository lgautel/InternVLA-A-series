#!/usr/bin/env python3
"""Validate libero_raw_analyz2.md against the filesystem and the measurements.

Three classes of check:
  1. every relative link and image path resolves
  2. every absolute source path cited exists, and every `file:line` citation
     points at a line that still contains the thing it claims
  3. every number quoted in the prose matches the corresponding JSON artefact

Run after editing the document:
    python3 verify_doc.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
DOC = HERE.parent / "libero_raw_analyz2.md"

failures: list[str] = []
checked = 0


def ok(condition: bool, message: str) -> None:
    global checked
    checked += 1
    if not condition:
        failures.append(message)


def load(name: str) -> dict:
    return json.loads((HERE / name).read_text())


def check_links(text: str) -> None:
    for label, target in re.findall(r"\[([^\]]+)\]\(([^)]+)\)", text):
        if target.startswith(("http://", "https://", "#")):
            continue
        path_part = target.split("#", 1)[0]
        if not path_part:
            continue
        resolved = Path(path_part) if path_part.startswith("/") else (DOC.parent / path_part)
        ok(resolved.exists(), f"broken link [{label}]({target}) -> {resolved}")


def check_anchors(text: str) -> None:
    headings = re.findall(r"^#{1,6}\s+(.*)$", text, re.MULTILINE)
    slugs = set()
    for h in headings:
        s = h.strip().lower()
        s = re.sub(r"[`*]", "", s)
        s = re.sub(r"[^\w\s\u4e00-\u9fff-]", "", s)
        slugs.add(re.sub(r"\s+", "-", s.strip()))
    for anchor in re.findall(r"\]\(#([^)]+)\)", text):
        ok(anchor in slugs, f"dangling anchor #{anchor}")


# What each cited line must still contain. Keeps the document honest when the
# referenced source moves.
CITATION_CONTENT = {
    ("robosuite/robots/single_arm.py", 304): "def eef_pos",
    ("robosuite/robots/single_arm.py", 308): "def eef_quat",
    ("robosuite/devices/spacemouse.py", 67): "axis_scale=350.0",
    ("robosuite/macros.py", 28): 'IMAGE_CONVENTION = "opengl"',
    ("LIBERO/libero/libero/envs/env_wrapper.py", 27): "control_freq=20",
    ("LIBERO/scripts/collect_demonstration.py", 237): "pos-sensitivity",
    ("LIBERO2/keypoint_utils.py", 6): "-0.56",
    ("LIBERO2/keypoint_utils.py", 21): "DEFAULT_R_PAD",
    ("LIBERO2/keypoint_utils.py", 27): "EEF_BODY_CANDIDATES",
    ("LIBERO2/keypoint_utils.py", 107): "xyzw = np.array",
    ("LIBERO2/keypoint_utils.py", 137): "max_len: int = 200",
    ("LIBERO2/model2libero_interface.py", 143): "def _extract_state",
    ("LIBERO2/model2libero_interface.py", 156): "def _maybe_rotate",
    ("LIBERO2/model2libero_interface.py", 176): "gripper_convention",
    ("LIBERO2/model2libero_interface.py", 183): "agentview_image",
    ("LIBERO2/model2libero_interface.py", 129): "def push_keypoint",
    ("LIBERO2/eval_libero_std.py", 38): "LIBERO_DUMMY_ACTION",
    ("LIBERO2/eval_libero_std.py", 335): "num_steps_wait",
    ("internvla_a1_5/configuration_internvla_a1_5.py", 599): "keypoint_3d_delta_indices",
    ("launch/internvla_a15_finetune_libero_geop.sh", 232): "keypoint_history_max_len=200",
}


# This repo comes first: /B/SRC also holds an unmodified upstream clone
# (InternVLA-A-series) with identically named files, and the document always
# means the local copy.
SEARCH_ROOTS = [
    Path("/B/SRC/itvlaGpLibPlus"),
    Path("/B/SRC"),
    Path("/B/VENV/libero_plus_client/lib/python3.10/site-packages"),
]


def resolve(suffix: str) -> Path | None:
    """`LIBERO2/keypoint_utils.py` -> the one absolute path ending with it.

    Several basenames are ambiguous in this tree (there is both an
    `evaluation/LIBERO/` and an `evaluation/LIBERO2/` copy of
    `model2libero_interface.py`), which is why CITATION_CONTENT keys carry
    enough of the parent path to disambiguate.
    """
    for root in SEARCH_ROOTS:
        hits = [p for p in root.glob(f"**/{suffix}") if str(p).endswith(suffix)]
        if hits:
            return sorted(hits)[0]
    return None


def check_line_citations(text: str) -> None:
    """Every cited `file:line` must appear in the prose AND still say what we claim.

    Line numbers live in the markdown link *label*, not the URL, so the
    document is scanned as plain text. A citation may list several lines
    (`file.py:38,335`) or a range (`file.py:143-154`); the leading number is
    the one pinned to an expected content string.
    """
    cited: set[tuple[str, int]] = set()
    for token, first, second in re.findall(
        r"([\w./+-]+\.(?:py|json|sh|xml)):(\d+)(?:[,-](\d+))?", text
    ):
        base = token.rsplit("/", 1)[-1]
        for n in (first, second):
            if n:
                cited.add((base, int(n)))

    for (suffix, line_no), needle in CITATION_CONTENT.items():
        base = suffix.rsplit("/", 1)[-1]
        ok_cited = (base, line_no) in cited
        if not ok_cited:
            ok(False, f"expected citation {suffix}:{line_no} is absent from the document")
            continue
        ok(ok_cited, f"{suffix}:{line_no} is cited")
        path = resolve(suffix)
        if path is None:
            ok(False, f"cited file cannot be resolved: {suffix}")
            continue
        lines = path.read_text(errors="ignore").splitlines()
        if line_no > len(lines):
            ok(False, f"{suffix}:{line_no} is past EOF ({len(lines)} lines)")
            continue
        ok(
            needle in lines[line_no - 1],
            f"{suffix}:{line_no} no longer contains {needle!r}; "
            f"it now reads {lines[line_no - 1].strip()[:70]!r}",
        )


def check_numbers(text: str) -> None:
    raw = load("raw_stats.json")
    kpt = load("kpt_stats.json")
    meta = load("merged_kpt_meta.json")
    orient = load("orientation_report.json")
    chain = load("chain_orientation_report.json")
    plus = load("libero_plus_report.json")
    contract = load("contract_report.json")

    def quoted(needle: str) -> None:
        ok(needle in text, f"document does not contain the measured value {needle!r}")

    def quoted_num(value: float, suffix: str = "", max_dp: int = 4) -> None:
        """Accept any rounding of `value` that the prose might reasonably use.

        Prose rounds (6.793% -> 6.79%) while the JSON keeps full precision, so
        an exact string match would fail on formatting alone. Rounding to
        fewer digits still catches genuine drift.
        """
        forms = {f"{round(value, dp):g}{suffix}" for dp in range(max_dp + 1)}
        forms |= {f"{value:.{dp}f}{suffix}" for dp in range(max_dp + 1)}
        ok(
            any(f in text for f in forms),
            f"document quotes none of {sorted(forms)} for the measured value {value}{suffix}",
        )

    # scale
    quoted(str(raw["totals"]["episodes"]))
    quoted(str(raw["totals"]["frames"]))
    quoted_num(raw["totals"]["retention_pct"], "%")
    for suite, s in raw["subsets"].items():
        ok(str(s["episodes"]) in text, f"missing episode count for {suite}")
        ok(str(s["frames"]) in text, f"missing frame count for {suite}")
        quoted_num(s["frame_share_pct"], "%")

    # action
    quoted(str(raw["action"]["abs_max_first6"]))
    quoted_num(raw["action"]["frames_at_clip_pct"], "%")
    for group in ("position_xyz", "rotation_rpy"):
        quoted(str(raw["action"]["quantisation"][group]["grid"]))
        quoted(str(raw["action"]["quantisation"][group]["levels_used"]))
    for value, count in raw["action"]["gripper_values"].items():
        quoted(str(count))

    # state
    dc = raw["state"]["axisangle_double_cover"]
    quoted(str(dc["frames_with_norm_over_pi"]))
    quoted_num(dc["frames_with_norm_over_pi_pct"], "%")
    quoted(str(dc["largest_single_step_jump_rad"]))
    quoted(str(raw["joint_state"]["max_limit_overshoot_rad"]))
    quoted_num(raw["noop_audit"]["tiny_eef_motion_pct"], "%")

    # r_pad
    quoted(str(meta["keypoints_meta"]["bbox_radius"]))
    quoted(str(kpt["r_pad"]["recomputed_from_this_dataset"]))
    quoted(str(meta["info"]["fps"]))

    # arena bases
    for name, entry in kpt["arena_bases"].items():
        quoted(name)
        quoted(str(entry["episodes"]))
        quoted_num(entry["frame_share_pct"], "%")

    # normalisation utilisation
    for key, scheme in kpt["normalisation_schemes"].items():
        if "axis_utilisation" in scheme and key != "C_base_frame_per_axis":
            for u in scheme["axis_utilisation"]:
                quoted(f"{u * 100:.1f}%")

    # redundancy
    quoted(str(kpt["redundancy"]["moving_position_dims_of_24"]))
    quoted(str(kpt["redundancy"]["independent_position_dims_of_24"]))

    # quaternion
    q = kpt["quaternion"]
    quoted(str(q["total_sign_flips"]))
    for body in ("link7", "eef"):
        quoted(str(q["frames_with_qw_below_0.05"][body]))
        quoted(str(q["sign_flips_between_consecutive_frames"][body]))
        pct = 100 * q["frames_with_qw_below_0.05"][body] / kpt["frames"]
        quoted(f"{pct:.1f}%")

    # history window
    hw = kpt["history_window"]
    quoted(f"{hw['padding_fraction'] * 100:.2f}%")
    quoted(f"{hw['episodes_longer_than_H_pct']}%")
    quoted(str(hw["per_suite"]["libero_spatial"]["max_history_reached"]))

    # orientation
    quoted(str(chain["verdict"]["mse"]))
    quoted(str(int(chain["verdict"]["margin_ratio"])))
    quoted_num(abs(orient["trajectory_correlation"]["row_median"]))
    quoted_num(abs(orient["trajectory_correlation"]["col_median"]))
    ok(
        orient["verdict"]["stored_as"] == "rot180"
        and chain["verdict"]["transform_from_rlds_to_merged"] == "rot180",
        "orientation verdicts are no longer both rot180",
    )

    # libero-plus
    quoted(str(plus["verdict"]["tasks_total"]))
    quoted(str(plus["verdict"]["tasks_invisible_to_keypoints"]))
    quoted(f"{100 * plus['verdict']['fraction_of_catalogue_invisible_to_keypoints']:.1f}%")
    for tier, s in plus["init_qpos_perturbation"]["tiers"].items():
        quoted(f"{s['eef_disp_mean_m'] * 100:.2f} cm")
    tf = plus["training_first_frame_distribution"]
    quoted(f"{tf['eef_radius_p95_m'] * 100:.2f} cm")
    for cat, n in plus["catalogue"]["overall_by_category"].items():
        quoted(cat)
        quoted(str(n))

    # contract
    s = contract["summary"]
    ok(s["blocking_failures"] == 0, f"contract has {s['blocking_failures']} blocking failures")
    quoted(f"{s['passed']}/{s['total']}")
    for c in contract["checks"]:
        ok(f"**{c['id']}**" in text, f"contract check {c['id']} is missing from the table")


def check_figures(text: str) -> None:
    referenced = set(re.findall(r"!\[[^\]]*\]\((asset/[^)]+)\)", text))
    on_disk = {f"asset/{p.name}" for p in HERE.glob("fig*.png")}
    for fig in sorted(on_disk - referenced):
        ok(False, f"figure {fig} exists but is never shown in the document")
    for fig in sorted(referenced - on_disk):
        ok(False, f"document references a missing figure {fig}")


def main() -> int:
    if not DOC.exists():
        print(f"missing {DOC}")
        return 1
    text = DOC.read_text()

    check_links(text)
    check_anchors(text)
    check_line_citations(text)
    check_figures(text)
    check_numbers(text)

    print(f"{checked - len(failures)}/{checked} document checks pass")
    for f in failures:
        print(f"  FAIL {f}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
