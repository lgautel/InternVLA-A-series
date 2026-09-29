"""Offline image-orientation probes for the Goal data. No simulator needed.

Question answered: is a stored 256x256 frame in robosuite RAW orientation, in
RLDS orientation (= rot180 of raw), or something else?

Idea. The camera poses are known exactly, and the robot's own joint angles give
the gripper position in the world. If the arm moves, the pixels that change
between two frames are the arm. Project the FK gripper position into the image
under each of the four orientation hypotheses and see which one tracks the
centroid of the changed pixels. The sign of the correlation does not depend on
the (unknown to us) field of view, so the verdict is robust to that.

Conventions (MuJoCo, and robosuite 1.4 which LIBERO uses):
  * ``upright`` = the picture a camera would show with rows growing downward.
  * robosuite hands the numpy array back bottom-up, so RAW = flipV(upright).
  * LIBERO/OpenVLA store RLDS = rot180(RAW) = flipH(upright).

Hypotheses are expressed relative to ``upright`` because that is what the
projection produces:  identity | flipV (=RAW) | flipH (=RLDS) | rot180.

agentview camera: LIBERO-Plus/libero/libero/envs/bddl_base_domain.py::_setup_camera
  pos  = [0.58861, 0, 1.49035]  quat(wxyz) = [0.63802, 0.30485, 0.30485, 0.63802]
wrist camera: robot0_right_hand + pos [0.05, 0, 0], quat(wxyz) [0, 0.707108, 0.707108, 0]
(from the robosuite Panda model; see assets/panda_kin.xml).
"""

from __future__ import annotations

import numpy as np

AGENT_POS = np.array([0.5886131746834771, 0.0, 1.4903500240372423])
AGENT_QUAT_WXYZ = np.array([0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349])
WRIST_POS_IN_HAND = np.array([0.05, 0.0, 0.0])
WRIST_QUAT_WXYZ = np.array([0.0, 0.707108, 0.707108, 0.0])

HYPOTHESES = {
    "identity": (False, False),
    "flipV_RAW": (False, True),
    "flipH_RLDS": (True, False),
    "rot180": (True, True),
}


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def project_agentview(points_world: np.ndarray, size: int = 256, fovy_deg: float = 75.0) -> np.ndarray:
    """[N,3] world points -> [N,2] (col,row) in the UPRIGHT picture."""
    rot = quat_to_mat(AGENT_QUAT_WXYZ)
    f = (size / 2) / np.tan(np.radians(fovy_deg) / 2)
    pc = (points_world - AGENT_POS) @ rot  # rows are R^T (p - c)
    u = f * pc[:, 0] / (-pc[:, 2])
    v = f * pc[:, 1] / (-pc[:, 2])
    return np.stack([size / 2 + u, size / 2 - v], axis=1)


def _apply_flip(cols: np.ndarray, rows: np.ndarray, flip_h: bool, flip_v: bool, size: int):
    return (size - 1 - cols if flip_h else cols), (size - 1 - rows if flip_v else rows)


def motion_centroids(frames_gray: np.ndarray, stride: int = 2, thresh: float = 25.0, min_pixels: int = 30):
    """Return (frame_index, col, row) of the changed-pixel centroid between t-stride and t."""
    out = []
    for t in range(stride, len(frames_gray), stride):
        d = np.abs(frames_gray[t].astype(np.float32) - frames_gray[t - stride].astype(np.float32))
        m = d > thresh
        if m.sum() < min_pixels:
            continue
        ys, xs = np.nonzero(m)
        out.append((t, xs.mean(), ys.mean()))
    return np.array(out, dtype=np.float64).reshape(-1, 3)


def agentview_orientation(eef_pos_per_episode: list[np.ndarray], gray_per_episode: list[np.ndarray]) -> dict:
    """Score the four hypotheses. Inputs are lists over episodes of [T,3] world eef and [T,H,W] gray."""
    pts, cx, cy = [], [], []
    for eef, gray in zip(eef_pos_per_episode, gray_per_episode):
        cen = motion_centroids(gray)
        for t, c, r in cen:
            pts.append(eef[int(t)])
            cx.append(c)
            cy.append(r)
    if len(pts) < 50:
        raise RuntimeError(f"only {len(pts)} motion samples; need at least 50")
    pts = np.asarray(pts)
    cx = np.asarray(cx)
    cy = np.asarray(cy)
    uv = project_agentview(pts)
    size = gray_per_episode[0].shape[-1]
    scores = {}
    for name, (fh, fv) in HYPOTHESES.items():
        pc, pr = _apply_flip(uv[:, 0], uv[:, 1], fh, fv, size)
        scores[name] = {
            "corr_col": float(np.corrcoef(pc, cx)[0, 1]),
            "corr_row": float(np.corrcoef(pr, cy)[0, 1]),
        }
        scores[name]["min_corr"] = min(scores[name]["corr_col"], scores[name]["corr_row"])
    best = max(scores, key=lambda k: scores[k]["min_corr"])
    return {"samples": int(len(pts)), "scores": scores, "best": best}


def wrist_orientation(
    cam_pos_world: list[np.ndarray],
    cam_rot_world: list[np.ndarray],
    gray_per_episode: list[np.ndarray],
    stride: int = 2,
    max_rot_deg: float = 0.6,
    min_move_m: float = 0.004,
) -> dict:
    """Score hypotheses from the global image shift caused by camera translation.

    For a static scene the picture shifts opposite to the camera motion. With
    camera-frame translation dc, the UPRIGHT picture shifts by
    (d_col, d_row) proportional to (-dc_x, +dc_y). Rotation-dominated pairs are skipped.
    """
    import cv2

    h_, w_ = gray_per_episode[0].shape[1:3]
    win = cv2.createHanningWindow((int(w_), int(h_)), cv2.CV_32F)
    rows = []
    for pos, rot, gray in zip(cam_pos_world, cam_rot_world, gray_per_episode):
        for t in range(0, len(gray) - stride):
            r0, r1 = rot[t], rot[t + stride]
            rel = r0.T @ r1
            ang = np.degrees(np.arccos(np.clip((np.trace(rel) - 1) / 2, -1, 1)))
            if ang > max_rot_deg:
                continue
            dc = r0.T @ (pos[t + stride] - pos[t])
            if np.linalg.norm(dc) < min_move_m:
                continue
            (dx, dy), resp = cv2.phaseCorrelate(gray[t].astype(np.float32), gray[t + stride].astype(np.float32), win)
            if resp < 0.1:
                continue
            rows.append((-dc[0], dc[1], dx, dy))
    if len(rows) < 50:
        raise RuntimeError(f"only {len(rows)} wrist samples; need at least 50")
    r = np.asarray(rows)
    scores = {}
    for name, (fh, fv) in HYPOTHESES.items():
        pc = -r[:, 0] if fh else r[:, 0]
        pr = -r[:, 1] if fv else r[:, 1]
        c1 = float(np.corrcoef(pc, r[:, 2])[0, 1])
        c2 = float(np.corrcoef(pr, r[:, 3])[0, 1])
        scores[name] = {"corr_col": c1, "corr_row": c2, "min_corr": min(c1, c2)}
    best = max(scores, key=lambda k: scores[k]["min_corr"])
    return {"samples": int(len(r)), "scores": scores, "best": best}


def to_gray(rgb_stack: np.ndarray) -> np.ndarray:
    """[T,H,W,3] uint8 -> [T,H,W] float32 luminance."""
    x = rgb_stack.astype(np.float32)
    return 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]
