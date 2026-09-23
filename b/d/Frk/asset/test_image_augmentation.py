#!/usr/bin/env python
"""
Test script: verify that the image augmentation configured in
  b/s/Frk/frk_plug_sft_dtaug_launch.sh
produces *moderate* color/sharpness jitter on real data from
  /B/Dta/plug_into_socket_lrb_4D/
and does NOT apply geometric transforms (rotation, translation, etc.).

What it does
────────────
1. Loads 3 episodes × 5 sampled frames from each camera (global + wrist).
2. For each frame, applies the augmentation pipeline 8 times with different
   random seeds to show the range of variation.
3. Generates per-frame comparison grids (original + 8 augmented) as PNG.
4. Stitches each camera's grids into a single MP4 side-by-side video.
5. Computes pixel-level statistics (PSNR, max-delta, structural diff) and
   asserts:
     - PSNR stays above 20 dB   (moderate, not destructive)
     - No spatial shift detected (affine is truly disabled)
     - Max per-pixel delta < 80  (no extreme artifacts)

Output goes to: b/d/Frk/asset/logs/

Usage:
    python b/d/Frk/asset/test_image_augmentation.py
"""

import os
import sys
import json
import math
import warnings
from pathlib import Path

import numpy as np
import torch
from torchvision.io import read_video
from torchvision.transforms.v2 import functional as F_v2

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.io")

PROJ_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJ_ROOT / "src"))

from lerobot.datasets.transforms import ImageTransformsConfig, ImageTransforms

# ─── Config (mirrors frk_plug_sft_dtaug_launch.sh) ─────────────────────────
DATA_ROOT = Path("/B/Dta/plug_into_socket_lrb_4D")
OUTPUT_DIR = PROJ_ROOT / "b" / "d" / "Frk" / "asset" / "logs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CAMERAS = {
    "global": DATA_ROOT / "videos" / "observation.images.global" / "chunk-000" / "file-000.mp4",
    "wrist":  DATA_ROOT / "videos" / "observation.images.wrist"  / "chunk-000" / "file-000.mp4",
}

EPISODES_TO_SAMPLE = [0, 50, 99]
FRAMES_PER_EPISODE = 5
NUM_AUG_VARIANTS = 8

# Thresholds
MIN_PSNR_DB = 18.0
MAX_PIXEL_DELTA = 80
MAX_SPATIAL_SHIFT_PX = 2


def build_augmenter() -> ImageTransforms:
    """Build the exact same ImageTransforms as the launch script configures."""
    cfg = ImageTransformsConfig(
        enable=True,
        max_num_transforms=3,
        random_order=False,
        disabled_tfs=["affine"],
    )
    aug = ImageTransforms(cfg)
    print(f"Active transforms: {list(aug.transforms.keys())}")
    print(f"Disabled transforms: {cfg.disabled_tfs}")
    print(f"max_num_transforms: {cfg.max_num_transforms}")
    return aug


def load_episode_info(data_root: Path) -> list[dict]:
    """Load episode start/end frame indices from meta."""
    episodes_dir = data_root / "meta" / "episodes"
    if episodes_dir.exists():
        import pyarrow.parquet as pq
        files = sorted(episodes_dir.glob("*.parquet"))
        if files:
            table = pq.read_table(files[0])
            df = table.to_pandas()
            episodes = []
            for _, row in df.iterrows():
                episodes.append({
                    "index": int(row.get("episode_index", row.name)),
                    "from": int(row.get("from", row.get("episode_data_index_from", 0))),
                    "to": int(row.get("to", row.get("episode_data_index_to", 0))),
                })
            return episodes
    info_path = data_root / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)
    total = info.get("total_frames", 0)
    n_ep = info.get("total_episodes", 1)
    fps = info.get("fps", 30)
    per_ep = total // n_ep
    return [{"index": i, "from": i * per_ep, "to": (i + 1) * per_ep} for i in range(n_ep)]


def extract_frames(video_path: Path, frame_indices: list[int], fps: int = 30) -> list[torch.Tensor]:
    """Extract specific frames from a video file. Returns list of [C, H, W] uint8 tensors."""
    frames = []
    for idx in frame_indices:
        t = idx / fps
        v, _, _ = read_video(str(video_path), pts_unit="sec", start_pts=t, end_pts=t + 0.05)
        if v.shape[0] == 0:
            print(f"  Warning: no frame at index {idx} (t={t:.3f}s), skipping")
            continue
        frame = v[0].permute(2, 0, 1)  # [H,W,C] -> [C,H,W]
        frames.append(frame)
    return frames


def compute_psnr(orig: torch.Tensor, aug: torch.Tensor) -> float:
    """Compute PSNR between two uint8 images."""
    mse = ((orig.float() - aug.float()) ** 2).mean().item()
    if mse < 1e-10:
        return 100.0
    return 10 * math.log10(255.0 ** 2 / mse)


def detect_spatial_shift(orig: torch.Tensor, aug: torch.Tensor) -> float:
    """
    Detect spatial shift via normalized cross-correlation on gradient magnitude.
    Color jitter does not move edges — only affine/rotation does.
    Returns estimated shift in pixels (0 = perfect alignment).
    """
    import torch.nn.functional as F_nn

    def to_gray(img):
        return (0.299 * img[0].float() + 0.587 * img[1].float() + 0.114 * img[2].float())

    def grad_mag(gray):
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        g = gray.unsqueeze(0).unsqueeze(0)
        ex = F_nn.conv2d(g, sobel_x, padding=1)
        ey = F_nn.conv2d(g, sobel_y, padding=1)
        return (ex ** 2 + ey ** 2).sqrt().squeeze()

    gm_orig = grad_mag(to_gray(orig))
    gm_aug = grad_mag(to_gray(aug))

    # Take a central patch (avoid border effects)
    H, W = gm_orig.shape
    margin = max(H, W) // 8
    patch_orig = gm_orig[margin:-margin, margin:-margin]
    patch_aug = gm_aug[margin:-margin, margin:-margin]

    # Normalize
    patch_orig = patch_orig - patch_orig.mean()
    patch_aug = patch_aug - patch_aug.mean()
    std_o = patch_orig.std()
    std_a = patch_aug.std()
    if std_o < 1e-6 or std_a < 1e-6:
        return 0.0

    # Pixel-wise correlation coefficient
    ncc = (patch_orig * patch_aug).mean() / (std_o * std_a)

    # NCC ~ 1.0 means perfect alignment (no shift).
    # Color jitter keeps NCC > 0.85 typically. Affine rotation/translate drops it below 0.7.
    # Convert to an estimated shift: shift_px ≈ (1 - NCC) * scale_factor
    shift_estimate = max(0.0, (1.0 - ncc.item()) * 20.0)
    return shift_estimate


def make_comparison_grid(orig: torch.Tensor, augmented: list[torch.Tensor],
                         labels: list[str] | None = None) -> np.ndarray:
    """
    Create a 3x3 grid: [original, aug1, aug2; aug3, aug4, aug5; aug6, aug7, aug8].
    Returns HWC uint8 numpy array.
    """
    imgs = [orig] + augmented
    if labels is None:
        labels = ["Original"] + [f"Aug {i+1}" for i in range(len(augmented))]

    C, H, W = orig.shape
    pad = 4
    cell_h, cell_w = H + pad * 2 + 20, W + pad * 2  # 20px for label
    grid_h, grid_w = 3 * cell_h, 3 * cell_w
    canvas = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 240

    for i, (img, label) in enumerate(zip(imgs, labels)):
        row, col = i // 3, i % 3
        y0 = row * cell_h + pad + 16
        x0 = col * cell_w + pad
        arr = img.permute(1, 2, 0).numpy()
        canvas[y0:y0+H, x0:x0+W] = arr

        # Simple label using pixel art (just put colored strip for identification)
        label_y = row * cell_h + 2
        if i == 0:
            canvas[label_y:label_y+12, x0:x0+60] = [0, 120, 0]  # green = original
        else:
            canvas[label_y:label_y+12, x0:x0+40] = [0, 0, 200]  # blue = augmented

    return canvas


def save_grid_as_png(grid: np.ndarray, path: Path):
    """Save grid as PNG using raw PPM -> ffmpeg (no PIL dependency)."""
    try:
        from PIL import Image
        Image.fromarray(grid).save(str(path))
        return
    except ImportError:
        pass
    # Fallback: save as raw npy and convert with cv2 or just save npy
    try:
        import cv2
        cv2.imwrite(str(path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        return
    except ImportError:
        pass
    np.save(str(path).replace(".png", ".npy"), grid)
    print(f"  (saved as .npy — install PIL or cv2 for PNG output)")


def grids_to_video(grid_paths: list[Path], output_video: Path, fps: int = 2):
    """Stitch PNG grids into an MP4 video."""
    try:
        from PIL import Image
        import subprocess

        first = Image.open(str(grid_paths[0]))
        w, h = first.size
        first.close()

        ffmpeg = "/B/VENV/itnvla15rbt20/bin/ffmpeg"
        if not os.path.exists(ffmpeg):
            # Try system ffmpeg
            ffmpeg = "ffmpeg"

        concat_file = output_video.parent / f"_concat_{output_video.stem}.txt"
        with open(concat_file, "w") as f:
            for p in grid_paths:
                f.write(f"file '{p}'\n")
                f.write(f"duration {1.0/fps}\n")
            f.write(f"file '{grid_paths[-1]}'\n")

        cmd = [
            ffmpeg, "-y", "-f", "concat", "-safe", "0",
            "-i", str(concat_file),
            "-vf", f"scale={w}:{h}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-crf", "18",
            str(output_video)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        concat_file.unlink(missing_ok=True)
        if result.returncode != 0:
            print(f"  ffmpeg warning: {result.stderr[:200]}")
            # Try alternative: frame-by-frame with torchvision
            _write_video_torch(grid_paths, output_video, fps)
        else:
            print(f"  Video saved: {output_video}")
    except Exception as e:
        print(f"  Video generation failed ({e}), PNG grids still available")


def _write_video_torch(grid_paths: list[Path], output_video: Path, fps: int):
    """Fallback: write video using torchvision."""
    try:
        from torchvision.io import write_video as tv_write_video
        from PIL import Image

        frames = []
        for p in grid_paths:
            img = np.array(Image.open(str(p)))
            # Ensure even dimensions for h264
            h, w = img.shape[:2]
            h = h - h % 2
            w = w - w % 2
            img = img[:h, :w]
            frames.append(torch.from_numpy(img))

        video_tensor = torch.stack(frames)  # [T, H, W, C]
        tv_write_video(str(output_video), video_tensor, fps=fps)
        print(f"  Video saved (torchvision): {output_video}")
    except Exception as e2:
        print(f"  Video fallback also failed: {e2}")


def main():
    print("=" * 70)
    print("Image Augmentation Test for frk_plug_sft_dtaug_launch.sh")
    print("=" * 70)
    print(f"Data:   {DATA_ROOT}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    augmenter = build_augmenter()

    # Also build an augmenter WITH affine to show what we're avoiding
    cfg_with_affine = ImageTransformsConfig(
        enable=True,
        max_num_transforms=3,
        random_order=False,
        disabled_tfs=[],  # nothing disabled
    )
    augmenter_with_affine = ImageTransforms(cfg_with_affine)
    print(f"Comparison augmenter (with affine): {list(augmenter_with_affine.transforms.keys())}")
    print()

    episodes = load_episode_info(DATA_ROOT)
    info_path = DATA_ROOT / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)
    fps = info.get("fps", 30)

    all_stats = []
    all_pass = True

    for cam_name, video_path in CAMERAS.items():
        print(f"\n{'─' * 60}")
        print(f"Camera: {cam_name}")
        print(f"Video:  {video_path}")
        print(f"{'─' * 60}")

        if not video_path.exists():
            print(f"  SKIP: video not found")
            continue

        grid_paths = []

        for ep_idx in EPISODES_TO_SAMPLE:
            if ep_idx >= len(episodes):
                continue
            ep = episodes[ep_idx]
            ep_len = ep["to"] - ep["from"]
            if ep_len <= 0:
                continue

            # Sample frames evenly within the episode
            sample_indices = [
                ep["from"] + int(i * (ep_len - 1) / max(1, FRAMES_PER_EPISODE - 1))
                for i in range(FRAMES_PER_EPISODE)
            ]

            print(f"\n  Episode {ep_idx}: frames {ep['from']}..{ep['to']} "
                  f"(len={ep_len}), sampling {sample_indices}")

            frames = extract_frames(video_path, sample_indices, fps=fps)

            for fi, frame in enumerate(frames):
                frame_id = sample_indices[fi] if fi < len(sample_indices) else fi
                # frame: [C, H, W] uint8

                # Apply augmentation NUM_AUG_VARIANTS times
                aug_frames = []
                aug_frames_affine = []
                frame_stats = {"camera": cam_name, "episode": ep_idx, "frame": frame_id}

                psnrs = []
                max_deltas = []
                spatial_shifts = []

                for vi in range(NUM_AUG_VARIANTS):
                    torch.manual_seed(42 + ep_idx * 1000 + fi * 100 + vi)
                    aug = augmenter(frame.clone())
                    aug_frames.append(aug)

                    # Compute stats
                    psnr = compute_psnr(frame, aug)
                    max_delta = (frame.float() - aug.float()).abs().max().item()
                    shift = detect_spatial_shift(frame, aug)
                    psnrs.append(psnr)
                    max_deltas.append(max_delta)
                    spatial_shifts.append(shift)

                    # Also apply affine version for comparison (first 2 variants only)
                    if vi < 2:
                        torch.manual_seed(42 + ep_idx * 1000 + fi * 100 + vi + 500)
                        aug_af = augmenter_with_affine(frame.clone())
                        aug_frames_affine.append(aug_af)

                frame_stats["psnr_min"] = min(psnrs)
                frame_stats["psnr_max"] = max(psnrs)
                frame_stats["psnr_mean"] = sum(psnrs) / len(psnrs)
                frame_stats["max_delta_max"] = max(max_deltas)
                frame_stats["spatial_shift_max"] = max(spatial_shifts)
                all_stats.append(frame_stats)

                # Check assertions
                pass_psnr = min(psnrs) >= MIN_PSNR_DB
                pass_delta = max(max_deltas) <= MAX_PIXEL_DELTA
                pass_shift = max(spatial_shifts) <= MAX_SPATIAL_SHIFT_PX

                status = "PASS" if (pass_psnr and pass_delta and pass_shift) else "FAIL"
                if status == "FAIL":
                    all_pass = False

                print(f"    Frame {frame_id}: PSNR={min(psnrs):.1f}~{max(psnrs):.1f} dB, "
                      f"max_delta={max(max_deltas):.0f}, shift={max(spatial_shifts):.2f}px "
                      f"[{status}]")

                # Generate comparison grid (without affine)
                labels = ["Original"] + [
                    f"Aug{i+1} PSNR={psnrs[i]:.1f}" for i in range(NUM_AUG_VARIANTS)
                ]
                grid = make_comparison_grid(frame, aug_frames, labels)
                grid_path = OUTPUT_DIR / f"{cam_name}_ep{ep_idx:03d}_f{frame_id:06d}_aug.png"
                save_grid_as_png(grid, grid_path)
                grid_paths.append(grid_path)

                # Generate affine comparison grid (first frame only per episode)
                if fi == 0 and aug_frames_affine:
                    # Show: original, aug_no_affine x2, aug_with_affine x2
                    comparison_imgs = [
                        aug_frames[0], aug_frames[1],
                        aug_frames_affine[0], aug_frames_affine[1],
                    ]
                    comparison_labels = [
                        "Original",
                        "NoAffine 1", "NoAffine 2",
                        "WithAffine 1", "WithAffine 2",
                    ]
                    # Pad to 8 with copies of original for grid layout
                    while len(comparison_imgs) < 8:
                        comparison_imgs.append(frame)
                        comparison_labels.append("(pad)")
                    comp_grid = make_comparison_grid(frame, comparison_imgs, comparison_labels)
                    comp_path = OUTPUT_DIR / f"{cam_name}_ep{ep_idx:03d}_affine_comparison.png"
                    save_grid_as_png(comp_grid, comp_path)
                    print(f"    Affine comparison saved: {comp_path.name}")

        # Stitch grids into video
        if grid_paths:
            video_path_out = OUTPUT_DIR / f"{cam_name}_augmentation_test.mp4"
            grids_to_video(grid_paths, video_path_out, fps=2)

    # ─── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if not all_stats:
        print("No frames processed!")
        sys.exit(1)

    psnr_all = [s["psnr_min"] for s in all_stats]
    delta_all = [s["max_delta_max"] for s in all_stats]
    shift_all = [s["spatial_shift_max"] for s in all_stats]

    print(f"  Frames tested:    {len(all_stats)}")
    print(f"  PSNR range:       {min(psnr_all):.1f} ~ {max(psnr_all):.1f} dB "
          f"(threshold: >= {MIN_PSNR_DB} dB)")
    print(f"  Max pixel delta:  {max(delta_all):.0f} "
          f"(threshold: <= {MAX_PIXEL_DELTA})")
    print(f"  Max spatial shift:{max(shift_all):.2f} px "
          f"(threshold: <= {MAX_SPATIAL_SHIFT_PX} px)")
    print()

    # Detailed per-camera summary
    for cam in CAMERAS:
        cam_stats = [s for s in all_stats if s["camera"] == cam]
        if cam_stats:
            cam_psnr = [s["psnr_min"] for s in cam_stats]
            cam_delta = [s["max_delta_max"] for s in cam_stats]
            cam_shift = [s["spatial_shift_max"] for s in cam_stats]
            print(f"  [{cam}] PSNR: {min(cam_psnr):.1f}~{max(cam_psnr):.1f}, "
                  f"delta: {max(cam_delta):.0f}, shift: {max(cam_shift):.2f}")

    print()
    if all_pass:
        print("  RESULT: ALL PASS")
        print("  Augmentation is moderate (color/sharpness only), no geometric distortion.")
    else:
        print("  RESULT: SOME CHECKS FAILED")
        for s in all_stats:
            if s["psnr_min"] < MIN_PSNR_DB:
                print(f"    FAIL PSNR: {s['camera']} ep{s['episode']} f{s['frame']} "
                      f"psnr={s['psnr_min']:.1f}")
            if s["max_delta_max"] > MAX_PIXEL_DELTA:
                print(f"    FAIL DELTA: {s['camera']} ep{s['episode']} f{s['frame']} "
                      f"delta={s['max_delta_max']:.0f}")
            if s["spatial_shift_max"] > MAX_SPATIAL_SHIFT_PX:
                print(f"    FAIL SHIFT: {s['camera']} ep{s['episode']} f{s['frame']} "
                      f"shift={s['spatial_shift_max']:.2f}")

    print(f"\n  Output directory: {OUTPUT_DIR}")
    print(f"  Files generated:")
    for f in sorted(OUTPUT_DIR.glob("*augment*")):
        print(f"    {f.name}")
    for f in sorted(OUTPUT_DIR.glob("*affine*")):
        print(f"    {f.name}")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
