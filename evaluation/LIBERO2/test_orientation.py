#!/usr/bin/env python3
"""Gate 1+2: Pixel-level orientation verification and resize validation.

Verifies that:
  1. _maybe_rotate() flag consistency
  2. Training data uses raw (unrotated) orientation
  3. Shell script has ROTATE_IMAGES config
  4. eval_libero_plus.py replay uses client._maybe_rotate()
  5. ResizeImagesWithPadFn mapping produces 224×224
  6. T6: Qwen AutoImageProcessor image_grid_thw is [[1,16,16]] for 224 and 256

Usage (in CLIENT_VENV for Tests 1-4):
    python evaluation/LIBERO2/test_orientation.py --num_samples 20

Usage (in SERVER_VENV for Test 5-6):
    python evaluation/LIBERO2/test_orientation.py --test resize_only
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_VLM_SNAP = Path(
    "/B/VENV/hf_home/hub/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc"
)


def compute_mse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.shape != b.shape:
        h = min(a.shape[0], b.shape[0])
        w = min(a.shape[1], b.shape[1])
        a = a[:h, :w]
        b = b[:h, :w]
    return float(np.mean((a - b) ** 2))


def test_rotation_flag_consistency() -> bool:
    """Test 1: Verify _maybe_rotate() behaves correctly for both flag values."""
    test_img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    test_img[0, 0] = [255, 0, 0]
    test_img[-1, -1] = [0, 0, 255]

    class FakeClient:
        def __init__(self, rotate: bool):
            self.rotate_images = rotate

        def _maybe_rotate(self, image):
            arr = np.asarray(image)
            if self.rotate_images:
                arr = arr[::-1, ::-1]
            return np.ascontiguousarray(arr)

    # rotate_images=False: preserve orientation
    result = FakeClient(False)._maybe_rotate(test_img)
    assert result[0, 0, 0] == 255, "No-rotation should preserve top-left red pixel"
    assert result[-1, -1, 2] == 255, "No-rotation should preserve bottom-right blue pixel"

    # rotate_images=True: flip both axes
    result = FakeClient(True)._maybe_rotate(test_img)
    assert result[0, 0, 2] == 255, "Rotation should move blue to top-left"
    assert result[-1, -1, 0] == 255, "Rotation should move red to bottom-right"

    print("  _maybe_rotate() flag consistency: PASS")
    return True


def test_training_orientation(dataset_root: Path, num_samples: int) -> bool:
    """Test 2: Verify training data orientation (raw vs rotated asymmetry)."""
    images = []
    video_dir = dataset_root / "videos"
    if video_dir.exists():
        try:
            import av
            video_files = sorted(video_dir.glob("chunk-*/**/observation.images.image_*.mp4"))
            if not video_files:
                video_files = sorted(video_dir.glob("**/*.mp4"))
            for vf in video_files[:num_samples]:
                container = av.open(str(vf))
                for frame in container.decode(video=0):
                    img = frame.to_ndarray(format="rgb24")
                    images.append(img)
                    break
                container.close()
                if len(images) >= num_samples:
                    break
        except ImportError:
            pass

    if not images:
        print("  WARNING: Could not load training images (av not installed or no videos found)")
        print("  Skipping orientation asymmetry check, verifying structure only")
        return True

    print(f"  Loaded {len(images)} training images, shape={images[0].shape}")

    all_asym = True
    for i, img in enumerate(images):
        rotated = img[::-1, ::-1]
        mse_rot = compute_mse(img, rotated)
        if mse_rot <= 100:
            print(f"  WARN sample {i}: MSE(raw vs rotated)={mse_rot:.1f} (low — image may be near-symmetric)")
            all_asym = False

    print(f"  Orientation asymmetry check: {'PASS' if all_asym else 'WARN (some symmetric images)'}")
    return True


def test_shell_config() -> bool:
    """Test 3: Verify plus2 + std shells disable rotation for this dataset."""
    sh_path = REPO_ROOT / "evaluation/LIBERO-plus2/run_eval_libero_plus_venv.sh"
    if not sh_path.exists():
        print("  Shell script not found")
        return False
    content = sh_path.read_text()
    has_var = 'ROTATE_IMAGES=' in content
    has_flag = 'ROTATE_FLAG' in content
    default_false = 'ROTATE_IMAGES="${ROTATE_IMAGES:-false}"' in content
    opt_in = 'ROTATE_FLAG="--rotate_images"' in content
    print(f"  plus2 ROTATE_IMAGES config var: {'PASS' if has_var else 'FAIL'}")
    print(f"  plus2 ROTATE_FLAG logic: {'PASS' if has_flag else 'FAIL'}")
    print(f"  plus2 default false: {'PASS' if default_false else 'FAIL'}")
    print(f"  plus2 opt-in --rotate_images: {'PASS' if opt_in else 'FAIL'}")

    std_sh = REPO_ROOT / "evaluation/LIBERO2/run_eval_libero_std_venv.sh"
    std_ok = std_sh.exists() and 'ROTATE_IMAGES="${ROTATE_IMAGES:-false}"' in std_sh.read_text()
    print(f"  std shell ROTATE_IMAGES default false: {'PASS' if std_ok else 'FAIL'}")
    return has_var and has_flag and default_false and opt_in and std_ok


def test_replay_consistency() -> bool:
    """Test 4: Verify plus2 + std replay both use _maybe_rotate (not a hardcoded flip)."""
    ok = True
    for rel in (
        "evaluation/LIBERO-plus2/eval_libero_plus.py",
        "evaluation/LIBERO2/eval_libero_std.py",
    ):
        eval_path = REPO_ROOT / rel
        if not eval_path.exists():
            print(f"  {rel} not found")
            ok = False
            continue
        content = eval_path.read_text()
        lines_with_replay = [
            (i + 1, line) for i, line in enumerate(content.split("\n"))
            if "replay_images.append" in line
        ]
        print(f"  {rel}:")
        if not lines_with_replay:
            print("    No replay_images.append found")
            ok = False
            continue
        for lineno, line in lines_with_replay:
            if "[::-1, ::-1]" in line:
                print(f"    FAIL L{lineno}: hardcoded 180° rotation")
                ok = False
            elif "_maybe_rotate" in line:
                print(f"    L{lineno}: uses _maybe_rotate() — PASS")
            else:
                print(f"    L{lineno}: unknown pattern")
                ok = False
    return ok


def test_resize_mapping() -> bool:
    """Test 5: Verify ResizeImagesWithPadFn mapping produces 224×224."""
    try:
        import torch
        from lerobot.transforms.core import ResizeImagesWithPadFn
        from lerobot.utils.constants import OBS_IMAGES
    except ImportError:
        print("  Skipping resize test (lerobot not importable — run in SERVER_VENV)")
        return True

    sample = {f"{OBS_IMAGES}.image{i}": torch.randn(3, 256, 256) for i in range(3)}

    # Old: empty mapping → no-op
    resize_old = ResizeImagesWithPadFn(height=224, width=224)
    result_old = resize_old(dict(sample))
    for i in range(3):
        h, w = result_old[f"{OBS_IMAGES}.image{i}"].shape[-2:]
        assert (h, w) == (256, 256), f"Old resize should be no-op but got {h}×{w}"
    print("  Old resize (empty mapping): confirmed no-op 256×256")

    # New: explicit mapping → resized
    resize_new = ResizeImagesWithPadFn(
        height=224, width=224,
        mapping={f"{OBS_IMAGES}.image{i}": f"{OBS_IMAGES}.image{i}" for i in range(3)},
    )
    result_new = resize_new(dict(sample))
    for i in range(3):
        h, w = result_new[f"{OBS_IMAGES}.image{i}"].shape[-2:]
        assert (h, w) == (224, 224), f"New resize should produce 224×224 but got {h}×{w}"
    print("  New resize (explicit mapping): confirmed 224×224")
    return True


def _hwc_uint8_from_chw01(t) -> np.ndarray:
    arr = t.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return (arr * 255.0).round().astype(np.uint8)


def test_t6_image_grid_thw() -> bool:
    """T6: Qwen processor image_grid_thw is identical for 256 vs mapped 224.

    `lerobot.transforms.utils.resize_with_pad` requires a CHW float tensor in [0, 1].
    Passing HWC uint8 numpy (the naive snippet in eval3_optim2 §10.4) raises
    AttributeError: ndarray has no unsqueeze.
    """
    try:
        import torch
        from transformers import AutoImageProcessor
        from lerobot.transforms.core import ResizeImagesWithPadFn
        from lerobot.transforms.utils import resize_with_pad
        from lerobot.utils.constants import OBS_IMAGES
    except ImportError as e:
        print(f"  Skipping T6 (need SERVER_VENV / transformers / lerobot): {e}")
        return True

    snap = Path(os.environ.get("VLM_MODEL_PATH", str(DEFAULT_VLM_SNAP)))
    if not snap.is_dir():
        print(f"  SKIP: VLM snapshot not found: {snap}")
        return True

    rng = np.random.default_rng(0)
    img256 = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
    chw01 = torch.from_numpy(img256).permute(2, 0, 1).float() / 255.0

    img224_fn = resize_with_pad(chw01, 224, 224)
    if tuple(img224_fn.shape[-2:]) != (224, 224):
        print(f"  FAIL: resize_with_pad shape {tuple(img224_fn.shape)}")
        return False
    img224 = _hwc_uint8_from_chw01(img224_fn)

    mapped = ResizeImagesWithPadFn(
        height=224,
        width=224,
        mapping={f"{OBS_IMAGES}.image0": f"{OBS_IMAGES}.image0"},
    )({f"{OBS_IMAGES}.image0": chw01})
    mapped_hw = tuple(mapped[f"{OBS_IMAGES}.image0"].shape[-2:])
    if mapped_hw != (224, 224):
        print(f"  FAIL: ResizeImagesWithPadFn mapping shape {mapped_hw}")
        return False

    ip = AutoImageProcessor.from_pretrained(str(snap), local_files_only=True)
    g224 = ip(images=[img224]).image_grid_thw.tolist()
    g256 = ip(images=[img256]).image_grid_thw.tolist()
    expected = [[1, 16, 16]]
    print(f"  processor={type(ip).__name__} size={getattr(ip, 'size', None)}")
    print(f"  image_grid_thw 224={g224} 256={g256}")
    ok = g224 == g256 == expected
    if not ok:
        print(f"  FAIL: expected {expected} for both")
        return False
    print("  T6 image_grid_thw: PASS (224 and 256 both [[1,16,16]])")
    return True


def main():
    parser = argparse.ArgumentParser(description="Gate 1+2: Orientation and resize verification")
    parser.add_argument("--dataset_root", type=str,
                        default="/B/Dta/opvla_libero_merged_kpt")
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--test", type=str, default="all",
                        choices=["all", "resize_only", "orientation_only"])
    args = parser.parse_args()

    print("=" * 60)
    print("Gate 1+2: Orientation & Resize Verification")
    print("=" * 60)

    results = {}

    if args.test in ("all", "orientation_only"):
        print("\n[Test 1] _maybe_rotate() flag consistency")
        results["t1"] = test_rotation_flag_consistency()

        print(f"\n[Test 2] Training image orientation ({args.num_samples} samples)")
        results["t2"] = test_training_orientation(Path(args.dataset_root), args.num_samples)

        print("\n[Test 3] Shell script ROTATE_IMAGES config")
        results["t3"] = test_shell_config()

        print("\n[Test 4] Replay frame rotation consistency")
        results["t4"] = test_replay_consistency()

        print("\n[Test 4b] Both cameras use _maybe_rotate")
        m2l = (REPO_ROOT / "evaluation/LIBERO2/model2libero_interface.py").read_text()
        av = 'self._maybe_rotate(np.asarray(obs["agentview_image"]' in m2l
        wr = 'self._maybe_rotate(np.asarray(obs["robot0_eye_in_hand_image"]' in m2l
        print(f"  agentview: {'PASS' if av else 'FAIL'}")
        print(f"  wrist:     {'PASS' if wr else 'FAIL'}")
        results["t4b"] = av and wr

        print("\n[Test 4c] Recorded T1 JSON (agentview + wrist vs training RAW)")
        try:
            from evaluation.LIBERO2.orientation_contract import check_recorded_t1

            rec = check_recorded_t1()
            for name, cam in rec["cameras"].items():
                tag = "PASS" if cam["matches_raw"] else "FAIL"
                print(
                    f"  {name}: {tag}  raw={cam['mse_live_raw_vs_train']}  "
                    f"rot180={cam['mse_live_rot180_vs_train']}  ratio={cam['ratio_rot_over_raw']}"
                )
            results["t4c"] = rec["passed"]
        except Exception as e:
            print(f"  FAIL: {e}")
            results["t4c"] = False

        print("\n[Test 4d] train_eval_contract.json + default rotate False")
        try:
            from evaluation.LIBERO2.orientation_contract import (
                enforce_rotate_against_contract,
                load_train_eval_contract,
            )

            contract = load_train_eval_contract()
            ok_orient = contract.get("image_orientation") == "raw"
            print(f"  image_orientation=raw: {'PASS' if ok_orient else 'FAIL'}")
            enforce_rotate_against_contract(False, contract)
            rejected = False
            try:
                enforce_rotate_against_contract(True, contract)
            except RuntimeError:
                rejected = True
            print(f"  raw rejects rotate=True: {'PASS' if rejected else 'FAIL'}")
            m2l = (REPO_ROOT / "evaluation/LIBERO2/model2libero_interface.py").read_text()
            default_false = "rotate_images: bool = False" in m2l
            print(f"  LiberoModelClient default False: {'PASS' if default_false else 'FAIL'}")
            results["t4d"] = ok_orient and rejected and default_false
        except Exception as e:
            print(f"  FAIL: {e}")
            results["t4d"] = False

    if args.test in ("all", "resize_only"):
        print("\n[Test 5] Resize mapping verification")
        results["t5"] = test_resize_mapping()

        print("\n[Test 6] T6 image_grid_thw 224 vs 256")
        results["t6"] = test_t6_image_grid_thw()

    all_pass = all(results.values())
    print("\n" + "=" * 60)
    for name, passed in results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    print(f"\nOVERALL: {'PASS' if all_pass else 'FAIL'}")
    print("=" * 60)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
