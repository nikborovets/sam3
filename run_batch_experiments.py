"""
Batch SAM3 experiment runner (with overlap at batch boundaries).

For each sequence, frames are split into overlapping batches so that the
last annotation frame of one batch is also the first annotation frame of
the next batch (1-frame overlap):

  batch 0 : frames [  0 .. 300]  -> annotations  0,  75, 150, 225, 300
  batch 1 : frames [300 .. 600]  -> annotations 300, 375, 450, 525, 600
  batch 2 : frames [600 .. 900]  -> annotations 600, 675, 750, 825, 900
  ...

Step between batches = BATCH_SIZE - 1.

Per-batch pipeline:
  1_track_and_save.py  ->  2_visualize_results.py

Results accumulate in:
  RESULTS_ROOT_DIR/<anno_dir_name>_results/
    masks_npz/   <- NPZ files (all batches, boundary frames overwritten by later batch)
    masks_out/   <- segmentation mask PNGs (all batches)
    overlay/     <- overlay PNGs (all batches, only when SAVE_OVERLAY=True)
"""

import os
import shutil
import subprocess
from pathlib import Path
import time
import json, html
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
try:
    from tg_notifier import notify_error, send_message
except ImportError:
    def notify_error(e, msg=""): print(f"Notifier not found: {msg} {e}")
    def send_message(msg): print(msg)

# ===================== CONFIG =====================
# INPUT_BASE_DIR   = Path("/workspace/data_mount/third_wave_tracker_rgb_input/cvat_share_mount/E-3023-A5-input-rgb")
# ANNO_BASE_DIR    = Path("/workspace/3023-cvat-annotations/2stage")
# RESULTS_ROOT_DIR = Path("/workspace/3023-A5-SAM3-results/2stage")
# INPUT_BASE_DIR   = Path("/workspace/data_mount/third_wave_tracker_rgb_input/cvat_share_mount/03-25-26_E-R1-1023-input-rgb")
# ANNO_BASE_DIR    = Path("/workspace/03-25-26_E-R1-1023-cvat-annotations/2stage")
# RESULTS_ROOT_DIR = Path("/workspace/03-25-26_E-R1-1023-SAM3-results/2stage")
# INPUT_BASE_DIR   = Path("/workspace/data_mount/third_wave_tracker_rgb_input/cvat_share_mount/03-27-26_E-A4-3024-input-rgb")
# ANNO_BASE_DIR    = Path("/workspace/03-27-26_E-A4-3024-cvat-annotations/1stage")
# RESULTS_ROOT_DIR = Path("/workspace/03-27-26_E-A4-3024-SAM3-results/1stage")
# INPUT_BASE_DIR   = Path("/workspace/data_mount/third_wave_tracker_rgb_input/cvat_share_mount/04-02-26-biology-lab-input-rgb")
# ANNO_BASE_DIR    = Path("/workspace/04-02-26-biology-lab-cvat-annotations/1stage")
# RESULTS_ROOT_DIR = Path("/workspace/04-02-26-biology-lab-SAM3-results/1stage")
# INPUT_BASE_DIR   = Path("/workspace/data_mount/third_wave_tracker_rgb_input/cvat_share_mount/E-3023-A5-input-rgb")
# ANNO_BASE_DIR    = Path("/workspace/3023-cvat-annotations/3stage")
# RESULTS_ROOT_DIR = Path("/workspace/3023-A5-SAM3-results/3stage")
# INPUT_BASE_DIR   = Path("/workspace/data_mount/third_wave_tracker_rgb_input/cvat_share_mount/04-17-26-C3-2040-input-rgb")
# ANNO_BASE_DIR    = Path("/workspace/sam_anno_input_output/04-17-26-C3-2040-cvat-annotations/1stage")
# RESULTS_ROOT_DIR = Path("/workspace/sam_anno_input_output/04-17-26-C3-2040-SAM3-results/1stage")
# INPUT_BASE_DIR   = Path("/workspace/data_mount/third_wave_tracker_rgb_input/cvat_share_mount/04-23-2026-living-room-1-input-rgb")
# ANNO_BASE_DIR    = Path("/workspace/sam_anno_input_output/04-23-2026-living-room-1-cvat-annotations/1stage")
# RESULTS_ROOT_DIR = Path("/workspace/sam_anno_input_output/04-23-2026-living-room-1-SAM3-results/1stage")
# SCENE_NAME = "04-27-26-living-room-2"
# SCENE_NAME = "04-30-26-children-room"
# SCENE_NAME = "04-30-26-kitchen"
# SCENE_NAME = "04-30-26-bathroom"
# SCENE_NAME = "05-08-26-E-B4-3011" # 160
# SCENE_NAME = "05-08-26-E-B4-3003" # 160
# SCENE_NAME = "05-13-26-bathroom-2" # 160
# SCENE_NAME = "05-13-26-kitchen-2" # 120
SCENE_NAME = "05-20-26-classroom-1" # 160
INPUT_BASE_DIR   = Path(f"/workspace/data_mount/third_wave_tracker_rgb_input/cvat_share_mount/{SCENE_NAME}-input-rgb")
ANNO_BASE_DIR    = Path(f"/workspace/sam_anno_input_output/{SCENE_NAME}-cvat-annotations/1stage")
RESULTS_ROOT_DIR = Path(f"/workspace/sam_anno_input_output/{SCENE_NAME}-SAM3-results/1stage")
SAM3_DIR         = Path("/workspace/sam3")

# SEQUENCES    = ["seq1", "seq2", "seq3", "seq4", "seq5", "seq6"]
SEQUENCES    = ["state1_art", "state1_nat", "state2_art", "state2_nat", "state3_nat"]
# SEQUENCES    = ["state2_art", "state2_nat", "state3_nat"]
# BATCH_SIZE   = 201       # frames per batch inclusive: e.g. 0..300 = 301 frames
# BATCH_SIZE   = 241
BATCH_SIZE   = 321
# BATCH_SIZE   = 281
SAVE_OVERLAY  = False    # True  -> also write overlay PNGs alongside masks
BIDIRECTIONAL = False    # True  -> run reverse propagation pass (sam3 only)

# Model version: "sam3" or "sam3.1"
# sam3.1 uses Object Multiplex — drastically lower GPU memory for multi-object tracking
# (~7x speedup at 128 objects; ~17.7 GB on H100 for 5 objects).
# Requires a separate checkpoint: sam3.1_multiplex.pt
SAM3_VERSION     = "sam3"
SAM3_WEIGHTS_31  = "/workspace/data_mount/model_weights/sam3.1/sam3.1_multiplex.pt"
# ==================================================

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
BATCH_STEP = BATCH_SIZE - 1  # overlap of 1 frame at boundaries

stats_dict = {}

def find_anno_dir(seq_name: str) -> Path | None:
    """Return first annotation directory matching <seq_name>_job* pattern."""
    candidates = sorted(ANNO_BASE_DIR.glob(f"{seq_name}_job*"))
    # candidates = sorted(ANNO_BASE_DIR.glob(f"{seq_name}-job*"))
    return candidates[0] if candidates else None


def get_image_frames(directory: Path) -> list[Path]:
    """Sorted list of image files inside a directory."""
    return sorted(p for p in directory.glob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def run_batch(
    batch_idx: int,
    rgb_frames: list[Path],
    anno_frames: list[Path],
    anno_dir: Path,
    results_dir: Path,
) -> None:
    """
    Symlink batch frames + masks into a temp dir, run tracking then
    visualization, move NPZ files to the shared results dir, clean up.
    """
    batch_tag = f"batch_{batch_idx:03d}"
    batch_tmp = results_dir / f"_tmp_{batch_tag}"
    rgb_tmp   = batch_tmp / "rgb"
    masks_tmp = batch_tmp / "masks"
    npz_tmp   = batch_tmp / "masks_npz"

    shared_npz    = results_dir / "masks_npz"
    masks_out_dir = results_dir / "masks_out"
    overlay_dir   = results_dir / "overlay"

    # Clean any leftover tmp from a previous failed run
    if batch_tmp.exists():
        shutil.rmtree(batch_tmp)

    rgb_tmp.mkdir(parents=True)
    masks_tmp.mkdir(parents=True)
    npz_tmp.mkdir(parents=True)
    shared_npz.mkdir(parents=True, exist_ok=True)
    masks_out_dir.mkdir(parents=True, exist_ok=True)
    if SAVE_OVERLAY:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    for frame_path in rgb_frames:
        os.symlink(frame_path.resolve(), rgb_tmp / frame_path.name)

    for mask_path in anno_frames:
        os.symlink(mask_path.resolve(), masks_tmp / mask_path.name)

    print(f"  [{batch_tag}] frames {rgb_frames[0].name}..{rgb_frames[-1].name} "
          f"({len(rgb_frames)} total), {len(anno_frames)} annotation(s): "
          f"{[p.stem for p in anno_frames]}")

    labelmap = anno_dir / "labelmap.txt"

    # --- 1. Track and save NPZ ---
    cmd_track = [
        "python", str(SAM3_DIR / "1_track_and_save.py"),
        "--version",  SAM3_VERSION,
        "--inputs",   str(rgb_tmp),
        "--masks",    str(masks_tmp),
        "--labelmap", str(labelmap),
        "--out-npz",  str(npz_tmp),
    ]
    if SAM3_VERSION == "sam3.1":
        cmd_track += ["--weights-31", SAM3_WEIGHTS_31]
    if BIDIRECTIONAL:
        cmd_track.append("--bidirectional")

    print(f"  [{batch_tag}] Tracking...")
    subprocess.run(cmd_track, check=True)

    # --- 2. Visualize ---
    cmd_viz = [
        "python", str(SAM3_DIR / "2_visualize_results.py"),
        "--inputs",      str(rgb_tmp),
        "--in-npz",      str(npz_tmp),
        "--labelmap",    str(labelmap),
        "--out-overlay", str(overlay_dir),
        "--out-masks",   str(masks_out_dir),
    ]
    if not SAVE_OVERLAY:
        cmd_viz.append("--no-overlay")

    print(f"  [{batch_tag}] Visualizing...")
    subprocess.run(cmd_viz, check=True)

    # Move NPZ files to shared dir (boundary frame NPZ gets overwritten by later batch)
    for npz_file in npz_tmp.glob("*.npz"):
        dest = shared_npz / npz_file.name
        if dest.exists():
            dest.unlink()
        shutil.move(str(npz_file), dest)

    shutil.rmtree(batch_tmp)
    print(f"  [{batch_tag}] Done. Temp dir removed.\n")


def process_sequence(seq_name: str) -> None:
    print(f"\n{'='*60}")
    print(f"  Sequence : {seq_name}")
    print(f"{'='*60}")

    anno_dir = find_anno_dir(seq_name)
    if not anno_dir:
        print(f"  SKIP: no annotation directory found for '{seq_name}'")
        return
    print(f"  Anno dir : {anno_dir.name}")

    # rgb_dir = INPUT_BASE_DIR / f"{seq_name}_color" / f"{seq_name}_rs_color"
    # rgb_dir = INPUT_BASE_DIR / f"{seq_name}_color_rs"
    rgb_dir = INPUT_BASE_DIR / f"{seq_name}_colors"
    # rgb_dir = INPUT_BASE_DIR / seq_name
    if not rgb_dir.exists():
        print(f"  SKIP: RGB directory not found: {rgb_dir}")
        return

    all_rgb = get_image_frames(rgb_dir)
    if not all_rgb:
        print(f"  SKIP: no RGB frames in {rgb_dir}")
        return
    print(f"  RGB frames : {len(all_rgb)}")

    # anno_mask_dir = anno_dir / "SegmentationClass" / "03-27-26_E-A4-3024-input-rgb" / f"{seq_name}_color" / f"{seq_name}_rs_color"
    # anno_mask_dir = anno_dir / "SegmentationClass" / "03-25-26_E-R1-1023-input-rgb" / f"{seq_name}_color" / f"{seq_name}_rs_color"
    # anno_mask_dir = anno_dir / "SegmentationClass" / "04-02-26-biology-lab-input-rgb" / f"{seq_name}_color_rs"
    # anno_mask_dir = anno_dir / "SegmentationClass" / "04-17-26-C3-2040-input-rgb" / f"{seq_name}_colors"
    # anno_mask_dir = anno_dir / "SegmentationClass" / "04-23-2026-living-room-1-input-rgb" / f"{seq_name}_colors"
    anno_mask_dir = anno_dir / "SegmentationClass" / INPUT_BASE_DIR.name / f"{seq_name}_colors"
    # anno_mask_dir = anno_dir / "SegmentationClass" / seq_name
    if not anno_mask_dir.exists():
        print(f"  SKIP: annotation mask dir not found: {anno_mask_dir}")
        return

    all_anno = get_image_frames(anno_mask_dir)
    print(f"  Annotations: {len(all_anno)}")

    anno_by_stem = {p.stem: p for p in all_anno}

    results_dir = RESULTS_ROOT_DIR / f"{anno_dir.name}_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Results    : {results_dir}\n")

    n = len(all_rgb)
    time_start = time.time()

    # Build batch start indices with step = BATCH_SIZE - 1 (overlap of 1 frame)
    batch_starts = list(range(0, n, BATCH_STEP))
    # Ensure the very last frame is always covered
    if batch_starts[-1] + BATCH_SIZE - 1 >= n:
        # last batch may be shorter — that's fine
        pass

    total_batches = len(batch_starts)
    print(f"  Batches    : {total_batches}  (step={BATCH_STEP}, size={BATCH_SIZE})\n")

    for batch_idx, batch_start in enumerate(batch_starts):
        batch_end_excl = min(batch_start + BATCH_SIZE, n)  # exclusive
        rgb_batch = all_rgb[batch_start:batch_end_excl]

        # Annotations whose stem matches any frame in this batch
        anno_batch = [anno_by_stem[f.stem] for f in rgb_batch if f.stem in anno_by_stem]

        if not anno_batch:
            print(f"  [batch_{batch_idx:03d}] No annotations in this range, skipping.")
            continue
        send_message(f"Processing batch {batch_idx + 1} of {total_batches} for {SCENE_NAME} — '{seq_name}'...")
        run_batch(batch_idx, rgb_batch, anno_batch, anno_dir, results_dir)

    print(f"  All {total_batches} batch(es) finished for '{seq_name}'.")
    time_end = time.time()
    time_elapsed = time.strftime("%H:%M:%S", time.gmtime(time_end - time_start))
    stats_dict[seq_name] = {
        "rgb_frames": len(all_rgb),
        "anno_frames": len(all_anno),
        "batch_size": BATCH_SIZE,
        "batches": total_batches,
        "time_elapsed": time_elapsed,
    }
    send_message(f"All {total_batches} batch(es) finished for '{seq_name}' in {time_elapsed}. Stats: {stats_dict[seq_name]}")


def main() -> None:
    RESULTS_ROOT_DIR.mkdir(parents=True, exist_ok=True)
    print("SAM3 Batch Experiment Runner")
    print(f"  Results root : {RESULTS_ROOT_DIR}")
    print(f"  Batch size   : {BATCH_SIZE} frames  (step={BATCH_STEP}, overlap=1)")
    print(f"  Save overlay : {SAVE_OVERLAY}")
    print(f"  Bidirectional: {BIDIRECTIONAL}")
    print(f"  Sequences    : {SEQUENCES}\n")

    for seq in SEQUENCES:
        process_sequence(seq)

    print("\n" + "="*60)
    print("  All sequences processed!")
    print(f"  Results in: {RESULTS_ROOT_DIR}")
    print("="*60)


if __name__ == "__main__":
    try:
        time_start = time.time()
        send_message(f"Starting {os.path.basename(__file__)}...")
        main()
    except Exception as e:
        time_end = time.time()
        time_elapsed = time.strftime("%H:%M:%S", time.gmtime(time_end - time_start))
        notify_error(e, f"Критическая ошибка при выполнении {os.path.basename(__file__)}. Time elapsed: {time_elapsed}")
        raise
    time_end = time.time()
    time_elapsed = time.strftime("%H:%M:%S", time.gmtime(time_end - time_start))
    send_message(f"{os.path.basename(__file__)} completed successfully in {time_elapsed}")
    send_message(f"Stats:<pre>{html.escape(json.dumps(stats_dict, indent=2, ensure_ascii=False))}</pre>")
