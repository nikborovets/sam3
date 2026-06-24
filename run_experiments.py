import os
import shutil
import subprocess
from pathlib import Path

# Paths
RGB_DIR = Path("/workspace/data_mount/third_wave_tracker_rgb_input/E-1023-R1-input-rgb/seq2_orbbec")
GT_MASKS_DIR = Path("/workspace/E-1023-R1-cvat-annotations/2stage/seq2_obbec_cvat_export_15-03-26/SegmentationClass/seq2_orbbec")
LABELMAP_PATH = Path("/workspace/E-1023-R1-cvat-annotations/2stage/seq2_obbec_cvat_export_15-03-26/labelmap.txt")
BASE_OUT_DIR = Path("/workspace/data_mount/experiments_tmp")

def setup_experiment(exp_id, start_idx, end_idx, mask_indices, all_rgb_frames):
    print(f"--- Setting up Experiment {exp_id} ---")
    print(f"Frames: {start_idx} to {end_idx}, Masks at: {mask_indices}")
    
    exp_dir = BASE_OUT_DIR / f"exp_{exp_id}"
    rgb_tmp_dir = exp_dir / "rgb"
    masks_tmp_dir = exp_dir / "masks"
    out_npz_dir = exp_dir / "results_npz"
    out_overlay_dir = exp_dir / "overlay"
    out_masks_dir = exp_dir / "masks_out"
    
    # Clean old run if exists
    if exp_dir.exists():
        shutil.rmtree(exp_dir)
        
    rgb_tmp_dir.mkdir(parents=True)
    masks_tmp_dir.mkdir(parents=True)
    out_npz_dir.mkdir(parents=True)
    out_overlay_dir.mkdir(parents=True)
    out_masks_dir.mkdir(parents=True)
    
    # 1. Symlink RGB frames
    frames_to_process = all_rgb_frames[start_idx : end_idx + 1]
    for frame_path in frames_to_process:
        symlink_target = rgb_tmp_dir / frame_path.name
        os.symlink(frame_path, symlink_target)
        
    # 2. Symlink Masks
    for mask_idx in mask_indices:
        mask_name = all_rgb_frames[mask_idx].name
        src_mask_path = GT_MASKS_DIR / mask_name
        if not src_mask_path.exists():
            print(f"WARNING: Mask {src_mask_path} does not exist!")
            continue
        dst_mask_path = masks_tmp_dir / mask_name
        os.symlink(src_mask_path, dst_mask_path)
        
    return exp_dir, rgb_tmp_dir, masks_tmp_dir, out_npz_dir, out_overlay_dir, out_masks_dir

def run_experiment(exp_dir, rgb_tmp_dir, masks_tmp_dir, out_npz_dir, out_overlay_dir, out_masks_dir, bidirectional=False):
    # Step 1: Track and save
    cmd_track = [
        "python", "/workspace/sam3/1_track_and_save.py",
        "--inputs", str(rgb_tmp_dir),
        "--masks", str(masks_tmp_dir),
        "--labelmap", str(LABELMAP_PATH),
        "--out-npz", str(out_npz_dir)
    ]
    if bidirectional:
        cmd_track.append("--bidirectional")
        
    print(f"Running Tracking for {exp_dir.name}...")
    subprocess.run(cmd_track, check=True)
    
    # Step 2: Visualize results
    cmd_viz = [
        "python", "/workspace/sam3/2_visualize_results.py",
        "--inputs", str(rgb_tmp_dir),
        "--in-npz", str(out_npz_dir),
        "--labelmap", str(LABELMAP_PATH),
        "--out-overlay", str(out_overlay_dir),
        "--out-masks", str(out_masks_dir)
    ]
    print(f"Running Visualization for {exp_dir.name}...")
    subprocess.run(cmd_viz, check=True)
    print(f"Finished Experiment {exp_dir.name}\n")

def main():
    # Load all RGB frames
    all_rgb_frames = sorted([p for p in RGB_DIR.glob("*.png")])
    print(f"Total original frames: {len(all_rgb_frames)}")
    
    # Check if we have enough frames
    if len(all_rgb_frames) < 601:
        print("Not enough frames for experiments up to index 600.")
        return

    # Define the 7 experiments
    experiments = [
        # 1. [N=150] Подаём N кадров, разметка только для первого кадра
        {"id": 1, "start": 300, "end": 450, "masks": [300], "bidirectional": False},
        
        # 2. [N=75] То же самое
        {"id": 2, "start": 300, "end": 375, "masks": [300], "bidirectional": False},
        
        # 3. [N=150] Подаём N кадров, разметка для первого и последнего кадров
        {"id": 3, "start": 300, "end": 450, "masks": [300, 450], "bidirectional": False},
        
        # 4. [N=75] То же самое
        {"id": 4, "start": 300, "end": 375, "masks": [300, 375], "bidirectional": False},
        
        # 5. [N=150] Подаём N×2 кадров, разметка для первого, среднего и последнего кадров
        {"id": 5, "start": 300, "end": 600, "masks": [300, 450, 600], "bidirectional": False},
        
        # 6. [N=75] То же самое
        {"id": 6, "start": 300, "end": 450, "masks": [300, 375, 450], "bidirectional": False},
        
        # 7. Подаём 300 кадров, разметка предоставляется для каждого 75-го кадра
        {"id": 7, "start": 300, "end": 600, "masks": [300, 375, 450, 525, 600], "bidirectional": False},
    ]

    for exp in experiments:
        paths = setup_experiment(exp["id"], exp["start"], exp["end"], exp["masks"], all_rgb_frames)
        exp_dir, rgb_tmp_dir, masks_tmp_dir, out_npz_dir, out_overlay_dir, out_masks_dir = paths
        
        run_experiment(exp_dir, rgb_tmp_dir, masks_tmp_dir, out_npz_dir, out_overlay_dir, out_masks_dir, bidirectional=exp["bidirectional"])

if __name__ == "__main__":
    main()
