import time
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm
import gc
import psutil
import numpy as np

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
try:
    from tg_notifier import notify_error, send_message
except ImportError:
    def notify_error(e, msg=""): print(f"Notifier not found: {msg} {e}")
    def send_message(msg): print(msg)

# DEFAULT VALUES
anno_name = "02-16-26_seq6_aruna_anno_1iter"
DEFAULT_INPUTS = "/workspace/02-16-26_seq6_data"
DEFAULT_IN_NPZ = f"/workspace/{anno_name}_results/masks_npz"
DEFAULT_LABELMAP = f"/workspace/{anno_name}/labelmap.txt"
DEFAULT_OUT_OVERLAY = f"/workspace/{anno_name}_results/overlay"
DEFAULT_OUT_MASKS = f"/workspace/{anno_name}_results/masks"
# ---------------------

def parse_args():
    parser = argparse.ArgumentParser(description="SAM3 Video Visualization from NPZ")
    parser.add_argument("--inputs", type=str, default=DEFAULT_INPUTS, help="Path to RGB images")
    parser.add_argument("--in-npz", type=str, default=DEFAULT_IN_NPZ, help="Path to saved numpy masks")
    parser.add_argument("--labelmap", type=str, default=DEFAULT_LABELMAP, help="Path to labelmap.txt")
    parser.add_argument("--out-overlay", type=str, default=DEFAULT_OUT_OVERLAY, help="Output folder for overlays")
    parser.add_argument("--out-masks", type=str, default=DEFAULT_OUT_MASKS, help="Output folder for binary masks")
    parser.add_argument("--no-overlay", action="store_true", help="Skip saving overlay images (saves disk space and time)")
    return parser.parse_args()

def get_mem_report():
    process = psutil.Process(os.getpid())
    ram_gb = process.memory_info().rss / (1024 ** 3)
    system_ram_percent = psutil.virtual_memory().percent
    return f"RAM: {ram_gb:.2f} GB (Sys: {system_ram_percent}%)"

def load_objects_from_labelmap(path):
    objects = []
    with open(path, 'r') as f:
        lines = f.readlines()
    for line in lines[2:]:
        parts = line.strip().split(':')
        if len(parts) < 2: continue
        r, g, b = map(int, parts[1].split(','))
        objects.append([r, g, b])
    return objects

def main():
    args = parse_args()

    input_path = Path(args.inputs)
    frame_names = sorted([p for p in input_path.glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]])
    print(f"Total frames in input: {len(frame_names)}")
    
    in_npz_path = Path(args.in_npz)
    npz_files = sorted(list(in_npz_path.glob("*.npz")))
    print(f"Total NPZ mask files found: {len(npz_files)}")
    
    save_overlay = not args.no_overlay
    out_overlay_path = Path(args.out_overlay)
    out_masks_path = Path(args.out_masks)
    if save_overlay:
        out_overlay_path.mkdir(parents=True, exist_ok=True)
    out_masks_path.mkdir(parents=True, exist_ok=True)

    objects = load_objects_from_labelmap(args.labelmap)
    print(f"Loaded {len(objects)} objects from labelmap.")

    # Match frames with NPZ files by stem
    frame_dict = {p.stem: p for p in frame_names}
    
    total_to_save = len(npz_files)
    report_every = max(1, total_to_save // 2)

    print("Generating and saving overlays...")
    for i, npz_path in enumerate(tqdm(npz_files)):
        fname = npz_path.stem
        
        if fname not in frame_dict:
            print(f"Warning: No matching original frame found for {fname}, skipping.")
            continue
            
        frame_path = frame_dict[fname]
        frame = cv2.imread(str(frame_path))
        if frame is None:
            print(f"Warning: Could not read frame {frame_path}, skipping.")
            continue
            
        res_mask_rgb = np.zeros_like(frame)
        overlay = frame.copy()

        # Load NPZ mask
        loaded_npz = np.load(npz_path)
        
        for obj_id_str in loaded_npz.files:
            obj_id = int(obj_id_str)
            out_mask = loaded_npz[obj_id_str]
            
            color_rgb = np.array(objects[obj_id], dtype=np.uint8)
            color_bgr = color_rgb[::-1]
            mask_bool = out_mask.astype(bool)

            empty_area = np.all(res_mask_rgb == 0, axis=-1)
            target_pixels = empty_area & mask_bool
            
            res_mask_rgb[target_pixels] = color_rgb
            overlay[mask_bool] = (overlay[mask_bool] * 0.4 + color_bgr * 0.6).astype(np.uint8)

        if save_overlay:
            cv2.imwrite(f"{out_overlay_path}/{fname}.png", overlay)
        cv2.imwrite(f"{out_masks_path}/{fname}.png", cv2.cvtColor(res_mask_rgb, cv2.COLOR_RGB2BGR))
        
        # Cleanup memory immediately
        loaded_npz.close()
        
        if i > 0 and i % report_every == 0:
            # send_message(f"Rendering progress: {i}/{total_to_save} ({i/total_to_save:.0%}). {get_mem_report()}")
            gc.collect()

    send_message(f"Rendering finished. {get_mem_report()}")

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
