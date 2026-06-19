import time
import torch
import numpy as np
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm
import gc
import psutil
from concurrent.futures import ThreadPoolExecutor
from sam3.model_builder import build_sam3_video_model

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
try:
    from tg_notifier import notify_error, send_message
except ImportError:
    def notify_error(e, msg=""): print(f"Notifier not found: {msg} {e}")
    def send_message(msg): print(msg)

# DEFAULT VALUES
# anno_name = "02-16-26_seq6_aruna_anno_1iter"
# DEFAULT_WEIGHTS = "/workspace/data_mount/model_weights/sam3/sam3.pt"
# DEFAULT_INPUTS = "/workspace/02-16-26_seq6_data"
# DEFAULT_MASKS = f"/workspace/{anno_name}/SegmentationClass/seq6"
# DEFAULT_LABELMAP = f"/workspace/{anno_name}/labelmap.txt"
# DEFAULT_OUT_NPZ = f"/workspace/{anno_name}_results/masks_npz"
# DEFAULT_DEVICE = "cuda"

anno_name = "seq1-job_53_24-03-26"
DEFAULT_WEIGHTS = "/workspace/data_mount/model_weights/sam3/sam3.pt"
DEFAULT_INPUTS = "/workspace/data_mount/third_wave_tracker_rgb_input/E-3023-A5-input-rgb/seq1"
DEFAULT_MASKS = f"/workspace/3023-cvat-annotations/2stage/{anno_name}/SegmentationClass/seq1"
DEFAULT_LABELMAP = f"/workspace/3023-cvat-annotations/2stage/{anno_name}/labelmap.txt"
DEFAULT_OUT_NPZ = f"/workspace/3023-A5-SAM3-results/2stage/{anno_name}_results/masks_npz"
DEFAULT_DEVICE = "cuda"
# ---------------------

def parse_args():
    parser = argparse.ArgumentParser(description="SAM3 Video Tracking & Save to NPZ")
    parser.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS, help="Path to model weights")
    
    parser.add_argument("--inputs", type=str, default=DEFAULT_INPUTS, help="Path to RGB images")
    parser.add_argument("--masks", type=str, default=DEFAULT_MASKS, help="Path to input masks (Ground Truth)")
    parser.add_argument("--labelmap", type=str, default=DEFAULT_LABELMAP, help="Path to labelmap.txt")
    parser.add_argument("--out-npz", type=str, default=DEFAULT_OUT_NPZ, help="Output folder for raw numpy masks")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE, help="Device to use (cuda/cpu)")
    parser.add_argument("--bidirectional", action="store_true", help="Run both forward and reverse propagation")
    return parser.parse_args()

def get_mem_report():
    process = psutil.Process(os.getpid())
    ram_gb = process.memory_info().rss / (1024 ** 3)
    system_ram_percent = psutil.virtual_memory().percent
    
    report = f"RAM: {ram_gb:.2f} GB (Sys: {system_ram_percent}%)"
    
    if torch.cuda.is_available():
        vram_alloc = torch.cuda.memory_allocated() / (1024 ** 3)
        vram_res = torch.cuda.memory_reserved() / (1024 ** 3)
        report += f" | VRAM: {vram_alloc:.2f}/{vram_res:.2f} GB"
    
    return report

def _save_npz(path, mask_dict):
    np.savez_compressed(path, **mask_dict)

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

@torch.inference_mode()
def main():
    args = parse_args()
    arg_lines = []
    for key, value in vars(args).items():
        arg_lines.append(f"{key}: {value}")
    send_message("Аргументы запуска:\n```\n" + "\n".join(arg_lines) + "\n```")

    device = torch.device(args.device)

    if device.type == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    sam3_model = build_sam3_video_model(
        checkpoint_path=args.weights, 
        load_from_HF=False, 
        bpe_path='/workspace/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz'
    )
    # send_message(f"Model loaded. {get_mem_report()}")
    predictor = sam3_model.tracker
    predictor.backbone = sam3_model.detector.backbone
    predictor.non_overlap_masks_for_output = True

    input_path = Path(args.inputs)
    frame_names = sorted([p for p in input_path.glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]])
    frame_names_stems = [p.stem for p in frame_names]
    print(f"Total frames in input: {len(frame_names)}")
    
    input_masks = sorted([p for p in Path(args.masks).glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]])
    
    out_npz_path = Path(args.out_npz)
    out_npz_path.mkdir(parents=True, exist_ok=True)

    objects = load_objects_from_labelmap(args.labelmap)
    print(f"Loaded {len(objects)} objects from labelmap.")

    inference_state = predictor.init_state(
        video_path=str(input_path), 
        async_loading_frames=True, 
        offload_video_to_cpu=True, 
        offload_state_to_cpu=True
    )

    for mask_path in input_masks:
        mask_image = cv2.imread(str(mask_path))
        mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)
        mask_path_stem = mask_path.stem
        
        try:
            frame_idx = frame_names_stems.index(mask_path_stem)
        except ValueError:
            int_mask_path = int(mask_path_stem) if mask_path_stem.isdigit() else int(mask_path_stem.split('frame')[1])
            frame_idx = int_mask_path

        added_objs = 0
        for obj_id, color in enumerate(objects):
            mask_np = np.all(mask_image == color, axis=-1)
            if mask_np.any():
                mask_tensor = torch.from_numpy(mask_np).to(device)
                predictor.add_new_mask(inference_state, frame_idx, obj_id, mask_tensor)
                added_objs += 1
        print(f"Annotated frame {frame_idx} added! ({added_objs} objects)")
    
    print("Propagating video (forward) and saving to disk...")
    len_frame_names = len(frame_names)
    report_every = max(1, len_frame_names // 2)

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for out_frame_idx, out_obj_ids, _, out_mask_logits, _ in predictor.propagate_in_video(inference_state, start_frame_idx=0, max_frame_num_to_track=None, reverse=False, propagate_preflight=True):

            frame_mask_dict = {
                str(out_obj_id): (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
            fname = frame_names[out_frame_idx].stem
            futures.append(executor.submit(_save_npz, out_npz_path / f"{fname}.npz", frame_mask_dict))

            # if out_frame_idx > 0 and out_frame_idx % report_every == 0:
            #     send_message(f"Forward progress: {out_frame_idx}/{len_frame_names} ({out_frame_idx/len_frame_names:.0%}). {get_mem_report()}")

        for f in futures:
            f.result()

    if args.bidirectional:
        print("Propagating video (reverse) and saving to disk...")
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for out_frame_idx, out_obj_ids, _, out_mask_logits, _ in predictor.propagate_in_video(inference_state, start_frame_idx=len_frame_names - 1, max_frame_num_to_track=None, reverse=True, propagate_preflight=True):

                frame_mask_dict = {
                    str(out_obj_id): (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
                fname = frame_names[out_frame_idx].stem
                futures.append(executor.submit(_save_npz, out_npz_path / f"{fname}.npz", frame_mask_dict))

                progress_idx = len_frame_names - 1 - out_frame_idx
                # if progress_idx > 0 and progress_idx % report_every == 0:
                #     send_message(f"Reverse progress: {progress_idx}/{len_frame_names} ({progress_idx/len_frame_names:.0%}). {get_mem_report()}")

            for f in futures:
                f.result()

    send_message(f"Propagation finished. {get_mem_report()}")

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
