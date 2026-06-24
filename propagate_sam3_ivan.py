import time
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm
import gc
import psutil
from sam3.model_builder import build_sam3_video_model

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
try:
    from tg_notifier import notify_error, send_message
except ImportError:
    def notify_error(e, msg=""): print(f"Notifier not found: {msg} {e}")

# DEFAULT VALUES
anno_name = "02-16-26_seq6_aruna_anno_1iter"
DEFAULT_WEIGHTS = "/workspace/data_mount/model_weights/sam3/sam3.pt"
# DEFAULT_INPUTS = "/workspace/seq2img"
DEFAULT_INPUTS = "/workspace/02-16-26_seq6_data"
DEFAULT_MASKS = f"/workspace/{anno_name}/SegmentationClass/seq6"
DEFAULT_LABELMAP = f"/workspace/{anno_name}/labelmap.txt"
DEFAULT_OUT_OVERLAY = f"/workspace/{anno_name}_results/overlay"
DEFAULT_OUT_MASKS = f"/workspace/{anno_name}_results/masks"
DEFAULT_DEVICE = "cuda"
# ---------------------

def parse_args():
    parser = argparse.ArgumentParser(description="SAM3 Video Propagation CLI")
    parser.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS, help="Path to model weights")
    
    parser.add_argument("--inputs", type=str, default=DEFAULT_INPUTS, help="Path to RGB images")
    parser.add_argument("--masks", type=str, default=DEFAULT_MASKS, help="Path to input masks")
    parser.add_argument("--labelmap", type=str, default=DEFAULT_LABELMAP, help="Path to labelmap.txt")
    parser.add_argument("--out-overlay", type=str, default=DEFAULT_OUT_OVERLAY, help="Output folder for overlays")
    parser.add_argument("--out-masks", type=str, default=DEFAULT_OUT_MASKS, help="Output folder for binary masks")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE, help="Device to use (cuda/cpu)")
    return parser.parse_args()

def get_mem_report():
    # RAM этого процесса
    process = psutil.Process(os.getpid())
    ram_gb = process.memory_info().rss / (1024 ** 3)
    # Общая загрузка системы
    system_ram_percent = psutil.virtual_memory().percent
    
    report = f"RAM: {ram_gb:.2f} GB (Sys: {system_ram_percent}%)"
    
    if torch.cuda.is_available():
        # Выделено прямо сейчас / Зарезервировано кэшем CUDA
        vram_alloc = torch.cuda.memory_allocated() / (1024 ** 3)
        vram_res = torch.cuda.memory_reserved() / (1024 ** 3)
        report += f" | VRAM: {vram_alloc:.2f}/{vram_res:.2f} GB"
    
    return report

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
    send_message(f"Model loaded. {get_mem_report()}")
    predictor = sam3_model.tracker
    predictor.backbone = sam3_model.detector.backbone
    predictor.non_overlap_masks_for_output = True

    input_path = Path(args.inputs)
    frame_names = sorted([p for p in input_path.glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]])
    frame_names_stems = [p.stem for p in frame_names]
    print(f"frame_names_stems: {len(frame_names_stems)}")
    input_masks = sorted([p for p in Path(args.masks).glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]])
    
    Path(args.out_overlay).mkdir(parents=True, exist_ok=True)
    Path(args.out_masks).mkdir(parents=True, exist_ok=True)

    objects = load_objects_from_labelmap(args.labelmap)

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
            # Fallback к старому способу, если имя маски не найдено напрямую
            int_mask_path = int(mask_path_stem) if mask_path_stem.isdigit() else int(mask_path_stem.split('frame')[1])
            frame_idx = int_mask_path

        for obj_id, color in enumerate(objects):
            mask_np = np.all(mask_image == color, axis=-1)
            if mask_np.any():
                mask_tensor = torch.from_numpy(mask_np).to(device)
                predictor.add_new_mask(inference_state, frame_idx, obj_id, mask_tensor)
        print(f'Annotated frame {frame_idx} added!')
    
    print("Propagating video...")
    len_frame_names = len(frame_names)
    report_every = max(1, len_frame_names // 4)
    video_segments = {}
    for out_frame_idx, out_obj_ids, _, out_mask_logits, _ in predictor.propagate_in_video(inference_state, start_frame_idx=0, max_frame_num_to_track=None, reverse=False, propagate_preflight=True):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
        if out_frame_idx > 0 and out_frame_idx % report_every == 0:
            send_message(f"Propagation progress: {out_frame_idx}/{len_frame_names} ({out_frame_idx/len_frame_names:.0%}). {get_mem_report()}")

    # for out_frame_idx, out_obj_ids, _, out_mask_logits, _ in predictor.propagate_in_video(inference_state, start_frame_idx=len(frame_names) - 1, max_frame_num_to_track=None, reverse=True, propagate_preflight=True):
    #     video_segments[out_frame_idx] = {
    #         out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
    #         for i, out_obj_id in enumerate(out_obj_ids)
    #     }
    #     if out_frame_idx > 0 and out_frame_idx % report_every == 0:
    #         send_message(f"Propagation progress: {out_frame_idx}/{len_frame_names} ({out_frame_idx/len_frame_names:.0%}). {get_mem_report()}")

    send_message(f"Propagation finished. {get_mem_report()}")

    del inference_state
    del sam3_model
    del predictor
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    send_message(f"GPU cleared. {get_mem_report()}")

    print("Saving results...")
    frame_indices = sorted(list(video_segments.keys()))
    total_to_save = len(frame_indices)
    report_every = max(1, total_to_save // 4)

    for i, out_frame_idx in enumerate(tqdm(frame_indices)):
        if i > 0 and i % report_every == 0:
            send_message(f"Saving progress: {i}/{total_to_save} ({i/total_to_save:.0%}). {get_mem_report()}")
            gc.collect()

        frame_data = video_segments.pop(out_frame_idx)
        frame = cv2.imread(str(frame_names[out_frame_idx]))
        if frame is None:
            print(f"Warning: Could not read frame {frame_names[out_frame_idx]}, skipping.")
            continue
        res_mask_rgb = np.zeros_like(frame)
        overlay = frame.copy()

        for obj_id, out_mask in frame_data.items():
            color_rgb = np.array(objects[obj_id], dtype=np.uint8)
            color_bgr = color_rgb[::-1]
            mask_bool = out_mask.squeeze().astype(bool)

            empty_area = np.all(res_mask_rgb == 0, axis=-1)
            target_pixels = empty_area & mask_bool
            
            res_mask_rgb[target_pixels] = color_rgb
            overlay[mask_bool] = (overlay[mask_bool] * 0.4 + color_bgr * 0.6).astype(np.uint8)

        fname = frame_names[out_frame_idx].stem
        cv2.imwrite(f"{args.out_overlay}/{fname}.png", overlay)
        cv2.imwrite(f"{args.out_masks}/{fname}.png", cv2.cvtColor(res_mask_rgb, cv2.COLOR_RGB2BGR))
        
        del frame_data
        del frame
        del res_mask_rgb
        del overlay

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
