import os
import json
import torch
import numpy as np
import cv2
import glob
import logging
import argparse
from tqdm import tqdm
import sys
import colorsys
from sam3.logger import get_logger

# Настройка путей
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CURRENT_DIR, ".."))

try:
    from sam3.model_builder import build_sam3_video_model
    from sam3.visualization_utils import load_frame
except ImportError:
    print("Error: Could not import sam3 modules. Ensure the script is in the correct directory.")
    sys.exit(1)

try:
    from pycocotools import mask as mask_utils
except ImportError:
    print("Error: pycocotools not found. Install it with 'pip install pycocotools'.")
    sys.exit(1)

# Настройка логгера
logger = get_logger("sam3_redo_coco", level=logging.INFO)

def generate_bright_colors(n=256):
    """Генерация ярких цветов (из sam3_batch_run.py)."""
    colors = []
    golden_ratio_conjugate = 0.618033988749895
    h = np.random.random()
    for i in range(n):
        h += golden_ratio_conjugate
        h %= 1
        s = 0.6 + np.random.random() * 0.3
        v = 0.8 + np.random.random() * 0.2
        rgb = colorsys.hsv_to_rgb(h, s, v)
        colors.append(rgb)
    return np.array(colors)

def decode_rle_mask(segmentation, size):
    """Декодирует RLE маску из формата COCO."""
    if isinstance(segmentation, dict):
        if "counts" in segmentation:
            if isinstance(segmentation["counts"], list):
                segmentation = mask_utils.frPyObjects(segmentation, size[0], size[1])
            return mask_utils.decode(segmentation)
    return mask_utils.decode(mask_utils.frPyObjects(segmentation, size[0], size[1]))

def main():
    # --- DEFAULT PATHS ---
    DEFAULT_VIDEO_PATH = "/workspace/2-half-blind-no-light-day_e3f"
    DEFAULT_COCO_JSON = "/workspace/sam3/0example_coco_anno.json"
    DEFAULT_OUTPUT_DIR = "/workspace/sam3_redo_results"
    # ---------------------

    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, default=DEFAULT_VIDEO_PATH)
    parser.add_argument("--coco_json", type=str, default=DEFAULT_COCO_JSON)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--obj_id_offset", type=int, default=100)
    args = parser.parse_args()

    logger.info(f"Loading video from {args.video_path}")
    logger.info(f"Loading COCO from {args.coco_json}")

    # 1. Init Model
    sam3_model = build_sam3_video_model(device=args.device)
    predictor = sam3_model.tracker
    predictor.backbone = sam3_model.detector.backbone
    
    if args.device == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

    # 2. Init State
    inference_state = predictor.init_state(video_path=args.video_path)
    video_width = inference_state["video_width"]
    video_height = inference_state["video_height"]
    logger.info(f"Video resolution: {video_width}x{video_height}")

    # 3. Frame Mapping
    frame_files = [f for f in os.listdir(args.video_path) 
                   if os.path.splitext(f)[-1].lower() in [".jpg", ".jpeg", ".png"]]
    try:
        frame_files.sort(key=lambda p: int(os.path.splitext(p)[0]))
    except ValueError:
        frame_files.sort()
    filename_to_idx = {f: i for i, f in enumerate(frame_files)}

    # 4. Parse COCO
    with open(args.coco_json, 'r') as f:
        coco_data = json.load(f)
    images_map = {img['id']: img['file_name'] for img in coco_data['images']}

    # 5. Add Prompts
    logger.info("--- INJECTING PROMPTS ---")
    prompt_frame_indices = set()
    
    for ann in tqdm(coco_data['annotations'], desc="Processing annotations"):
        img_filename = images_map.get(ann['image_id'])
        img_basename = os.path.basename(img_filename)
        
        if img_basename not in filename_to_idx:
            continue
            
        frame_idx = filename_to_idx[img_basename]
        ann_obj_id = ann['id'] + args.obj_id_offset
        
        # 5.1 Box
        bbox = ann.get('bbox')
        if bbox:
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]
            rel_box = np.array([[x1/video_width, y1/video_height, x2/video_width, y2/video_height]], dtype=np.float32)
            predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=frame_idx,
                obj_id=ann_obj_id,
                box=rel_box,
            )

        # 5.2 Point
        mask = decode_rle_mask(ann['segmentation'], [video_height, video_width])
        coords = np.argwhere(mask > 0)
        
        if coords.size > 0:
            y_c, x_c = coords[len(coords)//2]
            rel_points = torch.tensor([[x_c/video_width, y_c/video_height]], dtype=torch.float32)
            labels = torch.tensor([1], dtype=torch.int32)
            
            predictor.add_new_points(
                inference_state=inference_state,
                frame_idx=frame_idx,
                obj_id=ann_obj_id,
                points=rel_points,
                labels=labels,
                clear_old_points=False,
            )
            
            # 5.3 Mask (Advanced)
            # Добавляем полную бинарную маску как самый сильный промпт.
            # Благодаря propagate_preflight=True, она корректно консолидируется в память.
            mask_2d = torch.from_numpy(mask).to(args.device).to(torch.float32).squeeze()
            if mask_2d.ndim == 2:
                logger.info(f"Adding MASK prompt for obj_id={ann_obj_id} on frame {frame_idx}")
                predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=frame_idx,
                    obj_id=ann_obj_id,
                    mask=mask_2d
                )
        
        prompt_frame_indices.add(frame_idx)

    if not prompt_frame_indices:
        logger.error("No prompts were successfully added.")
        return

    # 6. Propagation
    logger.info("--- STARTING PROPAGATION ---")
    video_segments = {}
    MAX_TRACK = 100
    
    min_idx = min(prompt_frame_indices)
    max_idx = max(prompt_frame_indices)

    # Forward
    logger.info(f"Forward Pass from frame {min_idx}...")
    for frame_idx, obj_ids, low_res_masks, video_res_masks, obj_scores in predictor.propagate_in_video(
        inference_state, 
        start_frame_idx=min_idx, 
        max_frame_num_to_track=MAX_TRACK,
        reverse=False,
        propagate_preflight=True
    ):
        if frame_idx not in video_segments: video_segments[frame_idx] = {}
        for i, out_obj_id in enumerate(obj_ids):
            video_segments[frame_idx][out_obj_id] = (video_res_masks[i] > 0.0).cpu().numpy()

    # Backward
    logger.info(f"Backward Pass from frame {max_idx}...")
    for frame_idx, obj_ids, low_res_masks, video_res_masks, obj_scores in predictor.propagate_in_video(
        inference_state, 
        start_frame_idx=max_idx, 
        max_frame_num_to_track=MAX_TRACK,
        reverse=True,
        propagate_preflight=True
    ):
        if frame_idx not in video_segments: video_segments[frame_idx] = {}
        for i, out_obj_id in enumerate(obj_ids):
            video_segments[frame_idx][out_obj_id] = (video_res_masks[i] > 0.0).cpu().numpy()

    # 7. Visualization & Saving
    logger.info("--- VISUALIZING AND SAVING RESULTS ---")
    
    os.makedirs(args.output_dir, exist_ok=True)
    masks_dir = os.path.join(args.output_dir, "masks")
    os.makedirs(masks_dir, exist_ok=True)
    vis_dir = os.path.join(args.output_dir, "visualization")
    os.makedirs(vis_dir, exist_ok=True)
    
    COLORS = generate_bright_colors(max(100, len(prompt_frame_indices) + 10))
    temp_video_path = os.path.join(vis_dir, "temp_render.mp4")
    final_video_path = os.path.join(vis_dir, "output_redo.mp4")
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(temp_video_path, fourcc, 24, (video_width, video_height))

    for frame_idx in tqdm(sorted(video_segments.keys()), desc="Rendering video"):
        img_path = os.path.join(args.video_path, frame_files[frame_idx])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        overlay = img.copy()
        objects = video_segments[frame_idx]
        
        # Сохранение масок
        frame_name = os.path.splitext(frame_files[frame_idx])[0]
        f_dir = os.path.join(masks_dir, frame_name)
        os.makedirs(f_dir, exist_ok=True)
        
        for obj_id, mask in objects.items():
            color = COLORS[obj_id % len(COLORS)]
            color255 = tuple(int(x * 255) for x in color)
            mask_bool = mask.squeeze() > 0
            
            # Оверлей
            alpha = 0.5
            for c in range(3):
                overlay[..., c][mask_bool] = (alpha * color255[c] + (1 - alpha) * overlay[..., c][mask_bool]).astype(np.uint8)
            
            # Подпись
            coords = np.argwhere(mask_bool)
            if coords.size > 0:
                y_m, x_m = coords.mean(axis=0)
                cv2.putText(overlay, f"ID:{obj_id}", (int(x_m), int(y_m)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color255, 2)
            
            # Маска PNG
            cv2.imwrite(os.path.join(f_dir, f"obj_{obj_id}.png"), (mask.squeeze()*255).astype(np.uint8))

        video_writer.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    video_writer.release()
    os.system(f"ffmpeg -y -i {temp_video_path} -vcodec libx264 -crf 23 -pix_fmt yuv420p {final_video_path} > /dev/null 2>&1")
    if os.path.exists(temp_video_path): os.remove(temp_video_path)

    logger.info(f"SUCCESS: Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
