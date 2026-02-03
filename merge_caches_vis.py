import os
import argparse
import glob
import numpy as np
import cv2
import colorsys
from tqdm import tqdm
from collections import defaultdict
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Дефолтные пути из запроса
DEFAULT_TEXT_CACHE = "/workspace/sam3_batch_results_3023_16_01_2026/cache"
DEFAULT_MASK_CACHE = "/workspace/sam3_redo_results/02-02-1433/cache"
DEFAULT_VIDEO_PATH = "/workspace/2-half-blind-no-light-day_e3f" # Предполагаемый путь к видео, нужно уточнить если отличается
DEFAULT_OUTPUT_DIR = "/workspace/sam3_merged_results"
NO_BG = True
NO_BOX = True
NO_LABEL = True
NO_MASK = False

def generate_bright_colors(n=256):
    """Генерация списка ярких цветов (как в других скриптах SAM3)"""
    colors = []
    golden_ratio_conjugate = 0.618033988749895
    h = np.random.random()
    for i in range(n):
        h += golden_ratio_conjugate
        h %= 1
        s = 0.6 + np.random.random() * 0.3
        v = 0.8 + np.random.random() * 0.2
        rgb = colorsys.hsv_to_rgb(h, s, v)
        colors.append(tuple(int(c * 255) for c in rgb))
    return colors

def load_frames_from_folder(video_path):
    """Загрузка путей к кадрам"""
    if not os.path.isdir(video_path):
        raise ValueError(f"Path is not a directory: {video_path}")
        
    exts = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']
    frames = []
    for ext in exts:
        frames.extend(glob.glob(os.path.join(video_path, ext)))
    
    try:
        # Пытаемся сортировать по номеру кадра в имени файла
        frames.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    except:
        logger.warning("Could not sort frames numerically, falling back to lexicographic sort")
        frames.sort()
        
    return frames

def load_cache_data(text_cache_dir, mask_cache_dir):
    """
    Загружает кэши и объединяет их в структуру:
    frame_idx -> list of objects {mask, box, label, obj_id, source_type}
    """
    frame_data = defaultdict(list)
    
    # 1. Load Text Prompts Cache (Batch Run)
    if text_cache_dir and os.path.exists(text_cache_dir):
        files = glob.glob(os.path.join(text_cache_dir, "*.npz"))
        logger.info(f"Loading {len(files)} text prompt cache files from {text_cache_dir}...")
        
        for fpath in tqdm(files, desc="Text Cache"):
            prompt_name = os.path.splitext(os.path.basename(fpath))[0]
            try:
                data = np.load(fpath, allow_pickle=True)
                if 'outputs' not in data:
                    logger.warning(f"Skipping {fpath}: no 'outputs' key")
                    continue
                    
                outputs = data['outputs'].item()
                
                for f_idx, content in outputs.items():
                    # Content keys: out_obj_ids, out_binary_masks, out_boxes_xywh, etc.
                    num_objs = len(content['out_obj_ids'])
                    for i in range(num_objs):
                        mask = content['out_binary_masks'][i]
                        # Fix dims if needed
                        if mask.ndim == 3: mask = mask[0]
                        
                        box = content['out_boxes_xywh'][i] if len(content['out_boxes_xywh']) > i else None
                        
                        obj_info = {
                            'mask': mask,
                            'box': box,
                            'label': prompt_name,
                            'obj_id': int(content['out_obj_ids'][i]),
                            'source': 'text'
                        }
                        frame_data[int(f_idx)].append(obj_info)
                        
            except Exception as e:
                logger.error(f"Error loading {fpath}: {e}")

    # 2. Load Mask Prompts Cache (Redo from COCO)
    if mask_cache_dir and os.path.exists(mask_cache_dir):
        files = glob.glob(os.path.join(mask_cache_dir, "*.npz"))
        logger.info(f"Loading {len(files)} mask prompt cache files from {mask_cache_dir}...")
        
        for fpath in tqdm(files, desc="Mask Cache"):
            # Filename usually obj_101.npz
            basename = os.path.splitext(os.path.basename(fpath))[0]
            try:
                data = np.load(fpath, allow_pickle=True)
                if 'outputs' not in data:
                    # Fallback for old simple format (just masks)
                    # Но мы обновили скрипт, так что ожидаем новый формат
                    logger.warning(f"Skipping {fpath}: no 'outputs' key (old format?)")
                    continue

                outputs = data['outputs'].item()
                
                for f_idx, content in outputs.items():
                    # Content structure similar to batch run now
                    num_objs = len(content['out_obj_ids'])
                    for i in range(num_objs):
                        mask = content['out_binary_masks'][i]
                        if mask.ndim == 3: mask = mask[0]
                        
                        box = content['out_boxes_xywh'][i] if len(content['out_boxes_xywh']) > i else None
                        
                        # Get label
                        if 'prompt_source' in content and len(content['prompt_source']) > i:
                            label = content['prompt_source'][i]
                        else:
                            label = basename # obj_101
                            
                        obj_info = {
                            'mask': mask,
                            'box': box,
                            'label': label,
                            'obj_id': int(content['out_obj_ids'][i]),
                            'source': 'mask'
                        }
                        frame_data[int(f_idx)].append(obj_info)
                        
            except Exception as e:
                logger.error(f"Error loading {fpath}: {e}")
                
    return frame_data

def main():
    parser = argparse.ArgumentParser(description="Merge and visualize SAM3 caches")
    
    # Paths
    parser.add_argument("--video_path", type=str, default=DEFAULT_VIDEO_PATH, help="Path to folder with frames")
    parser.add_argument("--text_cache_dir", type=str, default=DEFAULT_TEXT_CACHE, help="Path to batch_run cache")
    parser.add_argument("--mask_cache_dir", type=str, default=DEFAULT_MASK_CACHE, help="Path to redo_coco cache")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Where to save result video")
    parser.add_argument("--output_name", type=str, default="merged_visualization.mp4")
    
    # Visualization Flags
    parser.add_argument("--no_bg", action="store_true", default=NO_BG, help="If set, background will be black (no original image)")
    parser.add_argument("--no_mask", action="store_true", default=NO_MASK, help="Do not draw binary masks")
    parser.add_argument("--no_box", action="store_true", default=NO_BOX, help="Do not draw bounding boxes")
    parser.add_argument("--no_label", action="store_true", default=NO_LABEL, help="Do not draw text labels")
    parser.add_argument("--fps", type=int, default=10)
    
    args = parser.parse_args()
    
    # 1. Setup
    os.makedirs(args.output_dir, exist_ok=True)
    frame_paths = load_frames_from_folder(args.video_path)
    if not frame_paths:
        logger.error("No frames found!")
        return
        
    # Get dimensions from first frame
    first_img = cv2.imread(frame_paths[0])
    h, w = first_img.shape[:2]
    logger.info(f"Video resolution: {w}x{h}, Frames: {len(frame_paths)}")
    
    # 2. Load Data
    merged_data = load_cache_data(args.text_cache_dir, args.mask_cache_dir)
    frames_with_data = sorted(merged_data.keys())
    if not frames_with_data:
        logger.error("No cache data loaded!")
        return
    logger.info(f"Found data for {len(frames_with_data)} frames.")

    # 3. Prepare Renderer
    temp_out = os.path.join(args.output_dir, "temp_merged.mp4")
    final_out = os.path.join(args.output_dir, args.output_name)
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(temp_out, fourcc, args.fps, (w, h))
    
    # Assign persistent colors based on (label + obj_id) hash to keep them stable
    COLORS = generate_bright_colors(500)
    def get_color(label, obj_id):
        # Hash combination string
        h = hash(f"{label}_{obj_id}")
        return COLORS[h % len(COLORS)]

    # 4. Render Loop
    # Render all frames that exist in video folder
    for i, frame_path in enumerate(tqdm(frame_paths, desc="Rendering")):
        
        # Load background
        if args.no_bg:
            frame = np.zeros((h, w, 3), dtype=np.uint8)
        else:
            frame = cv2.imread(frame_path)
            if frame is None:
                frame = np.zeros((h, w, 3), dtype=np.uint8)
                
        # Draw Objects
        if i in merged_data:
            objects = merged_data[i]
            
            # Draw Masks First (Alpha Blending)
            if not args.no_mask:
                overlay = frame.copy()
                # If no background (black), make masks opaque for brightness
                alpha = 1.0 if args.no_bg else 0.5
                
                for obj in objects:
                    mask = obj['mask']
                    color = get_color(obj['label'], obj['obj_id'])
                    
                    # Ensure mask is bool/uint8
                    if mask.dtype != bool:
                        mask = mask > 0.5
                        
                    # Resize mask if needed (shouldn't happen if cache is correct)
                    if mask.shape != (h, w):
                         mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                         
                    # Add color
                    frame[mask] = (frame[mask] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)

            # Draw Lines/Boxes/Text
            for obj in objects:
                color = get_color(obj['label'], obj['obj_id'])
                # BGR for OpenCV
                color_bgr = color[::-1] 
                
                # Box
                if not args.no_box and obj['box'] is not None:
                    # Box format: xywh normalized or absolute? 
                    # In scripts: out_boxes_xywh is usually normalized (0-1) or absolute.
                    # Redo script saved: normalized [x_rel, y_rel, w_rel, h_rel]
                    # Batch run script saved: [x_rel, y_rel, w_rel, h_rel] (Need to verify this assumption from code analysis, usually SAM2 output is unnormalized, but my previous edit normalized it)
                    
                    # Let's assume normalized for safety if < 2.0
                    bx, by, bw, bh = obj['box']
                    
                    # Simple heuristic check
                    is_normalized = (bx <= 1.5 and by <= 1.5 and bw <= 1.5 and bh <= 1.5)
                    
                    if is_normalized:
                        x1 = int(bx * w)
                        y1 = int(by * h)
                        x2 = int((bx + bw) * w)
                        y2 = int((by + bh) * h)
                    else:
                        x1, y1 = int(bx), int(by)
                        x2, y2 = int(bx + bw), int(by + bh)
                        
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 2)
                    
                    # Label Position
                    lx, ly = x1, max(y1 - 10, 10)
                else:
                    # Find mask centroid for label if box missing
                    if obj['mask'].any():
                        coords = np.argwhere(obj['mask'])
                        ly, lx = coords.mean(axis=0).astype(int)
                    else:
                        lx, ly = 0, 0

                # Label
                if not args.no_label:
                    label_text = f"{obj['label']} ({obj['obj_id']})"
                    cv2.putText(frame, label_text, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.putText(frame, label_text, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 1)

        writer.write(frame)

    writer.release()
    
    # 5. Convert with FFMPEG
    if os.path.exists(final_out):
        os.remove(final_out)
        
    logger.info("Running FFMPEG compression...")
    os.system(f"ffmpeg -y -i {temp_out} -vcodec libx264 -crf 23 -pix_fmt yuv420p {final_out} > /dev/null 2>&1")
    
    if os.path.exists(temp_out):
        os.remove(temp_out)
        
    logger.info(f"Done! Video saved to: {final_out}")

if __name__ == "__main__":
    main()
