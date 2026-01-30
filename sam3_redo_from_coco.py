from dataclasses import dataclass
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
import yaml
from collections import defaultdict
from sam3.logger import get_logger
from sam3.model.sam3_video_inference import Sam3VideoInferenceWithInstanceInteractivity
from sam3.model.sam3_tracking_predictor import Sam3TrackerPredictor
from sam3.model.sam3_image import Sam3ImageOnVideoMultiGPU
from sam3.model.vl_combiner import SAM3VLBackbone

# Настройка путей
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CURRENT_DIR, ".."))

from sam3.model_builder import build_sam3_video_model
from pycocotools import mask as mask_utils

# Конфигурация логгера
logger = get_logger("sam3_redo_coco", level=logging.INFO)

try:
    from tg_notifier import notify_error, send_message
except ImportError:
    logger.warning("Notifier not found, using dummy function")
    def notify_error(e, msg=""): print(f"Notifier not found: {msg} {e}")
    def send_message(): pass


# ---------------------
DEFAULT_VIDEO_PATH = "/workspace/2-half-blind-no-light-day_e3f"
DEFAULT_COCO_JSON = "/workspace/sam3/coco_all_redo_29-01_00-38.json"
DEFAULT_OUTPUT_DIR = "/workspace/sam3_redo_results"
DEFAULT_JOIN_OBJECT_PATH = "/workspace/sam3/coco_join_objects.yml"

USE_MASK_PROMPT = True
USE_BBOX_PROMPT = False
USE_POINT_PROMPT = False
# ---------------------

@dataclass
class ScriptArgs:
    video_path: str
    coco_json: str
    output_dir: str
    device: str
    obj_id_offset: int
    use_mask_prompt: bool
    use_bbox_prompt: bool
    use_point_prompt: bool
    id_groups: str

def get_args() -> ScriptArgs:
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--video_path", type=str, default=DEFAULT_VIDEO_PATH)
    parser.add_argument("--coco_json", type=str, default=DEFAULT_COCO_JSON)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--obj_id_offset", type=int, default=100)
    
    parser.add_argument("--use_mask_prompt", action="store_true", default=USE_MASK_PROMPT)
    parser.add_argument("--use_bbox_prompt", action="store_true", default=USE_BBOX_PROMPT)
    parser.add_argument("--use_point_prompt", action="store_true", default=USE_POINT_PROMPT)
    
    parser.add_argument("--id_groups", type=str, default=DEFAULT_JOIN_OBJECT_PATH, 
                        help="Строка JSON/YAML со списком групп ID для склейки: '[[105, 108], [200, 205]]'")
    
    args = parser.parse_args()
    
    return ScriptArgs(**vars(args))

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

def reset_inference_state(predictor: Sam3TrackerPredictor, inference_state: dict):
    """
    Reset inference state to clear inputs and tracking memory for a new object,
    keeping loaded images and their features intact.
    """
    if hasattr(predictor, "clear_all_points_in_video"):
        predictor.clear_all_points_in_video(inference_state)
    else:
        logger.warning("Method clear_all_points_in_video not found in predictor! Using manual reset.")
        # Fallback manual reset (including missing first_ann_frame_idx)
        inference_state["obj_ids"] = []
        inference_state["obj_idx_to_id"] = {}
        inference_state["obj_id_to_idx"] = {}
        inference_state["point_inputs_per_obj"] = {}
        inference_state["mask_inputs_per_obj"] = {}
        inference_state["output_dict"] = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
        inference_state["output_dict_per_obj"] = {}
        inference_state["temp_output_dict_per_obj"] = {}
        inference_state["consolidated_frame_inds"] = {"cond_frame_outputs": set(), "non_cond_frame_outputs": set()}
        inference_state["tracking_has_started"] = False
        inference_state["frames_already_tracked"] = {}
        inference_state["first_ann_frame_idx"] = None

def log_memory_usage(tag=""):
    if torch.cuda.is_available():
        mem_alloc = torch.cuda.memory_allocated() / 1024**3
        mem_res = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"[{tag}] VRAM: {mem_alloc:.2f} GB allocated, {mem_res:.2f} GB reserved")

def main():
    args = get_args()

    # --- 1. PREPARATION ---
    
    # Загрузка маппинга ID
    id_map = {}
    if args.id_groups:
        try:
            if os.path.isfile(args.id_groups):
                with open(args.id_groups, 'r') as f:
                    if args.id_groups.lower().endswith(('.yaml', '.yml')):
                        groups = yaml.safe_load(f)
                        logger.info(f"Loaded ID groups from YAML: {args.id_groups}")
                    else:
                        groups = json.load(f)
                        logger.info(f"Loaded ID groups from JSON: {args.id_groups}")
            else:
                try:
                    groups = json.loads(args.id_groups)
                except:
                    groups = yaml.safe_load(args.id_groups)
            
            if not isinstance(groups, list):
                raise ValueError(f"id_groups must be a list of lists, got {type(groups)}")

            # Преобразование [[A, B], [C, D, E]] -> {A:A, B:A, C:C, D:C, E:C}
            for group in groups:
                if not group: continue
                # Используем первый ID группы как Target ID для всех членов группы
                target_id = int(group[0]) + args.obj_id_offset # Добавляем оффсет сразу, чтобы не конфликтовать
                for member_id in group:
                    id_map[int(member_id)] = target_id
            
            logger.info(f"Loaded ID groups. Generated mapping size: {len(id_map)}")
            logger.debug(f"Mapping: {id_map}")

        except Exception as e:
            logger.error(f"Failed to parse id_groups: {e}")
            return
    logger.info(groups)
    logger.info(f"Loading video from {args.video_path}")
    logger.info(f"Loading COCO from {args.coco_json}")

    # Init Model
    sam3_model: Sam3VideoInferenceWithInstanceInteractivity = build_sam3_video_model(device=args.device)
    predictor: Sam3TrackerPredictor = sam3_model.tracker
    predictor.backbone: SAM3VLBackbone = sam3_model.detector.backbone
    
    if args.device == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

    # Init State (Load Frames once)
    inference_state = predictor.init_state(
        video_path=args.video_path, 
        # async_loading_frames=False,
        )
    video_width = inference_state["video_width"]
    video_height = inference_state["video_height"]
    logger.info(f"Video resolution: {video_width}x{video_height}")

    # Frame Mapping
    frame_files = [f for f in os.listdir(args.video_path) 
                   if os.path.splitext(f)[-1].lower() in [".jpg", ".jpeg", ".png"]]
    try:
        frame_files.sort(key=lambda p: int(os.path.splitext(p)[0]))
    except ValueError:
        frame_files.sort()
    filename_to_idx = {f: i for i, f in enumerate(frame_files)}
    logger.info(f"Total frames in folder: {len(filename_to_idx)}")

    # Parse COCO and Group Prompts by Object ID
    with open(args.coco_json, 'r') as f:
        coco_data = json.load(f)
    images_map = {img['id']: img['file_name'] for img in coco_data['images']}

    # Группируем промпты по объектам
    obj_prompts = defaultdict(list)
    
    logger.info("Parsing annotations and grouping by object...")
    for ann in tqdm(coco_data['annotations'], desc="Grouping annotations"):
        img_filename = images_map.get(ann['image_id'])
        img_basename = os.path.basename(img_filename)
        
        if img_basename not in filename_to_idx:
            continue
            
        frame_idx = filename_to_idx[img_basename]
        
        # Определяем целевой ID объекта для SAM3
        raw_id = ann['id']
        
        if raw_id in id_map:
            ann_obj_id = id_map[raw_id]
            logger.info(f"ReID: Merging CVAT ID {raw_id} -> SAM ID {ann_obj_id}")
        else:
            ann_obj_id = raw_id + args.obj_id_offset
            
        obj_prompts[ann_obj_id].append({
            'frame_idx': frame_idx,
            'bbox': ann.get('bbox'),
            'segmentation': ann.get('segmentation'),
            'raw_id': raw_id
        })

    logger.info(f"Found {len(obj_prompts)} unique objects to process.")
    
    # DEBUG: Показать состав групп
    for target_id, prompts in obj_prompts.items():
        raw_ids = set(p['raw_id'] for p in prompts)
        frames = sorted(list(set(p['frame_idx'] for p in prompts)))
        logger.info(f"Target Obj {target_id}: composed of raw IDs {raw_ids}. Prompts on {len(frames)} frames: {frames}")

    # Prepare directories
    os.makedirs(args.output_dir, exist_ok=True)
    cache_dir = os.path.join(args.output_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    masks_dir = os.path.join(args.output_dir, "masks")
    os.makedirs(masks_dir, exist_ok=True)
    
    # --- 2. SEQUENTIAL PROCESSING (OBJECT-WISE) ---
    MAX_TRACK = None # None means track until end/start of video
    
    # Проходим по каждому объекту отдельно
    for obj_id, prompts in tqdm(obj_prompts.items(), desc="Processing Objects"):
        cache_path = os.path.join(cache_dir, f"obj_{obj_id}.npz")
        
        # Если кэш уже есть, можно пропустить (опционально, сейчас перезаписываем для надежности)
        # if os.path.exists(cache_path): continue

        # ВАЖНО: Очищаем состояние трекера перед новым объектом
        log_memory_usage("Before Reset")
        
        # Освобождаем память от предыдущего объекта
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        reset_inference_state(predictor, inference_state)
        
        log_memory_usage("After Reset")
        
        prompt_frame_indices = set()
        
        # Инъекция всех промптов для текущего объекта
        for p in prompts:
            frame_idx = p['frame_idx']
            bbox = p['bbox']

            # 2. Point & Mask
            if p['segmentation'] and (args.use_mask_prompt or args.use_point_prompt):
                mask = decode_rle_mask(p['segmentation'], [video_height, video_width])
                coords = np.argwhere(mask > 0)
                
                if coords.size > 0:
                    y_c, x_c = coords[len(coords)//2]
                    rel_points = torch.tensor([[x_c/video_width, y_c/video_height]], dtype=torch.float32)
                    labels = torch.tensor([1], dtype=torch.int32)

                    # Mask prompt
                    mask_2d = torch.from_numpy(mask).to(args.device).to(torch.float32).squeeze()
                    if mask_2d.ndim == 2 and args.use_mask_prompt:
                        predictor.add_new_mask(
                            inference_state=inference_state,
                            frame_idx=frame_idx,
                            obj_id=obj_id,
                            mask=mask_2d
                        )                  
                    # Point refinement
                    if args.use_point_prompt:
                        predictor.add_new_points_or_box(
                            inference_state=inference_state,
                            frame_idx=frame_idx,
                            obj_id=obj_id,
                            points=rel_points,
                            labels=labels,
                            clear_old_points=False,
                        )
 
            # 1. Box
            if bbox and args.use_bbox_prompt:
                x1, y1, x2, y2 = bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]
                rel_box = np.array([[x1/video_width, y1/video_height, x2/video_width, y2/video_height]], dtype=np.float32)
                
                predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    box=rel_box,
                )

            
            prompt_frame_indices.add(frame_idx)

        if not prompt_frame_indices:
            continue

        # Propagation for this single object
        obj_results = {}
        min_idx = min(prompt_frame_indices)
        max_idx = max(prompt_frame_indices)

        # Forward
        for frame_idx, obj_ids, _, video_res_masks, _ in predictor.propagate_in_video(
            inference_state, 
            start_frame_idx=min_idx, 
            max_frame_num_to_track=MAX_TRACK,
            reverse=False,
            propagate_preflight=True
        ):
            # frame_idx, obj_ids, _            , video_res_masks, _          = response
            # frame_idx, obj_ids, low_res_masks, video_res_masks, obj_scores = response

            # frame_idx (int): индекс текущего кадра в видео
            # obj_ids (list): отслеживаемые ID объектов на кадре
            # low_res_masks (torch.Tensor): маски объектов в низком разрешении (N, 1, H_low, W_low),  H_low, W_low = 288, 288 получились в нашем слчае
            # video_res_masks (torch.Tensor): маски объектов в оригинальном разрешении видео (N, 1, H_video, W_video) H_video, W_video = 720, 1280
            # obj_scores (torch.Tensor): уверенность модели для каждого объекта
            
            for i, out_obj_id in enumerate(obj_ids):
                if out_obj_id == obj_id:
                    mask_bool = (video_res_masks[i] > 0.0).cpu().numpy().astype(bool)
                    obj_results[frame_idx] = mask_bool

        # Backward
        for frame_idx, obj_ids, _, video_res_masks, _ in predictor.propagate_in_video(
            inference_state, 
            start_frame_idx=max_idx, 
            max_frame_num_to_track=MAX_TRACK,
            reverse=True,
            propagate_preflight=True
        ):
            for i, out_obj_id in enumerate(obj_ids):
                if out_obj_id == obj_id:
                    mask_bool = (video_res_masks[i] > 0.0).cpu().numpy().astype(bool)
                    obj_results[frame_idx] = mask_bool

        # Save Object Cache
        # Мы сохраняем dict {frame_idx: mask} в NPZ
        # Чтобы ключи были строками (требование savez), конвертируем frame_idx в str
        save_dict = {str(k): v for k, v in obj_results.items()}
        np.savez_compressed(cache_path, **save_dict)
        
        # Сразу сохраняем PNG маски (опционально, можно вынести в конец)
        # Это увеличивает I/O, но позволяет видеть прогресс
        for f_idx, mask in obj_results.items():
            frame_name = os.path.splitext(frame_files[f_idx])[0]
            f_dir = os.path.join(masks_dir, frame_name)
            os.makedirs(f_dir, exist_ok=True)
            cv2.imwrite(os.path.join(f_dir, f"obj_{obj_id}.png"), (mask.squeeze()*255).astype(np.uint8))

    # --- 3. MERGE & VISUALIZE ---
    logger.info("--- MERGING AND VISUALIZING ---")
    
    vis_dir = os.path.join(args.output_dir, "visualization")
    os.makedirs(vis_dir, exist_ok=True)
    temp_video_path = os.path.join(vis_dir, "temp_render.mp4")
    final_video_path = os.path.join(vis_dir, "output_redo.mp4")
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(temp_video_path, fourcc, 24, (video_width, video_height))
    
    COLORS = generate_bright_colors(max(100, len(obj_prompts) + 100)) # запас цветов

    # Собираем все обработанные объекты из кэша
    cached_files = glob.glob(os.path.join(cache_dir, "*.npz"))
    
    # Чтобы рендерить кадр за кадром, нам нужно инвертировать структуру:
    # Cache: Obj -> Frame -> Mask
    # Need: Frame -> List[(Obj, Mask)]
    
    # Это может занять много памяти, если грузить всё сразу.
    # Оптимизация: читаем NPZ лениво? Нет, npz читается сразу.
    # Если видео длинное, лучше проходить по кадрам и дергать данные из NPZ (медленно).
    # Или загрузить всё в RAM (может быть много, но это bool маски, они легкие).
    
    logger.info("Loading cache into memory for visualization...")
    frame_to_objects = defaultdict(list)
    
    for cf in tqdm(cached_files, desc="Loading cache"):
        # obj_101.npz
        oid = int(os.path.basename(cf).split('_')[1].split('.')[0])
        data = np.load(cf)
        for frame_key, mask in data.items():
            frame_idx = int(frame_key)
            frame_to_objects[frame_idx].append((oid, mask))
            
    # Рендеринг
    sorted_frames = sorted(frame_to_objects.keys())
    if not sorted_frames:
        logger.warning("No results found to visualize.")
        return

    for frame_idx in tqdm(sorted_frames, desc="Rendering Video"):
        img_path = os.path.join(args.video_path, frame_files[frame_idx])
        img = cv2.imread(img_path)
        if img is None: continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        overlay = img.copy()
        
        for oid, mask in frame_to_objects[frame_idx]:
            color = COLORS[oid % len(COLORS)]
            color255 = tuple(int(x * 255) for x in color)
            
            mask_bool = mask.squeeze()
            if mask_bool.ndim == 3: mask_bool = mask_bool[0]
            
            # Overlay
            alpha = 0.5
            for c in range(3):
                overlay[..., c][mask_bool] = (alpha * color255[c] + (1 - alpha) * overlay[..., c][mask_bool]).astype(np.uint8)
            
            # Label
            coords = np.argwhere(mask_bool)
            if coords.size > 0:
                y_m, x_m = coords.mean(axis=0)
                cv2.putText(overlay, f"ID:{oid}", (int(x_m), int(y_m)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color255, 2)

        video_writer.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    video_writer.release()
    os.system(f"ffmpeg -y -i {temp_video_path} -vcodec libx264 -crf 23 -pix_fmt yuv420p {final_video_path} > /dev/null 2>&1")
    if os.path.exists(temp_video_path): os.remove(temp_video_path)

    logger.info(f"SUCCESS: Results saved to {args.output_dir}")

if __name__ == "__main__":
    try:
        send_message("Batch run started")
        main()
        send_message("Batch run completed successfully")
    except Exception as e:
        notify_error(e, "Критическая ошибка при выполнении batch_run_configs.py")
        raise
