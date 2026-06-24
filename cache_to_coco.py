import os
import argparse
import glob
import numpy as np
import cv2
import json
import logging
from tqdm import tqdm
from collections import defaultdict
from pycocotools import mask as mask_utils
import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

DEFAULT_TEXT_CACHE = "/workspace/sam3_batch_results_3023_16_01_2026/cache"
DEFAULT_MASK_CACHE = "/workspace/sam3_redo_results/02-02-1433/cache"
DEFAULT_VIDEO_PATH = "/workspace/2-half-blind-no-light-day_e3f"
DEFAULT_OUTPUT_JSON = "/workspace/sam3_cache_to_coco/output_coco.json"

ALLOWED_PROMPTS = [
        # --- Архитектура и фон (Room Structure) ---
        "wall",
        "ceiling",
        "illumination",
        "floor",
        "baseboard",
        "concrete",
        "column",
        "window",
        "door",
        "pipe",
        # "radiator", # мусор, убрал ранее
        "ventilation",

        # --- Элементы на стенах (Fixtures) ---
        "socket",
        "switch",
        "light switch",
        "window blind",
        "door handle",
        "blackboard",
        "black plate",
        "paperboard", # мусор, в cvat увидел, TODO: объединить с box

        # --- Крупная мебель (Large Furniture) ---
        "cabinet",
        # "wardrobe", # мусор, в cvat увидел (вообще нет его кажется)
        # "cabinet wall", # мусор, в cvat увидел (вообще нет его кажется)
        "shelves",
        # "cabinet door", # мусор, в cvat увидел, TODO: объединить с cabinet
        "glass",
        # "inside of the cabinet", # мусор, в cvat увидел, TODO: объединить с cabinet
        "sofa",
        "armrest",
        "cushion",
        "chair",
        "table",
        "bin",

        # --- Предметы в шкафу/на полках (Objects on shelves/background) ---
        # "cardboard boxes", # мусор, в cvat увидел, TODO: объединить с box
        "box",
        # "router box", # мусор, в cvat увидел, TODO: объединить с box
        # "package", # мусор, в cvat увидел, TODO: объединить с box
        "frame",
        "book",
        "helmet",
        # "vase", # добавлено через mask prompt
        "mug",
        # "black thing", # мусор, в cvat увидел (вообще нет его кажется)
        # "statuette", # добавлено через mask prompt
        "backpack",
        # "pump", # добавлено через mask prompt
        
        # --- Техника на столе ---
        "monitor",
        "imac",

        # --- Мелкие предметы на столе/переднем плане (Objects on desk/foreground) ---
        "keyboard",
        # "touchpad", # мусор, в cvat увидел, TODO: объединить с keyboard
        "mouse",
        # "usb adapter", # мусор, в cvat увидел (вообще нет его кажется)
        "wire",
        # "plug", # мусор, в cvat увидел (вообще нет его кажется)
        # "battery", # мусор, в cvat увидел (вообще нет его кажется)
        "paper",
        # "screwdriver", # мусор, в cvat увидел (вообще нет его кажется)
    ]

def parse_args():
    parser = argparse.ArgumentParser(description="Convert SAM3 cache to COCO JSON format")
    
    # Paths
    parser.add_argument("--text_cache_dir", type=str, default=DEFAULT_TEXT_CACHE, help="Path to batch run cache (text prompts)")
    parser.add_argument("--mask_cache_dir", type=str, default=DEFAULT_MASK_CACHE, help="Path to redo cache (mask prompts)")
    parser.add_argument("--video_path", type=str, default=DEFAULT_VIDEO_PATH, help="Path to folder with frames (source images)")
    parser.add_argument("--output_json", type=str, default=DEFAULT_OUTPUT_JSON, help="Path to save result JSON")
    
    # Optional prompt list (if not provided, all found categories will be used)
    parser.add_argument("--prompts", nargs="*", default=ALLOWED_PROMPTS, help="List of text prompts to export. If empty, all found prompts are used.")
    
    return parser.parse_args()

def load_frames_metadata(video_path):
    """Load frame file names and dimensions"""
    if not os.path.isdir(video_path):
        raise ValueError(f"Path is not a directory: {video_path}")
        
    exts = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']
    frames = []
    for ext in exts:
        frames.extend(glob.glob(os.path.join(video_path, ext)))
    
    try:
        frames.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    except:
        logger.warning("Could not sort frames numerically, falling back to lexicographic sort")
        frames.sort()
        
    if not frames:
        raise ValueError(f"No frames found in {video_path}")

    # Read dimensions from first frame
    first_img = cv2.imread(frames[0])
    if first_img is None:
        raise ValueError(f"Could not read first frame: {frames[0]}")
    h, w = first_img.shape[:2]
    
    # Create image entries for COCO
    images_info = []
    filename_to_id = {}
    
    for idx, path in enumerate(frames):
        file_name = os.path.basename(path)
        img_id = idx # 0-based ID for internal mapping
        
        images_info.append({
            "id": img_id,
            "width": w,
            "height": h,
            "file_name": file_name,
            "license": 0,
            "flickr_url": "",
            "coco_url": "",
            "date_captured": 0
        })
        filename_to_id[file_name] = img_id
        
    return images_info, w, h

def load_cache_data(text_cache_dir, mask_cache_dir, allowed_prompts=None):
    """
    Load data from caches.
    Returns list of dicts: {frame_idx, mask, category_name, score, bbox}
    """
    all_detections = [] # List of {frame_idx, mask, category, score, bbox, obj_id}
    
    # Helper to check if prompt allowed
    def is_allowed(p):
        if not allowed_prompts: return True
        return p in allowed_prompts

    # 1. Text Cache
    if text_cache_dir and os.path.exists(text_cache_dir):
        files = glob.glob(os.path.join(text_cache_dir, "*.npz"))
        logger.info(f"Loading {len(files)} text cache files...")
        
        for fpath in tqdm(files, desc="Text Cache"):
            prompt_name = os.path.splitext(os.path.basename(fpath))[0].replace("_", " ")
            
            # Filter by allowed prompts if provided
            if not is_allowed(prompt_name):
                continue
                
            try:
                data = np.load(fpath, allow_pickle=True)
                if 'outputs' not in data: continue
                outputs = data['outputs'].item()
                
                for f_idx, content in outputs.items():
                    f_idx = int(f_idx)
                    num_objs = len(content['out_obj_ids'])
                    
                    for i in range(num_objs):
                        mask = content['out_binary_masks'][i]
                        if mask.ndim == 3: mask = mask[0]
                        
                        # Only add non-empty masks
                        if not mask.any(): continue
                        
                        box = content['out_boxes_xywh'][i] if 'out_boxes_xywh' in content else None
                        score = float(content['out_probs'][i]) if 'out_probs' in content else 1.0
                        
                        # Unique category name for tracking instances
                        # Format: <prompt_name>_<obj_id>
                        # This ensures distinct chairs (chair_0, chair_1) get distinct category IDs in COCO
                        unique_cat_name = f"{prompt_name}_{content['out_obj_ids'][i]}"
                        
                        all_detections.append({
                            'frame_idx': f_idx,
                            'mask': mask,
                            'category': unique_cat_name,
                            'score': score,
                            'bbox_norm': box, # usually normalized xywh
                            'source': 'text'
                        })
            except Exception as e:
                logger.error(f"Error reading {fpath}: {e}")

    # 2. Mask Cache
    if mask_cache_dir and os.path.exists(mask_cache_dir):
        files = glob.glob(os.path.join(mask_cache_dir, "*.npz"))
        logger.info(f"Loading {len(files)} mask cache files...")
        
        for fpath in tqdm(files, desc="Mask Cache"):
            # obj_101.npz
            basename = os.path.splitext(os.path.basename(fpath))[0]
            obj_id_str = basename.split('_')[1] if '_' in basename else "0"
            
            try:
                data = np.load(fpath, allow_pickle=True)
                outputs = {}
                if 'outputs' in data:
                    outputs = data['outputs'].item()
                else:
                    # Fallback old format
                    for k, v in data.items():
                        if k != 'outputs': outputs[int(k)] = {'out_binary_masks': [v], 'out_obj_ids': [int(obj_id_str)]}

                for f_idx, content in outputs.items():
                    f_idx = int(f_idx)
                    num_objs = len(content.get('out_obj_ids', []))
                    
                    for i in range(num_objs):
                        mask = content['out_binary_masks'][i]
                        if mask.ndim == 3: mask = mask[0]
                        if not mask.any(): continue

                        # Determine category name
                        # Use source name from COCO + object ID to keep instances distinct
                        # Format: <source_name>_<obj_id>
                        
                        obj_id = content['out_obj_ids'][i]
                        base_name = "mask_obj"
                        
                        if 'prompt_source' in content and len(content['prompt_source']) > i:
                             src = content['prompt_source'][i]
                             if src and src != "unknown":
                                 base_name = src
                        
                        unique_cat_name = f"{base_name}_{obj_id}"

                        box = content['out_boxes_xywh'][i] if 'out_boxes_xywh' in content else None
                        score = float(content['out_probs'][i]) if 'out_probs' in content else 1.0

                        all_detections.append({
                            'frame_idx': f_idx,
                            'mask': mask,
                            'category': unique_cat_name,
                            'score': score,
                            'bbox_norm': box,
                            'source': 'mask'
                        })

            except Exception as e:
                logger.error(f"Error reading {fpath}: {e}")
                
    return all_detections

def main():
    args = parse_args()
    
    logger.info("Starting conversion...")
    logger.info(f"Video Path: {args.video_path}")
    if args.text_cache_dir:
        logger.info(f"Text Cache: {args.text_cache_dir}")
    if args.mask_cache_dir:
        logger.info(f"Mask Cache: {args.mask_cache_dir}")
        
    if not args.text_cache_dir and not args.mask_cache_dir:
        logger.error("At least one cache directory (text or mask) must be provided!")
        return
    
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)

    # 1. Load Metadata
    try:
        images_info, width, height = load_frames_metadata(args.video_path)
        logger.info(f"Loaded {len(images_info)} frames. Size: {width}x{height}")
    except Exception as e:
        logger.error(f"Failed to load frames: {e}")
        return

    # 2. Load Cache
    detections = load_cache_data(args.text_cache_dir, args.mask_cache_dir, args.prompts)
    logger.info(f"Loaded {len(detections)} detection masks.")
    
    if not detections:
        logger.warning("No detections found! Exiting.")
        return

    # 3. Process Annotations
    categories = []
    cat_name_to_id = {}
    annotations = []
    ann_id_counter = 1
    
    # Process detections to create categories and annotations
    detections.sort(key=lambda x: x['frame_idx'])
    
    for det in tqdm(detections, desc="Processing Annotations"):
        cat_name = det['category']
        
        # Get or create category ID
        if cat_name not in cat_name_to_id:
            new_id = len(cat_name_to_id) + 1
            cat_name_to_id[cat_name] = new_id
            categories.append({
                "id": new_id,
                "name": cat_name,
                "supercategory": ""
            })
        cat_id = cat_name_to_id[cat_name]
        
        if det['frame_idx'] >= len(images_info):
            continue 
            
        # RLE Encoding (Uncompressed list of counts)
        # Ensure mask is strictly 0/1 uint8
        binary_mask = (det['mask'] > 0).astype(np.uint8) 
        
        # Calculate Area and BBox using pycocotools (reliable)
        mask_fortran = np.asfortranarray(binary_mask)
        rle_compressed = mask_utils.encode(mask_fortran)
        area = float(mask_utils.area(rle_compressed))
        bbox = list(mask_utils.toBbox(rle_compressed))
        
        # Custom RLE implementation (Binary Mask -> List of Counts)
        # Flatten column-major
        flat_pixels = binary_mask.flatten(order='F')
        
        # Find runs
        # We want lengths of consecutive sequences
        # e.g. [0, 0, 1, 1, 1, 0] -> [2, 3, 1]
        
        # Method: Find indices where value changes
        # Append -1 at start and len at end? No.
        # Use simple reliable method:
        
        # 1. Find indices where value changes
        # diff != 0 gives True at index i if a[i+1] != a[i]
        # We prepend/append values to force change at start/end if needed?
        # Actually simpler:
        # where(pixel[1:] != pixel[:-1]) gives indices before change
        
        tmp = np.concatenate([[0], flat_pixels, [0]])
        # This padding ensures we detect start and end of sequences if we look for transitions?
        # No, simpler:
        
        n = len(flat_pixels)
        if n == 0:
            counts = []
        else:
            # Positions where value changes
            # We compare array with itself shifted by 1
            # indices points to the END of a run
            # e.g. [0 0 1 1 0]
            # 0!=0 F, 0!=1 T (idx 1), 1!=1 F, 1!=0 T (idx 3)
            # changes at 1 and 3.
            # runs: 0..1 (len 2), 2..3 (len 2), 4..end (len 1)
            
            changes = flat_pixels[:-1] != flat_pixels[1:]
            idx = np.where(changes)[0] + 1 # indices where new run STARTS
            
            # Combine [0, idx1, idx2, ..., length]
            run_starts = np.concatenate(([0], idx, [n]))
            
            # Lengths are diffs
            counts = np.diff(run_starts).tolist()
        
        # COCO RLE must start with background (0).
        # If mask starts with 1, prepend 0.
        if flat_pixels[0] == 1:
            counts = [0] + counts
            
        # Validation
        if sum(counts) != height * width:
            logger.error(f"RLE Error: Sum({sum(counts)}) != Size({height}*{width}) for frame {det['frame_idx']}")
            # Fallback to single run if something is super broken (empty or full)
            if not counts:
                # Should be full background?
                counts = [height * width]
            continue
            
        segmentation = {
            "counts": counts,
            "size": [height, width]
        }

        annotations.append({
            "id": ann_id_counter,
            "image_id": det['frame_idx'], # Using 0-based index as per logic
            "category_id": cat_id,
            "segmentation": segmentation,
            "area": area,
            "bbox": bbox,
            "iscrowd": 0,
            "attributes": {
                "occluded": False,
                "score": det['score'],
                "source": det['source']
            }
        })
        ann_id_counter += 1

    # 4. Construct Final JSON
    coco_output = {
        "licenses": [{"name": "", "id": 0, "url": ""}],
        "info": {
            "contributor": "SAM3",
            "date_created": datetime.datetime.now().isoformat(),
            "description": "Converted from SAM3 Cache",
            "url": "",
            "version": "1.0",
            "year": str(datetime.datetime.now().year)
        },
        "categories": categories,
        "images": images_info,
        "annotations": annotations
    }
    
    # 5. Save
    logger.info(f"Saving {len(annotations)} annotations to {args.output_json}...")
    with open(args.output_json, 'w') as f:
        json.dump(coco_output, f)
        
    logger.info("Done!")

if __name__ == "__main__":
    main()
