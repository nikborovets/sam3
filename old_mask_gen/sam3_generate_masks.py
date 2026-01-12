import torch
import numpy as np
import cv2
import sys
import os
import logging
import math
from itertools import product
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torchvision.ops import nms, batched_nms

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("sam3_generate_masks.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("SAM3Generator")

# Add sam3 to python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from sam3.model_builder import build_sam3_image_model
    # We no longer use Sam3Processor for the main generation logic in this advanced generator
    # but we still import it if needed or for reference.
    from sam3.model.sam3_image_processor import Sam3Processor
except ImportError as e:
    logger.error(f"Error importing SAM3 modules: {e}")
    logger.error(f"Make sure you are running in an environment with sam3 installed or the sam3 folder is in {current_dir}")
    sys.exit(1)

def save_anns(anns, save_path):
    if len(anns) == 0:
        logger.info(f"No annotations to save for {save_path}")
        return
    
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    
    h, w = sorted_anns[0]['segmentation'].shape
    img = np.zeros((h, w, 3))
    
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.random.random(3) * 255
        img[m] = color_mask 

    cv2.imwrite(str(save_path), img)
    logger.info(f"Saved masks to {save_path} ({len(anns)} masks)")

def mask_nms(masks, scores, iou_threshold=0.5):
    """
    Performs NMS based on mask IoU.
    masks: list of binary masks (H, W) or (N, H, W)
    scores: list of scores or (N,)
    iou_threshold: threshold for dropping
    """
    if len(masks) == 0:
        return []
    
    # Sort by score descending
    idxs = np.argsort(scores)[::-1]
    
    keep = []
    while len(idxs) > 0:
        current = idxs[0]
        keep.append(current)
        
        if len(idxs) == 1:
            break
            
        current_mask = masks[current]
        remaining_idxs = idxs[1:]
        
        # Compute IoU with remaining
        # This can be slow if many masks and large size.
        # Optimization: use bounding box pre-filtering or downsample for NMS
        
        # Let's verify containment/IoU
        # Vectorized implementation for numpy? 
        # A loop is safer for memory with large masks.
        
        keep_mask = []
        current_area = current_mask.sum()
        
        for other_idx in remaining_idxs:
            other_mask = masks[other_idx]
            intersection = np.logical_and(current_mask, other_mask).sum()
            
            if intersection == 0:
                keep_mask.append(True)
                continue
                
            union = current_area + other_mask.sum() - intersection
            iou = intersection / (union + 1e-6)
            
            if iou > iou_threshold:
                keep_mask.append(False) # Suppress
            else:
                # Also suppress if "other" is highly contained in "current" (duplicate part)
                # overlap_ratio = intersection / other_mask.sum()
                # if overlap_ratio > 0.8: # Heuristic
                #     keep_mask.append(False)
                # else:
                keep_mask.append(True)
        
        idxs = remaining_idxs[keep_mask]
        
    return keep

class SAM3AutomaticMaskGenerator:
    """
    SAM3 version of AutomaticMaskGenerator with features ported from SAM2:
    - M2M Refinement (iterative mask refinement)
    - Stability Score filtering
    - Cropping (processing crops for better small object detection)
    
    Uses SAM3InteractiveImagePredictor for prompt-based mask generation.
    """
    
    def __init__(
        self,
        image_model,
        points_per_side: int = 32,
        points_per_batch: int = 64,
        pred_iou_thresh: float = 0.8,
        stability_score_thresh: float = 0.95,
        stability_score_offset: float = 1.0,
        box_nms_thresh: float = 0.7,
        crop_n_layers: int = 0,
        crop_nms_thresh: float = 0.7,
        crop_overlap_ratio: float = 512 / 1500,
        crop_n_points_downscale_factor: int = 1,
        min_mask_region_area: int = 0,
        use_m2m: bool = True,
        device: str = "cuda"
    ):
        self.image_model = image_model
        
        # Access the interactive predictor directly
        if not hasattr(image_model, 'inst_interactive_predictor') or image_model.inst_interactive_predictor is None:
            raise ValueError("The provided SAM3 model does not have 'inst_interactive_predictor' enabled. "
                             "Please build the model with enable_inst_interactivity=True.")
        
        self.predictor = image_model.inst_interactive_predictor
        
        # FIX: The tracker used in SAM3InteractiveImagePredictor might be initialized without a backbone
        if self.predictor.model.backbone is None:
            logger.info("Tracker backbone is None, assigning image model backbone to tracker...")
            self.predictor.model.backbone = image_model.backbone

        self.points_per_side = points_per_side
        self.points_per_batch = points_per_batch
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.stability_score_offset = stability_score_offset
        self.box_nms_thresh = box_nms_thresh
        self.crop_n_layers = crop_n_layers
        self.crop_nms_thresh = crop_nms_thresh
        self.crop_overlap_ratio = crop_overlap_ratio
        self.crop_n_points_downscale_factor = crop_n_points_downscale_factor
        self.min_mask_region_area = min_mask_region_area
        self.use_m2m = use_m2m
        self.device = device
        
        logger.info(f"Initialized SAM3Generator (Advanced) with:")
        logger.info(f"  Grid: {points_per_side}x{points_per_side}")
        logger.info(f"  Refinement (M2M): {use_m2m}")
        logger.info(f"  Stability Thresh: {stability_score_thresh}")
        logger.info(f"  Crops: {crop_n_layers} layers")
    
    @torch.no_grad()
    def generate(self, image: np.ndarray):
        """Generate masks for the given image."""
        
        # Calculate crop boxes for all layers
        crop_boxes, layer_idxs = self._generate_crop_boxes(
            image.shape[:2], self.crop_n_layers, self.crop_overlap_ratio
        )
        
        all_masks = []
        
        # Iterate over crops
        for crop_box, layer_idx in zip(crop_boxes, layer_idxs):
            x0, y0, x1, y1 = crop_box
            cropped_im = image[y0:y1, x0:x1, :]
            
            # Process this crop
            crop_data = self._process_crop(
                cropped_im, 
                crop_box=crop_box, 
                crop_layer_idx=layer_idx
            )
            all_masks.extend(crop_data)

        # Remove duplicates (NMS) and merge results from different crops
        if not all_masks:
            return []
            
        # Post-process (NMS, etc)
        # Convert to dictionary format expected by NMS/save
        return self._postprocess_small_regions(
            self._apply_nms(all_masks), 
            self.min_mask_region_area, 
            self.box_nms_thresh
        )

    def _process_crop(self, image, crop_box, crop_layer_idx):
        # Set image in predictor
        self.predictor.set_image(image)
        
        # Generate grid points for this crop
        points_scale = np.array(image.shape[1::-1])[None, ::-1]
        points_for_image = self._build_point_grid(crop_layer_idx) * points_scale
        
        # Process in batches
        data = []
        n_points = points_for_image.shape[0]
        for i in range(0, n_points, self.points_per_batch):
            batch_points = points_for_image[i : i + self.points_per_batch]
            batch_data = self._process_batch(batch_points, image.shape[:2], crop_box)
            
            # Filter masks touching crop boundaries (only for crops, not full image)
            if crop_layer_idx > 0:
                batch_data = self._filter_border_masks(batch_data, image.shape[:2])
                
            data.extend(batch_data)
            
        return data
        
    def _filter_border_masks(self, data, crop_size, margin=2):
        """Remove masks that touch the boundary of the crop."""
        h, w = crop_size
        filtered_data = []
        for d in data:
            mask = d['segmentation']
            # Check boundaries: top, bottom, left, right
            # mask is boolean HxW
            # If any pixel in margin is True, discard
            if (mask[:margin, :].any() or mask[-margin:, :].any() or 
                mask[:, :margin].any() or mask[:, -margin:].any()):
                continue
            filtered_data.append(d)
        return filtered_data

    def _process_batch(self, points, im_size, crop_box):
        # Points: (B, 2)
        # Transform to (B, 1, 2) for predictor (B objects, 1 point each)
        transformed_points = points[:, None, :]
        
        # Predict
        masks, iou_preds, low_res_masks = self.predictor.predict(
            point_coords=transformed_points,
            point_labels=np.ones((transformed_points.shape[0], 1), dtype=np.int32),
            multimask_output=True
        )
        
        # Convert to Tensor for PyTorch operations
        masks_t = torch.from_numpy(masks).to(self.device)
        iou_preds_t = torch.from_numpy(iou_preds).to(self.device)
        low_res_masks_t = torch.from_numpy(low_res_masks).to(self.device)

        # Refinement (Mask-to-Mask)
        if self.use_m2m:
            # Pick best mask from first pass to refine
            best_idx = torch.argmax(iou_preds_t, dim=1) # (B,)
            batch_idxs = torch.arange(best_idx.shape[0], device=best_idx.device)
            best_masks_logits = low_res_masks[batch_idxs.cpu().numpy(), best_idx.cpu().numpy()][:, None, :, :] # (B, 1, H, W) numpy
            
            # Second pass - mask_input expects numpy
            masks, iou_preds, low_res_masks = self.predictor.predict(
                point_coords=transformed_points,
                point_labels=np.ones((transformed_points.shape[0], 1), dtype=np.int32),
                mask_input=best_masks_logits,
                multimask_output=True
            )
            # Re-convert to tensors for stability calculation
            masks_t = torch.from_numpy(masks).to(self.device)
            iou_preds_t = torch.from_numpy(iou_preds).to(self.device)
            low_res_masks_t = torch.from_numpy(low_res_masks).to(self.device)
        
        # Calculate Stability Score
        high_thresh = 0.0 + self.stability_score_offset
        low_thresh = 0.0 - self.stability_score_offset
        
        # Operations on logits (B, 3, H, W)
        high_masks = low_res_masks_t > high_thresh
        low_masks = low_res_masks_t > low_thresh
        
        high_sum = high_masks.flatten(2).sum(dim=2)
        low_sum = low_masks.flatten(2).sum(dim=2)
        stability_scores = high_sum / (low_sum + 1e-8) # (B, 3)
        
        # Convert to numpy
        masks_np = masks
        iou_preds_np = iou_preds
        stability_scores_np = stability_scores.cpu().numpy()
        
        data = []
        for i in range(len(points)):
            # Filter masks for this point
            for m_idx in range(3): # multimask=3
                iou = iou_preds_np[i, m_idx]
                stability = stability_scores_np[i, m_idx]
                mask = masks_np[i, m_idx]
                
                if iou >= self.pred_iou_thresh and stability >= self.stability_score_thresh:
                    data.append({
                        'segmentation': mask,
                        'iou': float(iou),
                        'stability_score': float(stability),
                        'point': points[i],
                        'crop_box': crop_box
                    })
                    
        return data

    def _apply_nms(self, all_masks):
        # Improved NMS using Mask IoU
        
        # Find global W, H
        global_w, global_h = 0, 0
        for m in all_masks:
             if m['crop_box'][0] == 0 and m['crop_box'][1] == 0:
                 global_w = max(global_w, m['crop_box'][2])
                 global_h = max(global_h, m['crop_box'][3])
        
        if global_w == 0 or global_h == 0:
             # Fallback
             for m in all_masks:
                 global_w = max(global_w, m['crop_box'][2])
                 global_h = max(global_h, m['crop_box'][3])

        masks_full = []
        scores = []
        
        # Reconstruct all masks first
        for m in all_masks:
            crop_mask = m['segmentation']
            x0, y0, x1, y1 = m['crop_box']
            
            full_mask = np.zeros((global_h, global_w), dtype=bool)
            h_c, w_c = crop_mask.shape
            y1_act = min(y0 + h_c, global_h)
            x1_act = min(x0 + w_c, global_w)
            full_mask[y0:y1_act, x0:x1_act] = crop_mask[:(y1_act-y0), :(x1_act-x0)]
            
            masks_full.append(full_mask)
            scores.append(m['iou'])

        if not masks_full:
            return []
            
        # Apply Mask NMS
        # This is slower but handles "split lines" and "containment" much better than Box NMS
        keep_idxs = mask_nms(masks_full, scores, iou_threshold=self.box_nms_thresh)
        
        final_results = []
        for idx in keep_idxs:
            full_mask = masks_full[idx]
            
            # Re-calculate bbox
            y_idx, x_idx = np.where(full_mask)
            if len(y_idx) == 0: continue
            
            final_results.append({
                'segmentation': full_mask,
                'bbox_xyxy': [int(x_idx.min()), int(y_idx.min()), 
                              int(x_idx.max()), int(y_idx.max())],
                'area': int(full_mask.sum()),
                'predicted_iou': scores[idx],
                'point_coords': [0, 0] 
            })
            
        return final_results

    def _postprocess_small_regions(self, mask_data, min_area, nms_thresh):
        # Filter by area
        new_data = []
        for m in mask_data:
            if m['area'] < min_area:
                continue
            new_data.append(m)
        return new_data

    def _generate_crop_boxes(self, im_size, n_layers, overlap_ratio):
        crop_boxes = []
        layer_idxs = []
        h, w = im_size
        short_side = min(h, w)
        
        # Layer 0: Full image
        crop_boxes.append([0, 0, w, h])
        layer_idxs.append(0)
        
        def crop_len(orig_len, n_crops, overlap):
            return int(math.ceil((overlap * (n_crops - 1) + orig_len) / n_crops))

        for layer in range(1, n_layers + 1):
            n_crops_per_side = 1 << layer
            overlap = int(overlap_ratio * short_side * (2 / n_crops_per_side))
            crop_w = crop_len(w, n_crops_per_side, overlap)
            crop_h = crop_len(h, n_crops_per_side, overlap)
            crop_w_stride = crop_w - overlap
            crop_h_stride = crop_h - overlap
            
            for i_h in range(n_crops_per_side):
                for i_w in range(n_crops_per_side):
                    y0 = i_h * crop_h_stride
                    x0 = i_w * crop_w_stride
                    y1 = min(y0 + crop_h, h)
                    x1 = min(x0 + crop_w, w)
                    y0 = max(0, y1 - crop_h)
                    x0 = max(0, x1 - crop_w)
                    
                    crop_boxes.append([x0, y0, x1, y1])
                    layer_idxs.append(layer)
                    
        return crop_boxes, layer_idxs

    def _build_point_grid(self, crop_layer_idx):
        n_points = int(self.points_per_side / (self.crop_n_points_downscale_factor**crop_layer_idx))
        offset = 1 / (2 * n_points)
        xs = np.linspace(offset, 1 - offset, n_points)
        ys = np.linspace(offset, 1 - offset, n_points)
        xv, yv = np.meshgrid(xs, ys)
        points = np.stack([xv.flatten(), yv.flatten()], axis=1)
        return points


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    if torch.cuda.is_available():
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("TF32 enabled for Ampere GPU")

    logger.info("Building SAM3 model...")
    sam3_model = build_sam3_image_model(
        device=str(device),
        eval_mode=True,
        enable_inst_interactivity=True
    )

    mask_generator = SAM3AutomaticMaskGenerator(
        image_model=sam3_model,
        points_per_side=32, 
        points_per_batch=64,
        pred_iou_thresh=0.7,
        stability_score_thresh=0.92,
        stability_score_offset=1.0,
        crop_n_layers=1,
        min_mask_region_area=25.0,
        use_m2m=True,
        device=str(device)
    )

    input_path = Path('/workspace/ivan_images_slice')
    if not input_path.exists():
        logger.warning(f"Input path {input_path} does not exist.")
        local_input = Path('input_images')
        if local_input.exists():
            input_path = local_input
            logger.info(f"Using local input path: {input_path}")
        else:
            logger.error("No valid input directory found.")

    output_path = Path('/workspace/output_masks_ivan_slice/')
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Reading images from: {input_path}")
    logger.info(f"Saving masks to: {output_path}")

    try:
        input_images = sorted(input_path.iterdir())
        input_images = [x for x in input_images if x.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
        logger.info(f"Found {len(input_images)} images to process")
    except FileNotFoundError:
        logger.error("Input directory not found.")
        return

    output_images_existing = sorted(output_path.iterdir())
    output_images_names = [image.name for image in output_images_existing]

    for image_path in tqdm(input_images, desc="Processing Images"):
        image_name = image_path.name
        if image_name in output_images_names:
            logger.info(f"Skipping {image_name} (already exists)")
            continue

        try:
            logger.info(f"Processing {image_name}...")
            image = Image.open(image_path)
            image_np = np.array(image.convert("RGB"))
            
            masks = mask_generator.generate(image_np)
            
            save_anns(masks, output_path / image_name)
        except Exception as e:
            logger.error(f"Error processing {image_name}: {e}", exc_info=True)

if __name__ == "__main__":
    main()
