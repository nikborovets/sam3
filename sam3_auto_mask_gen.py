import torch
import numpy as np
import cv2
import sys
import os
import logging
import math
import json
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torchvision.ops import nms, batched_nms

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("sam3_auto_mask_gen.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("SAM3AutoMaskGen")

# Add sam3 to python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from sam3.model_builder import build_sam3_image_model
except ImportError as e:
    logger.error(f"Error importing SAM3 modules: {e}")
    sys.exit(1)

def save_anns(anns, save_path):
    if len(anns) == 0:
        logger.info(f"No annotations to save for {save_path}")
        return
    
    # Sort by area
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    
    # Create output image
    h, w = sorted_anns[0]['segmentation'].shape
    img = np.zeros((h, w, 3), dtype=np.uint8)
    
    for ann in sorted_anns:
        m = ann['segmentation']
        # Ensure mask is boolean for indexing
        if m.dtype != bool:
            m = m > 0
            
        color_mask = np.random.randint(0, 255, (3,), dtype=np.uint8)
        img[m] = color_mask

    cv2.imwrite(str(save_path), img)
    logger.info(f"Saved masks to {save_path} ({len(anns)} masks)")

# --- UTILS PORTED FROM SAM2 ---

def calculate_stability_score(
    masks: torch.Tensor, mask_threshold: float, threshold_offset: float
) -> torch.Tensor:
    """
    Computes the stability score for a batch of masks. The stability
    score is the IoU between the binary masks obtained by thresholding
    the predicted mask logits at high and low values.
    """
    # One-sided error: masks are logits
    intersections = (
        (masks > (mask_threshold + threshold_offset))
        .sum(-1, dtype=torch.int16)
        .sum(-1, dtype=torch.int32)
    )
    unions = (
        (masks > (mask_threshold - threshold_offset))
        .sum(-1, dtype=torch.int16)
        .sum(-1, dtype=torch.int32)
    )
    return intersections / (unions + 1e-8)

def build_point_grid(n_per_side: int) -> np.ndarray:
    """Generates a 2D grid of points evenly spaced in [0,1]x[0,1]."""
    offset = 1 / (2 * n_per_side)
    points_one_side = np.linspace(offset, 1 - offset, n_per_side)
    points_x = np.tile(points_one_side[None, :], (n_per_side, 1))
    points_y = np.tile(points_one_side[:, None], (1, n_per_side))
    points = np.stack([points_x, points_y], axis=-1).reshape(-1, 2)
    return points

def generate_crop_boxes(
    im_size: Tuple[int, ...], n_layers: int, overlap_ratio: float
) -> Tuple[List[List[int]], List[int]]:
    """Generates a list of crop boxes of different sizes."""
    crop_boxes, layer_idxs = [], []
    im_h, im_w = im_size
    short_side = min(im_h, im_w)

    # Original image
    crop_boxes.append([0, 0, im_w, im_h])
    layer_idxs.append(0)

    def crop_len(orig_len, n_crops, overlap):
        return int(math.ceil((overlap * (n_crops - 1) + orig_len) / n_crops))

    for i_layer in range(n_layers):
        n_crops_per_side = 2 ** (i_layer + 1)
        overlap = int(overlap_ratio * short_side * (2 / n_crops_per_side))

        crop_w = crop_len(im_w, n_crops_per_side, overlap)
        crop_h = crop_len(im_h, n_crops_per_side, overlap)

        crop_w_stride = crop_w - overlap
        crop_h_stride = crop_h - overlap

        # Generate crops
        for i_h in range(n_crops_per_side):
            for i_w in range(n_crops_per_side):
                crop_y = i_h * crop_h_stride
                crop_x = i_w * crop_w_stride

                crop_y_end = min(crop_y + crop_h, im_h)
                crop_x_end = min(crop_x + crop_w, im_w)
                
                # Adjust start to ensure crop size is maintained if near edge
                crop_y = max(0, crop_y_end - crop_h)
                crop_x = max(0, crop_x_end - crop_w)

                crop_boxes.append([crop_x, crop_y, crop_x_end, crop_y_end])
                layer_idxs.append(i_layer + 1)

    return crop_boxes, layer_idxs

def uncrop_masks(
    masks: torch.Tensor, crop_box: List[int], orig_h: int, orig_w: int
) -> torch.Tensor:
    """Put binary masks from a crop back into the full image coordinate system."""
    x0, y0, x1, y1 = crop_box
    # masks is (N, H_crop, W_crop)
    if x0 == 0 and y0 == 0 and x1 == orig_w and y1 == orig_h:
        return masks
    
    # Pad to original size
    # We create a full-size tensor and place the crop in it
    # Note: This can be memory intensive for huge images/batches.
    # SAM2 uses specific optimized uncropping, here we do simple padding
    n, h, w = masks.shape
    full_masks = torch.zeros((n, orig_h, orig_w), device=masks.device, dtype=masks.dtype)
    full_masks[:, y0:y1, x0:x1] = masks
    return full_masks

def batched_mask_to_box(masks: torch.Tensor) -> torch.Tensor:
    """
    Calculates boxes in XYXY format around masks.
    masks: (N, H, W)
    """
    if torch.numel(masks) == 0:
        return torch.zeros((0, 4), device=masks.device)

    # Simple implementation via max/min indices
    # Much faster on GPU than loop
    n = masks.shape[0]
    boxes = torch.zeros((n, 4), device=masks.device)
    
    for i in range(n):
        y, x = torch.where(masks[i])
        if len(y) > 0:
            boxes[i, 0] = x.min()
            boxes[i, 1] = y.min()
            boxes[i, 2] = x.max() + 1 # exclusive max
            boxes[i, 3] = y.max() + 1
    
    return boxes

def remove_small_regions(
    mask: np.ndarray, area_thresh: float, mode: str
) -> Tuple[np.ndarray, bool]:
    """
    Removes small disconnected regions and holes in a mask. Returns the
    processed mask and an indicator of if the mask was modified.
    """
    import cv2  # type: ignore

    assert mode in ["holes", "islands"]
    correct_holes = mode == "holes"
    
    # Convert mask to uint8 for cv2
    if mask.dtype == bool:
        mask_uint8 = mask.astype(np.uint8)
    else:
        mask_uint8 = (mask > 0).astype(np.uint8)
        
    # correct_holes True -> remove holes (fill 0s in 1s)
    # correct_holes False -> remove islands (remove small 1s)
    
    if correct_holes:
        # Invert mask to find holes (0s becoming 1s)
        working_mask = (1 - mask_uint8)
    else:
        # Working on islands (1s)
        working_mask = mask_uint8

    n_labels, regions, stats, _ = cv2.connectedComponentsWithStats(working_mask, 8)
    sizes = stats[:, -1][1:]  # Row 0 is background label
    small_regions = [i + 1 for i, s in enumerate(sizes) if s < area_thresh]
    
    if len(small_regions) == 0:
        return mask, False
        
    if not correct_holes:
        # For islands: we want to KEEP large regions.
        labels_to_keep = [i + 1 for i, s in enumerate(sizes) if s >= area_thresh]
        if not labels_to_keep: # If all small, keep largest
             labels_to_keep = [int(np.argmax(sizes)) + 1]
             
        new_mask = np.isin(regions, labels_to_keep)
        return new_mask, True
        
    else: # correct_holes
        # regions where label is in small_regions are holes we want to fill (set to 1)
        # original mask has 0s there.
        holes_mask = np.isin(regions, small_regions)
        
        # Combine original mask (boolean or uint8) with holes_mask (boolean)
        if mask.dtype == bool:
            new_mask = mask | holes_mask
        else:
            new_mask = np.logical_or(mask > 0, holes_mask)
            
        return new_mask, True

class SAM3AutomaticMaskGenerator:
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
        
        # Store configuration for saving later
        self.config = {
            "points_per_side": points_per_side,
            "points_per_batch": points_per_batch,
            "pred_iou_thresh": pred_iou_thresh,
            "stability_score_thresh": stability_score_thresh,
            "stability_score_offset": stability_score_offset,
            "box_nms_thresh": box_nms_thresh,
            "crop_n_layers": crop_n_layers,
            "crop_nms_thresh": crop_nms_thresh,
            "crop_overlap_ratio": crop_overlap_ratio,
            "crop_n_points_downscale_factor": crop_n_points_downscale_factor,
            "min_mask_region_area": min_mask_region_area,
            "use_m2m": use_m2m,
            "device": device
        }
        
        # --- SAM3 Setup ---
        self.predictor = image_model.inst_interactive_predictor
        if self.predictor is None:
             raise ValueError("Model must have inst_interactive_predictor enabled")
        
        if self.predictor.model.backbone is None:
            logger.info("Injecting backbone into tracker...")
            self.predictor.model.backbone = image_model.backbone

    @torch.no_grad()
    def generate(self, image: np.ndarray) -> List[Dict[str, Any]]:
        # 1. Generate Crop Boxes
        crop_boxes, layer_idxs = generate_crop_boxes(
            image.shape[:2], self.crop_n_layers, self.crop_overlap_ratio
        )
        
        # 2. Process Crops
        data_list = []
        for crop_box, layer_idx in zip(crop_boxes, layer_idxs):
            crop_data = self._process_crop(image, crop_box, layer_idx)
            data_list.append(crop_data)
            
        # 3. Concatenate and Post-process
        # data_list contains dicts with keys: masks, ious, points, boxes, crop_boxes
        
        # Flatten all lists
        all_masks = []
        all_ious = []
        all_boxes = []
        
        # We need to reconstruct full masks for everything
        orig_h, orig_w = image.shape[:2]
        
        for d in data_list:
            if d['masks'] is None: continue
            
            # d['masks'] is (N, CropH, CropW)
            # Uncrop to (N, OrigH, OrigW)
            uncropped = uncrop_masks(d['masks'], d['crop_box'], orig_h, orig_w)
            
            # Recalculate boxes on uncropped masks to be precise in global coords
            # Optimization: Shift boxes from crop coords to global coords
            boxes_crop = d['boxes'] # xyxy in crop
            boxes_global = boxes_crop.clone()
            boxes_global[:, 0] += d['crop_box'][0]
            boxes_global[:, 2] += d['crop_box'][0]
            boxes_global[:, 1] += d['crop_box'][1]
            boxes_global[:, 3] += d['crop_box'][1]
            
            all_masks.append(uncropped)
            all_ious.append(d['ious'])
            all_boxes.append(boxes_global)
            
        if not all_masks:
            return []
            
        masks = torch.cat(all_masks, dim=0)
        ious = torch.cat(all_ious, dim=0)
        boxes = torch.cat(all_boxes, dim=0)
        
        # 4. Global NMS
        # Prefer masks from smaller crops? SAM2 does this via setting scores based on crop size.
        # Here we just use IoU scores for simplicity, or we can use batched_nms.
        
        # Using simple NMS on boxes first to reduce load
        keep_by_nms = batched_nms(
            boxes.float(),
            ious,
            torch.zeros_like(boxes[:, 0]), # all same category
            iou_threshold=self.box_nms_thresh
        )
        
        masks = masks[keep_by_nms]
        ious = ious[keep_by_nms]
        boxes = boxes[keep_by_nms]
        
        # 5. Convert to Output Format & Post-process small regions
        results = []
        masks_np = masks.cpu().numpy()
        ious_np = ious.cpu().numpy()
        boxes_np = boxes.cpu().numpy()
        
        for i in range(len(masks_np)):
            mask = masks_np[i]
            
            # Remove small disconnected regions
            if self.min_mask_region_area > 0:
                mask, modified = remove_small_regions(mask, self.min_mask_region_area, "islands")
                if not modified:
                    # If mask wasn't modified, check if it was empty to begin with?
                    pass
                
                # Check area again
                if mask.sum() < self.min_mask_region_area:
                    continue
            
            # Remove holes (optional, SAM2 does it)
            if self.min_mask_region_area > 0:
                mask, _ = remove_small_regions(mask, self.min_mask_region_area, "holes")

            y_idx, x_idx = np.where(mask)
            if len(y_idx) == 0: continue
            
            bbox = [int(x_idx.min()), int(y_idx.min()), int(x_idx.max()), int(y_idx.max())]
            
            results.append({
                "segmentation": mask,
                "area": int(mask.sum()),
                "bbox_xyxy": bbox,
                "predicted_iou": float(ious_np[i]),
            })
            
        return results

    def _process_crop(self, image, crop_box, crop_layer_idx):
        x0, y0, x1, y1 = crop_box
        cropped_im = image[y0:y1, x0:x1, :]
        crop_h, crop_w = cropped_im.shape[:2]
        
        # Set Image
        self.predictor.set_image(cropped_im)
        
        # Generate Points
        points_scale = np.array([crop_w, crop_h])[None, :]
        points_for_image = build_point_grid(
            int(self.points_per_side / (self.crop_n_points_downscale_factor**crop_layer_idx))
        ) * points_scale
        
        # Batch Inference
        points_list = []
        masks_list = []
        ious_list = []
        
        for i in range(0, len(points_for_image), self.points_per_batch):
            batch_points = points_for_image[i : i + self.points_per_batch]
            
            # Predict
            masks, iou_preds, low_res_masks = self.predictor.predict(
                point_coords=batch_points[:, None, :], # expects numpy
                point_labels=np.ones((len(batch_points), 1), dtype=np.int32),
                multimask_output=True
            )
            
            # Convert to torch
            masks = torch.from_numpy(masks).to(self.device)
            iou_preds = torch.from_numpy(iou_preds).to(self.device)
            low_res_masks = torch.from_numpy(low_res_masks).to(self.device)
            
            # M2M Refinement
            if self.use_m2m:
                best_idx = torch.argmax(iou_preds, dim=1)
                batch_range = torch.arange(len(best_idx), device=self.device)
                best_low_res = low_res_masks[batch_range, best_idx][:, None, :, :]
                
                # Refine (convert tensor back to numpy for predict)
                masks, iou_preds, low_res_masks = self.predictor.predict(
                    point_coords=batch_points[:, None, :],
                    point_labels=np.ones((len(batch_points), 1), dtype=np.int32),
                    mask_input=best_low_res.cpu().numpy(),
                    multimask_output=True
                )
                masks = torch.from_numpy(masks).to(self.device)
                iou_preds = torch.from_numpy(iou_preds).to(self.device)
                low_res_masks = torch.from_numpy(low_res_masks).to(self.device)

            # Filter by Stability and IoU
            # 1. Stability
            stability_score = calculate_stability_score(
                low_res_masks, 0.0, self.stability_score_offset
            )
            
            # Select valid masks
            # Check thresholds
            # masks shape: (B, 3, H, W)
            # We want to select the best mask per point OR all valid masks?
            # SAM2 usually flattens everything: (B*3, H, W)
            
            masks = masks.flatten(0, 1) # (B*3, H, W)
            iou_preds = iou_preds.flatten(0, 1)
            stability_score = stability_score.flatten(0, 1)
            
            valid_mask = (iou_preds > self.pred_iou_thresh) & (stability_score > self.stability_score_thresh)
            
            if not valid_mask.any():
                continue
                
            masks = masks[valid_mask]
            iou_preds = iou_preds[valid_mask]
            
            # Filter masks touching crop borders (if not base layer)
            if crop_layer_idx > 0:
                # masks is (N, H, W) boolean
                # Check 1px borders
                touch_border = (
                    masks[:, 0, :].any(dim=1) | 
                    masks[:, -1, :].any(dim=1) |
                    masks[:, :, 0].any(dim=1) |
                    masks[:, :, -1].any(dim=1)
                )
                keep_inner = ~touch_border
                masks = masks[keep_inner]
                iou_preds = iou_preds[keep_inner]
                
            if masks.shape[0] > 0:
                masks_list.append(masks)
                ious_list.append(iou_preds)
                
        if not masks_list:
            return {'masks': None, 'ious': None, 'boxes': None, 'crop_box': crop_box}
            
        final_masks = torch.cat(masks_list, dim=0)
        final_ious = torch.cat(ious_list, dim=0)
        final_boxes = batched_mask_to_box(final_masks)
        
        # Run local NMS inside crop to reduce memory usage before global merge
        keep = batched_nms(
            final_boxes.float(),
            final_ious,
            torch.zeros_like(final_ious),
            iou_threshold=self.box_nms_thresh
        )
        final_masks = final_masks[keep]
        final_ious = final_ious[keep]
        final_boxes = final_boxes[keep]
        
        return {
            'masks': final_masks,
            'ious': final_ious,
            'boxes': final_boxes,
            'crop_box': crop_box
        }

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


    # mask_generator = SAM3AutomaticMaskGenerator(
    #     image_model=sam3_model,
    #     points_per_side=32, 
    #     points_per_batch=64,
    #     pred_iou_thresh=0.8,
    #     stability_score_thresh=0.95,
    #     stability_score_offset=1.0,
    #     crop_n_layers=1, # 1 layer = 1 global + 4 crops
    #     crop_nms_thresh=0.7,
    #     crop_overlap_ratio=512/1500,
    #     min_mask_region_area=100, # Filter tiny trash
    #     use_m2m=True,
    #     device=str(device)
    # )
    mask_generator = SAM3AutomaticMaskGenerator(
        image_model=sam3_model,
        points_per_side=48, 
        points_per_batch=128,
        pred_iou_thresh=0.7,
        stability_score_thresh=0.92,
        stability_score_offset=0.7,
        crop_n_layers=1, # 1 layer = 1 global + 4 crops
        box_nms_thresh=0.7,
        crop_nms_thresh=0.7,
        crop_n_points_downscale_factor=2,
        crop_overlap_ratio=512/1500,
        min_mask_region_area=25, # Filter tiny trash
        use_m2m=True,
        device=str(device)
    )

    input_path = Path('/workspace/ivan_images_slice')
    output_path = Path('/workspace/output_masks_ivan_slice_v5/')
    output_path.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        logger.error(f"Input path {input_path} does not exist.")
        return

    logger.info(f"Reading images from: {input_path}")
    logger.info(f"Saving masks to: {output_path}")
    
    # Save configuration to output directory
    config_path = output_path / "generation_config.json"
    try:
        with open(config_path, "w") as f:
            json.dump(mask_generator.config, f, indent=4)
        logger.info(f"Saved configuration to {config_path}")
    except Exception as e:
        logger.error(f"Failed to save configuration: {e}")

    input_images = sorted([x for x in input_path.iterdir() if x.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']])
    
    for image_path in tqdm(input_images, desc="Processing Images"):
        try:
            image = Image.open(image_path)
            image_np = np.array(image.convert("RGB"))
            masks = mask_generator.generate(image_np)
            save_anns(masks, output_path / image_path.name)
        except Exception as e:
            logger.error(f"Error processing {image_path.name}: {e}", exc_info=True)

if __name__ == "__main__":
    main()
