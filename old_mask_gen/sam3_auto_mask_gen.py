import logging
import sys
import os

# Configure logging at the very beginning
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("sam3_auto_mask_gen.log"),
        logging.StreamHandler(sys.stdout)
    ],
    force=True
)
logger = logging.getLogger("SAM3AutoMaskGen")

import torch
import numpy as np
import cv2
import math
import json
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torchvision.ops import nms, batched_nms

# Import MaskData from the ported utils
try:
    from sam3.utils.amg import (
        MaskData,
        generate_crop_boxes,
        uncrop_masks,
        batched_mask_to_box,
        calculate_stability_score,
        build_point_grid,
        remove_small_regions,
        is_box_near_crop_edge,
    )
except ImportError:
    # If not installed as package, try relative import or path hack
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from sam3.utils.amg import (
        MaskData,
        generate_crop_boxes,
        uncrop_masks,
        batched_mask_to_box,
        calculate_stability_score,
        build_point_grid,
        remove_small_regions,
        is_box_near_crop_edge,
    )

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

class SAM3AutomaticMaskGenerator:
    def __init__(
        self,
        image_model,
        points_per_side: int = 32,
        points_per_batch: int = 64,
        pred_iou_thresh: float = 0.88,
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
        
        # Store the model itself to access predictor and backbone if needed
        self.image_model = image_model 
        
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
        data = MaskData()
        for crop_box, layer_idx in zip(crop_boxes, layer_idxs):
            crop_data = self._process_crop(image, crop_box, layer_idx)
            data.cat(crop_data)
            
        # 3. Post-processing (Remove duplicates)
        if len(crop_boxes) > 1:
            # Prefer masks from smaller crops
            # SAM2: scores = 1 / box_area(data["crop_boxes"])
            crop_areas = (data["crop_boxes"][:, 2] - data["crop_boxes"][:, 0]) * \
                         (data["crop_boxes"][:, 3] - data["crop_boxes"][:, 1])
            scores = 1.0 / (crop_areas.float() + 1e-6)
            
            # Using batched_nms from torchvision
            keep_by_nms = batched_nms(
                data["boxes"].float(),
                scores,
                torch.zeros_like(data["boxes"][:, 0]),
                iou_threshold=self.crop_nms_thresh,
            )
            data.filter(keep_by_nms)
        
        # 4. Final Global NMS based on IoU (standard)
        keep_by_nms = batched_nms(
            data["boxes"].float(),
            data["iou_preds"],
            torch.zeros_like(data["boxes"][:, 0]),
            iou_threshold=self.box_nms_thresh,
        )
        data.filter(keep_by_nms)

        # 5. Convert to Output Format & Post-process small regions
        data.to_numpy()
        results = []
        
        for i in range(len(data["masks"])):
            mask = data["masks"][i] > 0 # Convert to boolean
            
            # Remove small disconnected regions
            if self.min_mask_region_area > 0:
                mask, modified = remove_small_regions(mask, self.min_mask_region_area, "islands")
                if not modified:
                    pass
                
                if mask.sum() < self.min_mask_region_area:
                    continue
            
            # Remove holes
            if self.min_mask_region_area > 0:
                mask, _ = remove_small_regions(mask, self.min_mask_region_area, "holes")

            y_idx, x_idx = np.where(mask)
            if len(y_idx) == 0: continue
            
            bbox = [int(x_idx.min()), int(y_idx.min()), int(x_idx.max()), int(y_idx.max())]
            
            results.append({
                "segmentation": mask,
                "area": int(mask.sum()),
                "bbox_xyxy": bbox,
                "predicted_iou": float(data["iou_preds"][i]),
                "stability_score": float(data["stability_score"][i]),
                "point_coords": [data["points"][i].tolist()]
            })
            
        return results

    def _process_crop(self, image, crop_box, crop_layer_idx):
        x0, y0, x1, y1 = crop_box
        cropped_im = image[y0:y1, x0:x1, :]
        crop_h, crop_w = cropped_im.shape[:2]
        
        self.predictor.set_image(cropped_im)
        
        points_scale = np.array([crop_w, crop_h])[None, :]
        points_for_image = build_point_grid(
            int(self.points_per_side / (self.crop_n_points_downscale_factor**crop_layer_idx))
        ) * points_scale
        
        data = MaskData()
        
        for i in range(0, len(points_for_image), self.points_per_batch):
            batch_points = points_for_image[i : i + self.points_per_batch]
            
            transformed_points = batch_points[:, None, :]
            
            masks, iou_preds, low_res_masks = self.predictor.predict(
                point_coords=transformed_points,
                point_labels=np.ones((len(batch_points), 1), dtype=np.int32),
                multimask_output=True
            )
            
            masks = torch.from_numpy(masks).to(self.device)
            iou_preds = torch.from_numpy(iou_preds).to(self.device)
            low_res_masks = torch.from_numpy(low_res_masks).to(self.device)
            
            if self.use_m2m:
                best_idx = torch.argmax(iou_preds, dim=1)
                batch_range = torch.arange(len(best_idx), device=self.device)
                best_low_res = low_res_masks[batch_range, best_idx][:, None, :, :]
                
                masks, iou_preds, low_res_masks = self.predictor.predict(
                    point_coords=transformed_points,
                    point_labels=np.ones((len(batch_points), 1), dtype=np.int32),
                    mask_input=best_low_res.cpu().numpy(),
                    multimask_output=True
                )
                masks = torch.from_numpy(masks).to(self.device)
                iou_preds = torch.from_numpy(iou_preds).to(self.device)
                low_res_masks = torch.from_numpy(low_res_masks).to(self.device)

            stability_score = calculate_stability_score(
                low_res_masks, 0.0, self.stability_score_offset
            )
            
            masks = masks.flatten(0, 1) # (B*3, H, W)
            iou_preds = iou_preds.flatten(0, 1)
            stability_score = stability_score.flatten(0, 1)
            
            # Expand points to match flattened masks
            points_repeated = torch.as_tensor(batch_points, device=self.device).repeat_interleave(3, dim=0)
            
            valid_mask = (iou_preds > self.pred_iou_thresh) & (stability_score > self.stability_score_thresh)
            
            if not valid_mask.any():
                continue
                
            masks = masks[valid_mask]
            iou_preds = iou_preds[valid_mask]
            stability_score = stability_score[valid_mask]
            points_repeated = points_repeated[valid_mask]
            
            # Calculate boxes in crop coords
            boxes = batched_mask_to_box(masks > 0)
            
            if crop_layer_idx > 0:
                # Filter masks touching crop boundaries
                
                # Check if touching crop edge BUT NOT image edge
                # Left edge is image edge if x0 == 0
                # Top edge is image edge if y0 == 0
                # Right edge is image edge if x1 == orig_w
                # Bottom edge is image edge if y1 == orig_h
                
                # We need original image size to check if crop edge is image edge
                orig_h, orig_w = image.shape[:2] # image passed to generate is full image
                
                touch_left = (boxes[:, 0] <= 1) & (x0 > 0)
                touch_top = (boxes[:, 1] <= 1) & (y0 > 0)
                touch_right = (boxes[:, 2] >= crop_w - 1) & (x1 < orig_w)
                touch_bottom = (boxes[:, 3] >= crop_h - 1) & (y1 < orig_h)
                
                discard = touch_left | touch_top | touch_right | touch_bottom
                keep_inner = ~discard
                
                masks = masks[keep_inner]
                iou_preds = iou_preds[keep_inner]
                stability_score = stability_score[keep_inner]
                points_repeated = points_repeated[keep_inner]
                boxes = boxes[keep_inner]
            
            if masks.shape[0] > 0:
                # Uncrop masks to global size
                orig_h, orig_w = image.shape[:2]
                uncropped_masks = uncrop_masks(masks, crop_box, orig_h, orig_w)
                
                # Global boxes
                boxes[:, 0] += x0
                boxes[:, 2] += x0
                boxes[:, 1] += y0
                boxes[:, 3] += y0
                
                crop_box_torch = torch.tensor([x0, y0, x1, y1], device=self.device).repeat(len(masks), 1)
                
                batch_data = MaskData(
                    masks=uncropped_masks,
                    iou_preds=iou_preds,
                    stability_score=stability_score,
                    boxes=boxes,
                    points=points_repeated,
                    crop_boxes=crop_box_torch
                )
                data.cat(batch_data)
        
        return data

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
        points_per_side=32, 
        points_per_batch=128,
        pred_iou_thresh=0.81,
        stability_score_thresh=0.88,
        stability_score_offset=1.0,
        crop_n_layers=1, # 1 layer = 1 global + 4 crops
        box_nms_thresh=0.9,
        crop_nms_thresh=0.7,
        crop_n_points_downscale_factor=1,
        crop_overlap_ratio=512/1500,
        min_mask_region_area=25, # Filter tiny trash
        use_m2m=True,
        device=str(device)
    )

    input_path = Path('/workspace/ivan_images_slice')
    output_path = Path('/workspace/output_masks_ivan_slicy_v8/')
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
            sys.exit(1)

if __name__ == "__main__":
    main()
