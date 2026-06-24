import logging
import sys
import os

# Configure logging at the very beginning
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("sam3_auto_mask_gen2.log"),
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
from typing import Any, Dict, List, Optional, Tuple, Generator
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torchvision.ops import batched_nms, box_area
from copy import deepcopy

# Import MaskData from the ported utils
try:
    from sam3.utils.amg import (
        MaskData,
        generate_crop_boxes,
        uncrop_masks,
        batched_mask_to_box,
        calculate_stability_score,
        build_point_grid,
        build_all_layer_point_grids,
        remove_small_regions,
        is_box_near_crop_edge,
        uncrop_boxes_xyxy,
        coco_encode_rle,
        uncrop_points,
        mask_to_rle_pytorch,
        rle_to_mask,
        area_from_rle,
        box_xyxy_to_xywh,
        batch_iterator,
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
        build_all_layer_point_grids,
        remove_small_regions,
        is_box_near_crop_edge,
        uncrop_boxes_xyxy,
        coco_encode_rle,
        uncrop_points,
        mask_to_rle_pytorch,
        rle_to_mask,
        area_from_rle,
        box_xyxy_to_xywh,
        batch_iterator,
    )

try:
    from sam3.model_builder import build_sam3_image_model
except ImportError as e:
    logger.error(f"Error importing SAM3 modules: {e}")
    sys.exit(1)

# --- Missing Utils from SAM2 ---
# (Removed manually defined utils as they are now imported from sam3.utils.amg)
# --- End Missing Utils ---

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
        mask_threshold: float = 0.0,
        box_nms_thresh: float = 0.7,
        crop_n_layers: int = 0,
        crop_nms_thresh: float = 0.7,
        crop_overlap_ratio: float = 512 / 1500,
        crop_n_points_downscale_factor: int = 1,
        point_grids: Optional[List[np.ndarray]] = None,
        min_mask_region_area: int = 0,
        output_mode: str = "binary_mask",
        use_m2m: bool = True,
        multimask_output: bool = True,
        device: str = "cuda"
    ):
        self.points_per_side = points_per_side
        self.points_per_batch = points_per_batch
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.stability_score_offset = stability_score_offset
        self.mask_threshold = mask_threshold
        self.box_nms_thresh = box_nms_thresh
        self.crop_n_layers = crop_n_layers
        self.crop_nms_thresh = crop_nms_thresh
        self.crop_overlap_ratio = crop_overlap_ratio
        self.crop_n_points_downscale_factor = crop_n_points_downscale_factor
        self.min_mask_region_area = min_mask_region_area
        self.output_mode = output_mode
        self.use_m2m = use_m2m
        self.multimask_output = multimask_output
        self.device = device
        
        self.image_model = image_model
        
        self.config = {
            "points_per_side": points_per_side,
            "points_per_batch": points_per_batch,
            "pred_iou_thresh": pred_iou_thresh,
            "stability_score_thresh": stability_score_thresh,
            "stability_score_offset": stability_score_offset,
            "mask_threshold": mask_threshold,
            "box_nms_thresh": box_nms_thresh,
            "crop_n_layers": crop_n_layers,
            "crop_nms_thresh": crop_nms_thresh,
            "crop_overlap_ratio": crop_overlap_ratio,
            "crop_n_points_downscale_factor": crop_n_points_downscale_factor,
            "min_mask_region_area": min_mask_region_area,
            "output_mode": output_mode,
            "use_m2m": use_m2m,
            "multimask_output": multimask_output,
            "device": device
        }

        if points_per_side is not None:
            self.point_grids = build_all_layer_point_grids(
                points_per_side,
                crop_n_layers,
                crop_n_points_downscale_factor,
            )
        elif point_grids is not None:
            self.point_grids = point_grids
        else:
            raise ValueError("Can't have both points_per_side and point_grid be None.")
        
        self.predictor = image_model.inst_interactive_predictor
        if self.predictor is None:
             raise ValueError("Model must have inst_interactive_predictor enabled")
        
        if self.predictor.model.backbone is None:
            logger.info("Injecting backbone into tracker...")
            self.predictor.model.backbone = image_model.backbone

    @torch.no_grad()
    def generate(self, image: np.ndarray) -> List[Dict[str, Any]]:
        # Generate masks
        mask_data = self._generate_masks(image)

        # Filter small disconnected regions and holes in masks
        if self.min_mask_region_area > 0:
            mask_data = self.postprocess_small_regions(
                mask_data,
                self.min_mask_region_area,
                max(self.box_nms_thresh, self.crop_nms_thresh),
            )

        # Encode masks
        if self.output_mode == "coco_rle":
            mask_data["segmentations"] = [
                coco_encode_rle(rle) for rle in mask_data["rles"]
            ]
        elif self.output_mode == "binary_mask":
            mask_data["segmentations"] = [rle_to_mask(rle) for rle in mask_data["rles"]]
        else:
            mask_data["segmentations"] = mask_data["rles"]

        # Write mask records
        curr_anns = []
        for idx in range(len(mask_data["segmentations"])):
            ann = {
                "segmentation": mask_data["segmentations"][idx],
                "area": area_from_rle(mask_data["rles"][idx]),
                "bbox": box_xyxy_to_xywh(mask_data["boxes"][idx]).tolist(),
                "predicted_iou": mask_data["iou_preds"][idx].item(),
                "point_coords": [mask_data["points"][idx].tolist()],
                "stability_score": mask_data["stability_score"][idx].item(),
                "crop_box": box_xyxy_to_xywh(mask_data["crop_boxes"][idx]).tolist(),
            }
            curr_anns.append(ann)

        return curr_anns

    def _generate_masks(self, image: np.ndarray) -> MaskData:
        orig_size = image.shape[:2]
        crop_boxes, layer_idxs = generate_crop_boxes(
            orig_size, self.crop_n_layers, self.crop_overlap_ratio
        )

        # Iterate over image crops
        data = MaskData()
        for crop_box, layer_idx in zip(crop_boxes, layer_idxs):
            crop_data = self._process_crop(image, crop_box, layer_idx, orig_size)
            data.cat(crop_data)

        # Remove duplicate masks between crops
        if len(crop_boxes) > 1:
            # Prefer masks from smaller crops
            scores = 1 / box_area(data["crop_boxes"])
            scores = scores.to(data["boxes"].device)
            keep_by_nms = batched_nms(
                data["boxes"].float(),
                scores,
                torch.zeros_like(data["boxes"][:, 0]),  # categories
                iou_threshold=self.crop_nms_thresh,
            )
            data.filter(keep_by_nms)
        
        data.to_numpy()
        return data

    def _process_crop(
        self,
        image: np.ndarray,
        crop_box: List[int],
        crop_layer_idx: int,
        orig_size: Tuple[int, ...],
    ) -> MaskData:
        # Crop the image
        x0, y0, x1, y1 = crop_box
        cropped_im = image[y0:y1, x0:x1, :]
        cropped_im_size = cropped_im.shape[:2]
        orig_h, orig_w = orig_size
        
        self.predictor.set_image(cropped_im)

        # Get points for this crop
        points_scale = np.array(cropped_im_size)[None, ::-1]
        points_for_image = self.point_grids[crop_layer_idx] * points_scale

        # Generate masks for this crop in batches
        data = MaskData()
        for (points,) in batch_iterator(self.points_per_batch, points_for_image):
            batch_data = self._process_batch(
                points, cropped_im_size, crop_box, orig_size, normalize=True
            )
            data.cat(batch_data)
            del batch_data
        
        # SAM3 specific: Clear predictor state if needed? SAM2 does reset_predictor.
        # self.predictor.reset_predictor() # SAM3 inst_interactive_predictor might not need/have this? 
        # Checking usage in user code: no reset called. Assuming OK.

        # Remove duplicates within this crop.
        keep_by_nms = batched_nms(
            data["boxes"].float(),
            data["iou_preds"],
            torch.zeros_like(data["boxes"][:, 0]),  # categories
            iou_threshold=self.box_nms_thresh,
        )
        data.filter(keep_by_nms)

        # Return to the original image frame
        data["boxes"] = uncrop_boxes_xyxy(data["boxes"], crop_box)
        data["points"] = uncrop_points(data["points"], crop_box)
        data["crop_boxes"] = torch.tensor([crop_box for _ in range(len(data["rles"]))])

        return data

    def _process_batch(
        self,
        points: np.ndarray,
        im_size: Tuple[int, ...],
        crop_box: List[int],
        orig_size: Tuple[int, ...],
        normalize=False,
    ) -> MaskData:
        orig_h, orig_w = orig_size
        
        # Run model on this batch
        # SAM3 takes points in (B, 1, 2) format for single point prompt per object
        points_torch = torch.as_tensor(points, dtype=torch.float32, device=self.device)
        
        # Transform coords
        in_points = self.predictor._transforms.transform_coords(
            points_torch, normalize=normalize, orig_hw=im_size
        )
        
        in_labels = torch.ones(
            in_points.shape[0], dtype=torch.int, device=self.device
        )

        # SAM3 predict call
        # Returns: masks, iou_preds, low_res_masks
        masks, iou_preds, low_res_masks = self.predictor._predict(
            point_coords=in_points[:, None, :],
            point_labels=in_labels[:, None],
            multimask_output=self.multimask_output,
            return_logits=True,
        )

        # Serialize predictions and store in MaskData
        # masks shape: (B, 3, H, W) -> flatten to (B*3, H, W)
        data = MaskData(
            masks=masks.flatten(0, 1),
            iou_preds=iou_preds.flatten(0, 1),
            points=points_torch.repeat_interleave(masks.shape[1], dim=0),
            low_res_masks=low_res_masks.flatten(0, 1),
        )
        del masks

        if not self.use_m2m:
            # Filter by predicted IoU
            if self.pred_iou_thresh > 0.0:
                keep_mask = data["iou_preds"] > self.pred_iou_thresh
                data.filter(keep_mask)

            # Calculate and filter by stability score
            # Note: low_res_masks are usually logits. 
            data["stability_score"] = calculate_stability_score(
                data["low_res_masks"], self.mask_threshold, self.stability_score_offset
            )
            if self.stability_score_thresh > 0.0:
                keep_mask = data["stability_score"] >= self.stability_score_thresh
                data.filter(keep_mask)
        else:
            # One step refinement using previous mask predictions
            # SAM3 m2m
            # Re-implementing M2M to match SAM3 logic properly:
            # We need to do the second predict call.
            # Extract best low res masks from the first pass
            
            # Reshape back to (B, 3, ...)
            # data["low_res_masks"] is flattened (B*3, 256, 256)
            curr_low_res = data["low_res_masks"].view(len(points), -1, *data["low_res_masks"].shape[-2:])
            curr_iou = data["iou_preds"].view(len(points), -1)
            
            best_idx = torch.argmax(curr_iou, dim=1)
            batch_range = torch.arange(len(best_idx), device=self.device)
            best_low_res = curr_low_res[batch_range, best_idx][:, None, :, :] # (B, 1, 256, 256)
            
            # No need to transform points again, they are already transformed in `in_points`
            
            masks, iou_preds, low_res_masks = self.predictor._predict(
                point_coords=in_points[:, None, :],
                point_labels=in_labels[:, None],
                mask_input=best_low_res, # Accepts tensor Bx1xHxW
                multimask_output=self.multimask_output,
                return_logits=True,
            )

            data["masks"] = masks.flatten(0, 1)
            data["iou_preds"] = iou_preds.flatten(0, 1)
            data["low_res_masks"] = low_res_masks.flatten(0, 1)

            if self.pred_iou_thresh > 0.0:
                keep_mask = data["iou_preds"] > self.pred_iou_thresh
                data.filter(keep_mask)

            data["stability_score"] = calculate_stability_score(
                data["low_res_masks"], self.mask_threshold, self.stability_score_offset
            )
            if self.stability_score_thresh > 0.0:
                keep_mask = data["stability_score"] >= self.stability_score_thresh
                data.filter(keep_mask)

        # Threshold masks and calculate boxes
        # SAM3 masks might already be binary if predict returns them?
        # But we treated them as logits in calculate_stability_score (which expects logits).
        # So we assume masks/low_res_masks are logits.
        # But wait, SAM3 code usually returns binary masks for `masks` and logits for `low_res_masks`?
        # Let's check `sam3_auto_mask_gen.py`:
        # `masks = masks > 0`
        # This implies `masks` from predictor are NOT boolean. They are likely logits or probability maps.
        # SAM2 `_predict` returns logits for `masks` (if return_logits=True).
        # We will assume `masks` are logits.
        
        data["masks"] = data["masks"] > self.mask_threshold
        data["boxes"] = batched_mask_to_box(data["masks"])

        # Filter boxes that touch crop boundaries
        keep_mask = ~is_box_near_crop_edge(
            data["boxes"], crop_box, [0, 0, orig_w, orig_h]
        )
        if not torch.all(keep_mask):
            data.filter(keep_mask)

        # Compress to RLE
        data["masks"] = uncrop_masks(data["masks"], crop_box, orig_h, orig_w)
        data["rles"] = mask_to_rle_pytorch(data["masks"])
        del data["masks"]
        del data["low_res_masks"] # Don't need low res anymore

        return data

    @staticmethod
    def postprocess_small_regions(
        mask_data: MaskData, min_area: int, nms_thresh: float
    ) -> MaskData:
        """
        Removes small disconnected regions and holes in masks, then reruns
        box NMS to remove any new duplicates.
        """
        if len(mask_data["rles"]) == 0:
            return mask_data

        # Filter small disconnected regions and holes
        new_masks = []
        scores = []
        for rle in mask_data["rles"]:
            mask = rle_to_mask(rle)

            mask, changed = remove_small_regions(mask, min_area, mode="holes")
            unchanged = not changed
            mask, changed = remove_small_regions(mask, min_area, mode="islands")
            unchanged = unchanged and not changed

            new_masks.append(torch.as_tensor(mask).unsqueeze(0))
            # Give score=0 to changed masks and score=1 to unchanged masks
            # so NMS will prefer ones that didn't need postprocessing
            scores.append(float(unchanged))

        # Recalculate boxes and remove any new duplicates
        masks = torch.cat(new_masks, dim=0)
        boxes = batched_mask_to_box(masks)
        keep_by_nms = batched_nms(
            boxes.float(),
            torch.as_tensor(scores),
            torch.zeros_like(boxes[:, 0]),  # categories
            iou_threshold=nms_thresh,
        )

        # Only recalculate RLEs for masks that have changed
        for i_mask in keep_by_nms:
            if scores[i_mask] == 0.0:
                mask_torch = masks[i_mask].unsqueeze(0)
                mask_data["rles"][i_mask] = mask_to_rle_pytorch(mask_torch)[0]
                mask_data["boxes"][i_mask] = boxes[i_mask]  # update res directly
        mask_data.filter(keep_by_nms)

        return mask_data

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
        points_per_side=48, 
        points_per_batch=128,
        pred_iou_thresh=0.82,
        stability_score_thresh=0.89,
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

    input_path = Path('/workspace/ivan_input_images')
    output_path = Path('/workspace/output_masks_ivan_48_ALL1200/')
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
