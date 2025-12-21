import torch
import numpy as np
import cv2
import sys
import os
import logging
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torchvision.ops import nms

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

class SAM3AutomaticMaskGenerator:
    """
    SAM3 version of AutomaticMaskGenerator.
    Generates masks for entire image using grid prompts.
    Adapted from sam3/uvost_sam3.py
    
    NOTE on 'Jagged Edges' vs SAM2:
    This implementation uses a single-pass grid prompt approach with Sam3Processor.
    SAM2's AutomaticMaskGenerator typically produces smoother/better masks because it uses:
    1. Iterative Refinement (use_m2m=True): Uses the mask output as input for a second pass.
    2. Stability Score: Filters out masks that change shape significantly with threshold changes.
    3. Cropping (crop_n_layers > 0): Runs inference on zoomed-in crops for better small object detail.
    
    Current implementation does not include these features (M2M, Stability Score, Cropping), 
    which may result in lower edge quality or 'jagged' edges compared to the full SAM2 generator.
    """
    
    def __init__(
        self,
        image_model,
        points_per_side: int = 32,
        points_per_batch: int = 64, # Not fully used in current implementation but kept for API compatibility
        pred_iou_thresh: float = 0.7,
        box_nms_thresh: float = 0.7,
        min_mask_region_area: int = 100,
        device: str = "cuda"
    ):
        self.image_model = image_model
        self.processor = Sam3Processor(image_model, confidence_threshold=0.5)
        self.points_per_side = points_per_side
        self.points_per_batch = points_per_batch
        self.pred_iou_thresh = pred_iou_thresh
        self.box_nms_thresh = box_nms_thresh
        self.min_mask_region_area = min_mask_region_area
        self.device = device
        
        # Build point grid [0,1] x [0,1]
        offset = 1 / (2 * points_per_side)
        xs = np.linspace(offset, 1 - offset, points_per_side)
        ys = np.linspace(offset, 1 - offset, points_per_side)
        xv, yv = np.meshgrid(xs, ys)
        self.point_grids = np.stack([xv.flatten(), yv.flatten()], axis=1)
        
        logger.info(f"Initialized SAM3Generator with {points_per_side}x{points_per_side} grid")
    
    @torch.no_grad()
    def generate(self, image: np.ndarray):
        """Generate masks using grid prompts."""
        h, w = image.shape[:2]
        pil_image = Image.fromarray(image)
        inference_state = self.processor.set_image(pil_image)
        
        all_masks = []
        all_scores = []
        all_points = []
        
        # Note: current implementation iterates point by point. 
        # For optimization, batching could be implemented if supported by Sam3Processor/Model
        # Using tqdm for progress tracking but silencing it for logs to avoid clutter, relying on logger
        for point in tqdm(self.point_grids, desc="Grid prompts", leave=False, disable=True):
            self.processor.reset_all_prompts(inference_state)
            
            if "geometric_prompt" not in inference_state:
                inference_state["geometric_prompt"] = self.image_model._get_dummy_prompt()
            
            points_tensor = torch.tensor([point], device=self.device, dtype=torch.float32).view(1, 1, 2)
            labels_tensor = torch.tensor([1], device=self.device, dtype=torch.long).view(1, 1)
            inference_state["geometric_prompt"].append_points(points_tensor, labels_tensor)
            
            if "language_features" not in inference_state.get("backbone_out", {}):
                dummy_text = self.image_model.backbone.forward_text(["visual"], device=self.device)
                inference_state["backbone_out"].update(dummy_text)
            
            try:
                inference_state = self.processor._forward_grounding(inference_state)
                
                if "masks" in inference_state and inference_state["scores"].numel() > 0:
                    masks = inference_state["masks"]
                    scores = inference_state["scores"]
                    best_idx = torch.argmax(scores)
                    score = scores[best_idx].item()
                    
                    if score >= self.pred_iou_thresh:
                        mask = masks[best_idx].squeeze().cpu().numpy()
                        if mask.sum() >= self.min_mask_region_area:
                            all_masks.append(mask)
                            all_scores.append(score)
                            all_points.append(point * np.array([w, h]))
            except Exception as e:
                # logger.debug(f"Error processing point {point}: {e}")
                continue
        
        if not all_masks:
            return []
        
        return self._apply_nms(all_masks, all_scores, all_points, h, w)
    
    def _apply_nms(self, masks, scores, points, h, w):
        """Apply NMS to filter overlapping masks."""
        boxes = []
        valid_indices = []
        
        for idx, m in enumerate(masks):
            m = m.squeeze() if m.ndim == 3 else m
            y_idx, x_idx = np.where(m > 0)
            if len(y_idx) > 0:
                boxes.append([float(x_idx.min()), float(y_idx.min()), 
                             float(x_idx.max()), float(y_idx.max())])
                valid_indices.append(idx)
        
        if not boxes:
            return []
        
        boxes_t = torch.tensor(boxes, dtype=torch.float32)
        scores_t = torch.tensor([scores[i] for i in valid_indices], dtype=torch.float32)
        keep = nms(boxes_t, scores_t, self.box_nms_thresh)
        
        results = []
        for k in keep:
            idx = valid_indices[k.item()]
            mask = masks[idx].squeeze() if masks[idx].ndim == 3 else masks[idx]
            y_idx, x_idx = np.where(mask > 0)
            
            results.append({
                'segmentation': mask > 0,
                'bbox_xyxy': [int(x_idx.min()), int(y_idx.min()), 
                              int(x_idx.max()), int(y_idx.max())],
                'area': int(mask.sum()),
                'predicted_iou': scores[idx],
                'point_coords': points[idx].tolist()
            })
        
        return results

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # use bfloat16 for the entire notebook
    if torch.cuda.is_available():
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("TF32 enabled for Ampere GPU")

    logger.info("Building SAM3 model...")
    # Initialize SAM3 Image Model
    sam3_model = build_sam3_image_model(
        device=str(device),
        eval_mode=True,
    )

    mask_generator = SAM3AutomaticMaskGenerator(
        image_model=sam3_model,
        points_per_side=32, # Adjusted for SAM3/UVOST defaults, was 64 in SAM2 example
        points_per_batch=128,
        pred_iou_thresh=0.7,
        box_nms_thresh=0.7,
        min_mask_region_area=25.0, # Matching SAM2 example
        device=str(device)
    )

    # Input and Output paths
    # input_path = Path('/sam2_custom/input_images/')
    input_path = Path('/workspace/images_30/')
    if not input_path.exists():
        logger.warning(f"Input path {input_path} does not exist.")
        # Fallback to local directory or user should configure
        local_input = Path('input_images')
        if local_input.exists():
            input_path = local_input
            logger.info(f"Using local input path: {input_path}")
        else:
            logger.error("No valid input directory found.")
            # Don't return, let it fail or user fix

    # output_path = Path('/sam3_custom/output_images/')
    output_path = Path('/workspace/output_masks_new/')
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
            logger.error(f"Error processing {image_name}: {e}")

if __name__ == "__main__":
    main()
