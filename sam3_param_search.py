import logging
import sys
import os
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Ensure we can import from the current directory structure
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    # from sam3_auto_mask_gen import SAM3AutomaticMaskGenerator, save_anns
    from sam3_auto_mask_gen2 import SAM3AutomaticMaskGenerator, save_anns
    from sam3.model_builder import build_sam3_image_model
except ImportError as e:
    # Fallback for imports if running directly inside the package structure
    sys.path.append(os.path.join(current_dir, ".."))
    from sam3_auto_mask_gen2 import SAM3AutomaticMaskGenerator, save_anns
    from sam3.model_builder import build_sam3_image_model

# Configure logging to output to stdout and file
# Must be done AFTER imports because imported modules might force-reset logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("sam3_param_search.log"),
        logging.StreamHandler(sys.stdout)
    ],
    force=True
)
logger = logging.getLogger("SAM3ParamSearch")

def get_experiment_configs():
    """
    Returns a dictionary of experiment configurations.
    """
    base_config = {
        "points_per_side": 32,
        "points_per_batch": 64,
        "pred_iou_thresh": 0.88,
        "stability_score_thresh": 0.95,
        "stability_score_offset": 1.0,
        "box_nms_thresh": 0.7,
        "crop_n_layers": 0,
        "crop_nms_thresh": 0.7,
        "crop_overlap_ratio": 512 / 1500,
        "crop_n_points_downscale_factor": 1,
        "min_mask_region_area": 0,
        "use_m2m": True,
    }

    experiments = {}

    # 1. Balanced Detail (Baseline for office)
    # Good starting point for tables/rooms
    conf1 = base_config.copy()
    conf1.update({
        "crop_n_layers": 1, 
        "min_mask_region_area": 50,
        "points_per_side": 32
    })
    experiments["1_balanced_detail"] = conf1

    # 2. Deep Crop (Small Objects)
    # Focus on tiny details like switches, pens
    # conf2 = base_config.copy()
    # conf2.update({
    #     "crop_n_layers": 2,
    #     "points_per_side": 48,
    #     "pred_iou_thresh": 0.85, # Slightly lower to catch faint small objects
    #     "stability_score_thresh": 0.92,
    #     "min_mask_region_area": 25
    # })
    # experiments["2_deep_crop_small_obj"] = conf2

    # 3. High Precision (No Junk)
    # Focus on clean walls/tables, ignore ambiguous noise
    conf3 = base_config.copy()
    conf3.update({
        "crop_n_layers": 1,
        "pred_iou_thresh": 0.92, # Very strict
        "stability_score_thresh": 0.96, # Very strict
        "box_nms_thresh": 0.5, # Aggressive deduplication
        "min_mask_region_area": 100
    })
    experiments["3_high_precision"] = conf3

    # 4. Global Dense (No Crops)
    # Fast, checks if dense grid is enough without cropping
    conf4 = base_config.copy()
    conf4.update({
        "crop_n_layers": 0,
        "points_per_side": 64, # Dense grid
    })
    experiments["4_global_dense"] = conf4

    # 5. Edge Master (High Overlap)
    # For objects cut by crop boundaries
    conf5 = base_config.copy()
    conf5.update({
        "crop_n_layers": 1,
        "crop_overlap_ratio": 0.8, # Significant overlap
        "points_per_side": 32,
    })
    experiments["5_high_overlap"] = conf5

    # 6. Raw Sensitive (No M2M)
    # Experimental: See raw outputs without refinement
    conf6 = base_config.copy()
    conf6.update({
        "crop_n_layers": 1,
        "use_m2m": False,
        "pred_iou_thresh": 0.80, # Allow lower confidence
        "stability_score_thresh": 0.90,
    })
    experiments["6_raw_sensitive"] = conf6

    # 7. Ultra Detail (Slow & Thorough)
    # Deep crops with high overlap to catch everything
    conf7 = base_config.copy()
    conf7.update({
        "crop_n_layers": 2,
        "crop_overlap_ratio": 0.5, # 50% overlap!
        "points_per_side": 32,
        "min_mask_region_area": 15,
    })
    experiments["7_ultra_detail"] = conf7

    # 8. Relaxed Sensitivity (Max Recall)
    # Lower thresholds to see what the model "barely" sees
    conf8 = base_config.copy()
    conf8.update({
        "crop_n_layers": 1,
        "pred_iou_thresh": 0.75,
        "stability_score_thresh": 0.85,
        "box_nms_thresh": 0.8, # Allow more overlaps
    })
    experiments["8_relaxed_sensitivity"] = conf8

    conf9 = {
        "points_per_side": 32,
        "points_per_batch": 128,
        "pred_iou_thresh": 0.82,
        "stability_score_thresh": 0.88,
        "stability_score_offset": 1.0,
        "crop_n_layers": 1, # 1 layer = 1 global + 4 crops
        "box_nms_thresh": 0.9,
        "crop_nms_thresh": 0.7,
        "crop_n_points_downscale_factor": 1,
        "crop_overlap_ratio": 512/1500,
        "min_mask_region_area": 25, # Filter tiny trash
        "use_m2m": True,
    }
    experiments["9_pred_iou_thresh_0.82"] = conf9

    conf10 = conf9.copy()
    conf10["pred_iou_thresh"] = 0.7
    experiments["10_pred_iou_thresh_0.7"] = conf10

    conf11 = conf9.copy()
    conf11["pred_iou_thresh"] = 0.73
    experiments["11_pred_iou_thresh_0.73"] = conf11

    conf12 = conf9.copy()
    conf12["pred_iou_thresh"] = 0.75
    experiments["12_pred_iou_thresh_0.75"] = conf12

    conf13 = conf9.copy()
    conf13["pred_iou_thresh"] = 0.77
    experiments["13_pred_iou_thresh_0.77"] = conf13

    conf14 = conf9.copy()
    conf14["pred_iou_thresh"] = 0.79
    experiments["14_pred_iou_thresh_0.79"] = conf14

    conf15 = conf9.copy()
    conf15.update({
        "pred_iou_thresh": 0.82,
        "stability_score_thresh": 0.85,
    })
    experiments["15_pred_iou_thresh_0.82_stability_0.85"] = conf15

    conf16 = conf9.copy()
    conf16.update({
        "pred_iou_thresh": 0.82,
        "stability_score_thresh": 0.87,
    })
    experiments["16_pred_iou_thresh_0.82_stability_0.87"] = conf16

    conf17 = conf9.copy()
    conf17.update({
        "pred_iou_thresh": 0.82,
        "stability_score_thresh": 0.89,
    })
    experiments["17_pred_iou_thresh_0.82_stability_0.89"] = conf17

    conf18 = conf9.copy()
    conf18.update({
        "pred_iou_thresh": 0.82,
        "stability_score_thresh": 0.91,
    })
    experiments["18_pred_iou_thresh_0.82_stability_0.91"] = conf18

    conf19 = conf9.copy()

    conf20 = conf9.copy()
    conf20.update({
        "pred_iou_thresh": 0.82,
        "stability_score_thresh": 0.93,
    })
    experiments["20_pred_iou_thresh_0.82_stability_0.93"] = conf20

    return experiments

def main():
    # 1. Setup paths
    input_path = Path('/workspace/ivan_images_slice')
    output_base_path = Path('/workspace/output_masks_param_search')
    
    if not input_path.exists():
        logger.error(f"Input path {input_path} does not exist.")
        return

    # Get first 2 images
    all_images = sorted([x for x in input_path.iterdir() if x.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']])
    target_images = all_images[:2]
    
    if not target_images:
        logger.error("No images found in input directory.")
        return
        
    logger.info(f"Selected images for testing: {[img.name for img in target_images]}")

    # 2. Initialize Model (Once)
    # We initialize model later ONLY if we have work to do, to save time? 
    # But checking experiments is fast. Let's keep logic simple.
    
    # Check if we have any work to do before loading model
    experiments = get_experiment_configs()
    experiments_to_run = []
    
    for exp_name, config in experiments.items():
        exp_output_path = output_base_path / exp_name
        
        # Check if all target images already exist
        all_done = True
        if not exp_output_path.exists():
            all_done = False
        else:
            for img in target_images:
                if not (exp_output_path / img.name).exists():
                    all_done = False
                    break
        
        if all_done:
            logger.info(f"Skipping experiment {exp_name}: already completed.")
            continue
            
        experiments_to_run.append((exp_name, config))
    
    if not experiments_to_run:
        logger.info("All experiments are already completed!")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        # Enable TF32/BF16 optimizations
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    logger.info("Building SAM3 model...")
    sam3_model = build_sam3_image_model(
        device=str(device),
        eval_mode=True,
        enable_inst_interactivity=True
    )

    # 3. Run Experiments
    for exp_name, config in experiments_to_run:
        logger.info(f"\n{'='*20}\nRunning Experiment: {exp_name}\n{'='*20}")
        
        # Prepare output dir
        exp_output_path = output_base_path / exp_name
        exp_output_path.mkdir(parents=True, exist_ok=True)
        
        # Save config
        with open(exp_output_path / "config.json", "w") as f:
            json.dump(config, f, indent=4)
        logger.info(f"Saved configuration to {exp_output_path / 'config.json'}")
            
        # Initialize Generator with current config
        mask_generator = SAM3AutomaticMaskGenerator(
            image_model=sam3_model,
            device=str(device),
            **config
        )
        
        # Process images
        for image_path in target_images:
            if (exp_output_path / image_path.name).exists():
                logger.info(f"Skipping {image_path.name} (exists)")
                continue

            logger.info(f"Processing {image_path.name}...")
            try:
                image = Image.open(image_path)
                image_np = np.array(image.convert("RGB"))
                
                masks = mask_generator.generate(image_np)
                
                # Save visualization
                save_anns(masks, exp_output_path / image_path.name)
                
                logger.info(f"Generated {len(masks)} masks for {image_path.name}")
                
            except Exception as e:
                logger.error(f"Failed to process {image_path.name} in {exp_name}: {e}")

    logger.info(f"\nAll experiments completed. Results in {output_base_path}")

if __name__ == "__main__":
    main()
