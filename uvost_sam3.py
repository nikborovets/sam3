"""
UVOST SAM3: Unsupervised Video Object Segmentation & Tracking using SAM 3

Grid Prompting approach (inspired by SAM2 AutomaticMaskGenerator):
1. Generate grid of points on keyframes
2. Run Sam3Processor to get masks for each point  
3. Filter via NMS
4. Track objects using Sam3TrackerPredictor (SAM2-style API)
"""

import os
import sys
import cv2
import shutil
import tempfile
import numpy as np
import torch
from typing import List, Dict, Any, Tuple
from PIL import Image
from tqdm import tqdm

# sys.path.insert(0, os.path.join(os.path.dirname(__file__), "sam3"))

try:
    from sam3.model_builder import build_sam3_video_model
    from sam3.model.sam3_image_processor import Sam3Processor
    from torchvision.ops import nms
except ImportError as e:
    print(f"Error importing SAM3 modules: {e}")
    sys.exit(1)


class SAM3AutomaticMaskGenerator:
    """
    SAM3 version of AutomaticMaskGenerator.
    Generates masks for entire image using grid prompts.
    """
    
    def __init__(
        self,
        image_model,
        points_per_side: int = 32,
        points_per_batch: int = 128,
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
    
    @torch.no_grad()
    def generate(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Generate masks using grid prompts."""
        h, w = image.shape[:2]
        pil_image = Image.fromarray(image)
        inference_state = self.processor.set_image(pil_image)
        
        all_masks = []
        all_scores = []
        all_points = []
        
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
            except:
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


class UVOST_SAM3:
    """
    Unsupervised Video Object Segmentation & Tracking using SAM 3.
    Uses Sam3TrackerPredictor (SAM2-style API) for tracking.
    """
    
    def __init__(
        self,
        grid_interval: int = 30,
        points_per_side: int = 16,
        iou_threshold_nms: float = 0.4,
        iou_threshold_new_obj: float = 0.3,
        pred_iou_thresh: float = 0.5,
        min_mask_area: int = 500,
        device: str = "cuda",
        output_dir: str = "/workspace/output_uvost"
    ):
        self.grid_interval = grid_interval
        self.points_per_side = points_per_side
        self.iou_threshold_nms = iou_threshold_nms
        self.iou_threshold_new_obj = iou_threshold_new_obj
        self.pred_iou_thresh = pred_iou_thresh
        self.min_mask_area = min_mask_area
        self.device = device
        self.output_dir = output_dir
        
        os.makedirs(self.output_dir, exist_ok=True)

        print(f"[UVOST] Initializing SAM 3 Model...")
        self.sam3_model = build_sam3_video_model()
        
        # Get tracker (SAM2-style API)
        self.tracker = self.sam3_model.tracker
        self.tracker.backbone = self.sam3_model.detector.backbone
        
        print("[UVOST] Initializing Automatic Mask Generator...")
        self.mask_generator = SAM3AutomaticMaskGenerator(
            image_model=self.sam3_model.detector,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            box_nms_thresh=iou_threshold_nms,
            min_mask_region_area=min_mask_area,
            device=device
        )
        
        self.results_per_frame: Dict[int, Dict[int, np.ndarray]] = {}
        self.next_obj_id = 1
        self.temp_dir = None

    def _compute_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Compute IoU between two binary masks."""
        m1 = mask1 > 0 if mask1.dtype != np.bool_ else mask1
        m2 = mask2 > 0 if mask2.dtype != np.bool_ else mask2
        intersection = np.logical_and(m1, m2).sum()
        union = np.logical_or(m1, m2).sum()
        return float(intersection) / float(union + 1e-8)

    def process_video(self, video_path: str, max_frames: int = None):
        """Main entry point."""
        # Load frames
        frames = self._load_frames(video_path)
        if not frames:
            print("[UVOST] No frames loaded!")
            return
        
        if max_frames:
            frames = frames[:max_frames]
            print(f"[UVOST] Limited to {len(frames)} frames")
        
        h, w = frames[0].shape[:2]
        total_frames = len(frames)
        print(f"[UVOST] Processing {total_frames} frames ({w}x{h})")
        
        # Prepare video path for tracker
        if os.path.isfile(video_path) and video_path.endswith('.mp4'):
            tracker_video_path = video_path
        else:
            # Save frames to temp directory for tracker
            self.temp_dir = tempfile.mkdtemp()
            print(f"[UVOST] Saving frames to temp dir...")
            for i, frame in enumerate(frames):
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(self.temp_dir, f"{i:05d}.jpg"), frame_bgr)
            tracker_video_path = self.temp_dir
        
        # Initialize tracker state
        print("[UVOST] Initializing tracker state...")
        inference_state = self.tracker.init_state(video_path=tracker_video_path)
        
        # Step 1: Grid Prompting on frame 0
        print(f"\n[UVOST] === Step 1: Grid Prompting ===")
        proposals = self.mask_generator.generate(frames[0])
        print(f"[UVOST] Found {len(proposals)} objects")
        
        if not proposals:
            print("[UVOST] No objects found!")
            return
        
        # Step 2: Add objects to tracker using boxes
        print(f"\n[UVOST] === Step 2: Add Objects to Tracker ===")
        
        # Sort by area (largest first)
        proposals = sorted(proposals, key=lambda x: x['area'], reverse=True)
        
        # Add each detected object
        for prop in proposals[:20]:  # Limit to top 20 objects
            bbox_xyxy = prop['bbox_xyxy']
            
            # Normalize to [0,1]
            bbox_norm = np.array([[
                bbox_xyxy[0] / w,
                bbox_xyxy[1] / h,
                bbox_xyxy[2] / w,
                bbox_xyxy[3] / h
            ]], dtype=np.float32)
            
            obj_id = self.next_obj_id
            self.next_obj_id += 1
            
            try:
                _, out_obj_ids, low_res_masks, video_res_masks = self.tracker.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=0,
                    obj_id=obj_id,
                    box=bbox_norm,
                )
                
                if len(out_obj_ids) > 0:
                    print(f"[UVOST]   Added object {obj_id}")
                    
                    # Store initial mask
                    if 0 not in self.results_per_frame:
                        self.results_per_frame[0] = {}
                    
                    for i, oid in enumerate(out_obj_ids):
                        if i < len(video_res_masks):
                            mask = (video_res_masks[i] > 0.0).cpu().numpy()
                            if mask.ndim == 4:
                                mask = mask[0, 0]
                            elif mask.ndim == 3:
                                mask = mask[0]
                            self.results_per_frame[0][int(oid)] = mask
                            
            except Exception as e:
                print(f"[UVOST]   Failed to add object {obj_id}: {e}")
        
        print(f"[UVOST] Added {self.next_obj_id - 1} objects")
        
        # Step 3: Propagate with periodic re-detection
        print(f"\n[UVOST] === Step 3: Propagate with Re-detection ===")
        
        # Get keyframes for re-detection
        keyframes = set(range(0, total_frames, self.grid_interval))
        
        count = 0
        for frame_idx, obj_ids, low_res_masks, video_res_masks, obj_scores in self.tracker.propagate_in_video(
            inference_state, 
            start_frame_idx=0, 
            max_frame_num_to_track=total_frames,
            reverse=False,
            propagate_preflight=True
        ):
            if count < 5 or count % 50 == 0:
                print(f"[UVOST] Frame {frame_idx}: {len(obj_ids)} objects")
            
            # Store current masks
            current_masks = {}
            if len(obj_ids) > 0:
                self.results_per_frame[frame_idx] = {}
                for i, oid in enumerate(obj_ids):
                    if i < len(video_res_masks):
                        mask = (video_res_masks[i] > 0.0).cpu().numpy()
                        if mask.ndim == 4:
                            mask = mask[0, 0]
                        elif mask.ndim == 3:
                            mask = mask[0]
                        self.results_per_frame[frame_idx][int(oid)] = mask
                        current_masks[int(oid)] = mask
            
            # Periodic re-detection for new objects
            if frame_idx in keyframes and frame_idx > 0:
                print(f"[UVOST] Re-detecting on keyframe {frame_idx}...")
                new_proposals = self.mask_generator.generate(frames[frame_idx])
                
                # Find NEW objects (low IoU with existing)
                new_objects = []
                for prop in new_proposals:
                    is_new = True
                    prop_mask = prop['segmentation']
                    
                    for existing_mask in current_masks.values():
                        iou = self._compute_iou(prop_mask, existing_mask)
                        if iou > self.iou_threshold_new_obj:
                            is_new = False
                            break
                    
                    if is_new:
                        new_objects.append(prop)
                
                if new_objects:
                    print(f"[UVOST]   Found {len(new_objects)} NEW objects!")
                    # Add new objects (limit to 5 per keyframe)
                    for prop in new_objects[:5]:
                        bbox_xyxy = prop['bbox_xyxy']
                        bbox_norm = np.array([[
                            bbox_xyxy[0] / w,
                            bbox_xyxy[1] / h,
                            bbox_xyxy[2] / w,
                            bbox_xyxy[3] / h
                        ]], dtype=np.float32)
                        
                        obj_id = self.next_obj_id
                        self.next_obj_id += 1
                        
                        try:
                            self.tracker.add_new_points_or_box(
                                inference_state=inference_state,
                                frame_idx=frame_idx,
                                obj_id=obj_id,
                                box=bbox_norm,
                            )
                            print(f"[UVOST]     Added new object {obj_id}")
                        except Exception as e:
                            pass
            
            count += 1
        
        print(f"[UVOST] Propagation complete: {count} frames")
        print(f"[UVOST] Total objects tracked: {self.next_obj_id - 1}")
        print(f"[UVOST] Frames with masks: {len(self.results_per_frame)}")
        
        # Save results
        print(f"\n[UVOST] === Saving Results ===")
        self._save_visualization(frames)
        
        # Cleanup temp dir
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print("[UVOST] Cleanup complete")

    def _load_frames(self, video_path: str) -> List[np.ndarray]:
        """Load video frames."""
        frames = []
        
        if os.path.isdir(video_path):
            print(f"[UVOST] Reading frames from: {video_path}")
            exts = {".jpg", ".jpeg", ".png", ".bmp"}
            files = [f for f in os.listdir(video_path) if os.path.splitext(f)[1].lower() in exts]
            try:
                files.sort(key=lambda x: int(os.path.splitext(x)[0]))
            except ValueError:
                files.sort()
            
            for f in files:
                img = cv2.imread(os.path.join(video_path, f))
                if img is not None:
                    frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            print(f"[UVOST] Reading video: {video_path}")
            cap = cv2.VideoCapture(video_path)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
        
        return frames

    def _save_visualization(self, frames: List[np.ndarray]):
        """Save output video."""
        if not self.results_per_frame:
            print("[UVOST] No results to save!")
            return
        
        h, w = frames[0].shape[:2]
        out_path = os.path.join(self.output_dir, "uvost_result.mp4")
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (w, h))
        
        np.random.seed(42)
        colors = np.random.randint(50, 255, (1000, 3), dtype=np.uint8)
        
        frames_with_masks = 0
        for f_idx, frame in enumerate(frames):
            vis = cv2.cvtColor(frame.copy(), cv2.COLOR_RGB2BGR)
            
            if f_idx in self.results_per_frame:
                masks_dict = self.results_per_frame[f_idx]
                if masks_dict:
                    frames_with_masks += 1
                
                for obj_id, mask in masks_dict.items():
                    if mask is None:
                        continue
                    mask = mask.squeeze() if mask.ndim > 2 else mask
                    mask_bool = mask > 0 if mask.dtype != np.bool_ else mask
                    if not mask_bool.any():
                        continue
                    
                    color = colors[obj_id % 1000].tolist()
                    overlay = np.zeros_like(vis)
                    overlay[mask_bool] = color
                    vis = cv2.addWeighted(vis, 1.0, overlay, 0.5, 0)
                    
                    contours, _ = cv2.findContours(
                        mask_bool.astype(np.uint8) * 255,
                        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    cv2.drawContours(vis, contours, -1, color, 2)
                    
                    y_idx, x_idx = np.where(mask_bool)
                    if len(y_idx) > 0:
                        cx, cy = int(x_idx.mean()), int(y_idx.mean())
                        cv2.putText(vis, str(obj_id), (cx-10, cy+5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            
            cv2.putText(vis, f"Frame: {f_idx}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
            writer.write(vis)
        
        writer.release()
        print(f"[UVOST] Frames with masks: {frames_with_masks}/{len(frames)}")
        print(f"[UVOST] Video saved: {out_path}")
        
        stats_path = os.path.join(self.output_dir, "stats.txt")
        with open(stats_path, 'w') as f:
            f.write(f"Total frames: {len(frames)}\n")
            f.write(f"Frames with masks: {frames_with_masks}\n")
            f.write(f"Objects tracked: {self.next_obj_id - 1}\n")
            f.write(f"Grid size: {self.points_per_side}x{self.points_per_side}\n")
        print(f"[UVOST] Stats saved: {stats_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="UVOST with SAM3")
    parser.add_argument("video_path", help="Path to video or frame directory")
    parser.add_argument("--grid-interval", type=int, default=30)
    parser.add_argument("--points-per-side", type=int, default=16)
    parser.add_argument("--iou-threshold", type=float, default=0.2)
    parser.add_argument("--min-mask-area", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default="/workspace/output_uvost")
    parser.add_argument("--max-frames", type=int, default=None)
    
    args = parser.parse_args()
    
    pipeline = UVOST_SAM3(
        grid_interval=args.grid_interval,
        points_per_side=args.points_per_side,
        iou_threshold_new_obj=args.iou_threshold,
        min_mask_area=args.min_mask_area,
        output_dir=args.output_dir
    )
    
    pipeline.process_video(args.video_path, max_frames=args.max_frames)


if __name__ == "__main__":
    main()
