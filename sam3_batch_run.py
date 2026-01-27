import os
import torch
import numpy as np
import cv2
import glob
import gc
import sys
import logging
from tqdm import tqdm
from PIL import Image

# Append workspace root to path if needed
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from sam3.model_builder import build_sam3_video_predictor
from sam3.visualization_utils import load_frame
from sam3.logger import get_logger

# Configure logging
logger = get_logger(__name__, level=logging.INFO)

try:
    from tg_notifier import notify_error
except ImportError:
    logger.warning("Notifier not found, using dummy function")
    def notify_error(e, msg=""): print(f"Notifier not found: {msg} {e}")

class SAM3BatchProcessor:
    def __init__(self, video_path, output_dir, device='cuda'):
        self.video_path = video_path
        self.output_dir = output_dir
        self.cache_dir = os.path.join(output_dir, "cache")
        self.vis_dir = os.path.join(output_dir, "visualization")
        self.masks_dir = os.path.join(output_dir, "masks")
        
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.vis_dir, exist_ok=True)
        os.makedirs(self.masks_dir, exist_ok=True)
        
        self.device = device
        self.predictor = None
        self.session_id = None
        self.video_frames = self._load_video_frames()
        
    def _load_video_frames(self):
        """Loads video frames paths or extracts them from video"""
        logger.info(f"Loading video from {self.video_path}")
        if os.path.isdir(self.video_path):
            frames = sorted(glob.glob(os.path.join(self.video_path, "*.png")) + 
                          glob.glob(os.path.join(self.video_path, "*.jpg")))
            # Try numeric sort first
            try:
                frames.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
            except:
                logger.warning("Could not sort frames numerically, falling back to lexicographic sort")
                frames.sort()
            logger.info(f"Found {len(frames)} frames")
            return frames
        else:
            raise NotImplementedError("Direct video file support not fully implemented, please provide directory with frames")

    def initialize_predictor(self):
        if self.predictor is None:
            logger.info("Initializing SAM3 predictor...")
            # Revert to build_sam3_video_predictor which returns Sam3VideoPredictorMultiGPU
            # This class has handle_request and handle_stream_request methods
            self.predictor = build_sam3_video_predictor()
            logger.info("SAM3 predictor initialized")

    def start_session(self, resource=None):
        self.initialize_predictor()
        logger.info("Starting inference session...")
        try:
            # Use handle_request for start_session
            video_path = resource if resource is not None else self.video_path
            response = self.predictor.handle_request(
                request=dict(
                    type="start_session",
                    resource_path=video_path,
                )
            )
            self.session_id = response["session_id"]
            logger.info(f"Session started with ID: {self.session_id}")
            return self.session_id
        except Exception as e:
            logger.error(f"Failed to start session: {e}")
            raise

    def reset_session(self):
        if self.session_id and self.predictor:
            logger.info(f"Closing session {self.session_id} to free memory...")
            try:
                self.predictor.handle_request(
                    request=dict(
                        type="close_session",
                        session_id=self.session_id,
                    )
                )
            except Exception as e:
                logger.warning(f"Error closing session: {e}")
            
            self.session_id = None
            # Force garbage collection
            gc.collect()
            torch.cuda.empty_cache()

    def _create_random_chunks(self):
        """
        Splits the total list of video frames into segments of random length
        (e.g., 350 to 450 frames) to avoid OOM.
        """

        slice_size = [370, 450] # [min, max]
        # slice_size = [350, 450] # [min, max]
        # slice_size = [300, 350] # [min, max]
        # slice_size = [220, 350] # [min, max]
        # slice_size = [20, 25] # [min, max]
        total_frames = len(self.video_frames)
        chunks = []
        current_idx = 0
        
        while current_idx < total_frames:
            remaining = total_frames - current_idx
            if remaining <= slice_size[1]:
                chunk_size = remaining
            else:
                chunk_size = np.random.randint(slice_size[0], slice_size[1] + 1)
            
            end_idx = current_idx + chunk_size
            chunks.append((current_idx, end_idx))
            current_idx = end_idx
            
        logger.info(f"Created {len(chunks)} chunks with sizes: {[(e-s) for s,e in chunks]}")
        return chunks
    
    def _create_sequential_chunks(self):
        """
        Splits the total list of video frames into segments of equal length
        (e.g., 600 frames) to avoid OOM.
        """
        slice_size = 605 # frames
        total_frames = len(self.video_frames)
        chunks = []
        current_idx = 0
        while current_idx < total_frames:
            remaining = total_frames - current_idx
            if remaining <= slice_size:
                chunk_size = remaining
            else:
                chunk_size = slice_size
            end_idx = current_idx + chunk_size
            chunks.append((current_idx, end_idx))
            current_idx = end_idx
        logger.info(f"Created {len(chunks)} chunks with sizes: {[(e-s) for s,e in chunks]}")
        return chunks

    def process_prompt(self, prompt_text, force_recompute=False):
        """Process a single text prompt and save results using chunking"""
        safe_prompt = prompt_text.replace(" ", "_")
        cache_file = os.path.join(self.cache_dir, f"{safe_prompt}.npz")
        
        if os.path.exists(cache_file) and not force_recompute:
            logger.info(f"Loading cached results for '{prompt_text}'...")
            return np.load(cache_file, allow_pickle=True)['outputs'].item()

        # Check if predictor/session is initialized before processing new prompts
        if self.predictor is None:
            self.initialize_predictor()

        logger.info(f"Processing prompt: '{prompt_text}' with chunking...")
        
        # chunks = self._create_random_chunks()
        chunks = self._create_sequential_chunks()
        all_outputs = {}
        
        try:
            for chunk_idx, (start_idx, end_idx) in enumerate(chunks):
                retry_count = 0
                max_retries = 1
                success = False
                
                while not success:
                    try:
                        logger.info(f"Processing chunk {chunk_idx+1}/{len(chunks)}: frames {start_idx} to {end_idx} (Attempt {retry_count + 1})")
                        
                        # Load frames for this chunk
                        chunk_paths = self.video_frames[start_idx:end_idx]
                        chunk_images = [Image.open(p) for p in chunk_paths]
                        
                        # Reset session to clear previous state
                        self.reset_session()
                        
                        # Start session with this chunk
                        self.start_session(resource=chunk_images)
                        
                        # Add prompt
                        # Note: We apply the text prompt to the first frame of the chunk (relative index 0)
                        self.predictor.handle_request(
                            request=dict(
                                type="add_prompt",
                                session_id=self.session_id,
                                frame_index=0,
                                text=prompt_text,
                            )
                        )
                        
                        # Propagate
                        for response in self.predictor.handle_stream_request(
                            request=dict(
                                type="propagate_in_video",
                                session_id=self.session_id,
                                propagation_direction="both",
                            )
                        ):
                            local_idx = response["frame_index"]
                            global_idx = start_idx + local_idx
                            
                            # Add chunk offset to object IDs to ensure uniqueness across chunks
                            # This enables correct stitching later using IoU if needed, 
                            # though for text prompts usually ID=0 is the main one.
                            chunk_id_offset = chunk_idx * 1000
                            outputs = response["outputs"]
                            if "out_obj_ids" in outputs:
                                outputs["out_obj_ids"] = [oid + chunk_id_offset for oid in outputs["out_obj_ids"]]
                            
                            all_outputs[global_idx] = outputs
                        
                        # Clean up chunk resources
                        del chunk_images
                        self.reset_session()
                        success = True
                        
                    except torch.cuda.OutOfMemoryError:
                        retry_count += 1
                        logger.warning(f"OOM Error processing chunk {chunk_idx+1}. Attempt {retry_count}/{max_retries}")
                        
                        # Aggressive cleanup: destroy predictor and force full re-initialization
                        self.reset_session()
                        
                        if 'chunk_images' in locals():
                            del chunk_images
                            
                        # # Destroy the predictor to free any internal caches
                        # self.predictor = None
                        gc.collect()
                        torch.cuda.empty_cache()
                        
                        # # Re-initialize predictor for next attempt
                        # self.initialize_predictor()
                        
                        if retry_count >= max_retries:
                            logger.error(f"Failed to process chunk {chunk_idx+1} after {max_retries} attempts due to OOM.")
                            raise

            # Save to cache
            np.savez_compressed(cache_file, outputs=all_outputs)
            
            return all_outputs
            
        except Exception as e:
            if not isinstance(e, torch.cuda.OutOfMemoryError): # OOM is already handled/raised above
                logger.error(f"Error processing prompt '{prompt_text}': {e}")
                notify_error(e, f"Ошибка в SAM3BatchProcessor.process_prompt для промпта: {prompt_text}")
            raise

    def merge_results(self, all_outputs):
        """Merge outputs from multiple prompts into a single per-frame structure with consistent IDs"""
        merged_frames = {}
        
        # Get all frame indices
        all_frames = set()
        for out in all_outputs.values():
            all_frames.update(out.keys())
            
        logger.info("Merging results...")
        
        # Global registry to keep IDs consistent across frames
        # Key: (prompt_text, original_obj_id), Value: global_unique_id
        id_registry = {}
        next_global_id = 0
        
        for frame_idx in sorted(list(all_frames)):
            frame_merged = {
                "out_obj_ids": [],
                "out_probs": [],
                "out_boxes_xywh": [],
                "out_binary_masks": [],
                "prompt_source": [] # To track which prompt generated this object
            }
            
            for prompt, prompt_output in all_outputs.items():
                if frame_idx in prompt_output:
                    p_out = prompt_output[frame_idx]
                    
                    # Skip if empty
                    if len(p_out["out_obj_ids"]) == 0:
                        continue
                        
                    ids = p_out["out_obj_ids"]
                    masks = p_out["out_binary_masks"]
                    probs = p_out["out_probs"]
                    boxes = p_out["out_boxes_xywh"]
                    
                    for i in range(len(ids)):
                        original_id = ids[i]
                        
                        # Generate or retrieve unique persistent ID
                        # We assume SAM tracks ID consistently within a single prompt session
                        obj_key = (prompt, original_id)
                        
                        if obj_key not in id_registry:
                            id_registry[obj_key] = next_global_id
                            next_global_id += 1
                        
                        persistent_id = id_registry[obj_key]
                        
                        frame_merged["out_obj_ids"].append(persistent_id)
                        frame_merged["out_binary_masks"].append(masks[i])
                        frame_merged["out_probs"].append(probs[i])
                        frame_merged["out_boxes_xywh"].append(boxes[i])
                        frame_merged["prompt_source"].append(prompt)

            # Convert lists to numpy arrays
            if frame_merged["out_obj_ids"]:
                frame_merged["out_obj_ids"] = np.array(frame_merged["out_obj_ids"])
                frame_merged["out_probs"] = np.array(frame_merged["out_probs"])
                frame_merged["out_binary_masks"] = np.array(frame_merged["out_binary_masks"])
                frame_merged["out_boxes_xywh"] = np.array(frame_merged["out_boxes_xywh"])
            else:
                frame_merged["out_obj_ids"] = np.array([])
                frame_merged["out_probs"] = np.array([])
                frame_merged["out_boxes_xywh"] = np.array([])
                frame_merged["out_binary_masks"] = np.zeros((0, 0, 0)) # Empty
                
            merged_frames[frame_idx] = frame_merged
            
        return merged_frames

    def visualize_merged(self, merged_frames, output_video_name="merged_output.mp4", fps=10, 
                        show_box=True, show_label=True, show_mask=True, 
                        show_original_image=True, save_frames=True):
        """Create video with merged masks and optionally save frames"""
        logger.info("Generating visualization video...")
        
        first_frame = load_frame(self.video_frames[0])
        h, w = first_frame.shape[:2]
        
        video_out_path = os.path.join(self.vis_dir, output_video_name)
        
        # Directory for saved frames
        frames_out_dir = os.path.join(self.vis_dir, os.path.splitext(output_video_name)[0] + "_frames")
        if save_frames:
            os.makedirs(frames_out_dir, exist_ok=True)
        
        # Temp video file
        temp_path = os.path.join(self.vis_dir, "temp_render.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(temp_path, fourcc, fps, (w, h))
        
        # from sam3.visualization_utils import COLORS
        
        # Generate bright distinct colors ensuring visibility on black background
        def generate_bright_colors(n=256):
            import colorsys
            colors = []
            # Use golden ratio to distribute hues evenly
            golden_ratio_conjugate = 0.618033988749895
            h = np.random.random()
            for i in range(n):
                h += golden_ratio_conjugate
                h %= 1
                # High saturation and value/brightness for visibility on black
                # Saturation 0.5-0.9, Value 0.7-1.0
                s = 0.6 + np.random.random() * 0.3
                v = 0.8 + np.random.random() * 0.2
                rgb = colorsys.hsv_to_rgb(h, s, v)
                colors.append(rgb)
            return np.array(colors)
            
        COLORS = generate_bright_colors(max(100, len(self.video_frames))) # Ensure enough colors
        
        for frame_idx in tqdm(sorted(merged_frames.keys())):
            if frame_idx >= len(self.video_frames):
                break
                
            img = load_frame(self.video_frames[frame_idx])
            
            # Ensure img is uint8
            if img.dtype != np.uint8:
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)
            
            # Start with original image or black background
            if show_original_image:
                overlay = img.copy()
            else:
                overlay = np.zeros_like(img)
                
            outputs = merged_frames[frame_idx]
            
            # Custom rendering loop to control what to show
            if len(outputs["out_obj_ids"]) > 0:
                # Iterate over objects in this frame
                for i in range(len(outputs["out_obj_ids"])):
                    obj_id = int(outputs["out_obj_ids"][i])
                    # Use prompt_source to color or obj_id
                    # Here we stick to simple coloring by obj_id
                    color = COLORS[obj_id % len(COLORS)]
                    color255 = tuple(int(x * 255) for x in color)
                    
                    if show_mask:
                        mask = outputs["out_binary_masks"][i]
                        # Resize if needed (though usually matches frame)
                        if mask.shape != img.shape[:2]:
                            mask = cv2.resize(mask.astype(np.float32), 
                                           (img.shape[1], img.shape[0]), 
                                           interpolation=cv2.INTER_NEAREST)
                        
                        # Handle both bool and float masks
                        if mask.dtype == bool:
                            mask_bool = mask
                        else:
                            mask_bool = mask > 0.5
                        # If showing on black background (no original image), use full opacity for mask
                        alpha = 0.5 if show_original_image else 1.0
                        
                        for c in range(3):
                            overlay[..., c][mask_bool] = (
                                alpha * color255[c] + (1 - alpha) * overlay[..., c][mask_bool]
                            ).astype(np.uint8)
                            
                    if show_box:
                        box_xywh = outputs["out_boxes_xywh"][i]
                        x, y, w_box, h_box = box_xywh
                        x1 = int(x * w)
                        y1 = int(y * h)
                        x2 = int((x + w_box) * w)
                        y2 = int((y + h_box) * h)
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), color255, 2)
                        
                    if show_label:
                        # Find box for label position if box is not shown
                        if not show_box:
                            box_xywh = outputs["out_boxes_xywh"][i]
                            x, y, w_box, h_box = box_xywh
                            x1 = int(x * w)
                            y1 = int(y * h)
                        else:
                            # x1, y1 already defined
                            pass
                            
                        # prob = outputs["out_probs"][i]
                        prompt = outputs["prompt_source"][i] # Optional: show prompt name
                        label_text = f"id={obj_id}"
                        # label_text = f"{obj_id}, {prompt}, {prob:.2f}"
                        
                        cv2.putText(overlay, label_text, (x1, max(y1 - 10, 0)),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color255, 1, cv2.LINE_AA)

            bgr_frame = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            writer.write(bgr_frame)
            
            if save_frames:
                # Use original filename if possible, otherwise fall back to index
                original_path = self.video_frames[frame_idx]
                original_name = os.path.basename(original_path)
                frame_filename = os.path.join(frames_out_dir, original_name)
                # If extension is not png/jpg in original, we force it to match what we write or keep it?
                # cv2.imwrite determines format by extension.
                # Let's ensure we save as png or jpg correctly.
                # If user wants exact original name, we trust cv2 handles the extension provided in original_name
                # unless it's not an image extension supported by imwrite.
                cv2.imwrite(frame_filename, bgr_frame)
            
        writer.release()
        
        # Convert with ffmpeg for compatibility
        if os.path.exists(video_out_path):
            os.remove(video_out_path)
            
        os.system(f"ffmpeg -y -i {temp_path} -vcodec libx264 {video_out_path}")
        os.remove(temp_path)
        logger.info(f"Video saved to {video_out_path}")

    def run(self, prompts):
        self.start_session()
        
        all_outputs = {}
        for prompt in prompts:
            all_outputs[prompt] = self.process_prompt(prompt)
            
        merged = self.merge_results(all_outputs)
        self.visualize_merged(merged)
        
        logger.info("Done!")

if __name__ == "__main__":
    VIDEO_PATH = "/workspace/ivan_images_slice" 
    OUTPUT_DIR = "/workspace/sam3_batch_results"
    PROMPTS = [
        "chair", 
        "table", 
        "keyboard", 
        "touchpad",
        "mouse",
        "usb hub",
        "monitor",
        "imac",
        "wires",
        "cushion",
        "sofa",
        "bin",
        "screwdriver",
        "window",
        "window blind",
        "door",
        "door handle",
        "floor",
        "wall",
        "ceiling",
        "pipe",
        "socket",
        "plug",
        "switch",
        "column",
        "concrete",
        ]
    
    processor = SAM3BatchProcessor(VIDEO_PATH, OUTPUT_DIR)
    
    all_outputs = {}
    for prompt in PROMPTS:
        all_outputs[prompt] = processor.process_prompt(prompt)
        
    merged = processor.merge_results(all_outputs)
    
    processor.visualize_merged(
        merged, 
        output_video_name="merged_output_no_image.mp4",
        show_box=False,
        show_label=False,
        show_mask=True,
        show_original_image=False,
        save_frames=True
    )
    
    # processor.visualize_merged(merged, output_video_name="merged_full.mp4", show_box=True, show_label=True)
    
    logger.info("Done!")

