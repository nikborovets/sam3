import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm
from sam3.model_builder import build_sam3_video_model

def parse_args():
    parser = argparse.ArgumentParser(description="SAM3 Video Propagation CLI")
    parser.add_argument("--weights", type=str, help="Path to model weights")
    parser.add_argument("--inputs", type=str, help="Path to RGB images")
    parser.add_argument("--masks", type=str, help="Path to input masks")
    parser.add_argument("--labelmap", type=str, help="Path to labelmap.txt")
    parser.add_argument("--out-overlay", type=str, help="Output folder for overlays")
    parser.add_argument("--out-masks", type=str, help="Output folder for binary masks")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")
    return parser.parse_args()

def load_objects_from_labelmap(path):
    objects = []
    with open(path, 'r') as f:
        lines = f.readlines()
    for line in lines[2:]:
        parts = line.strip().split(':')
        if len(parts) < 2: continue
        r, g, b = map(int, parts[1].split(','))
        objects.append([r, g, b])
    return objects

@torch.inference_mode()
def main():
    args = parse_args()
    device = torch.device(args.device)

    if device.type == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    sam3_model = build_sam3_video_model(
        checkpoint_path=args.weights, 
        load_from_HF=False, 
        bpe_path='/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz'
    )
    predictor = sam3_model.tracker
    predictor.backbone = sam3_model.detector.backbone
    predictor.non_overlap_masks_for_output = True

    input_path = Path(args.inputs)
    frame_names = sorted(list(input_path.glob("*")))
    input_masks = sorted(list(Path(args.masks).glob("*")))
    
    Path(args.out_overlay).mkdir(parents=True, exist_ok=True)
    Path(args.out_masks).mkdir(parents=True, exist_ok=True)

    objects = load_objects_from_labelmap(args.labelmap)

    inference_state = predictor.init_state(
        video_path=str(input_path), 
        async_loading_frames=True, 
        offload_video_to_cpu=True, 
        offload_state_to_cpu=True
    )

    for mask_path in input_masks:
        mask_image = cv2.imread(str(mask_path))
        mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)
        frame_idx = int(mask_path.stem)

        for obj_id, color in enumerate(objects):
            mask_np = np.all(mask_image == color, axis=-1)
            if mask_np.any():
                mask_tensor = torch.from_numpy(mask_np).to(device)
                predictor.add_new_mask(inference_state, frame_idx, obj_id, mask_tensor)
        print(f'Annotated frame {frame_idx} added!')

    print("Propagating video...")
    video_segments = {}
    for out_frame_idx, out_obj_ids, _, out_mask_logits, _ in predictor.propagate_in_video(inference_state, start_frame_idx=0, max_frame_num_to_track=None, reverse=False, propagate_preflight=True):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    print("Saving results...")
    for out_frame_idx in tqdm(video_segments.keys()):
        frame = cv2.imread(str(frame_names[out_frame_idx]))
        res_mask_rgb = np.zeros_like(frame)
        overlay = frame.copy()

        for obj_id, out_mask in video_segments[out_frame_idx].items():
            color_rgb = np.array(objects[obj_id], dtype=np.uint8)
            color_bgr = color_rgb[::-1]
            mask_bool = out_mask.squeeze().astype(bool)

            empty_area = np.all(res_mask_rgb == 0, axis=-1)
            target_pixels = empty_area & mask_bool
            
            res_mask_rgb[target_pixels] = color_rgb
            overlay[mask_bool] = (overlay[mask_bool] * 0.4 + color_bgr * 0.6).astype(np.uint8)

        fname = frame_names[out_frame_idx].stem
        cv2.imwrite(f"{args.out_overlay}/{fname}.png", overlay)
        cv2.imwrite(f"{args.out_masks}/{fname}.png", cv2.cvtColor(res_mask_rgb, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    main()
