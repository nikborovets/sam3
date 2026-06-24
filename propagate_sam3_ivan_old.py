import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2

from sam3.model_builder import build_sam3_video_model
from pathlib import Path
from PIL import Image

from tqdm import tqdm

device = torch.device("cuda")

torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

sam3_model = build_sam3_video_model(checkpoint_path='/workspace/data_mount/model_weights/sam3/sam3.pt', load_from_HF=False, bpe_path='/workspace/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz')
predictor = sam3_model.tracker
predictor.backbone = sam3_model.detector.backbone
predictor.non_overlap_masks_for_output = True

input_path = Path('/workspace/30_01_seq2_images/')
frame_names = sorted(input_path.iterdir())

inference_state = predictor.init_state(video_path=str(input_path), async_loading_frames=True, offload_video_to_cpu=True, offload_state_to_cpu=True)

with open('/workspace/segm_mask_ivan_09-02-2026/labelmap.txt') as f:
    lines = f.readlines()

objects = []
for line in lines[2:]:
    _, rgb, _, _ = line.split(':')
    r, g, b = rgb.split(',')
    r, g, b = int(r), int(g), int(b)
    objects.append([r, g, b])

input_masks = sorted(Path('/workspace/segm_mask_ivan_09-02-2026/SegmentationClass/').iterdir())
for mask_path in input_masks:
    mask_image = cv2.imread(mask_path)
    mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)

    index = int(mask_path.stem)

    for i, object_color in enumerate(objects):
        mask = np.all(mask_image == object_color, axis=-1)
        if mask.sum() > 0:
            mask = torch.tensor(mask, dtype=torch.bool)
            predictor.add_new_mask(inference_state, index, i, mask)

    print(f'Annotated frame {index} added!')

video_segments = {}
for out_frame_idx, out_obj_ids, _, out_mask_logits, _ in predictor.propagate_in_video(inference_state, start_frame_idx=0, max_frame_num_to_track=None, reverse=False, propagate_preflight=True):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

for out_frame_idx in tqdm(list(video_segments.keys())):
    frame_path = frame_names[out_frame_idx]
    frame = cv2.imread(str(frame_path))
    res_mask_image = np.zeros_like(frame)
    overlay = frame.copy()

    for out_obj_id, out_mask in video_segments[out_frame_idx].items():
        color = np.array(objects[out_obj_id], dtype=np.uint8)
        mask_bool = out_mask.squeeze().astype(bool)
        empty_area = np.all(res_mask_image == 0, axis=-1)
        res_mask_image[empty_area & mask_bool] = color
        overlay[mask_bool] = (overlay[mask_bool] * 0.4 + color[::-1] * 0.6).astype(np.uint8)
    save_path = f"/workspace/sam3_ivan_seq2_results/overlay/{str(out_frame_idx).zfill(6)}.png"
    cv2.imwrite(save_path, overlay)

    if out_frame_idx % 25 == 0:
        save_mask_path = f"/workspace/sam3_ivan_seq2_results/masks_inter/{frame_names[out_frame_idx].stem}.png"
        cv2.imwrite(save_mask_path, cv2.cvtColor(res_mask_image, cv2.COLOR_RGB2BGR))

    save_mask_path = f"/workspace/sam3_ivan_seq2_results/masks/{frame_names[out_frame_idx].stem}.png"
    cv2.imwrite(save_mask_path, cv2.cvtColor(res_mask_image, cv2.COLOR_RGB2BGR))