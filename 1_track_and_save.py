"""
SAM3 / SAM3.1 video tracking — saves per-frame masks to NPZ files.

SAM3   : uses the low-level tracker API with pixel-level mask prompts.
SAM3.1 : uses the multiplex predictor with low-level mask prompts
         (via _tracker_add_new_objects / tracker.add_new_masks) so that
         all N objects are batched into fixed-capacity buckets, giving
         ~7x speedup at many objects and drastically lower GPU memory.

         SAM3.1 requires a separate checkpoint (sam3.1_multiplex.pt).
         By default we look in DEFAULT_WEIGHTS_31.
"""
import gc
import time
import torch
import numpy as np
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import torch.nn.functional as F
from sam3.model_builder import build_sam3_video_model

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
try:
    from tg_notifier import notify_error, send_message
except ImportError:
    def notify_error(e, msg=""): print(f"Notifier not found: {msg} {e}")
    def send_message(msg): print(msg)

# ─── DEFAULT PATHS ───────────────────────────────────────────────────────────
anno_name = "seq1-job_53_24-03-26"
DEFAULT_WEIGHTS    = "/workspace/data_mount/model_weights/sam3/sam3.pt"
DEFAULT_WEIGHTS_31 = "/workspace/data_mount/model_weights/sam3.1/sam3.1_multiplex.pt"
DEFAULT_INPUTS     = "/workspace/data_mount/third_wave_tracker_rgb_input/E-3023-A5-input-rgb/seq1"
DEFAULT_MASKS      = f"/workspace/3023-cvat-annotations/2stage/{anno_name}/SegmentationClass/seq1"
DEFAULT_LABELMAP   = f"/workspace/3023-cvat-annotations/2stage/{anno_name}/labelmap.txt"
DEFAULT_OUT_NPZ    = f"/workspace/3023-A5-SAM3-results/2stage/{anno_name}_results/masks_npz"
DEFAULT_DEVICE     = "cuda"
# ─────────────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(description="SAM3/SAM3.1 Video Tracking → NPZ")
    parser.add_argument("--version",   type=str, default="sam3",
                        choices=["sam3", "sam3.1"],
                        help="Model version: 'sam3' (mask API) or 'sam3.1' (multiplex)")
    parser.add_argument("--weights",   type=str, default=DEFAULT_WEIGHTS,
                        help="Path to sam3 checkpoint (used when --version sam3)")
    parser.add_argument("--weights-31", type=str, default=DEFAULT_WEIGHTS_31,
                        dest="weights_31",
                        help="Path to sam3.1 multiplex checkpoint")
    parser.add_argument("--inputs",    type=str, default=DEFAULT_INPUTS)
    parser.add_argument("--masks",     type=str, default=DEFAULT_MASKS)
    parser.add_argument("--labelmap",  type=str, default=DEFAULT_LABELMAP)
    parser.add_argument("--out-npz",   type=str, default=DEFAULT_OUT_NPZ)
    parser.add_argument("--device",    type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--bidirectional", action="store_true",
                        help="Run reverse propagation pass (sam3 only; sam3.1 forward-only)")
    return parser.parse_args()


def get_mem_report():
    import psutil
    process = psutil.Process(os.getpid())
    ram_gb = process.memory_info().rss / (1024 ** 3)
    report = f"RAM: {ram_gb:.2f} GB (Sys: {psutil.virtual_memory().percent}%)"
    if torch.cuda.is_available():
        vram_alloc = torch.cuda.memory_allocated() / (1024 ** 3)
        vram_res   = torch.cuda.memory_reserved()  / (1024 ** 3)
        report += f" | VRAM: {vram_alloc:.2f}/{vram_res:.2f} GB"
    return report


def _save_npz(path, mask_dict):
    np.savez_compressed(path, **mask_dict)


def load_objects_from_labelmap(path):
    objects = []
    with open(path, 'r') as f:
        lines = f.readlines()
    for line in lines[2:]:
        parts = line.strip().split(':')
        if len(parts) < 2:
            continue
        r, g, b = map(int, parts[1].split(','))
        objects.append([r, g, b])
    return objects


# ─────────────────────────────────────────────────────────────────────────────
#  SAM3 path  (unchanged from before)
# ─────────────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def run_sam3(args, device, frame_names, frame_names_stems, input_masks, objects, out_npz_path):
    sam3_model = build_sam3_video_model(
        checkpoint_path=args.weights,
        load_from_HF=False,
        bpe_path='/workspace/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz'
    )
    predictor = sam3_model.tracker
    predictor.backbone = sam3_model.detector.backbone
    predictor.non_overlap_masks_for_output = True

    inference_state = predictor.init_state(
        video_path=str(Path(args.inputs)),
        async_loading_frames=True,
        offload_video_to_cpu=True,
        offload_state_to_cpu=True,
    )

    for mask_path in input_masks:
        mask_image = cv2.imread(str(mask_path))
        mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)
        mask_path_stem = mask_path.stem
        try:
            frame_idx = frame_names_stems.index(mask_path_stem)
        except ValueError:
            int_val = int(mask_path_stem) if mask_path_stem.isdigit() \
                      else int(mask_path_stem.split('frame')[1])
            frame_idx = int_val

        added = 0
        for obj_id, color in enumerate(objects):
            mask_np = np.all(mask_image == color, axis=-1)
            if mask_np.any():
                mask_tensor = torch.from_numpy(mask_np).to(device)
                predictor.add_new_mask(inference_state, frame_idx, obj_id, mask_tensor)
                added += 1
        print(f"Annotated frame {frame_idx} added! ({added} objects)")

    print("Propagating video (forward) and saving…")
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for out_frame_idx, out_obj_ids, _, out_mask_logits, _ in predictor.propagate_in_video(
            inference_state, start_frame_idx=0, max_frame_num_to_track=None,
            reverse=False, propagate_preflight=True,
        ):
            frame_mask_dict = {
                str(out_obj_id): (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
            fname = frame_names[out_frame_idx].stem
            futures.append(executor.submit(_save_npz, out_npz_path / f"{fname}.npz", frame_mask_dict))
        for f in futures:
            f.result()

    if args.bidirectional:
        print("Propagating video (reverse) and saving…")
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for out_frame_idx, out_obj_ids, _, out_mask_logits, _ in predictor.propagate_in_video(
                inference_state, start_frame_idx=len(frame_names) - 1,
                max_frame_num_to_track=None, reverse=True, propagate_preflight=True,
            ):
                frame_mask_dict = {
                    str(out_obj_id): (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
                fname = frame_names[out_frame_idx].stem
                futures.append(executor.submit(_save_npz, out_npz_path / f"{fname}.npz", frame_mask_dict))
            for f in futures:
                f.result()

    # GPU memory cleanup (mirrors Sam3BasePredictor.close_session)
    if hasattr(inference_state, 'clear'):
        inference_state.clear()
    del inference_state
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
#  SAM3.1 path  (multiplex with low-level mask prompts)
# ─────────────────────────────────────────────────────────────────────────────

def _build_tracker_metadata_31(all_obj_ids, device):
    """Build the tracker_metadata dict required by Sam3MultiplexTracking.propagate_in_video."""
    n_objects = len(all_obj_ids)
    obj_ids_arr = np.array(all_obj_ids, dtype=np.int64)
    return {
        "obj_ids_per_gpu":              [obj_ids_arr],
        "obj_ids_all_gpu":              obj_ids_arr,
        "num_obj_per_gpu":              [n_objects],
        "obj_id_to_score":              {oid: 1.0 for oid in all_obj_ids},
        "obj_id_to_sam2_score_frame_wise": defaultdict(dict),
        "obj_id_to_last_occluded":      {},
        "max_obj_id":                   max(all_obj_ids) if all_obj_ids else 0,
        "rank0_metadata": {
            "masklet_confirmation": {
                "status":               np.zeros(n_objects, dtype=np.int64),
                "consecutive_det_num":  np.zeros(n_objects, dtype=np.int64),
            },
            "removed_obj_ids":   set(),
            "suppressed_obj_ids": defaultdict(set),
        },
        "gpu_metadata": {
            "N_obj":                    n_objects,
            "obj_first_frame":          torch.zeros(n_objects, dtype=torch.long,  device=device),
            "consecutive_unmatch_count":torch.zeros(n_objects, dtype=torch.long,  device=device),
            "trk_keep_alive":           torch.ones(n_objects,  dtype=torch.bool,  device=device),
            "removed_mask":             torch.zeros(n_objects, dtype=torch.bool,  device=device),
            "overlap_pair_counts":      torch.zeros((n_objects, n_objects), dtype=torch.long, device=device),
            "last_occluded_tensor":     torch.zeros(n_objects, dtype=torch.long,  device=device),
        },
    }


@torch.inference_mode()
def run_sam31(args, device, frame_names, frame_names_stems, input_masks, objects, out_npz_path):
    """
    SAM3.1 multiplex tracking with pixel-level mask prompts.

    Low-level path:
      1. start_session  →  init_state (loads video frames, builds input_batch)
      2. _prepare_backbone_feats  →  runs backbone+detector on annotation frame,
         populates feature_cache[frame_idx] with SAM2-compatible features
      3. _tracker_add_new_objects (first annotation frame)  →  creates SAM2 sub-states,
         registers obj_ids, runs propagate_in_video_preflight (memory encoder)
      4. tracker.add_new_masks (subsequent annotation frames)  →  adds conditioning
         memory for already-registered objects on new frames
      5. _init_backbone_out  →  runs text encoder, initialises backbone_out in state
         (required by _run_single_frame_inference before propagation)
      6. _build_tracker_metadata_31  →  fills tracker_metadata required by propagate_in_video
      7. handle_stream_request("propagate_in_video")  →  yields per-frame outputs
      8. close_session  →  gc.collect + torch.cuda.empty_cache
    """
    import uuid
    from sam3.model_builder import build_sam3_predictor

    ckpt = args.weights_31
    if not os.path.isfile(ckpt):
        print(f"[SAM3.1] checkpoint not found at {ckpt!r}, downloading from HuggingFace…")
        ckpt = None

    # use_rope_real=False matches the published sam3.1_multiplex.pt checkpoint format;
    # set True only when using a checkpoint trained with real-valued RoPE.
    predictor = build_sam3_predictor(
        version="sam3.1",
        checkpoint_path=ckpt,
        bpe_path='/workspace/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz',
        use_fa3=False,          # set True only on H100/H200 with FA3 installed
        use_rope_real=False,
        async_loading_frames=True,
    )
    demo_model = predictor.model   # Sam3MultiplexTrackingWithInteractivity

    input_path = Path(args.inputs)

    # ── 1. Init state directly (bypasses Sam3BasePredictor.start_session which
    #        always forwards offload_state_to_cpu, a kwarg Sam3MultiplexTracking
    #        does not accept) ────────────────────────────────────────────────
    inference_state = demo_model.init_state(
        resource_path=str(input_path),
        offload_video_to_cpu=True,
        async_loading_frames=True,
    )
    session_id = str(uuid.uuid4())
    predictor._all_inference_states[session_id] = {
        "state":         inference_state,
        "session_id":    session_id,
        "start_time":    time.time(),
        "last_use_time": time.time(),
    }

    try:
        # ── 2. Parse annotation frames ────────────────────────────────────
        # frame_to_masks : {frame_idx: {obj_id: mask_np [H, W] bool}}
        frame_to_masks = {}
        for mask_path in input_masks:
            mask_image = cv2.imread(str(mask_path))
            mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)
            stem = mask_path.stem
            try:
                frame_idx = frame_names_stems.index(stem)
            except ValueError:
                frame_idx = int(stem) if stem.isdigit() else int(stem.split('frame')[1])

            frame_masks = {}
            for obj_id, color in enumerate(objects):
                mask_np = np.all(mask_image == color, axis=-1)
                if mask_np.any():
                    frame_masks[obj_id] = mask_np
            if frame_masks:
                frame_to_masks[frame_idx] = frame_masks
                print(f"Annotated frame {frame_idx}: {len(frame_masks)} objects")

        all_obj_ids = sorted({oid for masks in frame_to_masks.values() for oid in masks})
        if not all_obj_ids:
            print("No annotated objects found – skipping SAM3.1 tracking.")
            return

        # ── 3. Backbone features + mask registration ──────────────────────
        # Each object is initialised from its FIRST annotated frame.
        # add_new_masks(add_to_existing_state=True) requires the frame to have
        # been propagated already, so we never call it on unvisited frames.
        # Instead we group new objects by their first annotated frame and call
        # _tracker_add_new_objects once per group.
        obj_first_frame: dict[int, int] = {}
        for frame_idx in sorted(frame_to_masks.keys()):
            for obj_id in frame_to_masks[frame_idx]:
                if obj_id not in obj_first_frame:
                    obj_first_frame[obj_id] = frame_idx

        frame_to_new_objs: dict[int, list[int]] = defaultdict(list)
        for obj_id, first_frame in obj_first_frame.items():
            frame_to_new_objs[first_frame].append(obj_id)

        for frame_idx in sorted(frame_to_new_objs.keys()):
            demo_model._prepare_backbone_feats(inference_state, frame_idx, reverse=False)
            obj_ids_here = sorted(frame_to_new_objs[frame_idx])
            frame_masks  = frame_to_masks[frame_idx]
            masks_float  = torch.stack([
                torch.from_numpy(frame_masks[oid].astype(np.float32))
                for oid in obj_ids_here
            ]).to(device)

            inference_state["sam2_inference_states"] = demo_model._tracker_add_new_objects(
                frame_idx=frame_idx,
                num_frames=inference_state["num_frames"],
                new_obj_ids=obj_ids_here,
                new_obj_masks=masks_float,
                tracker_states_local=inference_state["sam2_inference_states"],
                orig_vid_height=inference_state["orig_height"],
                orig_vid_width=inference_state["orig_width"],
                feature_cache=inference_state["feature_cache"],
            )
            print(f"  [SAM3.1] Objects {obj_ids_here} registered on frame {frame_idx}.")

        # ── 4. Initialise backbone_out (text features for _run_single_frame_inference) ──
        inference_state["backbone_out"] = demo_model._init_backbone_out(inference_state)

        # ── 5. Build tracker_metadata ─────────────────────────────────────
        inference_state["tracker_metadata"] = _build_tracker_metadata_31(all_obj_ids, device)

        # ── 6. Forward propagation ────────────────────────────────────────
        print("Propagating video (forward, SAM3.1) and saving…")
        len_frame_names = len(frame_names)

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for resp in predictor.handle_stream_request({
                "type":                  "propagate_in_video",
                "session_id":            session_id,
                "propagation_direction": "forward",
            }):
                out_frame_idx    = resp["frame_index"]
                outputs          = resp["outputs"]
                out_obj_ids      = outputs["out_obj_ids"]      # np.ndarray[int]
                out_binary_masks = outputs["out_binary_masks"] # np.ndarray[bool] [N,H,W]

                frame_mask_dict = {
                    str(out_obj_ids[i]): out_binary_masks[i]
                    for i in range(len(out_obj_ids))
                }
                fname = frame_names[out_frame_idx].stem
                futures.append(executor.submit(
                    _save_npz, out_npz_path / f"{fname}.npz", frame_mask_dict
                ))
            for f in futures:
                f.result()

        if args.bidirectional:
            # NOTE: SAM3.1 action_history after forward propagation causes the next call
            # to return "propagation_fetch" (cached results), not true reverse propagation.
            # For bidirectional support, use --version sam3 instead.
            print("  [SAM3.1] WARNING: bidirectional not supported for sam3.1, skipping reverse pass.")

    finally:
        # ── 7. Close session (gc + empty_cache) ──────────────────────────
        predictor.handle_request({"type": "close_session", "session_id": session_id})


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def main():
    args = parse_args()
    arg_lines = "\n".join(f"{k}: {v}" for k, v in vars(args).items())
    send_message("Аргументы запуска:\n```\n" + arg_lines + "\n```")

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    input_path  = Path(args.inputs)
    frame_names = sorted([
        p for p in input_path.glob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    ])
    frame_names_stems = [p.stem for p in frame_names]
    print(f"Total frames in input: {len(frame_names)}")

    input_masks = sorted([
        p for p in Path(args.masks).glob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    ])

    out_npz_path = Path(args.out_npz)
    out_npz_path.mkdir(parents=True, exist_ok=True)

    objects = load_objects_from_labelmap(args.labelmap)
    print(f"Loaded {len(objects)} objects from labelmap.")

    if args.version == "sam3.1":
        run_sam31(args, device, frame_names, frame_names_stems, input_masks, objects, out_npz_path)
    else:
        run_sam3(args, device, frame_names, frame_names_stems, input_masks, objects, out_npz_path)

    send_message(f"Propagation finished. {get_mem_report()}")


if __name__ == "__main__":
    try:
        time_start = time.time()
        send_message(f"Starting {os.path.basename(__file__)}...")
        main()
    except Exception as e:
        time_end = time.time()
        time_elapsed = time.strftime("%H:%M:%S", time.gmtime(time_end - time_start))
        notify_error(e, f"Критическая ошибка при выполнении {os.path.basename(__file__)}. Time elapsed: {time_elapsed}")
        raise
    time_end = time.time()
    time_elapsed = time.strftime("%H:%M:%S", time.gmtime(time_end - time_start))
    send_message(f"{os.path.basename(__file__)} completed successfully in {time_elapsed}")
