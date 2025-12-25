import os
import numpy as np
import glob
import sys

def analyze_cache(cache_dir):
    """
    Analyzes SAM3 cache files (.npz) to count detections per prompt.
    """
    cache_files = glob.glob(os.path.join(cache_dir, "*.npz"))
    if not cache_files:
        print(f"No .npz files found in {cache_dir}")
        return

    results = []

    for file_path in cache_files:
        prompt_name = os.path.basename(file_path).replace(".npz", "").replace("_", " ")
        try:
            # allow_pickle=True is needed because outputs is a pickled dict
            data = np.load(file_path, allow_pickle=True)
            if 'outputs' not in data:
                print(f"Skipping {file_path}: 'outputs' key not found.")
                continue
                
            outputs = data['outputs'].item()
            
            total_frames = len(outputs)
            frames_with_masks = 0
            total_masks = 0
            
            for frame_idx, frame_data in outputs.items():
                # Check if out_obj_ids exists and has elements
                obj_ids = frame_data.get("out_obj_ids", [])
                num_masks = len(obj_ids) if hasattr(obj_ids, "__len__") else 0
                
                if num_masks > 0:
                    frames_with_masks += 1
                    total_masks += num_masks
            
            results.append({
                "prompt": prompt_name,
                "total_frames": total_frames,
                "frames_with_masks": frames_with_masks,
                "total_masks": total_masks,
                "detection_rate": frames_with_masks / total_frames if total_frames > 0 else 0
            })
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # Sort results: first by those with masks (descending), then by name
    results.sort(key=lambda x: (x["total_masks"], x["frames_with_masks"]), reverse=True)

    # Print table
    header = f"{'Prompt':<24} | {'Frames':<8} | {'With Mask':<10} | {'Total Masks':<12} | {'Rate':<6}"
    print("\n" + header)
    print("-" * len(header))
    
    for res in results:
        print(f"{res['prompt']:<24} | {res['total_frames']:<8} | {res['frames_with_masks']:<10} | {res['total_masks']:<12} | {res['detection_rate']:.2%}")
    
    print("-" * len(header))
    print(f"Total prompts analyzed: {len(results)}")

if __name__ == "__main__":
    # Use the specific path found or take from command line
    default_path = "/workspace/sam3_batch_results_music_room/cache"
    target_dir = sys.argv[1] if len(sys.argv) > 1 else default_path
    
    if not os.path.exists(target_dir):
        print(f"Directory not found: {target_dir}")
    else:
        print(f"Analyzing cache in: {target_dir}")
        analyze_cache(target_dir)

