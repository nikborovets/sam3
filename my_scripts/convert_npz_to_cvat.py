import os
import sys
import numpy as np
import glob
from lxml import etree
from tqdm import tqdm
import argparse

def rle_encode(mask):
    """
    Encodes a binary mask into RLE format suitable for CVAT XML.
    CVAT XML RLE expects a comma-separated list of run lengths,
    starting with the count of background pixels (0).
    
    Args:
        mask (np.ndarray): Binary mask of shape (H, W), dtype=bool or uint8.
        
    Returns:
        str: Comma-separated string of integers (e.g., "100, 5, 20, ...").
    """
    # Flatten the mask
    pixels = mask.flatten()
    
    # Ensure it's boolean
    pixels = (pixels > 0)
    
    # Calculate lengths of runs
    n = len(pixels)
    if n == 0:
        return "0"

    # Find where values change
    # where(pixels[1:] != pixels[:-1]) returns indices i where pixels[i] != pixels[i+1]
    # So the run ends at i+1
    change_indices = np.where(pixels[1:] != pixels[:-1])[0] + 1
    
    # Add start (0) and end (n)
    run_starts = np.concatenate(([0], change_indices, [n]))
    
    # Lengths of runs
    run_lengths = np.diff(run_starts)
    
    # CVAT/COCO RLE requirement: First count is always for 0s (background).
    # If the mask starts with 1 (foreground), we must prepend a 0 count for background.
    if pixels[0]:
        # Starts with foreground, so 0 background pixels at start
        run_lengths = np.concatenate(([0], run_lengths))
        
    # Convert to string
    rle_string = ", ".join(map(str, run_lengths))
    
    return rle_string

def create_cvat_xml(cache_dir, output_file, video_width=1920, video_height=1080):
    """
    Converts SAM3 .npz cache files to CVAT Video XML 1.1 format.
    """
    print(f"Searching for .npz files in {cache_dir}...")
    npz_files = glob.glob(os.path.join(cache_dir, "*.npz"))
    
    if not npz_files:
        print("No .npz files found!")
        return

    # Structure to hold tracks:
    # tracks[(class_name, obj_id)] = { frame_idx: mask }
    tracks = {}
    
    # Global frame range
    min_frame = float('inf')
    max_frame = 0
    
    print("Reading .npz files and grouping tracks...")
    for file_path in tqdm(npz_files):
        # Class name from filename (e.g. "chair.npz" -> "chair")
        class_name = os.path.basename(file_path).replace(".npz", "").replace("_", " ")
        
        try:
            data = np.load(file_path, allow_pickle=True)
            if 'outputs' not in data:
                print(f"Skipping {file_path}: 'outputs' key not found.")
                continue
            
            outputs = data['outputs'].item()
            
            for frame_idx, frame_data in outputs.items():
                min_frame = min(min_frame, frame_idx)
                max_frame = max(max_frame, frame_idx)
                
                # Check for masks
                obj_ids = frame_data.get("out_obj_ids", [])
                masks = frame_data.get("out_binary_masks", [])
                
                if len(obj_ids) == 0:
                    continue
                    
                for i, obj_id in enumerate(obj_ids):
                    # Unique key for this track
                    # We combine class name and obj_id to distinguish "chair 1" from "table 1"
                    # Note: obj_ids are already chunk-aware (e.g. 1000, 2000) thanks to previous edits
                    track_key = (class_name, obj_id)
                    
                    if track_key not in tracks:
                        tracks[track_key] = {}
                    
                    # Store mask (ensure bool)
                    mask = masks[i]
                    if mask.dtype != bool:
                        mask = mask > 0.5
                        
                    tracks[track_key][frame_idx] = mask
                    
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    if not tracks:
        print("No tracks found in the cache files.")
        return

    print(f"Found {len(tracks)} unique tracks.")
    print(f"Video range: {min_frame} to {max_frame}")

    # --- Generate XML ---
    root = etree.Element("annotations")
    
    # FIX 1: Add version
    etree.SubElement(root, "version").text = "1.1"
    
    # Meta block
    meta = etree.SubElement(root, "meta")
    task = etree.SubElement(meta, "task")
    # Basic meta info required by CVAT
    etree.SubElement(task, "id").text = "0"
    etree.SubElement(task, "name").text = "sam3_export"
    etree.SubElement(task, "size").text = str(max_frame + 1)
    etree.SubElement(task, "mode").text = "interpolation"
    etree.SubElement(task, "start_frame").text = str(min_frame)
    etree.SubElement(task, "stop_frame").text = str(max_frame)
    etree.SubElement(task, "z_order").text = "False"

    # FIX 2: Add labels definition
    labels_elem = etree.SubElement(task, "labels")
    unique_labels = sorted(list(set(k[0] for k in tracks.keys())))
    
    for label_name in unique_labels:
        label_elem = etree.SubElement(labels_elem, "label")
        etree.SubElement(label_elem, "name").text = label_name
        
        # Define attributes for this label (required to use them in tracks)
        attributes_elem = etree.SubElement(label_elem, "attributes")
        attribute_elem = etree.SubElement(attributes_elem, "attribute")
        etree.SubElement(attribute_elem, "name").text = "origin"
        etree.SubElement(attribute_elem, "mutable").text = "False"
        etree.SubElement(attribute_elem, "input_type").text = "text"
        etree.SubElement(attribute_elem, "default_value").text = "manual"

    
    # Tracks
    # Sort tracks by class and id for consistent output
    # Convert obj_id to int if possible for proper numerical sorting
    def sort_key(k):
        name, oid = k
        try:
            return (name, int(oid))
        except:
            return (name, str(oid))

    sorted_keys = sorted(tracks.keys(), key=sort_key)
    
    for track_idx, (class_name, obj_id) in enumerate(tqdm(sorted_keys, desc="Encoding tracks")):
        track_frames = tracks[(class_name, obj_id)]
        
        # Create <track> element
        track_elem = etree.SubElement(root, "track")
        track_elem.set("id", str(track_idx))
        track_elem.set("label", str(class_name))
        track_elem.set("source", "auto")
        
        # Sort frames
        sorted_frames = sorted(track_frames.keys())
        
        for frame_idx in sorted_frames:
            mask = track_frames[frame_idx]
            
            # Encode RLE
            # We encode full frame (left=0, top=0) to simplify coordinate mapping
            rle_string = rle_encode(mask)
            
            # Create <mask> element
            mask_elem = etree.SubElement(track_elem, "mask")
            mask_elem.set("frame", str(frame_idx))
            mask_elem.set("outside", "0")
            mask_elem.set("occluded", "0")
            mask_elem.set("keyframe", "1")
            
            # Mask attributes
            mask_elem.set("rle", rle_string)
            mask_elem.set("left", "0")
            mask_elem.set("top", "0")
            mask_elem.set("width", str(video_width))
            mask_elem.set("height", str(video_height))
            
            # Custom attribute
            attr_elem = etree.SubElement(mask_elem, "attribute")
            attr_elem.set("name", "origin")
            attr_elem.text = "sam3_auto"

    # Write to file
    print(f"Writing XML to {output_file}...")
    try:
        tree = etree.ElementTree(root)
        tree.write(output_file, pretty_print=True, xml_declaration=True, encoding="utf-8")
        print(f"Successfully saved to {output_file}")
    except Exception as e:
        print(f"Failed to write XML: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert SAM3 .npz cache to CVAT XML 1.1")
    parser.add_argument("cache_dir", help="Directory containing .npz files")
    parser.add_argument("output_xml", help="Output .xml file path")
    parser.add_argument("--width", type=int, default=1280, help="Video width")
    parser.add_argument("--height", type=int, default=720, help="Video height")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.cache_dir):
        print(f"Error: Directory {args.cache_dir} does not exist.")
        sys.exit(1)
        
    create_cvat_xml(args.cache_dir, args.output_xml, args.width, args.height)
