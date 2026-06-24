import json
import argparse
import random
import colorsys
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

def generate_random_colors(n, seed=42):
    """Generate n distinct random colors in HEX format."""
    random.seed(seed)
    colors = set()
    while len(colors) < n:
        hue = random.random()
        # Bright colors preferred
        saturation = 0.6 + 0.4 * random.random()   # [0.6, 1.0]
        value = 0.7 + 0.3 * random.random()        # [0.7, 1.0]
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        hex_color = "#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255))
        colors.add(hex_color)
    return list(colors)

DEFAULT_COCO_JSON = "/workspace/sam3_cache_to_coco/output_coco.json"
DEFAULT_OUTPUT_JSON = "/workspace/sam3_cache_to_coco/labels.json"

def main():
    parser = argparse.ArgumentParser(description="Generate CVAT labels JSON from COCO JSON categories")
    parser.add_argument("--coco_json", type=str, default=DEFAULT_COCO_JSON, help="Path to input COCO JSON file")
    parser.add_argument("--output_json", type=str, default=DEFAULT_OUTPUT_JSON, help="Path to output labels JSON file")
    args = parser.parse_args()

    logger.info(f"Reading COCO JSON from: {args.coco_json}")
    
    try:
        with open(args.coco_json, 'r') as f:
            coco_data = json.load(f)
    except FileNotFoundError:
        logger.error(f"File not found: {args.coco_json}")
        return
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON file: {args.coco_json}")
        return

    categories = coco_data.get("categories", [])
    if not categories:
        logger.warning("No categories found in COCO file.")
        return

    # Extract category names (sort for deterministic color assignment)
    # We use the name as the label name
    cat_names = sorted([cat["name"] for cat in categories])
    logger.info(f"Found {len(cat_names)} categories.")

    # Generate colors
    palette = generate_random_colors(len(cat_names))

    # Generate labels structure
    cvat_labels = []
    for i, class_name in enumerate(cat_names):
        cvat_labels.append({
            "name": class_name,
            "type": "any",
            "color": palette[i],
            "attributes": [] 
        })

    logger.info(f"Saving labels to: {args.output_json}")
    with open(args.output_json, 'w') as f:
        json.dump(cvat_labels, f, indent=2)
    
    logger.info("Done!")

if __name__ == "__main__":
    main()
