import glob
import os
import json
import random
import colorsys

def generate_random_colors(n, seed=42):
    random.seed(seed)
    colors = set()
    while len(colors) < n:
        hue = random.random()
        saturation = 0.6 + 0.4 * random.random()   # [0.6, 1.0]
        value = 0.7 + 0.3 * random.random()        # [0.7, 1.0]
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        hex_color = "#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255))
        colors.add(hex_color)
    return list(colors)

cache_dir = "/workspace/sam3_batch_results_3023_16_01_2026/cache"
npz_files = glob.glob(os.path.join(cache_dir, "*.npz"))

classes = sorted(list(set(
    os.path.basename(f).replace(".npz", "").replace("_", " ") 
    for f in npz_files
)))

palette = generate_random_colors(len(classes))

cvat_labels = []
for i, class_name in enumerate(classes):
    cvat_labels.append({
        "name": class_name,
        "type": "any",
        "color": palette[i],
        "attributes": []
    })
# for i, class_name in enumerate(classes):
#     cvat_labels.append({
#         "name": class_name,
#         "type": "any",
#         "color": palette[i],
#         "attributes": [
#             {
#                 "name": "origin",
#                 "input_type": "text",
#                 "mutable": False,
#                 "default_value": "",
#                 "values": ["__text__"] # <--- Добавили фиктивное значение
#             }
#         ]
#     })

print(json.dumps(cvat_labels, indent=2))
