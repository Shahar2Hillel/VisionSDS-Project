import json
import os
from PIL import Image

# Paths
dataset_root = '../output_all_tools_sample'
json_path = os.path.join(dataset_root, 'annotations.json')
images_dir = os.path.join(dataset_root, 'images_with_background')
output_dir = os.path.join(dataset_root, 'labels_with_background')

# Keypoint order for YOLOv8
kp_names = ["bottom_left", "bottom_right", "top_left", "top_right", "middle_left", "middle_right"]

# Category to class_id mapping
categories = {}
class_counter = 0

os.makedirs(output_dir, exist_ok=True)

with open(json_path) as f:
    data = json.load(f)

for entry in data["images"]:
    img_path = os.path.join(images_dir, entry["file_name"])
    img_w, img_h = Image.open(img_path).size

    # Assign class_id
    category = entry["category"]
    if category not in categories:
        categories[category] = class_counter
        class_counter += 1
    class_id = categories[category]

    # BBox normalization
    x, y, w, h = entry["bbox"]
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    w_norm = w / img_w
    h_norm = h / img_h

    # Keypoints normalization (v=2 means visible)
    norm_kpts = []
    for name in kp_names:
        if name in entry["keypoints"]:
            kx, ky = entry["keypoints"][name]
            kx = max(0, min(kx / img_w, 1))
            ky = max(0, min(ky / img_h, 1))
            v = 2
        else:
            kx, ky, v = 0, 0, 0
        norm_kpts.extend([kx, ky, v])

    # YOLO label line
    line = f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f} " + " ".join(map(str, norm_kpts))

    # Save label
    label_path = os.path.join(output_dir, f"{os.path.splitext(entry['file_name'])[0]}.txt")
    with open(label_path, "w") as f_out:
        f_out.write(line + "\n")

print("✅ Conversion done.")
print("Classes:", categories)
