import json
import os

json_path = "annotations.json" # TODO: Update with the actual path to your JSON file
output_dir = "labels" # TODO: Update with the desired output directory

with open(json_path) as f:
    data = json.load(f)

os.makedirs(output_dir, exist_ok=True)

for ann in data['annotations']:
    image_id = ann['image_id']
    bbox = ann['bbox']  # [x, y, width, height]
    kpts = ann['keypoints']  # [x1, y1, v1, x2, y2, v2, ...]

    img_info = next(img for img in data['images'] if img['id'] == image_id)
    img_w, img_h = img_info['width'], img_info['height']

    # Normalize bbox
    x_center = (bbox[0] + bbox[2] / 2) / img_w
    y_center = (bbox[1] + bbox[3] / 2) / img_h
    w = bbox[2] / img_w
    h = bbox[3] / img_h

    # Normalize keypoints
    norm_kpts = []
    for i in range(0, len(kpts), 3):
        x = kpts[i] / img_w
        y = kpts[i+1] / img_h
        v = kpts[i+2]
        norm_kpts.extend([x, y, v])

    line = "0 " + " ".join(map(str, [x_center, y_center, w, h] + norm_kpts))
    
    label_path = os.path.join(output_dir, f"{os.path.splitext(img_info['file_name'])[0]}.txt")
    with open(label_path, "a") as f_out:
        f_out.write(line + "\n")
