import json
import os
import yaml
import random
import shutil
from PIL import Image

# === CONFIG ===
dataset_root = '../output_combine'
json_path = os.path.join(dataset_root, 'annotations.json')
yaml_output_path = os.path.join(dataset_root, "pose.yaml")
images_src_dir = os.path.join(dataset_root, 'images_with_background')  # source folder of all images
train_ratio = 0.8  # train/val split ratio

# === OUTPUT STRUCTURE ===
images_train_path = os.path.join(dataset_root, 'images', 'train')
images_val_path = os.path.join(dataset_root, 'images', 'val')
labels_train_path = os.path.join(dataset_root, 'labels', 'train')
labels_val_path = os.path.join(dataset_root, 'labels', 'val')

# Ensure output directories exist
for p in [images_train_path, images_val_path, labels_train_path, labels_val_path]:
    os.makedirs(p, exist_ok=True)

# === READ JSON ===
with open(json_path) as f:
    data = json.load(f)

# Extract categories
categories = []
for img in data["images"]:
    try:
        if "objects" in img: # Multi tool format
            for obj in img["objects"]:
                if obj["category"] not in categories:
                    categories.append(obj["category"])
        else: # Single tool format
            if img["category"] not in categories:
                categories.append(img["category"])
    except Exception as e:
        print(img)
        raise e

cat_to_id = {c: i for i, c in enumerate(categories)}

# Keypoint order
if "objects" in data["images"][0]:
    first_kp_names = list(data["images"][0]["objects"][0]["keypoints"].keys())
else:
    first_kp_names = list(data["images"][0]["keypoints"].keys())
num_kpts = len(first_kp_names)

# Shuffle and split images
random.shuffle(data["images"])
split_index = int(len(data["images"]) * train_ratio)
train_images = data["images"][:split_index]
val_images = data["images"][split_index:]

def save_yolo_labels_for_image(entry, label_path):
    """Write all objects in the image to one YOLOv8 Pose label file."""
    img_path = os.path.join(images_src_dir, entry["file_name"])
    img_w, img_h = Image.open(img_path).size

    lines = []
    if "objects" in entry:
        # Multi-object
        for obj in entry["objects"]:
            x, y, w, h = obj["bbox"]
            x_center = (x + w / 2) / img_w
            y_center = (y + h / 2) / img_h
            w_norm = w / img_w
            h_norm = h / img_h

            norm_kpts = []
            for name in first_kp_names:
                if name in obj["keypoints"]:
                    kx, ky = obj["keypoints"][name]
                    kx = max(0, min(kx / img_w, 1))
                    ky = max(0, min(ky / img_h, 1))
                    v = 2
                else:
                    kx, ky, v = 0, 0, 0
                norm_kpts.extend([kx, ky, v])

            class_id = cat_to_id[obj["category"]]
            line = f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f} " + " ".join(map(str, norm_kpts))
            lines.append(line)
    else:
        # Single object
        x, y, w, h = entry["bbox"]
        x_center = (x + w / 2) / img_w
        y_center = (y + h / 2) / img_h
        w_norm = w / img_w
        h_norm = h / img_h

        norm_kpts = []
        for name in first_kp_names:
            if name in entry["keypoints"]:
                kx, ky = entry["keypoints"][name]
                kx = max(0, min(kx / img_w, 1))
                ky = max(0, min(ky / img_h, 1))
                v = 2
            else:
                kx, ky, v = 0, 0, 0
            norm_kpts.extend([kx, ky, v])

        class_id = cat_to_id[entry["category"]]
        line = f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f} " + " ".join(map(str, norm_kpts))
        lines.append(line)

    with open(label_path, "w") as f:
        f.write("\n".join(lines) + "\n")

# === Copy images and save labels ===
for subset, img_list, img_dst, lbl_dst in [
    ('train', train_images, images_train_path, labels_train_path),
    ('val', val_images, images_val_path, labels_val_path)
]:
    for entry in img_list:
        # Copy image
        src_img_path = os.path.join(images_src_dir, entry["file_name"])
        dst_img_path = os.path.join(img_dst, entry["file_name"])
        shutil.copy(src_img_path, dst_img_path)

        # Save labels
        label_filename = os.path.splitext(entry["file_name"])[0] + ".txt"
        label_path = os.path.join(lbl_dst, label_filename)
        save_yolo_labels_for_image(entry, label_path)

print(f"✅ Dataset split complete: {len(train_images)} train, {len(val_images)} val")

# === Create pose.yaml ===
pose_yaml = {
    "path": dataset_root,
    "train": "images/train",
    "val": "images/val",
    "nc": len(categories),
    "names": categories,
    "kpt_shape": [num_kpts, 3],
    "keypoint_names": first_kp_names,
    "flip_idx": [1, 0, 3, 2, 4]
}

with open(yaml_output_path, "w") as f:
    yaml.dump(pose_yaml, f, sort_keys=False)

print(f"✅ pose.yaml created at: {yaml_output_path}")
print(yaml.dump(pose_yaml, sort_keys=False))
