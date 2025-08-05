import json
import os
import yaml
import random
import shutil
from PIL import Image

# === CONFIG ===
dataset_root = '../output_all_tools'
json_path = os.path.join(dataset_root, 'annotations.json')
yaml_output_path = os.path.join(dataset_root, "pose.yaml")
images_src_dir = os.path.join(dataset_root, 'images_with_background')  # source folder of all images
train_ratio = 0.8  # train/val split ratio

# === OUTPUT STRUCTURE ===
images_train_path = os.path.join(dataset_root, 'images', 'train')
images_val_path = os.path.join(dataset_root, 'images', 'val')
labels_train_path = os.path.join(dataset_root, 'labels', 'train')
labels_val_path = os.path.join(dataset_root, 'labels', 'val')



if os.path.exists("pose.yaml"):
    print("pose.yaml already exists. Please remove it before running this script.")
    exit(1)

if os.path.exists(os.path.join(dataset_root, 'images')):
    os.rename(os.path.join(dataset_root, 'images'), os.path.join(dataset_root, 'images_with_no_background'))

# Ensure output directories exist
for p in [images_train_path, images_val_path, labels_train_path, labels_val_path]:
    os.makedirs(p, exist_ok=True)

# === READ JSON ===
with open(json_path) as f:
    data = json.load(f)

# Extract categories
categories = []
for img in data["images"]:
    if img["category"] not in categories:
        categories.append(img["category"])
cat_to_id = {c: i for i, c in enumerate(categories)}

# Keypoint order from first sample
first_kp_names = list(data["images"][0]["keypoints"].keys())
num_kpts = len(first_kp_names)

# Shuffle and split data
random.shuffle(data["images"])
split_index = int(len(data["images"]) * train_ratio)
train_images = data["images"][:split_index]
val_images = data["images"][split_index:]

def save_yolo_label(entry, label_path):
    """Convert JSON entry to YOLOv8 Pose format and save as .txt"""
    img_path = os.path.join(images_src_dir, entry["file_name"])
    img_w, img_h = Image.open(img_path).size

    # BBox
    x, y, w, h = entry["bbox"]
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    w_norm = w / img_w
    h_norm = h / img_h

    # Keypoints
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

    with open(label_path, "w") as f:
        f.write(line + "\n")

# === Process and copy files ===
for subset, img_list, img_dst, lbl_dst in [
    ('train', train_images, images_train_path, labels_train_path),
    ('val', val_images, images_val_path, labels_val_path)
]:
    for entry in img_list:
        src_img_path = os.path.join(images_src_dir, entry["file_name"])
        dst_img_path = os.path.join(img_dst, entry["file_name"])
        shutil.copy(src_img_path, dst_img_path)

        label_filename = os.path.splitext(entry["file_name"])[0] + ".txt"
        label_path = os.path.join(lbl_dst, label_filename)
        save_yolo_label(entry, label_path)

print(f"✅ Dataset split complete: {len(train_images)} train, {len(val_images)} val")

# === CREATE YAML ===
pose_yaml = {
    "path": dataset_root,
    "train": "images/train",
    "val": "images/val",
    "nc": len(categories),
    "names": categories,
    "kpt_shape": [num_kpts, 3],
    "keypoint_names": first_kp_names
}

with open(yaml_output_path, "w") as f:
    yaml.dump(pose_yaml, f, sort_keys=False)

print(f"✅ pose.yaml created at: {yaml_output_path}")
print(yaml.dump(pose_yaml, sort_keys=False))
