#!/usr/bin/env python3
"""
prepare_dataset.py

Description:
    Reads COCO-like JSON annotations, extracts categories and keypoints,
    splits images into train/validation sets, converts annotations to YOLOv8
    pose label format, copies images and labels into the proper directory structure,
    and generates a YOLOv8-compatible `pose.yaml` config file.

Usage:
    python prepare_dataset.py
"""
import os
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List

import yaml
from PIL import Image

from config import Config


def load_annotations(json_path: Path) -> Dict:
    """Load and return the JSON content."""
    with open(json_path, 'r') as f:
        return json.load(f)


def extract_categories(images: List[Dict]) -> List[str]:
    """Gather unique category names from entries."""
    cats = []
    for entry in images:
        objs = entry.get('objects') or [entry]
        for obj in objs:
            name = obj.get('category')
            if name and name not in cats:
                cats.append(name)
    return cats


def determine_keypoints(images: List[Dict]) -> List[str]:
    """Extract the ordered list of keypoint names from the first entry."""
    first = images[0]
    sample = (first['objects'][0] if 'objects' in first else first)
    return list(sample['keypoints'].keys())


def split_entries(entries: List[Dict], ratio: float):
    """Shuffle and split entries into train/val lists."""
    random.shuffle(entries)
    split = int(len(entries) * ratio)
    return entries[:split], entries[split:]


def write_pose_yaml(path: Path, root: Path, train: str, val: str,
                    names: List[str], kpt_names: List[str]):
    """Write the YOLOv8 pose dataset config file."""
    cfg = {
        'path': str(root),
        'train': train,
        'val': val,
        'nc': len(names),
        'names': names,
        'kpt_shape': [len(kpt_names), 3],
        'keypoint_names': kpt_names,
        "flip_idx": [1, 0, 3, 2, 4]
    }
    with open(path, 'w') as f:
        yaml.dump(cfg, f, sort_keys=False)
    print(f"✔ pose.yaml created at: {path}")


def normalize_bbox(bbox: List[float], img_w: int, img_h: int) -> List[float]:
    """Convert COCO bbox [x, y, w, h] to normalized center x,y,w,h."""
    x, y, w, h = bbox
    xc = (x + w/2) / img_w
    yc = (y + h/2) / img_h
    return [xc, yc, w/img_w, h/img_h]


def make_label_lines(entry: Dict, cat_to_id: Dict[str,int], kpt_order: List[str],
                     img_w: int, img_h: int) -> List[str]:
    """Generate YOLOv8 pose label lines for one image entry."""
    lines = []
    objs = entry.get('objects') or [entry]
    for obj in objs:
        cls_id = cat_to_id[obj['category']]
        bbox_norm = normalize_bbox(obj['bbox'], img_w, img_h)
        kpt_vals = []
        for name in kpt_order:
            if name in obj['keypoints']:
                kx, ky = obj['keypoints'][name]
                # clamp and normalize
                kx_n = min(max(kx / img_w, 0.0), 1.0)
                ky_n = min(max(ky / img_h, 0.0), 1.0)
                v = 2
            else:
                kx_n, ky_n, v = 0.0, 0.0, 0
            kpt_vals.extend([kx_n, ky_n, v])
        parts = [str(cls_id)] + [f"{x:.6f}" for x in bbox_norm] + [str(x) for x in kpt_vals]
        lines.append(" ".join(parts))
    return lines


def process_subset(subset: List[Dict], img_dst: Path, lbl_dst: Path,
                   src_dir: Path, cat_to_id: Dict[str,int], kpt_order: List[str]):
    """Copy images and write label files for a subset (train or val)."""
    for entry in subset:
        fname = entry['file_name']
        src_img = src_dir / fname
        dst_img = img_dst / fname
        shutil.copy(src_img, dst_img)

        img_w, img_h = Image.open(src_img).size
        label_lines = make_label_lines(entry, cat_to_id, kpt_order, img_w, img_h)

        lbl_file = lbl_dst / f"{Path(fname).stem}.txt"
        with open(lbl_file, 'w') as f:
            f.write("\n".join(label_lines) + "\n")


def main():
    data = load_annotations(Config.ANNOTATIONS_JSON)
    images = data['images']

    # categories and mapping
    categories = extract_categories(images)
    cat_to_id = {c: i for i, c in enumerate(categories)}

    # keypoints
    kpt_order = determine_keypoints(images)

    # split
    train, val = split_entries(images, Config.TRAIN_RATIO)
    print(f"Splitting: {len(train)} train, {len(val)} val images")

    # process each subset
    os.makedirs(Config.IMAGES_TRAIN, exist_ok=True)
    os.makedirs(Config.IMAGES_VAL, exist_ok=True)
    os.makedirs(Config.LABELS_TRAIN, exist_ok=True)
    os.makedirs(Config.LABELS_VAL, exist_ok=True)
    process_subset(train, Config.IMAGES_TRAIN, Config.LABELS_TRAIN,
                   Config.IMAGES_SRC, cat_to_id, kpt_order)
    process_subset(val, Config.IMAGES_VAL, Config.LABELS_VAL,
                   Config.IMAGES_SRC, cat_to_id, kpt_order)

    # write YAML config
    write_pose_yaml(Config.OUTPUT_POSE_YAML, Config.DATASET_ROOT_PATH,
                    'images/train', 'images/val',
                    categories, kpt_order)

if __name__ == '__main__':
    main()
