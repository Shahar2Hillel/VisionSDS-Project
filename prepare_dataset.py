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
import math

import cv2
import numpy as np
from tqdm import tqdm
import yaml
from PIL import Image

from configs.prepare_dataset_config import Config


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


def determine_keypoints_union(images: list[dict], preferred_order: list[str] | None = None) -> list[str]:
    """
    Build a global, deterministic union of keypoint names across all objects.
    If preferred_order is provided, keep that relative order first, then append unseen keys alphabetically.
    """
    seen = set()
    # 1) collect all keys
    for entry in images:
        objs = entry['objects'] if 'objects' in entry else [entry]
        for obj in objs:
            for name in obj.get('keypoints', {}).keys():
                seen.add(name)

    if not seen:
        raise ValueError("No keypoints found in dataset.")

    if preferred_order:
        # keep only those that exist, in the given order
        ordered = [k for k in preferred_order if k in seen]
        # append any remaining keys not in preferred_order, sorted for determinism
        tail = sorted(seen - set(ordered))
        return ordered + tail
    else:
        # deterministic (alphabetical) if no preference is given
        return sorted(seen)


def split_entries(entries: List[Dict], ratio: float):
    """Shuffle and split entries into train/val lists."""
    random.shuffle(entries)
    split = int(len(entries) * ratio)
    return entries[:split], entries[split:]


# def write_pose_yaml(path: Path, root: Path, train: str, val: str,
#                     names: List[str], kpt_names: List[str]):
#     """Write the YOLOv8 pose dataset config file."""
#     cfg = {
#         'path': str(root),
#         'train': train,
#         'val': val,
#         'nc': len(names),
#         'names': names,
#         'kpt_shape': [len(kpt_names), 3],
#         'keypoint_names': kpt_names,
#         "flip_idx": [1, 0, 3, 2, 4]
#     }
#     with open(path, 'w') as f:
#         yaml.dump(cfg, f, sort_keys=False)
#     print(f"✔ pose.yaml created at: {path}")


def normalize_bbox(bbox: List[float], img_w: int, img_h: int) -> List[float]:
    """Convert COCO bbox [x, y, w, h] to normalized center x,y,w,h."""
    x, y, w, h = bbox
    xc = (x + w/2) / img_w
    yc = (y + h/2) / img_h
    return [xc, yc, w/img_w, h/img_h]

def make_label_lines(entry: Dict, cat_to_id: Dict[str, int], kpt_order: List[str],
                     img_w: int, img_h: int) -> List[str]:
    """Generate YOLOv8 pose label lines for one image entry with correct visibility handling."""
    def parse_kpt(raw):
        """
        Accepts [x,y], (x,y), [x,y,v], {'x':..,'y':..,'v':..} and returns (x, y, v_or_None).
        """
        if raw is None:
            return None, None, None
        # list/tuple
        if isinstance(raw, (list, tuple)):
            if len(raw) == 2:
                x, y = raw
                return x, y, None
            if len(raw) >= 3:
                x, y, v = raw[:3]
                return x, y, v
        # dict
        if isinstance(raw, dict):
            x = raw.get("x", raw.get(0))
            y = raw.get("y", raw.get(1))
            v = raw.get("v", raw.get(2))
            return x, y, v
        # unknown
        return None, None, None

    def valid_xy(x, y):
        return (
            x is not None and y is not None and
            not (isinstance(x, float) and math.isnan(x)) and
            not (isinstance(y, float) and math.isnan(y)) and
            0.0 <= x < img_w and 0.0 <= y < img_h
        )

    lines = []
    objs = entry["objects"] if "objects" in entry else [entry]

    for obj in objs:
        cls_id = cat_to_id[obj["category"]]
        xc, yc, w_n, h_n = normalize_bbox(obj["bbox"], img_w, img_h)

        kpt_vals = []
        kp_dict = obj.get("keypoints", {}) or {}

        for name in kpt_order:
            raw = kp_dict.get(name, -1)
            if raw == -1:
                # Keypoint not present in this object
                kpt_vals.extend([0.0, 0.0, 0])
                continue
            # Parse keypoint value
            x, y, v_in = parse_kpt(raw)

            # # this is for when NULL means occluded, not missing
            # if x is None or y is None:
            #     # Missing keypoint → unlabeled
            #     kpt_vals.extend([0.0, 0.0, 1])  # v=1 means "labeled but not visible"
            #     continue
            if not valid_xy(x, y):
                # Missing / NaN / out-of-image → unlabeled
                kpt_vals.extend([0.0, 0.0, 0])
                continue

            # Normalize
            x_n = x / img_w
            y_n = y / img_h

            # Visibility: honor provided 0/1/2; else default to 2 (visible)
            if v_in in (0, 1, 2):
                v_out = int(v_in)
                # Safety: if coords are valid but v_in==0/1, keep as given.
            else:
                v_out = 2

            kpt_vals.extend([float(f"{x_n:.6f}"), float(f"{y_n:.6f}"), v_out])

        parts = [str(cls_id), f"{xc:.6f}", f"{yc:.6f}", f"{w_n:.6f}", f"{h_n:.6f}"] + \
                [str(v) if isinstance(v, int) else f"{v:.6f}" for v in kpt_vals]
        lines.append(" ".join(parts))

    return lines



def visualize_pose_yolo_lines(image_path: Path, label_lines: list[str], save_path: Path,
                              kpt_names: list[str] | None = None, draw_names: bool = False):
    """
    Draw YOLOv8-pose labels (bbox + keypoints) onto an image and save it.
    label_lines are strings in the YOLO format written by make_label_lines().
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return
    ih, iw = img.shape[:2]

    for line in label_lines:
        parts = list(map(float, line.strip().split()))
        if len(parts) < 5:
            continue

        cls_id = int(parts[0])
        xc, yc, w, h = parts[1:5]

        # denormalize bbox
        x1 = int((xc - w / 2) * iw); y1 = int((yc - h / 2) * ih)
        x2 = int((xc + w / 2) * iw); y2 = int((yc + h / 2) * ih)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(img, f"cls:{cls_id}", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # keypoints (kpt_x, kpt_y, v) * K
        flat = parts[5:]
        if len(flat) % 3 != 0:
            continue
        kpts = np.array(flat, dtype=float).reshape(-1, 3)

        for i, (kx, ky, v) in enumerate(kpts):
            px = int(kx * iw); py = int(ky * ih)
            v = int(v)
            if v == 2:      # visible/trusted
                color = (0, 255, 0); thick = -1
                cv2.circle(img, (px, py), 4, color, thick)
            elif v == 1:    # labeled but not visible
                color = (0, 165, 255); thick = 1
                cv2.circle(img, (px, py), 4, color, thick)
            else:           # v == 0 (ignored)
                color = (0, 0, 255)
                cv2.drawMarker(img, (px, py), color, markerType=cv2.MARKER_TILTED_CROSS,
                               markerSize=8, thickness=1)

            if draw_names and kpt_names and i < len(kpt_names):
                cv2.putText(img, kpt_names[i], (px + 5, py - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), img)


def process_subset(subset: List[Dict], img_dst: Path, lbl_dst: Path, vis_dir: Path,
                   src_dir: Path, cat_to_id: Dict[str,int], kpt_order: List[str],
                   data_folder_number: str):
    """Copy images and write label files for a subset (train or val)."""
    for entry in tqdm(subset, desc=f"Processing {img_dst.name}"):
        fname = entry['file_name']
        file_name_output = "data_folder" + data_folder_number + "_" + fname

        src_img = src_dir / fname
        dst_img = img_dst / file_name_output
        shutil.copy(src_img, dst_img)

        img_w, img_h = Image.open(src_img).size
        label_lines = make_label_lines(entry, cat_to_id, kpt_order, img_w, img_h)

        lbl_file = lbl_dst / f"{Path(file_name_output).stem}.txt"
        with open(lbl_file, 'w') as f:
            f.write("\n".join(label_lines) + "\n")
        
        # visualize this sample
        vis_path = vis_dir / file_name_output
        visualize_pose_yolo_lines(dst_img, label_lines, vis_path, kpt_names=kpt_order, draw_names=True)


def build_flip_idx(kpt_names: list[str]) -> list[int]:
    """
    Build horizontal-flip index mapping for YOLOv8 pose.
    Each i maps to the index that keypoint i should swap with under H-flip.
    Non-paired (midline) points map to themselves.
    """
    name_to_idx = {n: i for i, n in enumerate(kpt_names)}
    flip = [i for i in range(len(kpt_names))]  # default: self

    def counterpart(name: str) -> str | None:
        # Generic *_left <-> *_right swap
        if name.endswith("_left"):
            return name[:-5] + "_right"
        if name.endswith("_right"):
            return name[:-6] + "_left"
        # Common aliases
        aliases = {
            "top_left": "top_right", "top_right": "top_left",
            "mid_left": "mid_right", "mid_right": "mid_left",
            "middle_left": "middle_right", "middle_right": "middle_left",
            "bottom_left": "bottom_right", "bottom_right": "bottom_left",
        }
        return aliases.get(name, None)

    for i, n in enumerate(kpt_names):
        c = counterpart(n)
        if c is not None and c in name_to_idx:
            flip[i] = name_to_idx[c]
        else:
            flip[i] = i  # midline / unpaired stays the same

    return flip


def write_pose_yaml(path: Path, root: Path, train: str, val: str,
                    names: list[str], kpt_names: list[str]):
    flip_idx = build_flip_idx(kpt_names)  # <-- compute from names

    cfg = {
        "path": str(root),
        "train": train,
        "val": val,
        "nc": len(names),
        "names": names,
        "kpt_shape": [len(kpt_names), 3],
        "keypoint_names": kpt_names,
        "flip_idx": flip_idx,
    }
    with open(path, "w") as f:
        yaml.dump(cfg, f, sort_keys=False)
    print(f"✔ pose.yaml created at: {path}")


def main():
    # Collect all data folders in format coco annotations and covert to YOLOv8 pose format
    for data_folder_num, data_folder_name in enumerate(Config.BASE_DATA_FOLDERS):
        data = load_annotations(data_folder_name / "annotations.json")
        images = data['images']

        images_src_folder = data_folder_name / "images_with_background"

        # categories and mapping
        categories = extract_categories(images)
        cat_to_id = {c: i for i, c in enumerate(categories)}

        # keypoints (global order across all classes)
        kpt_order = determine_keypoints_union(images, preferred_order=[
            "top_left","top_right","mid_left","mid_right","middle_left","middle_right",
            "bottom_tip","bottom_left","bottom_right","joint_center"
        ])

        # split
        train, val = split_entries(images, Config.TRAIN_RATIO)
        print(f"Splitting: {len(train)} train, {len(val)} val images")

        # process each subset
        for dir in [Config.IMAGES_TRAIN, Config.IMAGES_VAL, Config.LABELS_TRAIN, Config.LABELS_VAL,
                Config.VIS_TRAIN, Config.VIS_VAL]:
            os.makedirs(dir, exist_ok=True)


        process_subset(train, Config.IMAGES_TRAIN, Config.LABELS_TRAIN, Config.VIS_TRAIN,
                       images_src_folder, cat_to_id, kpt_order, str(data_folder_num + 1))
        process_subset(val, Config.IMAGES_VAL, Config.LABELS_VAL, Config.VIS_VAL,
                       images_src_folder, cat_to_id, kpt_order, str(data_folder_num + 1))

    # write YAML config
    write_pose_yaml(Config.OUTPUT_POSE_YAML,
                    Config.YOLO_DATASET_PATH,
                    "images/train",
                    "images/val",
                    categories,
                    kpt_order)

if __name__ == '__main__':
    main()
