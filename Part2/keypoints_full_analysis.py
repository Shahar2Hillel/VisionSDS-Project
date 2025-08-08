#!/usr/bin/env python3
"""
keypoint_full_analysis.py

Description:
    Combines inference, error logging, visualization, and summary plotting
    for YOLOv8 pose keypoints in a single script.

    Steps:
      1. Run model on validation images, compare predictions to ground truth,
         annotate & save images, and log mispredicted keypoints to CSV.
      2. Load the CSV report and generate a two-panel bar chart:
         - Left:  count of wrong detections per keypoint
         - Right: average pixel-distance error per keypoint

Usage:
    python keypoint_full_analysis.py
"""
import os
import csv
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO

from config import Config


def load_gt(label_path: Path) -> np.ndarray:
    parts = list(map(float, label_path.read_text().strip().split()))
    return np.array(parts[5:]).reshape(-1, 3)


def annotate_and_log(img, gt_kpts, pred_kpts, bbox, img_name):
    rows = []
    x1, y1, x2, y2 = bbox
    max_dim = max(x2 - x1, y2 - y1)
    for idx, (gx, gy, gv) in enumerate(gt_kpts):
        if gv == 0: continue
        gx_px, gy_px = int(gx * img.shape[1]), int(gy * img.shape[0])
        px_px, py_px = map(int, pred_kpts[idx])
        # draw
        cv2.circle(img, (gx_px, gy_px), 4, (255, 0, 0), -1)
        cv2.circle(img, (px_px, py_px), 4, (0, 255, 0), -1)
        name = Config.KEYPOINT_NAMES[idx]
        if name in Config.CHECK_NAMES:
            dist = np.hypot(gx_px - px_px, gy_px - py_px)
            rel  = dist / max_dim
            if rel > Config.PCK_THRESHOLD:
                cv2.circle(img, (px_px, py_px), 6, (0, 0, 255), 2)
                cv2.putText(img, f"Wrong {name}", (px_px+5, py_px-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255),1)
                rows.append([img_name, name, gx_px, gy_px, px_px, py_px, round(dist,2), round(rel*100,2)])
    return rows


def run_inference_and_log():
    # Ensure output directory exists
    Config.OUTPUT_VIS.mkdir(parents=True, exist_ok=True)

    # Load model
    model = YOLO(Config.MODEL_PATH)
    results = model.predict(
        source=str(Config.IMAGES_VAL),
        save=False,
        conf=Config.CONF_THRESH,
        imgsz=Config.IMGSZ,
        verbose=False,
    )

    # CSV header
    csv_rows = [["image","keypoint_name","gt_x","gt_y","pred_x","pred_y","pixel_dist","rel_dist_pct"]]

    for res in results:
        img_path = Path(res.path)
        lbl_path = Config.LABELS_VAL / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue

        img = cv2.imread(str(img_path))
        ih, iw = img.shape[:2]

        # Read *each* line (one object) in the label file
        with open(lbl_path) as f:
            lines = f.readlines()

        for line in lines:
            parts = list(map(float, line.strip().split()))
            cls_id, xc, yc, w, h = parts[:5]
            gt_kpts = np.array(parts[5:]).reshape(-1, 3)

            # Get predicted keypoints for same class
            preds = [i for i, b in enumerate(res.boxes) if int(b.cls) == int(cls_id)]
            if not preds:
                continue

            pred_kpts = np.array(res.keypoints.xy[preds[0]].cpu())

            # Draw bounding box for reference
            x1 = int((xc - w / 2) * iw)
            y1 = int((yc - h / 2) * ih)
            x2 = int((xc + w / 2) * iw)
            y2 = int((yc + h / 2) * ih)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 2)

            # Annotate and log
            csv_rows += annotate_and_log(img, gt_kpts, pred_kpts, (x1, y1, x2, y2), img_path.name)

        # save annotated image
        cv2.imwrite(str(Config.OUTPUT_VIS / img_path.name), img)

    # write CSV
    with open(Config.KEYPOINT_ANALYSIS_OUTPUT_CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)

    print(f"✅ CSV log saved to {Config.KEYPOINT_ANALYSIS_OUTPUT_CSV_FILE}")



def plot_summary():
    df = pd.read_csv(Config.KEYPOINT_ANALYSIS_OUTPUT_CSV_FILE)
    counts = df.groupby('keypoint_name').size()
    errors = df.groupby('keypoint_name')['pixel_dist'].mean()
    if Config.SORT_BARS:
        order = counts.sort_values(ascending=False).index
        counts = counts.loc[order]; errors = errors.loc[order]
    fig, axes = plt.subplots(1,2,figsize=(12,5))
    # counts
    bars=axes[0].bar(counts.index, counts.values, alpha=0.7, color="red")
    axes[0].set_title('Wrong Keypoints Count'); axes[0].set_ylabel('Count')
    for b in bars: axes[0].text(b.get_x()+b.get_width()/2, b.get_height()+0.5,int(b.get_height()), ha='center')
    # errors
    bars=axes[1].bar(errors.index, errors.values, alpha=0.7, color="blue")
    axes[1].set_title('Avg. Pixel Error'); axes[1].set_ylabel('Pixels')
    for b in bars: axes[1].text(b.get_x()+b.get_width()/2, b.get_height()+0.5,f"{b.get_height():.1f}", ha='center')
    plt.tight_layout(); plt.savefig(Config.KEYPOINT_ANALYSIS_OUTPUT_VIS, dpi=300); plt.close()
    print(f"✅ Summary plot saved to {Config.KEYPOINT_ANALYSIS_OUTPUT_VIS}")


def main():
    run_inference_and_log()
    plot_summary()

if __name__ == '__main__':
    main()
