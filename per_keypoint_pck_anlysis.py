import os
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from PIL import Image

# === CONFIG ===
MODEL_PATH = "runs/pose/train3/weights/best.pt"  # your trained model
DATASET_ROOT = "../output_all_tools"
IMG_DIR = os.path.join(DATASET_ROOT, "images", "val")
LBL_DIR = os.path.join(DATASET_ROOT, "labels", "val")
KEYPOINT_NAMES = ['bottom_left', 'bottom_right', 'top_left', 'top_right', 'middle_left', 'middle_right']
PCK_THRESHOLD = 0.05  # 5% of bbox size

# === LOAD MODEL ===
model = YOLO(MODEL_PATH)

# === Run inference on validation set ===
pred_results = model.predict(source=IMG_DIR, save=False, conf=0.001, imgsz=640, verbose=False)

# === Storage for PCK calculation ===
correct_counts = np.zeros(len(KEYPOINT_NAMES), dtype=int)
total_counts = np.zeros(len(KEYPOINT_NAMES), dtype=int)

# === Process each image ===
for r in pred_results:
    img_path = Path(r.path)
    lbl_path = Path(LBL_DIR) / (img_path.stem + ".txt")

    if not lbl_path.exists():
        continue

    # Read ground truth
    with open(lbl_path) as f:
        lines = f.readlines()

    img_w, img_h = Image.open(img_path).size

    for line in lines:
        parts = list(map(float, line.strip().split()))
        cls_id, xc, yc, w, h = parts[:5]
        gt_kpts = np.array(parts[5:]).reshape(-1, 3)

        # Find matching prediction (highest conf with same class)
        preds = [b for b in r.boxes if int(b.cls) == int(cls_id)]
        if not preds:
            continue
        pred = preds[0]  # just take first for now

        pred_kpts = np.array(r.keypoints.xy[0].cpu()) / np.array([[img_w, img_h]])  # normalize

        # Compare each keypoint
        for i, (gx, gy, gv) in enumerate(gt_kpts):
            if gv == 0:  # skip unlabeled
                continue
            total_counts[i] += 1

            # Denormalize ground truth to pixels
            gx_px, gy_px = gx * img_w, gy * img_h
            px_px, py_px = pred_kpts[i] * np.array([img_w, img_h])

            # Distance in pixels
            dist = np.linalg.norm([gx_px - px_px, gy_px - py_px])
            max_dim = max(w * img_w, h * img_h)  # object scale

            if dist <= PCK_THRESHOLD * max_dim:
                correct_counts[i] += 1

# === Compute per-keypoint PCK ===
pck_scores = correct_counts / np.maximum(total_counts, 1)

print("\nPer-Keypoint PCK@{:.0%} of bbox size".format(PCK_THRESHOLD))
for name, score, total in zip(KEYPOINT_NAMES, pck_scores, total_counts):
    print(f"{name:15s}: {score:.3f} ({correct_counts[KEYPOINT_NAMES.index(name)]}/{total})")
