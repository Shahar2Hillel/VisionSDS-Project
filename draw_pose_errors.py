import os
import cv2
import csv
import numpy as np
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

# === CONFIG ===
MODEL_PATH = "runs/pose/train3/weights/best.pt"   # your trained model path
DATASET_ROOT = "../output_all_tools"
IMG_DIR = os.path.join(DATASET_ROOT, "images", "val")
LBL_DIR = os.path.join(DATASET_ROOT, "labels", "val")
OUTPUT_DIR = os.path.join(DATASET_ROOT, "output_vis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_PATH = os.path.join(OUTPUT_DIR, "wrong_keypoints.csv")

KEYPOINT_NAMES = ['bottom_left', 'bottom_right', 'top_left', 'top_right', 'middle_left', 'middle_right']
CHECK_NAMES = {'bottom_left', 'bottom_right', 'middle_left', 'middle_right'}
PCK_THRESHOLD = 0.05  # 5% of bbox size to consider correct

# === LOAD MODEL ===
model = YOLO(MODEL_PATH)
pred_results = model.predict(source=IMG_DIR, save=False, conf=0.001, imgsz=640, verbose=False)

# === CSV SETUP ===
csv_rows = [["image", "keypoint_name", "gt_x", "gt_y", "pred_x", "pred_y", "pixel_dist", "rel_dist_pct"]]

# === PROCESS EACH IMAGE ===
for r in pred_results:
    img_path = Path(r.path)
    lbl_path = Path(LBL_DIR) / (img_path.stem + ".txt")
    if not lbl_path.exists():
        continue

    img = cv2.imread(str(img_path))
    img_h, img_w = img.shape[:2]

    # Read ground truth
    with open(lbl_path) as f:
        lines = f.readlines()

    for line in lines:
        parts = list(map(float, line.strip().split()))
        cls_id, xc, yc, w, h = parts[:5]
        gt_kpts = np.array(parts[5:]).reshape(-1, 3)

        # Get predicted keypoints for same class
        preds = [i for i, b in enumerate(r.boxes) if int(b.cls) == int(cls_id)]
        if not preds:
            continue

        pred_kpts = np.array(r.keypoints.xy[preds[0]].cpu())

        # Draw bounding box for reference
        x1 = int((xc - w / 2) * img_w)
        y1 = int((yc - h / 2) * img_h)
        x2 = int((xc + w / 2) * img_w)
        y2 = int((yc + h / 2) * img_h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 2)

        max_dim = max(w * img_w, h * img_h)  # bbox scale

        for i, (gx, gy, gv) in enumerate(gt_kpts):
            if gv == 0:
                continue  # skip unlabeled

            # Convert to pixel coords
            gx_px, gy_px = int(gx * img_w), int(gy * img_h)
            px_px, py_px = int(pred_kpts[i][0]), int(pred_kpts[i][1])

            # Draw GT in blue
            cv2.circle(img, (gx_px, gy_px), 4, (255, 0, 0), -1)

            # Draw predicted in green
            cv2.circle(img, (px_px, py_px), 4, (0, 255, 0), -1)

            # Check if this is a bottom/middle keypoint
            if KEYPOINT_NAMES[i] in CHECK_NAMES:
                dist = np.linalg.norm([gx_px - px_px, gy_px - py_px])
                rel_dist = dist / max_dim

                if dist > PCK_THRESHOLD * max_dim:
                    # Highlight as wrong in red
                    cv2.circle(img, (px_px, py_px), 6, (0, 0, 255), 2)
                    cv2.putText(img, f"Wrong {KEYPOINT_NAMES[i]}", (px_px+5, py_px-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

                    # Save row in CSV
                    csv_rows.append([
                        img_path.name,
                        KEYPOINT_NAMES[i],
                        gx_px, gy_px,
                        px_px, py_px,
                        round(dist, 2),
                        round(rel_dist * 100, 2)
                    ])

    # Save visualization
    save_path = os.path.join(OUTPUT_DIR, img_path.name)
    cv2.imwrite(save_path, img)

# === SAVE CSV ===
with open(CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(csv_rows)

print(f"✅ Visualization saved in: {OUTPUT_DIR}")
print(f"✅ CSV report saved at: {CSV_PATH}")
