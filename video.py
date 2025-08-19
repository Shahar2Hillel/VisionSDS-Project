"""
pose_video_processing.py

Description:
    Processes a video using a YOLOv8 pose model in either single-stage or two-stage mode.
    Single-stage: detect keypoints and boxes directly on each frame.
    Annotates frames with bounding boxes and keypoint labels, and writes an output video.

Usage:
    python pose_video_processing.py

Configuration:
    Adjust MODEL_PATH, CROP_MODEL_PATH, VIDEO_PATH, and other parameters in main().
"""
import os
import cv2
from ultralytics import YOLO

from configs.video_config import Config
import numpy as np
import cv2


def process_frame_single_stage(
    frame,
    model,
    keypoint_names,
    conf,
    imgsz=None,        # set to None to avoid any internal resize hint
    kpt_thr=0.25,
    skeleton=None,
    box_margin=0       # pixels to expand the "inside box" test
):
    """
    Draw ONLY keypoints and skeleton segments that fall INSIDE each detection box.
    No ratio mapping or resizing — assumes model outputs are in the same pixel space
    as the input frame (Ultralytics does this by default).
    """
    if skeleton is None:
        skeleton = [
            (0, 1), (0, 2), (1, 3),
            (2, 4), (3, 5),
            (4, 6), (5, 6),
            (7, 8),
            (4, 7), (5, 8),
            (4, 9), (5, 9),
        ]

    annotated = frame.copy()

    # Predict without any manual resizing logic
    if imgsz is None:
        results = model.predict(annotated, conf=conf, verbose=False)
    else:
        results = model.predict(annotated, conf=conf, imgsz=imgsz, verbose=False)

    for res in results:
        print(f"Detected {len(res.boxes)} boxes, {len(res.keypoints)} keypoints")
        # --- Boxes ---
        if getattr(res, "boxes", None) is not None and len(res.boxes):
            boxes_xyxy = res.boxes.xyxy.cpu().numpy()           # (M,4) in original frame coords
            classes = res.boxes.cls.cpu().numpy().astype(int).ravel()  # (M,)
        else:
            boxes_xyxy = np.empty((0, 4), dtype=float)
            classes = np.empty((0,), dtype=int)

        # --- Keypoints ---
        has_kpts = (getattr(res, "keypoints", None) is not None and
                    getattr(res.keypoints, "xy", None) is not None)
        if has_kpts:
            kpts_xy_all = res.keypoints.xy.cpu().numpy()  # (N,K,2) in original frame coords
            kpts_cf_all = (res.keypoints.conf.cpu().numpy()
                           if res.keypoints.conf is not None
                           else np.ones(kpts_xy_all.shape[:2], dtype=np.float32))  # (N,K)
        else:
            kpts_xy_all = np.empty((0, 0, 2), dtype=float)
            kpts_cf_all = np.empty((0, 0), dtype=float)

        # Only pair up as many instances as both sides have
        N = min(kpts_xy_all.shape[0], boxes_xyxy.shape[0])

        # If no matched instances: draw boxes (if any) and skip keypoints (enforces "only within box")
        if N == 0:
            for bi in range(boxes_xyxy.shape[0]):
                x1, y1, x2, y2 = boxes_xyxy[bi]
                p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
                cv2.rectangle(annotated, p1, p2, (0, 255, 255), 2)
                if hasattr(model, "names") and 0 <= classes[bi] < len(model.names):
                    cls_name = model.names[int(classes[bi])]
                    cv2.putText(annotated, cls_name, (p1[0], max(0, p1[1] - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            continue

        for i in range(N):
            x1, y1, x2, y2 = boxes_xyxy[i]
            # Optional margin around the box for inclusion
            x1m, y1m, x2m, y2m = x1 - box_margin, y1 - box_margin, x2 + box_margin, y2 + box_margin

            cls_name = (model.names[int(classes[i])]
                        if hasattr(model, "names") and 0 <= classes[i] < len(model.names)
                        else str(int(classes[i])))

            person_xy = kpts_xy_all[i]   # (K,2)
            person_cf = kpts_cf_all[i]   # (K,)

            # Keep only keypoints INSIDE the (possibly margin-expanded) box
            inside = (person_xy[:, 0] >= x1m) & (person_xy[:, 0] <= x2m) & \
                     (person_xy[:, 1] >= y1m) & (person_xy[:, 1] <= y2m)

            # Draw skeleton (both endpoints inside & confident)
            K = person_xy.shape[0]
            for a, b in skeleton:
                if a < K and b < K and inside[a] and inside[b] and \
                   person_cf[a] >= kpt_thr and person_cf[b] >= kpt_thr:
                    xA, yA = person_xy[a]
                    xB, yB = person_xy[b]
                    cv2.line(annotated, (int(xA), int(yA)), (int(xB), int(yB)),
                             (0, 200, 255), 2, cv2.LINE_AA)

            # Draw keypoints (inside + confident)
            for idx, (x, y) in enumerate(person_xy):
                if not inside[idx] or person_cf[idx] < kpt_thr:
                    continue
                cv2.circle(annotated, (int(x), int(y)), 4, (0, 255, 0), -1)
                if idx < len(keypoint_names):
                    cv2.putText(annotated, keypoint_names[idx], (int(x) + 5, int(y) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # Draw box last
            p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
            cv2.rectangle(annotated, p1, p2, (0, 255, 255), 2)
            if cls_name:
                cv2.putText(annotated, cls_name, (p1[0], max(0, p1[1] - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    return annotated


def main():
    os.makedirs(Config.VIDEO_OUTPUT.parent, exist_ok=True)

    pose_model = YOLO(Config.MODEL_PATH)

    # open video
    cap = cv2.VideoCapture(Config.VIDEO_INPUT)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {Config.VIDEO_INPUT}")
    fps    = cap.get(cv2.CAP_PROP_FPS)
    interval = max(1, int(fps / Config.FPS_SAMPLE))
    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(Config.VIDEO_OUTPUT, fourcc, fps, (w, h))

    print(f"Processing: {Config.VIDEO_INPUT} → {Config.VIDEO_OUTPUT} ({w}x{h}, {fps}FPS)")

    # frame loop
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval != 0:
            idx += 1
            continue

        ann = process_frame_single_stage(frame,
                                            pose_model,
                                            Config.KEYPOINT_NAMES,
                                            Config.CONF_THRESH,
                                            Config.IMGSZ)
        out.write(ann)
        idx += 1
    cap.release()
    out.release()
    print(f"✅ Saved: {Config.VIDEO_OUTPUT}")


if __name__ == "__main__":
    main()
