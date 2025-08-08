"""
pose_video_processing.py

Description:
    Processes a video using a YOLOv8 pose model in either single-stage or two-stage mode.
    - Single-stage: detect keypoints and boxes directly on each frame.
    - Two-stage: detect objects with a crop model, then run pose model on each crop.
    Annotates frames with bounding boxes and keypoint labels, and writes an output video.

Usage:
    python pose_video_processing.py

Configuration:
    Adjust MODEL_PATH, CROP_MODEL_PATH, VIDEO_PATH, and other parameters in main().
"""
import os
import cv2
from ultralytics import YOLO

from config import Config


def process_frame_single_stage(frame, model, keypoint_names, conf, imgsz):
    """
    Run pose detection on the full frame and annotate keypoints + boxes.
    """
    annotated = frame.copy()
    results = model.predict(source=frame, conf=conf, imgsz=imgsz, verbose=False)

    for res in results:
        # Draw keypoints
        if res.keypoints is not None and res.keypoints.xy.shape[0] > 0:
            for idx, (x, y) in enumerate(res.keypoints.xy[0].cpu().numpy()):
                xi, yi = int(x), int(y)
                cv2.circle(annotated, (xi, yi), 4, (0, 255, 0), -1)
                if idx < len(keypoint_names):
                    cv2.putText(
                        annotated,
                        keypoint_names[idx],
                        (xi + 5, yi - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
        # Draw boxes
        for box in res.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_name = model.names[int(box.cls[0])]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(
                annotated,
                cls_name,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )
    return annotated


def process_frame_two_stage(frame, detect_model, pose_model, keypoint_names, conf, imgsz):
    """
    Run object detection (crop stage), then pose detection per crop, annotate.
    """
    annotated = frame.copy()
    dets = detect_model.predict(source=frame, conf=conf, imgsz=imgsz, verbose=False)

    for det in dets:
        for box in det.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            # crop region and skip empty
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            # run pose on crop
            pose_res = pose_model.predict(source=crop, conf=conf, imgsz=imgsz, verbose=False)[0]
            # draw keypoints mapped to original
            if pose_res.keypoints is not None and pose_res.keypoints.xy.shape[0] > 0:
                for idx, (kx, ky) in enumerate(pose_res.keypoints.xy[0].cpu().numpy()):
                    kx_o = int(kx) + x1
                    ky_o = int(ky) + y1
                    cv2.circle(annotated, (kx_o, ky_o), 4, (0, 255, 0), -1)
                    if idx < len(keypoint_names):
                        cv2.putText(
                            annotated,
                            keypoint_names[idx],
                            (kx_o + 5, ky_o - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA,
                        )
            # draw box
            cls_name = detect_model.names[int(box.cls[0])]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(
                annotated,
                cls_name,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )
    return annotated


def main():
    
    # load models
    pose_model = YOLO(Config.MODEL_PATH)
    detect_model = YOLO(Config.CROP_MODEL_PATH) if Config.TWO_STAGE else pose_model

    # open video
    cap = cv2.VideoCapture(Config.VIDEO_INPUT)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {Config.VIDEO_INPUT}")
    fps    = cap.get(cv2.CAP_PROP_FPS)
    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(Config.VIDEO_OUTPUT, fourcc, fps, (w, h))

    print(f"Processing: {Config.VIDEO_INPUT} → {Config.VIDEO_OUTPUT} ({w}x{h}, {fps}FPS)")

    # frame loop
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if Config.TWO_STAGE:
            ann = process_frame_two_stage(frame,
                                          detect_model,
                                          pose_model,
                                          Config.KEYPOINT_NAMES,
                                          Config.CONF_THRESH,
                                          Config.IMGSZ)
        else:
            ann = process_frame_single_stage(frame,
                                             pose_model,
                                             Config.KEYPOINT_NAMES,
                                             Config.CONF_THRESH,
                                             Config.IMGSZ)
        out.write(ann)

    cap.release()
    out.release()
    print(f"✅ Saved: {Config.VIDEO_OUTPUT}")


if __name__ == "__main__":
    main()
