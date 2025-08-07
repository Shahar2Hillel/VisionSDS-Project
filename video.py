import cv2
from ultralytics import YOLO

# === CONFIG ===
MODEL_PATH = "runs/pose/train8/weights/best.pt"  # path to your trained YOLOv8 pose model
VIDEO_PATH = "/datashare/HW1/ood_video_data/surg_1.mp4"    # input video path
OUTPUT_PATH = "output_video.mp4"                  # annotated output video

IMGSZ = 320
CONF_THRESHOLD = 0.3  # detection confidence threshold
keypoint_names = ["top_left", "top_right", "mid_left", "mid_right", "bottom_tip"]
TWO_STAGE = True

# === LOAD MODEL ===
model = YOLO(MODEL_PATH)

# === OPEN VIDEO ===
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise FileNotFoundError(f"Could not open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# === OUTPUT VIDEO WRITER ===
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))



def process_frame_single_stage(frame, model):
    annotated = frame.copy()
    results = model.predict(source=frame, conf=CONF_THRESHOLD, imgsz=IMGSZ, verbose=False)

    for res in results:
        if (res.keypoints is not None) and (res.keypoints.xy.shape[0] != 0):
            for idx, kp in enumerate(res.keypoints.xy[0]):
                x, y = map(int, kp.tolist())
                cv2.circle(annotated, (x, y), 4, (0, 255, 0), -1)
                if idx < len(keypoint_names):
                    cv2.putText(annotated, keypoint_names[idx], (x + 5, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        for box in res.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_name = model.names[int(box.cls[0])]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(annotated, cls_name, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    return annotated
    
    
def process_frame_two_stage(frame, model):
    annotated = frame.copy()
    det_results = model.predict(source=frame, conf=CONF_THRESHOLD, imgsz=IMGSZ, verbose=False)

    for det in det_results:
        for box in det.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cropped = frame[y1:y2, x1:x2]
            if cropped.size == 0:
                continue

            pose_res = model.predict(source=cropped, conf=CONF_THRESHOLD, imgsz=IMGSZ, verbose=False)[0]

            if (pose_res.keypoints is not None) and (pose_res.keypoints.xy.shape[0] != 0):
                for idx, kp in enumerate(pose_res.keypoints.xy[0]):
                  kpx, kpy = kp.tolist()
                  # Map to original coordinates
                  kpx_orig = int(kpx + x1)
                  kpy_orig = int(kpy + y1)

                  # Draw keypoint
                  cv2.circle(annotated, (kpx_orig, kpy_orig), 4, (0, 255, 0), -1)

                  if len(keypoint_names) > 0:
                      if idx < len(keypoint_names):
                          cv2.putText(
                              annotated,
                              keypoint_names[idx],
                              (kpx_orig + 5, kpy_orig - 5),
                              cv2.FONT_HERSHEY_SIMPLEX,
                              0.5,
                              (255, 255, 255),
                              1,
                              cv2.LINE_AA
                          )
            cls_name = model.names[int(box.cls[0])]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(annotated, cls_name, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    return annotated


print(f"Processing video: {VIDEO_PATH} ({width}x{height}, {fps} FPS, {total_frames} frames)")

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    if TWO_STAGE:
        annotated_frame = process_frame_two_stage(frame, model)
    else:
        annotated_frame = process_frame_single_stage(frame, model)
        
    # Write annotated frame to output video
    out.write(annotated_frame)
    frame_idx += 1

cap.release()
out.release()

print(f"✅ Saved annotated video to: {OUTPUT_PATH}")
