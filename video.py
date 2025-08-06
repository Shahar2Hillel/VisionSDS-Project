import cv2
from ultralytics import YOLO

# === CONFIG ===
MODEL_PATH = "runs/pose/train2/weights/best.pt"  # path to your trained YOLOv8 pose model
VIDEO_PATH = "/datashare/HW1/ood_video_data/surg_1.mp4"    # input video path
OUTPUT_PATH = "output_video.mp4"                  # annotated output video

CONF_THRESHOLD = 0.3  # detection confidence threshold

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

print(f"Processing video: {VIDEO_PATH} ({width}x{height}, {fps} FPS, {total_frames} frames)")

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 pose prediction
    results = model.predict(source=frame, conf=CONF_THRESHOLD, imgsz=416, verbose=False)

    # Draw results
    annotated_frame = results[0].plot()

    # Write annotated frame to output video
    out.write(annotated_frame)

    frame_idx += 1

cap.release()
out.release()

print(f"✅ Saved annotated video to: {OUTPUT_PATH}")
