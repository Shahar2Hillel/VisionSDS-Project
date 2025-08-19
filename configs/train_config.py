from pathlib import Path

class Config:

    YOLO_DATASET_PATH = Path("yolo_dataset")
    OUTPUT_POSE_YAML = YOLO_DATASET_PATH / "pose.yaml"
    IMGSZ = 640
