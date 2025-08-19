from pathlib import Path

class Config:
    """
    Centralized configuration for dataset prep, full analysis, and video processing.
    """
    # Dataset paths and split
    DATASET_ROOT_PATH = Path("yolo_dataset")


    # Train/val split ratio
    TRAIN_RATIO = 0.8

    ORIGINAL_POSE_YAML = DATASET_ROOT_PATH / "pose.yaml"
    MERGED_POSE_YAML = DATASET_ROOT_PATH / "pose_merged.yaml"

    # Model configs:

    MODEL_PATH = Path("/home/student/Desktop/VisionSDS-Project/yolo_dataset/runs/pose/synthetic4/weights/best.pt")
    
    HIGH_CONF_THRESH = 0.7
    IMGSZ = 640

    MODEL_OUTPUT_PATH = Path("/home/student/Desktop/VisionSDS-Project/yolo_dataset_new/runs/pose")

    # Video Processing (video.py)
    VIDEOS_INPUT = [
        # Path("/datashare/project/vids_tune/20_2_24_1.mp4"),
        # Path("/datashare/project/vids_tune/4_2_24_B_2.mp4"),
        Path("/datashare/project/vids_test/4_2_24_A_1.mp4"),
    ]
