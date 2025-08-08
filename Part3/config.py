from pathlib import Path

class Config:
    """
    Centralized configuration for dataset prep, full analysis, and video processing.
    """
    # Dataset paths and split
    DATASET_ROOT_PATH = Path("output_combine")
    ANNOTATIONS_JSON = DATASET_ROOT_PATH / "annotations.json"

    # Source images (with background)
    IMAGES_SRC = DATASET_ROOT_PATH / "images_with_background"

    # Train/val split ratio
    TRAIN_RATIO = 0.8

    ORIGINAL_POSE_YAML = DATASET_ROOT_PATH / "pose.yaml"
    MERGED_DATASET_OUTPUT_PATH = DATASET_ROOT_PATH / "merged"

    # Model configs:

    MODEL_PATH = Path("VisionSDS-Project/runs_single_stage/pose/train8/weights/best.pt")
    
    HIGH_CONF_THRESH = 0.7
    IMGSZ = 320

    MODEL_OUTPUT_PATH = Path("VisionSDS-Project/runs_single_stage/pose/refined")

    # Video Processing (video.py)
    VIDEO_INPUT = Path("/datashare/HW1/ood_video_data/4_2_24_A_1.mp4")
    VIDEO_OUTPUT = Path("output_video.mp4")
    TWO_STAGE = False        
