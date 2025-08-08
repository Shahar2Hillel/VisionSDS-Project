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

    # Output directories (created by scripts)
    IMAGES_TRAIN = DATASET_ROOT_PATH / "images" / "train"
    IMAGES_VAL   = DATASET_ROOT_PATH / "images" / "val"
    LABELS_TRAIN = DATASET_ROOT_PATH / "labels" / "train"
    LABELS_VAL   = DATASET_ROOT_PATH / "labels" / "val"
    OUTPUT_VIS   = DATASET_ROOT_PATH / "output_vis"
    OUTPUT_POSE_YAML = DATASET_ROOT_PATH / "pose.yaml"


    # Model configs:

    MODEL_PATH = Path("VisionSDS-Project/runs_single_stage/pose/train8/weights/best.pt")
    CROP_MODEL_PATH = Path("VisionSDS-Project/runs_single_stage/pose/train8/weights/best.pt")
    
    CONF_THRESH = 0.3
    IMGSZ = 320

    KEYPOINT_NAMES = ['bottom_left','bottom_right', 
                      'top_left', 'top_right'
                      'middle_left','middle_right'
                      ]
    
    
    # Full Analysis (keypoint_full_analysis.py)
    CHECK_NAMES = { 'bottom_left','bottom_right', 'top_left','top_right',
                   'middle_left','middle_right' }
    
    PCK_THRESHOLD = 0.05
    SORT_BARS = True
    KEYPOINT_ANALYSIS_OUTPUT_CSV_FILE = OUTPUT_VIS / "wrong_keypoints.csv"
    KEYPOINT_ANALYSIS_OUTPUT_VIS = OUTPUT_VIS / "keypoint_error_analysis.png"

    # Video Processing (video.py)
    VIDEO_INPUT = Path("/datashare/HW1/ood_video_data/surg_1.mp4")
    VIDEO_OUTPUT = Path("output_video.mp4")
    TWO_STAGE = False        
