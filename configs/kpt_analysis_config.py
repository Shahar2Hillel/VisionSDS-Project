from pathlib import Path

class Config:

    # Output directories (created by scripts)
    YOLO_DATASET_PATH = Path("yolo_dataset")
    IMAGES_TRAIN = YOLO_DATASET_PATH / "images" / "train"
    IMAGES_VAL   = YOLO_DATASET_PATH / "images" / "val"
    LABELS_TRAIN = YOLO_DATASET_PATH / "labels" / "train"
    LABELS_VAL   = YOLO_DATASET_PATH / "labels" / "val"
    VIS_TRAIN = YOLO_DATASET_PATH / "visualize" / "train"
    VIS_VAL = YOLO_DATASET_PATH / "visualize" / "val"
    OUTPUT_POSE_YAML = YOLO_DATASET_PATH / "pose.yaml"


    # Model configs:
    
    # For synthetic model:
    MODEL_PATH = Path("/home/student/Desktop/VisionSDS-Project/yolo_dataset/runs/pose/synthetic4/weights/best.pt")

    # For refined model:
    # MODEL_PATH = Path("/home/student/Desktop/VisionSDS-Project/yolo_dataset/runs/pose/refined3/weights/best.pt")

    
    CONF_THRESH = 0.3
    IMGSZ = 1600

    KEYPOINT_NAMES = [
            "top_left","top_right","mid_left","mid_right","middle_left","middle_right",
            "bottom_tip","bottom_left","bottom_right","joint_center"
        ]
    
    
    # Full Analysis (keypoint_full_analysis.py)
    CHECK_NAMES = {
            "top_left","top_right","mid_left","mid_right","middle_left","middle_right",
            "bottom_tip","bottom_left","bottom_right","joint_center"
    }
    
    PCK_THRESHOLD = 0.05
    SORT_BARS = True
    KEYPOINT_ANALYSIS_OUTPUT   =  Path("keypoint_error_analysis")
    KEYPOINT_ANALYSIS_OUTPUT_IMAGES = KEYPOINT_ANALYSIS_OUTPUT / "images"
    KEYPOINT_ANALYSIS_OUTPUT_CSV_FILE = KEYPOINT_ANALYSIS_OUTPUT / "wrong_keypoints.csv"
    KEYPOINT_ANALYSIS_OUTPUT_VIS = KEYPOINT_ANALYSIS_OUTPUT / "keypoint_error_analysis.png"
      
