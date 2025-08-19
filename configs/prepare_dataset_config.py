from pathlib import Path

class Config:
    # Dataset paths and split
    BASE_DATA_FOLDERS = [
        Path("OUTPUT_multi_tool_NEW_FULL_DATASET"),
        Path("OUTPUT_multi_tool_NEW_FULL_DATASET_PART2"),
        Path("NEW_VERSION_DATASET_FULL_09_08")
        ]


    # Train/val split ratio
    TRAIN_RATIO = 0.8

    # Output directories (created by scripts)
    YOLO_DATASET_PATH = Path("yolo_dataset")
    IMAGES_TRAIN = YOLO_DATASET_PATH / "images" / "train"
    IMAGES_VAL   = YOLO_DATASET_PATH / "images" / "val"
    LABELS_TRAIN = YOLO_DATASET_PATH / "labels" / "train"
    LABELS_VAL   = YOLO_DATASET_PATH / "labels" / "val"
    VIS_TRAIN = YOLO_DATASET_PATH / "visualize" / "train"
    VIS_VAL = YOLO_DATASET_PATH / "visualize" / "val"
    OUTPUT_POSE_YAML = YOLO_DATASET_PATH / "pose.yaml"

    IMGSZ = 1600
