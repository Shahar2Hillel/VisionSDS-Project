from pathlib import Path

class Config:
    MODEL_PATH = Path("/home/student/Desktop/VisionSDS-Project/yolo_dataset/runs/pose/synthetic4/weights/best.pt")
    CONF_THRESH = 0.3
    IMGSZ = ['640']

    YOLO_DATASET_PATH = Path("yolo_dataset")
    OUTPUT_POSE_YAML = YOLO_DATASET_PATH / "pose.yaml"
        
    IMAGE_PATH = YOLO_DATASET_PATH / 'images' / 'val' / 'data_folder1_multi_T1_NH5_000038.png'
    OUTPUT_PATH = Path("outputs")
    OUTPUT_CSV_PATH = OUTPUT_PATH / f'{IMAGE_PATH.stem}.csv'
