from pathlib import Path
from ultralytics import YOLO
from configs.train_config import Config

def fine_tune_model(data_yaml: Path, output_dir: Path):
    """
    Fine-tune YOLOv8 pose model on combined dataset.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO("yolo11m-pose.pt")

    params =  {'lr0': 0.003750276708276617,
               'lrf': 0.038226644823317016,
               'momentum': 0.9538781764210313,
               'weight_decay': 0.00037138566239414383,
               'batch': 8,
               'optimizer': 'AdamW',
               'imgsz': Config.IMGSZ}
    results = model.train(
        data=str(data_yaml),
        epochs=1000,
        degrees=30,
        flipud=0.5,
        fliplr=0.5,
        erasing=0.4,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        mosaic=1.0,
        mixup=0.2,
        project=str(output_dir),
        name="synthetic",
        **params
    )
    print(f"Model fine-tuned and saved to {output_dir / 'synthetic'}")

if __name__ == '__main__':
    fine_tune_model(
        data_yaml=Config.OUTPUT_POSE_YAML,
        output_dir=Config.YOLO_DATASET_PATH / "runs" / "pose"
    )