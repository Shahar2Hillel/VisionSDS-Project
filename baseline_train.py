import os
import json
import cv2
import optuna
import albumentations as A
from ultralytics import YOLO

# From the other files, we need the dataset root and YAML path
dataset_root = '../OUTPUT_multi_tool_NEW_FULL_DATASET'
yaml_output_path = os.path.join(dataset_root, "pose.yaml")

DATA_PATH = yaml_output_path
IS_OPTUNA = False  # Set to True if you want to use Optuna for hyperparameter tuning

model = YOLO("yolo11m-pose.pt")

# Transformations to help the data to look like a an opration
def get_custom_transforms():
    return A.Compose([
        # Random rotation within ±30° (already in your training params, here for completeness)
        A.Rotate(limit=30, p=0.5, border_mode=cv2.BORDER_CONSTANT),

        # Random occlusion inside bounding boxes
        A.BBoxSafeRandomCrop(erosion_rate=0.1, p=0.3),  # Crop bbox area randomly
        A.CoarseDropout(
            max_holes=4,
            max_height=0.3, max_width=0.3,  # relative to bbox
            min_height=0.1, min_width=0.1,
            fill_value=0,
            p=0.5
        ),

        # Random Gaussian blur
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),

        # Glare simulation
        A.RandomBrightnessContrast(
            brightness_limit=0.4, contrast_limit=0.3, p=0.5
        ),
        A.RandomShadow(
            shadow_roi=(0, 0, 1, 1), num_shadows_lower=1, num_shadows_upper=2,
            shadow_dimension=5, p=0.3
        ),
        A.RandomSunFlare(
            flare_roi=(0.2, 0.2, 0.8, 0.8), angle_lower=0.5, p=0.2
        ),

        # Color jitter (HSV)
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=50, val_shift_limit=40, p=0.5),

        # Flip
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
    ],
    bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']),
    keypoint_params=A.KeypointParams(format='yolo', remove_invisible=False)
    )



def objective(trial):
    lr0 = trial.suggest_float("lr0", 1e-4, 1e-1, log=True)   # Initial learning rate
    lrf = trial.suggest_float("lrf", 0.01, 1.0)               # Final LR fraction
    momentum = trial.suggest_float("momentum", 0.7, 0.98)
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.001)
    batch = trial.suggest_categorical("batch", [8, 16, 32])
    optimizer = trial.suggest_categorical("optimizer", ["SGD", "Adam", "AdamW"])
    imgsz = trial.suggest_categorical("imgsz", [416, 512, 640])
    results = model.train(
        data=DATA_PATH,
        epochs=50,
        imgsz=imgsz,
        batch=batch,
        optimizer=optimizer,
        lr0=lr0,
        lrf=lrf,
        momentum=momentum,
        weight_decay=weight_decay,
        device=0,
        verbose=False
    )

    metrics = model.val()
    map50_95 = metrics.results_dict.get("metrics/mAP50-95(P)", 0.0)
    return map50_95

if not IS_OPTUNA:
    params =  {'lr0': 0.003750276708276617,
               'lrf': 0.038226644823317016,
               'momentum': 0.9538781764210313,
               'weight_decay': 0.00037138566239414383,
               'batch': 8,
               'optimizer': 'AdamW',
               'imgsz': 416}
    custom_transforms = get_custom_transforms()
    results = model.train(
        data=DATA_PATH,
        epochs=150,
        pretrained=True,
        degrees=30,
        flipud=0.5,
        fliplr=0.5,
        erasing=0.4,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        mosaic=1.0,
        mixup=0.2,
        #transforms=custom_transforms,
        **params
    )
else:
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)
    print("Best trial:")
    trial = study.best_trial
    print(f"  mAP50-95: {trial.value}")
    print("  Best hyperparameters:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    with open("best_hyperparameters.json", "w") as f:
        json.dump(trial.params, f, indent=4)
    print("Best hyperparameters saved to 'best_hyperparameters.json'")    

