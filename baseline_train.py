import json
import optuna
from ultralytics import YOLO

DATA_PATH = "pose.yaml"
IS_OPTUNA = False

model = YOLO("yolov8n-pose.pt")

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
    map50_95 = metrics.results_dict.get("metrics/mAP50-95(B)", 0.0)
    return map50_95

if not IS_OPTUNA:
    results = model.train(
        data="pose.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        workers=8,
        device=0,
        optimizer='Adam',
        lr0=0.01,
        pretrained=True     
    )
else:
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=15)
    print("Best trial:")
    trial = study.best_trial
    print(f"  mAP50-95: {trial.value}")
    print("  Best hyperparameters:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    with open("best_hyperparameters.json", "w") as f:
        json.dump(trial.params, f, indent=4)
    print("Best hyperparameters saved to 'best_hyperparameters.json'")    

