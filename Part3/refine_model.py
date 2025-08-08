#!/usr/bin/env python3
"""
refine_model.py

Description:
    Phase 3: Unsupervised Model Refinement on Unlabeled Real Data

    This script implements a pseudo-labeling self-training pipeline:
      1. Extract frames from the provided surgical video(s).
      2. Run the synthetic-only pose model to generate keypoint predictions.
      3. Filter out low-confidence detections to form pseudo-labels.
      4. Write YOLOv8-format label files for pseudo-labeled frames.
      5. Merge the pseudo-labeled frames with the synthetic training set.
      6. Fine-tune the pose model on the combined dataset.

Usage:
    python refine_model.py
"""
import os
import cv2
from pathlib import Path
import numpy as np
from ultralytics import YOLO
import shutil

import yaml

from config import Config

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


# TODO - maybe smarter mechanism like HW1 to filter out low-confidence detections
# because if it detectes object and drops it due to low confidence,
# but there are keypoints that are above the threshold, then we have missinformation
def generate_pseudo_labels(model, label_dir: Path, conf_thresh: float, 
                           video_path: Path, out_dir: Path, vis_dir: Path,
                           fps_sample: int = 1):
    """
    Run model on sampled frames, filter by keypoint confidence,
    write labels and save visualizations showing class, keypoints, and conf.
    Returns list of label file paths.
    """
    label_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = max(1, int(fps / fps_sample))
    idx=saved=0
    labels=[]
    while True:
        ret, frame = cap.read()
        if not ret: break
        if idx % interval == 0:
            ih, iw = frame.shape[:2]
            results = model.predict(source=frame, conf=conf_thresh,
                                     imgsz=Config.IMGSZ, verbose=False)
            lines=[]
            # prepare vis copy
            vis = frame.copy()
            for res in results:
                if res.boxes.shape[0]==0: continue
                # take first det
                b=res.boxes[0]
                kp=res.keypoints.xy[0].cpu().numpy()
                cf=(res.keypoints.conf[0].cpu().numpy()
                    if hasattr(res.keypoints,'conf') else np.ones(len(kp)))
                x1,y1,x2,y2 = b.xyxy[0].cpu().numpy()
                cls_id=int(b.cls[0])
                # normalize box
                xc, yc = ((x1+x2)/2/iw), ((y1+y2)/2/ih)
                w, h = (x2-x1)/iw, (y2-y1)/ih
                # build keypoint vals
                vals=[]
                keep=False
                for (x,y),c in zip(kp,cf):
                    if c>=conf_thresh:
                        keep=True; vals.extend([x/iw,y/ih,2])
                    else: vals.extend([0,0,0])
                if not keep: continue
                # record label
                line = f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f} " + \
                       " ".join(f"{v:.6f}" for v in vals)
                lines.append(line)
                # draw box + class
                class_name = model.names[cls_id]
                cv2.rectangle(vis,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
                cv2.putText(vis,f"cls:{class_name}",(int(x1),int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
                # draw keypoints + conf
                for (x,y),c in zip(kp,cf):
                    px,py=int(x),int(y)
                    if c>=conf_thresh:
                        cv2.circle(vis,(px,py),4,(0,0,255),-1)
                        cv2.putText(vis,f"{c:.2f}",(px+5,py+5),
                                    cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,255),1)
            if lines:
                print(f"Frame {idx}: saving {len(lines)} labels & visualization")
                img_name=f"frame_{saved:06d}.jpg"
                img_path=out_dir/img_name
                cv2.imwrite(str(img_path),frame)
                # save visualization
                cv2.imwrite(str(vis_dir/img_name),vis)
                # write label
                lbl_file=label_dir/ f"{Path(img_name).stem}.txt"
                lbl_file.write_text("\n".join(lines)+"\n")
                labels.append(lbl_file)
                saved+=1
        idx+=1
    cap.release()
    print(f"Processed ~{(total+interval-1)//interval} frames, saved {saved} pseudo-labeled frames.")
    return labels

def merge_datasets(synth_root: Path, pseudo_frames: Path, pseudo_labels: Path):
    """
    Copy synthetic and pseudo-labeled data into merged/train and merged/val.
    """
    merged = Config.MERGED_DATASET_OUTPUT_PATH
    img_train = merged / 'images' / 'train'
    img_val   = merged / 'images' / 'val'
    lbl_train = merged / 'labels' / 'train'
    lbl_val   = merged / 'labels' / 'val'
    for d in [img_train, img_val, lbl_train, lbl_val]:
        d.mkdir(parents=True, exist_ok=True)
    # copy synthetic train & val
    for src in synth_root.glob('images/train/*'):
        shutil.copy(src, img_train / src.name)
    for src in synth_root.glob('images/val/*'):
        shutil.copy(src, img_val / src.name)
    for src in synth_root.glob('labels/train/*'):
        shutil.copy(src, lbl_train / src.name)
    for src in synth_root.glob('labels/val/*'):
        shutil.copy(src, lbl_val / src.name)
    # copy pseudo
    for src in pseudo_frames.glob('*.jpg'):
        shutil.copy(src, img_train / src.name)
    for src in pseudo_labels.glob('*.txt'):
        shutil.copy(src, lbl_train / src.name)


def create_merged_yaml(orig_yaml: Path):
    """
    Read the original pose.yaml, update paths to merged dataset, and write new yaml.
    """
    merged_root = Config.MERGED_DATASET_OUTPUT_PATH

    cfg = yaml.safe_load(orig_yaml.read_text())
    merged_rel = 'merged'
    cfg['path'] = str(merged_root)
    cfg['train'] = "images/train"
    cfg['val']   = "images/val"
    out = merged_root / 'pose.yaml'
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out


def fine_tune_model(base_model_path: Path, data_yaml: Path, output_dir: Path):
    """
    Fine-tune YOLOv8 pose model on combined dataset.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(base_model_path))
    model.train(
        data=str(data_yaml),
        imgsz=Config.IMGSZ,
        epochs=100,
        project=str(output_dir),
        name="refined",
        pretrained=True
    )
    return output_dir / "refined" / "weights" / "best.pt"


if __name__ == '__main__':
    pseudo_frames_dir = Config.DATASET_ROOT_PATH / "pseudo_frames"
    pseudo_label_dir = Config.DATASET_ROOT_PATH / "pseudo_labels"
    pseudo_vis_dir = Config.DATASET_ROOT_PATH / "pseudo_visualization"

    # 1) Extract frames and Generate pseudo labels
    print("Generating pseudo labels...")
    synth_model = YOLO(Config.MODEL_PATH)
    generate_pseudo_labels(synth_model, pseudo_label_dir, Config.HIGH_CONF_THRESH,
                           Config.VIDEO_INPUT, pseudo_frames_dir, pseudo_vis_dir, fps_sample=5)
    print(f"Generated pseudo labels in {pseudo_label_dir}")

    # # 2) Merge datasets
    # print("Merging datasets...")
    # merge_datasets(Config.DATASET_ROOT_PATH, pseudo_frames_dir, pseudo_label_dir)
    # print(f"✅ Merged dataset at: {Config.MERGED_DATASET_OUTPUT_PATH}")

    # # 3) Create merged pose.yaml
    # print("Creating merged pose.yaml...")
    # merged_yaml_path = create_merged_yaml(Config.ORIGINAL_POSE_YAML)
    # print(f"✅ Merged pose.yaml created at: {merged_yaml_path}")

    # # 4) Fine-tune
    # print("Fine-tuning model on merged dataset...")
    # refined_weights = fine_tune_model(
    #     Config.MODEL_PATH,
    #     merged_yaml_path,
    #     Config.MODEL_OUTPUT_PATH
    # )
    # print(f"✅ Refined weights saved to: {refined_weights}")
