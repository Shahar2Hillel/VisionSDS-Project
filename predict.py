#!/usr/bin/env python3
"""
YOLOv8‑Pose image inference script (predict.py)

Features
- Loads a YOLOv8 pose model (.pt) and runs inference on a single image or a folder of images
- Saves annotated images to --save-dir
- Optionally saves per-image JSON outputs and a global CSV with boxes + keypoints
- Optional dataset YAML to name classes and keypoints in outputs

"""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Dict, Any

import numpy as np
import cv2

try:
    import yaml  # type: ignore
except Exception:
    yaml = None

try:
    from ultralytics import YOLO
except Exception as e:
    raise SystemExit(
        "Ultralytics is required. Install with `pip install ultralytics` (or add to requirements.txt).\n"
        f"Import error: {e}"
    )

from configs.predict_config import Config

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_size(vals: List[str]) -> Tuple[int, int]:
    """Parse --imgsz: accepts one int (square) or two ints W H; also "960x544"."""
    if len(vals) == 1:
        tok = vals[0].lower().replace("x", ",").replace("*", ",")
        parts = [p for p in tok.split(",") if p]
        if len(parts) == 1:
            s = int(parts[0])
            return (s, s)
        elif len(parts) == 2:
            return (int(parts[0]), int(parts[1]))
        else:
            raise argparse.ArgumentTypeError("--imgsz expects 1 or 2 integers, e.g., 960 544 or 960x544")
    elif len(vals) == 2:
        return (int(vals[0]), int(vals[1]))
    else:
        raise argparse.ArgumentTypeError("--imgsz expects 1 or 2 values")


def load_dataset_meta(data_yaml: Optional[Path]) -> Tuple[List[str], List[str]]:
    """Return (class_names, keypoint_names). Missing entries will be synthesized.
    If yaml lib not available or file omitted, returns empty lists (caller will synthesize)."""
    classes: List[str] = []
    kpt_names: List[str] = []
    if data_yaml and data_yaml.exists() and yaml is not None:
        with open(data_yaml, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if isinstance(cfg, dict):
            classes = [str(n) for n in (cfg.get("names") or [])]
            kpt_names = [str(n) for n in (cfg.get("keypoint_names") or [])]
    return classes, kpt_names


def list_images(src: Path) -> List[Path]:
    if src.is_file() and src.suffix.lower() in IMG_EXTS:
        return [src]
    if src.is_dir():
        out: List[Path] = []
        for p in sorted(src.rglob("*")):
            if p.suffix.lower() in IMG_EXTS:
                out.append(p)
        return out
    raise FileNotFoundError(f"--source not found or unsupported: {src}")


def to_serializable(o: Any) -> Any:
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    return o


def ensure_dir(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)


def draw_and_save(image_bgr: np.ndarray, results_obj, save_path: Path, line_thickness: int = 2) -> None:
    # ultralytics Results.plot returns BGR np.ndarray with default styling
    annotated = results_obj.plot(line_width=line_thickness)
    cv2.imwrite(str(save_path), annotated)


def results_to_dicts(results_obj, classes: List[str], kpt_names: List[str]) -> List[Dict[str, Any]]:
    boxes = getattr(results_obj, "boxes", None)
    kpts = getattr(results_obj, "keypoints", None)

    dets: List[Dict[str, Any]] = []
    if boxes is None or len(boxes) == 0:
        return dets

    # Prepare names
    def cname(cidx: int) -> str:
        try:
            return classes[cidx]
        except Exception:
            return str(cidx)

    # Keypoint label helper
    def kp_label(i: int) -> str:
        try:
            return kpt_names[i]
        except Exception:
            return f"kpt_{i}"

    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)

    kpt_xy = None
    kpt_conf = None
    if kpts is not None and kpts.xy is not None:
        kpt_xy = kpts.xy.cpu().numpy()  # (N, K, 2)
        # Some versions expose .conf, else infer from .data if available
        try:
            kpt_conf = kpts.conf.cpu().numpy()  # (N, K)
        except Exception:
            kpt_conf = None

    for i in range(xyxy.shape[0]):
        item: Dict[str, Any] = {
            "bbox_xyxy": [float(v) for v in xyxy[i].tolist()],
            "score": float(conf[i]),
            "class_id": int(cls[i]),
            "class_name": cname(int(cls[i])),
        }
        if kpt_xy is not None:
            kps = []
            for k in range(kpt_xy.shape[1]):
                entry = {
                    "name": kp_label(k),
                    "x": float(kpt_xy[i, k, 0]),
                    "y": float(kpt_xy[i, k, 1]),
                }
                if kpt_conf is not None:
                    entry["conf"] = float(kpt_conf[i, k])
                kps.append(entry)
            item["keypoints"] = kps
        dets.append(item)
    return dets


def append_csv_rows(csv_path: Path, header: List[str], rows: List[List[Any]]) -> None:
    import csv
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        for r in rows:
            w.writerow(r)


def detections_to_csv_rows(
    image_path: Path,
    dets: List[Dict[str, Any]],
    kpt_names: List[str]
) -> Tuple[List[str], List[List[Any]]]:
    # Header: image, det_id, class, score, x1,y1,x2,y2, then per-kpt triples
    base_header = ["image", "det_id", "class_id", "class_name", "score", "x1", "y1", "x2", "y2"]
    k_headers: List[str] = []
    for i in range(len(kpt_names) or (len(dets[0]["keypoints"]) if dets and "keypoints" in dets[0] else 0)):
        nm = kpt_names[i] if i < len(kpt_names) else f"kpt_{i}"
        k_headers += [f"{nm}_x", f"{nm}_y", f"{nm}_conf"]
    header = base_header + k_headers

    rows: List[List[Any]] = []
    for j, d in enumerate(dets):
        x1, y1, x2, y2 = d["bbox_xyxy"]
        row = [str(image_path), j, d.get("class_id", -1), d.get("class_name", ""), d.get("score", 0.0), x1, y1, x2, y2]
        kps: List[Dict[str, Any]] = d.get("keypoints", [])
        for kp in kps:
            row += [kp.get("x", None), kp.get("y", None), kp.get("conf", None)]
        rows.append(row)
    return header, rows


def main():
    p = argparse.ArgumentParser(description="YOLOv8‑Pose inference on images")
    p.add_argument("--weights", type=Path, default=Config.MODEL_PATH, help="Path to YOLOv8 pose .pt weights")
    p.add_argument("--source", type=Path, default=Config.IMAGE_PATH, help="Image file or folder of images")
    p.add_argument("--imgsz", nargs="*", default=Config.IMGSZ, help="Inference size: one int or W H (e.g., 960 544 or 960x544)")
    p.add_argument("--conf", type=float, default=Config.CONF_THRESH, help="Confidence threshold")
    p.add_argument("--save-dir", type=Path, default=Config.OUTPUT_PATH, help="Output directory for annotated images + JSON")
    p.add_argument("--data", type=Path, default=Config.OUTPUT_POSE_YAML, help="Optional dataset YAML with names/keypoint_names for nicer outputs")
    
    p.add_argument("--save-json", action="store_true", help="Save per-image JSON with boxes and keypoints")
    p.add_argument("--save-csv", type=Path, default=Config.OUTPUT_CSV_PATH, help="Append all detections to a CSV at this path")

    args = p.parse_args()

    imgsz = parse_size(args.imgsz)
    classes_meta, kpt_names = load_dataset_meta(args.data)

    save_dir = args.save_dir
    ensure_dir(save_dir)

    # Load model
    model = YOLO(str(args.weights))

    # Gather images
    images = list_images(args.source)
    if not images:
        raise SystemExit("No images found to process.")

    # Inference loop
    all_rows: List[List[Any]] = []
    csv_header: Optional[List[str]] = None

    for img_path in images:
        # Ultralytics handles reading; for plotting we want original path
        results = model.predict(
            source=str(img_path),
            imgsz=imgsz,
            conf=args.conf,
            verbose=False,
            stream=False,
        )

        if not results:
            continue

        res = results[0]

        # Save annotated image
        out_img_path = save_dir / (img_path.stem + "_pred" + img_path.suffix)
        draw_and_save(None, res, out_img_path)

        # JSON per image
        if args.save_json:
            dets = results_to_dicts(res, classes_meta, kpt_names)
            json_path = save_dir / (img_path.stem + ".json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump({
                    "image": str(img_path),
                    "detections": dets,
                }, f, ensure_ascii=False, indent=2, default=to_serializable)

        # CSV aggregate
        if args.save_csv:
            dets = results_to_dicts(res, classes_meta, kpt_names)
            if dets:
                header, rows = detections_to_csv_rows(img_path, dets, kpt_names)
                if csv_header is None:
                    csv_header = header
                # ensure consistent header length across images
                # (pad rows if needed)
                target_len = len(csv_header)
                for r in rows:
                    if len(r) < target_len:
                        r += [None] * (target_len - len(r))
                all_rows.extend(rows)

    if args.save_csv and all_rows:
        append_csv_rows(args.save_csv, csv_header or [], all_rows)
        print(f"[CSV] appended {len(all_rows)} rows -> {args.save_csv}")

    print(f"Done. Annotated outputs saved to: {save_dir}")


if __name__ == "__main__":
    main()
