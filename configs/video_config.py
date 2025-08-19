from pathlib import Path

class Config:
    # Video Processing (video.py)

    VIDEO_INPUT = Path("/datashare/project/vids_test/4_2_24_A_1.mp4")
    FPS_SAMPLE = 30

    # # For synthetic video:
    # VIDEO_OUTPUT = Path("outputs/results_synthetic_only.mp4")
    # MODEL_PATH = Path("/home/student/Desktop/VisionSDS-Project/yolo_dataset/runs/pose/synthetic4/weights/best.pt")
    # IMGSZ = 1600
    
    # For refined model video:
    VIDEO_OUTPUT = Path("outputs/results_refined.mp4")
    MODEL_PATH = Path("/home/student/Desktop/VisionSDS-Project/yolo_dataset/runs/pose/refined3/weights/best.pt")
    IMGSZ = 640

    CONF_THRESH = 0.3
    

    KEYPOINT_NAMES = [
            "top_left","top_right","mid_left","mid_right","middle_left","middle_right",
            "bottom_tip","bottom_left","bottom_right","joint_center"
        ]
