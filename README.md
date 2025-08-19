# Computer Vision - Surgical Applications (0970222) Final Project
## Phase 1 -Synthetic Data Generator
Short guide to run `synthetic_data_generator.py` to make labeled images of surgical tools.

### Install (short)
1) `requirements.txt`:
   ```txt
   blenderproc>=2.6
   numpy>=1.23
   matplotlib>=3.6
   Pillow>=9.4
   ```
2) (Optional) env & first run:
   ```bash
   conda create -n synth python=3.10
   conda activate synth
   pip install blenderproc
   blenderproc quickstart
   pip install -r requirements.txt
   blenderproc --version
   ```

### Data needed
- 3D models (.obj) in: `needle_holder/`, `tweezers/`
- HDRIs: `haven/hdris/.../*.hdr`
- `camera.json` (fx, fy, cx, cy, width, height)
- **Backgrounds (choose ONE mode below)**

### Background modes (TWO VERSIONS)
- **Single background** — use an image named **`surg_background.png`**.  
  Configure its path with `--single_background` (defaults to `/home/student/Desktop/VisionSDS-Project/surg_background.png`).  
- **Multi background (random)** — use a folder via `--backgrounds_dir` (current approach).

### Generate data
**1) Single background (`surg_background.png`)**
```bash
blenderproc run synthetic_data_generator.py   --tools_dir /path/to/surgical_tools_models   --camera_params /path/to/camera.json   --output_dir DATASET_SINGLE_BG   --num_images 10   --categories needle_holder tweezers   --single_background /path/to/surg_background.png   --haven_path /path/to/haven/
```

**2) Multi background (random)**
```bash
blenderproc run synthetic_data_generator.py   --tools_dir /path/to/surgical_tools_models   --camera_params /path/to/camera.json   --output_dir DATASET_MULTI_BG   --num_images 10   --categories needle_holder tweezers   --backgrounds_dir /path/to/backgrounds   --haven_path /path/to/haven/
```

### Outputs
- `images_with_background/`, `visualizations_with_background/`
- `annotations.json`, `processing_summary.json`

### Reproduce
1) Install  
2) Place assets  
3) Run one of the commands  
4) Check outputs
   
### Visualizations

#### Dataset 1 Samples
<p float="left">
  <img src="10%20Visualizations%20To%20Submit/dataset1_1.png" width="200"/>
  <img src="10%20Visualizations%20To%20Submit/dataset1_2.png" width="200"/>
  <img src="10%20Visualizations%20To%20Submit/dataset1_3.png" width="200"/>
  <img src="10%20Visualizations%20To%20Submit/dataset1_4.png" width="200"/>
  <img src="10%20Visualizations%20To%20Submit/dataset1_5.png" width="200"/>
</p>

#### Approach 2 Results
<p float="left">
  <img src="10%20Visualizations%20To%20Submit/approach2_img1.png" width="200"/>
  <img src="10%20Visualizations%20To%20Submit/approach2_img2.png" width="200"/>
  <img src="10%20Visualizations%20To%20Submit/approach2_img3.png" width="200"/>
  <img src="10%20Visualizations%20To%20Submit/approach2_img4.png" width="200"/>
  <img src="10%20Visualizations%20To%20Submit/approach2_img5.png" width="200"/>
</p>

## Phase 2 - Model Training
### Preparing YOLO Dataset
To run the yolo dataset preperation -> moving into yolo format labeling & pose.yaml creation run
```bash
python prepare_dataset.py
```
For configuartions and additions of datasets from phase 1, add them in "BASE_DATA_FOLDERS" in
 ```bash
configs/prepare_dataset_config.py
```

### Training YOLO
Simply run:
```bash
python train_yolo.py
```

## Phase 3 - Model Refinement
### Refining model
To create psuedo labeled dataset and refine the model run:
```bash
python refine_model.py
```
To change configuartions go to:
 ```bash
configs/refine_config.py
```

# Inferences
## Running on Video:
Configure:
- MODEL_PATH (Chosen model - refined or synthetic) 
- VIDEO_OUTPUT
- VIDEO_INPUT
 ```bash
configs/video_config.py
```
Then run
 ```bash
python video.py
```

## Running on Image
Configure:
- MODEL_PATH (Chosen model - refined or synthetic) 
- IMAGE_PATH
- OUTPUT_PATH
 ```bash
configs/predict_config.py
```
Then run
 ```bash
python predict.py
```

# Project Outputs
Link to required outputs:
https://drive.google.com/drive/folders/1zE6NhAeuovcwQ8BlNARvn4m6TjiPOkAm?usp=sharing
- models:
   - synthetic.pt
   - refined.pt
- videos:
   -  results_synthetic_only.mp4
   -  results_refined.mp4
