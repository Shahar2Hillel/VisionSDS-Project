# Computer Vision - Surgical Applications (0970222) Final Project
## Synthetic Data Generator
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
5) 
### Visualizations

#### Dataset 1 Samples
![dataset1_1](dataset1_1.png)
![dataset1_2](dataset1_2.png)
![dataset1_3](dataset1_3.png)
![dataset1_4](dataset1_4.png)
![dataset1_5](dataset1_5.png)

#### Approach 2 Results
![approach2_img1](approach2_img1.png)
![approach2_img2](approach2_img2.png)
![approach2_img3](approach2_img3.png)
![approach2_img4](approach2_img4.png)
![approach2_img5](approach2_img5.png)
