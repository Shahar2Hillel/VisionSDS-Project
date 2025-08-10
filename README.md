# Computer Vision - Surgical Applications (0970222) Final Project
## Synthetic Data Generator
Short guide to set up and run `synthetic_data_generator.py` to make labeled images of surgical tools.
### Install
1) Create `requirements.txt` with:
   ```txt
   blenderproc>=2.6
   numpy>=1.23
   matplotlib>=3.6
   Pillow>=9.4
   ```
2) (Optional) Virtual env:
   ```bash
   conda create -n synth python=3.10
   conda activate synth
   pip install blenderproc
   blenderproc quickstart
   ```
3) Install:
   ```bash
   pip install -r requirements.txt
   blenderproc --version   # first run may download Blender bundle
   ```

### Data needed
- 3D models (.obj) in folders: `needle_holder/`, `tweezers/`
- Background images folder
- HDRIs: `haven/hdris/.../*.hdr`
- `camera.json` with fx, fy, cx, cy, width, height

### Generate data
```bash
blenderproc run synthetic_data_generator.py   --tools_dir /path/to/surgical_tools_models   --camera_params /path/to/camera.json   --output_dir DATASET_01   --num_images 10   --categories needle_holder tweezers   --backgrounds_dir /path/to/backgrounds   --haven_path /path/to/haven/
```

### Outputs
- `images_with_background/`
- `visualizations_with_background/`
- `annotations.json`, `processing_summary.json`

### Reproduce
1) Install (see above)
2) Place assets (see Data needed)
3) Run the command (see Generate data)
4) Check outputs (see Outputs)
