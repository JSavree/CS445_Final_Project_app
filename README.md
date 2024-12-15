# CS445_Final_Project_app


### Dataset preparation:

I have prepared two gdrive links: [dataset](https://drive.google.com/drive/folders/1eRdSAqDloGXk04JK60v3io6GHWdomy2N?usp=sharing) and [output](https://drive.google.com/drive/folders/1KE8LSA_r-3f2LVlTLJ5k4SHENvbwdAfN?usp=drive_link). Basically, we need the following stuff for simulation:

- video frames (under `custom_camera_path/transforms_001/images` in [output](https://drive.google.com/drive/folders/1KE8LSA_r-3f2LVlTLJ5k4SHENvbwdAfN?usp=drive_link))
- corresponding camera poses (under `custom_camera_path/transforms_001/blender_cfg.json` [output](https://drive.google.com/drive/folders/1KE8LSA_r-3f2LVlTLJ5k4SHENvbwdAfN?usp=drive_link))
- surface meshes (under `mesh/mesh.obj` in [dataset](https://drive.google.com/drive/folders/1eRdSAqDloGXk04JK60v3io6GHWdomy2N?usp=sharing))
- hdr environmental lighting (under `hdr/transforms_001/00000_rotate.exr` in [output](https://drive.google.com/drive/folders/1KE8LSA_r-3f2LVlTLJ5k4SHENvbwdAfN?usp=drive_link))
- object asset to insert


### How to render:

I have prepared a Python script `utils/blender_render.py` for your reference. Specifically, it takes all stuff together for simulation and rendering. You need to first download [Blender](https://www.blender.org/download/release/Blender3.6/blender-3.6.11-linux-x64.tar.xz). And execute by this following command:
```bash
path/to/blender/blender-3.6.11-linux-x64/blender --background --python ./utils/blender_render.py
```
Note: you may need to change `config_path` in the `utils/blender_render.py` to the path to `blender_cfg.json`, and several folder/file paths within `blender_cfg.json`. 
Also, you will need to change the blender file path to whichever path stores your blender executable in the `frontend_page.py` file if you wish to run this code yourself with the ui. Specifically, you will need to change lines 69 and 166 for the blender commands.

### How to composite:

I have prepared a Python script `utils/composite.py` for your reference. Remember to also change several paths in the file. Then, simply run: 
```bash
python ./utils/composite.py
```