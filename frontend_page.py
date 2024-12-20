import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import os
import shutil
import cv2


def update_preview(obj_files, scale, light_intensity, render_option, x, y, z, angle):
    # change this to display an image. I can just displaythe image from rgb_all
    if obj_files is None:
        return "No file uploaded!"
    print(obj_files)
    obj_dir = "./input/"
    if os.path.exists(obj_dir):
        shutil.rmtree(obj_dir)

    os.makedirs(obj_dir, exist_ok=True)
    # for file in obj_files:
    temp_path = obj_files.name
    file_name = os.path.basename(temp_path)
    print(file_name)
    save_path = os.path.join(obj_dir, file_name)
    print(save_path)
    shutil.copy(temp_path, save_path)

    output_dir = './output/'
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    obj_positions = []
    obj_rotations = []
    # for obj_file in obj_files:
    #     obj_pos = np.array([float(x), float(y), float(z)])
    #     obj_positions.append(obj_pos)

    #     angle_rad = np.deg2rad(float(angle))
    #     obj_rot = np.array([
    #         [np.cos(angle_rad), -np.sin(angle_rad), 0.0],
    #         [np.sin(angle_rad), np.cos(angle_rad), 0.0],
    #         [0.0, 0.0, 1.0]
    #     ])
    #     obj_rotations.append(obj_rot)

    #     print(f"Processing file: {obj_file.name}")
    #     print(f"Object Position: {obj_pos}")
    #     print(f"Rotation Matrix: \n{obj_rot}")

    obj_pos = np.array([float(x), float(y), float(z)])
    obj_positions.append(obj_pos)

    angle_rad = np.deg2rad(float(angle))
    obj_rot = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad), 0.0],
        [np.sin(angle_rad), np.cos(angle_rad), 0.0],
        [0.0, 0.0, 1.0]
    ])
    obj_rotations.append(obj_rot)

    print(f"Processing file: {save_path}")
    print(f"Object Position: {obj_pos}")
    print(f"Rotation Matrix: \n{obj_rot}")

    script_path = os.path.abspath(os.path.join("utils", "blender_render.py"))
    print(script_path)
    blender_command = [
        r"C:/Program Files/Blender Foundation/Blender 4.2/blender",
        "--background",
        "--python", script_path,
        "--",
        temp_path,
        str(scale),
        str(light_intensity),
        render_option,
        str(x),
        str(y),
        str(z),
        str(angle),
        str(1)
    ]
    print("Blender Command:", blender_command)

    log_path = os.path.join(output_dir, "blender_error.log")
    with open(log_path, "w") as log_file:
        try:
            print(f"blender_command: {blender_command}")
            print(f"log_file: {log_file}")
            print(f"log_file type: {type(log_file)}")
            subprocess.run(blender_command, stdout=log_file, stderr=log_file, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Blender rendering failed. Check log file at {log_path}")
            if os.path.exists(log_path):
                with open(log_path, "r") as error_file:
                    print(error_file.read())
            return f"Blender rendering failed. Check log file at {log_path}"

    # display the result from rgb_all
    img_dir = './output/rgb_all/001.png'
    img_dir = os.path.abspath(img_dir)
    img = cv2.imread(img_dir)
    img_color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img_color


def render_object(obj_files, scale, light_intensity, render_option, x, y, z, angle):
    if obj_files is None:
        return "No file uploaded!"
    print(obj_files)
    obj_dir = "./input/"
    if os.path.exists(obj_dir):
        shutil.rmtree(obj_dir)

    os.makedirs(obj_dir, exist_ok=True)
    # for file in obj_files:
    temp_path = obj_files.name
    file_name = os.path.basename(temp_path)
    print(file_name)
    save_path = os.path.join(obj_dir, file_name)
    print(save_path)
    shutil.copy(temp_path, save_path)
    
    output_dir = './output/'
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    obj_positions = []
    obj_rotations = []
    # for obj_file in obj_files:
    #     obj_pos = np.array([float(x), float(y), float(z)])
    #     obj_positions.append(obj_pos)

    #     angle_rad = np.deg2rad(float(angle))
    #     obj_rot = np.array([
    #         [np.cos(angle_rad), -np.sin(angle_rad), 0.0],
    #         [np.sin(angle_rad), np.cos(angle_rad), 0.0],
    #         [0.0, 0.0, 1.0]
    #     ])
    #     obj_rotations.append(obj_rot)
        
    #     print(f"Processing file: {obj_file.name}")
    #     print(f"Object Position: {obj_pos}")
    #     print(f"Rotation Matrix: \n{obj_rot}")

    obj_pos = np.array([float(x), float(y), float(z)])
    obj_positions.append(obj_pos)

    angle_rad = np.deg2rad(float(angle))
    obj_rot = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad), 0.0],
        [np.sin(angle_rad), np.cos(angle_rad), 0.0],
        [0.0, 0.0, 1.0]
    ])
    obj_rotations.append(obj_rot)
        
    print(f"Processing file: {save_path}")
    print(f"Object Position: {obj_pos}")
    print(f"Rotation Matrix: \n{obj_rot}")    
    
    script_path = os.path.abspath(os.path.join("utils", "blender_render.py"))
    print(script_path)
    blender_command = [
        r"C:/Program Files/Blender Foundation/Blender 4.2/blender", #change this path to your blender.exe path
        "--background",
        "--python", script_path,
        "--",
        temp_path,
        str(scale),
        str(light_intensity),
        render_option,
        str(x),
        str(y),
        str(z),
        str(angle),
    ]
    print("Blender Command:", blender_command)

    log_path = os.path.join(output_dir, "blender_error.log")
    with open(log_path, "w") as log_file:
        try:
            print(f"blender_command: {blender_command}")
            print(f"log_file: {log_file}")
            print(f"log_file type: {type(log_file)}")
            subprocess.run(blender_command, stdout=log_file, stderr=log_file, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Blender rendering failed. Check log file at {log_path}")
            if os.path.exists(log_path):
                with open(log_path, "r") as error_file:
                    print(error_file.read())
            return f"Blender rendering failed. Check log file at {log_path}"
        
    composite_script = os.path.abspath("./utils/composite.py")
    python_command = ["python", composite_script]
    print("Composite Command:", python_command)

    composite_log_path = os.path.join(output_dir, "composite_error.log")
    with open(composite_log_path, "w") as log_file:
        try:
            subprocess.run(python_command, stdout=log_file, stderr=log_file, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Composite script failed. Check log file at {composite_log_path}")
            return f"Composite script failed. Check log file at {composite_log_path}"

    video_output = os.path.join(output_dir, "blended.mp4")
    if not os.path.exists(video_output):
        return "Composite script completed, but no video was generated."
    
    return video_output

    # return "done"
    
with gr.Blocks() as interface:
    
    with gr.Row():
        with gr.Column():
            obj_files = gr.File(
                label="Upload object File",
                file_types=[".obj", ".glb"]
                # file_count="multiple"
            )
            scale = gr.Slider(label="Scale", minimum=0.05, maximum=1.0, value=0.5)
            light_intensity = gr.Slider(label="Light Intensity", minimum=0.0, maximum=10.0, value=1.0)
            render_option = gr.Radio(choices=["Single Image"], label="Render Option")
            
            gr.Markdown("### Customize Object Position")
            x = gr.Textbox(label="X Position", value="0.20")
            y = gr.Textbox(label="Y Position", value="-0.03")
            z = gr.Textbox(label="Z Position", value="-0.45")
            angle = gr.Slider(label="Z-axis Rotation (degrees)", minimum=-180, maximum=180, value=0)
            #frames = gr.Textbox(label="Number of Frames to Render")
            
            render_button = gr.Button("Render")
        
        with gr.Column():
            preview_plot = gr.Image(label="Position and Rotation Preview")
            output_image = gr.Video(label="Rendered Output")

    scale.change(fn=update_preview, inputs=[obj_files, scale, light_intensity, render_option,
                                            x, y, z, angle], outputs=preview_plot)
    light_intensity.change(fn=update_preview, inputs=[obj_files, scale, light_intensity, render_option,
                                                      x, y, z, angle], outputs=preview_plot)
    x.change(fn=update_preview, inputs=[obj_files, scale, light_intensity, render_option,
                                        x, y, z, angle], outputs=preview_plot)
    y.change(fn=update_preview, inputs=[obj_files, scale, light_intensity, render_option,
                                        x, y, z, angle], outputs=preview_plot)
    z.change(fn=update_preview, inputs=[obj_files, scale, light_intensity, render_option,
                                        x, y, z, angle], outputs=preview_plot)
    angle.change(fn=update_preview, inputs=[obj_files, scale, light_intensity, render_option,
                                            x, y, z, angle], outputs=preview_plot)

    render_button.click(
        fn=render_object,
        inputs=[obj_files, scale, light_intensity, render_option, x, y, z, angle],
        outputs=output_image
    )

interface.launch()
