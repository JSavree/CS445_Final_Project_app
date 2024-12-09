import gradio as gr
import subprocess
import tempfile
from PIL import Image
import os
import shutil

def run_blender_render(scene_mesh_file, light_intensity, resolution):
    if scene_mesh_file is None:
        return "No file uploaded!"

    mesh_dir = "./input/"
    os.makedirs(mesh_dir, exist_ok=True)
    temp_path = scene_mesh_file.name
    file_name = os.path.basename(temp_path)
    save_path = os.path.join(mesh_dir, file_name)
    shutil.copy(temp_path, save_path)
    
    output_dir = './output/'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "blended.mp4")

    script_path = os.path.abspath(os.path.join("utils", "blender_render.py"))
    scene_mesh_path = os.path.abspath(save_path)
    output_video_path = os.path.abspath(output_path)
    output_dir_path = os.path.abspath(output_dir)

    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Script file not found: {script_path}")

    blender_command = [
        r"C:/Program Files/Blender Foundation/Blender 4.2/blender.exe",
        "--background",
        "--python", script_path,
        "--",
        scene_mesh_path,
        str(resolution),
        str(light_intensity),
        # f"--render_option={render_option}",
        output_dir_path,
    ]
    print("Blender Command:", blender_command)

    log_path = os.path.join(output_dir, "blender_error.log")
    with open(log_path, "w") as log_file:
        try:
            subprocess.run(blender_command, stdout=log_file, stderr=log_file, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Blender rendering failed. Check log file at {log_path}")
            if os.path.exists(log_path):
                with open(log_path, "r") as error_file:
                    print(error_file.read())
            return f"Blender rendering failed. Check log file at {log_path}"
        
    composite_script_path = os.path.abspath("./utils/composite.py")
    try:
        subprocess.run(
            ["python", composite_script_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as e:
        return f"Composite process failed: {e.stderr.decode('utf-8')}"

    if not os.path.exists(output_video_path):
        return f"Rendering failed: No output generated. Check log file at {log_path}"

    return output_video_path


with gr.Blocks() as interface:
    gr.Markdown("# Blender Rendering Client")
    
    with gr.Row():
        with gr.Column():
            obj_file = gr.File(label="Upload Mesh File (.obj, .fbx, .glb)", file_types=[".obj", ".fbx", ".glb"])
            resolution = gr.Slider(label="Resolution", minimum=256, maximum=2048, value=256.0)
            light_intensity = gr.Slider(label="Light Intensity", minimum=0.0, maximum=10.0, value=1.0)
            # render_option = gr.Radio(choices=["SINGLE_VIEW", "MULTI_VIEW"], label="Render Option")
            render_button = gr.Button("Render")
        
        with gr.Column():
            output_video = gr.Video(label="Rendered Output")

    render_button.click(
        fn=run_blender_render, # put render function here
        inputs=[obj_file,light_intensity, resolution],
        outputs=output_video
    )

interface.launch()
