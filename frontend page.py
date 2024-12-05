import gradio as gr


with gr.Blocks() as interface:
    gr.Markdown("# Blender Rendering Client")
    
    with gr.Row():
        with gr.Column():
            obj_file = gr.File(label="Upload .obj File")
            scale = gr.Slider(label="Scale", minimum=0.1, maximum=2.0, value=1.0)
            light_intensity = gr.Slider(label="Light Intensity", minimum=0.0, maximum=10.0, value=1.0)
            render_option = gr.Radio(choices=["Single Image", "Normal Map", "Depth"], label="Render Option")
            render_button = gr.Button("Render")
        
        with gr.Column():
            output_image = gr.Image(label="Rendered Output")

    render_button.click(
        # fn=render_object, # put render function here
        inputs=[obj_file, scale, light_intensity, render_option],
        outputs=output_image
    )

interface.launch()
