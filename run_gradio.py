import gradio as gr


# Building Gradio UI

with gr.Blocks() as demo:
    gr.Markdown("# MegaClassifier")
    with gr.Row():
        det_drop = gr.Dropdown(
            ["None", "MegaDetectorV5"],
            label="Detection model",
            info="Will add more detection models!",
            value="None",  # Default
        )
        det_version = gr.Dropdown(
            ["None"],
            label="Model version",
            info="Select the version of the model",
            value="None",
        )

    with gr.Column():
        load_but = gr.Button("Load Models!")
        load_out = gr.Text("NO MODEL LOADED!!", label="Loaded models:")

    def update_ui_elements(det_model):
        if det_model == "MegaDetectorV5":
            return gr.Dropdown(
                choices=["a", "b"], interactive=True, label="Model version", value="a"
            ), gr.update(visible=True)
        else:
            return gr.Dropdown(
                choices=["None"], interactive=True, label="Model version", value="None"
            ), gr.update(value="None", visible=False)

    det_drop.change(update_ui_elements, det_drop, [det_version])

    with gr.Tab("Single Image Process"):
        with gr.Row():
            with gr.Column():
                sgl_in = gr.Image(type="pil")
                sgl_conf_sl_det = gr.Slider(
                    0, 1, label="Detection Confidence Threshold", value=0.2
                )
            sgl_out = gr.Image()
        sgl_but = gr.Button("Detect Animals!")

if __name__ == "__main__":
    demo.launch(share=True)
