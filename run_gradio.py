import gradio as gr


# Building Gradio UI

with gr.Blocks() as demo:
    gr.Markdown("# MegaClassifier")
    with gr.Row():
        model_dropdown = gr.Dropdown(
            ["None", "MegaDetectorV5"],
            label="Model",
            info="Select the model to use",
            value="None",
        )
        model_version_dropdown = gr.Dropdown(
            ["None"],
            label="Model version",
            info="Select the version of the model",
            value="None",
        )

    with gr.Column():
        load_but = gr.Button("Load Model")
        load_out = gr.Text("NO MODEL LOADED", label="Loaded model:")

    def update_version_dropdown(dropdown):
        if dropdown == "MegaDetectorV5":
            return gr.update(choices=["a", "b"], value="a", visible=True, interactive=True)
        else:
            return gr.update(choices=["None"], value="None", visible=False, interactive=False)

    model_dropdown.change(update_version_dropdown, model_dropdown, [model_version_dropdown])

    with gr.Tab("Single Image Process"):
        with gr.Row():
            with gr.Column():
                sgl_in = gr.Image(type="pil")
                sgl_conf_sl_det = gr.Slider(
                    0, 1, label="Detection Confidence Threshold", value=0.2
                )
            sgl_out = gr.Image()
        sgl_but = gr.Button("Detect Animals")


if __name__ == "__main__":
    demo.launch(share=True)
