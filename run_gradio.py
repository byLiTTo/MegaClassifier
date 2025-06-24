import gradio as gr
import torch
import numpy as np
import supervision as sv

from MegaClassifier.models import detection as detection_models

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODELS = [
    "None",
    "MegaDetectorV5",
]
VERSIONS = {
    "None": ["None"],
    "MegaDetectorV5": ["a", "b"],
}

# Initializing the model
model = None

# Initializing a supervision box annotator for visualizing detections
dot_annotator = sv.DotAnnotator(radius=3)
box_annotator = sv.BoxAnnotator(thickness=2)
lab_annotator = sv.LabelAnnotator(text_color=sv.Color.BLACK, text_thickness=2, text_scale=1)


def load_model(model_name, version):
    global model
    if model_name != "None":
        if model_name == "MegaDetectorV5":
            model = detection_models.__dict__[model_name](
                device=DEVICE, pretrained=True, version=version
            )
    else:
        model = None
        return ""
    return f"{model_name}.{version}"


def single_image_detection(input_img, det_conf_thres, img_index=None):
    input_img = np.array(input_img)
    annotator = box_annotator

    results = model.single_image_detection(input_img, img_path=img_index, det_conf_thres=det_conf_thres)

    labels = results["labels"]
    annotated_img = lab_annotator.annotate(
        scene=annotator.annotate(
            scene=input_img,
            detections=results["detections"],
        ),
        detections=results["detections"],
        labels=labels,
    )

    return annotated_img


# Building Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# MegaClassifier")
    with gr.Row():
        model_dropdown = gr.Dropdown(
            MODELS,
            label="Model",
            info="Select the model to use",
            value=MODELS[0],
        )
        model_version_dropdown = gr.Dropdown(
            [MODELS[0]],
            label="Model version",
            info="Select the version of the model",
            value=VERSIONS[MODELS[0]][0],
        )
    with gr.Row():
        with gr.Column():
            load_out = gr.Text("", label="Loaded model:")
        load_but = gr.Button("LOAD")


    def update_version_dropdown(dropdown):
        if dropdown == MODELS[1]:
            return [
                gr.update(
                    choices=VERSIONS[MODELS[1]],
                    value=VERSIONS[MODELS[1]][0],
                    visible=True,
                    interactive=True,
                ),
                gr.Slider(0, 1, label="Detection Confidence Threshold", value=0.2),
            ]
        else:
            return [
                gr.update(
                    choices=VERSIONS[MODELS[0]],
                    value=VERSIONS[MODELS[0]][0],
                    visible=False,
                    interactive=False,
                ),
                gr.Slider(0, 1, label="Detection Confidence Threshold", value=0.2),
            ]


    with gr.Tab("Single Image Process"):
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil")
                threshold_slider = gr.Slider(
                    0, 1, label="Detection Confidence Threshold", value=0.2
                )
            image_output = gr.Image()
        image_button = gr.Button("Detect Animals")

    model_dropdown.change(
        update_version_dropdown, model_dropdown, [model_version_dropdown, threshold_slider]
    )
    load_but.click(
        load_model,
        inputs=[model_dropdown, model_version_dropdown],
        outputs=load_out,
    )
    image_button.click(single_image_detection, inputs=[image_input, threshold_slider], outputs=image_output)

if __name__ == "__main__":
    demo.launch(share=True)
