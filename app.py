import argparse
import os

import gradio as gr
import huggingface_hub
import numpy as np
import onnxruntime as rt
import pandas as pd
from PIL import Image
import traceback
import tempfile
import zipfile
from datetime import datetime

TITLE = "WaifuDiffusion Tagger"
DESCRIPTION = """
Demo for the WaifuDiffusion tagger models

Example image by [ほし☆☆☆](https://www.pixiv.net/en/users/43565085)
"""

# Dataset v3 series of models:
SWINV2_MODEL_DSV3_REPO = "SmilingWolf/wd-swinv2-tagger-v3"
CONV_MODEL_DSV3_REPO = "SmilingWolf/wd-convnext-tagger-v3"
VIT_MODEL_DSV3_REPO = "SmilingWolf/wd-vit-tagger-v3"
VIT_LARGE_MODEL_DSV3_REPO = "SmilingWolf/wd-vit-large-tagger-v3"
EVA02_LARGE_MODEL_DSV3_REPO = "SmilingWolf/wd-eva02-large-tagger-v3"

# Dataset v2 series of models:
MOAT_MODEL_DSV2_REPO = "SmilingWolf/wd-v1-4-moat-tagger-v2"
SWIN_MODEL_DSV2_REPO = "SmilingWolf/wd-v1-4-swinv2-tagger-v2"
CONV_MODEL_DSV2_REPO = "SmilingWolf/wd-v1-4-convnext-tagger-v2"
CONV2_MODEL_DSV2_REPO = "SmilingWolf/wd-v1-4-convnextv2-tagger-v2"
VIT_MODEL_DSV2_REPO = "SmilingWolf/wd-v1-4-vit-tagger-v2"

# Files to download from the repos
MODEL_FILENAME = "model.onnx"
LABEL_FILENAME = "selected_tags.csv"

# https://github.com/toriato/stable-diffusion-webui-wd14-tagger/blob/a9eacb1eff904552d3012babfa28b57e1d3e295c/tagger/ui.py#L368
kaomojis = [
    "0_0",
    "(o)_(o)",
    "+_+",
    "+_-",
    "._.",
    "<o>_<o>",
    "<|>_<|>",
    "=_=",
    ">_<",
    "3_3",
    "6_9",
    ">_o",
    "@_@",
    "^_^",
    "o_o",
    "u_u",
    "x_x",
    "|_|",
    "||_||",
]

tag_results = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-slider-step", type=float, default=0.05)
    parser.add_argument("--score-general-threshold", type=float, default=0.35)
    parser.add_argument("--score-character-threshold", type=float, default=0.85)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def load_labels(dataframe) -> list[str]:
    name_series = dataframe["name"]
    name_series = name_series.map(
        lambda x: x.replace("_", " ") if x not in kaomojis else x
    )
    tag_names = name_series.tolist()

    rating_indexes = list(np.where(dataframe["category"] == 9)[0])
    general_indexes = list(np.where(dataframe["category"] == 0)[0])
    character_indexes = list(np.where(dataframe["category"] == 4)[0])
    return tag_names, rating_indexes, general_indexes, character_indexes


def mcut_threshold(probs):
    """
    Maximum Cut Thresholding (MCut)
    Largeron, C., Moulin, C., & Gery, M. (2012). MCut: A Thresholding Strategy
     for Multi-label Classification. In 11th International Symposium, IDA 2012
     (pp. 172-183).
    """
    sorted_probs = probs[probs.argsort()[::-1]]
    difs = sorted_probs[:-1] - sorted_probs[1:]
    t = difs.argmax()
    thresh = (sorted_probs[t] + sorted_probs[t + 1]) / 2
    return thresh


class Predictor:
    def __init__(self):
        self.model_target_size = None
        self.last_loaded_repo = None

    def download_model(self, model_repo):
        csv_path = huggingface_hub.hf_hub_download(
            model_repo,
            LABEL_FILENAME,
        )
        model_path = huggingface_hub.hf_hub_download(
            model_repo,
            MODEL_FILENAME,
        )
        return csv_path, model_path

    def load_model(self, model_repo):
        if model_repo == self.last_loaded_repo:
            return

        csv_path, model_path = self.download_model(model_repo)

        tags_df = pd.read_csv(csv_path)
        sep_tags = load_labels(tags_df)

        self.tag_names = sep_tags[0]
        self.rating_indexes = sep_tags[1]
        self.general_indexes = sep_tags[2]
        self.character_indexes = sep_tags[3]

        model = rt.InferenceSession(model_path)
        _, height, width, _ = model.get_inputs()[0].shape
        self.model_target_size = height

        self.last_loaded_repo = model_repo
        self.model = model

    def prepare_image(self, path):
        image = Image.open(path)
        image = image.convert("RGBA")
        target_size = self.model_target_size

        canvas = Image.new("RGBA", image.size, (255, 255, 255))
        canvas.alpha_composite(image)
        image = canvas.convert("RGB")

        # Pad image to square
        image_shape = image.size
        max_dim = max(image_shape)
        pad_left = (max_dim - image_shape[0]) // 2
        pad_top = (max_dim - image_shape[1]) // 2

        padded_image = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
        padded_image.paste(image, (pad_left, pad_top))

        # Resize
        if max_dim != target_size:
            padded_image = padded_image.resize(
                (target_size, target_size),
                Image.BICUBIC,
            )

        # Convert to numpy array
        image_array = np.asarray(padded_image, dtype=np.float32)

        # Convert PIL-native RGB to BGR
        image_array = image_array[:, :, ::-1]

        return np.expand_dims(image_array, axis=0)

    def create_file(self, text: str, directory: str, fileName: str) -> str:
        # Write the text to a file
        with open(os.path.join(directory, fileName), 'w+', encoding="utf-8") as file:
            file.write(text)

        return file.name

    def predict(
        self,
        gallery,
        model_repo,
        general_thresh,
        general_mcut_enabled,
        character_thresh,
        character_mcut_enabled,
        characters_merge_enabled,
        additional_tags_prepend,
        additional_tags_append,
    ):
        self.load_model(model_repo)
        # Result
        txt_infos = []
        output_dir = tempfile.mkdtemp()
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        sorted_general_strings = ""
        rating = None
        character_res = None
        general_res = None

        tag_results.clear()

        prepend_list = [tag.strip() for tag in additional_tags_prepend.split(",") if tag.strip()]
        append_list = [tag.strip() for tag in additional_tags_append.split(",") if tag.strip()]
        if prepend_list and append_list:
            append_list = [item for item in append_list if item not in prepend_list]

        for idx, value in enumerate(gallery):
            try:
                image_path = value[0]
                image_name = os.path.splitext(os.path.basename(image_path))[0]

                image = self.prepare_image(image_path)

                input_name = self.model.get_inputs()[0].name
                label_name = self.model.get_outputs()[0].name
                preds = self.model.run([label_name], {input_name: image})[0]

                labels = list(zip(self.tag_names, preds[0].astype(float)))

                # First 4 labels are actually ratings: pick one with argmax
                ratings_names = [labels[i] for i in self.rating_indexes]
                rating = dict(ratings_names)

                # Then we have general tags: pick any where prediction confidence > threshold
                general_names = [labels[i] for i in self.general_indexes]

                if general_mcut_enabled:
                    general_probs = np.array([x[1] for x in general_names])
                    general_thresh = mcut_threshold(general_probs)

                general_res = [x for x in general_names if x[1] > general_thresh]
                general_res = dict(general_res)

                # Everything else is characters: pick any where prediction confidence > threshold
                character_names = [labels[i] for i in self.character_indexes]

                if character_mcut_enabled:
                    character_probs = np.array([x[1] for x in character_names])
                    character_thresh = mcut_threshold(character_probs)
                    character_thresh = max(0.15, character_thresh)

                character_res = [x for x in character_names if x[1] > character_thresh]
                character_res = dict(character_res)
                character_list = list(character_res.keys())

                sorted_general_list = sorted(
                    general_res.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )
                sorted_general_list = [x[0] for x in sorted_general_list]
                #Remove values from character_list that already exist in sorted_general_list
                character_list = [item for item in character_list if item not in sorted_general_list]
                #Remove values from sorted_general_list that already exist in prepend_list or append_list
                if prepend_list:
                    sorted_general_list = [item for item in sorted_general_list if item not in prepend_list]
                if append_list:
                    sorted_general_list = [item for item in sorted_general_list if item not in append_list]

                sorted_general_strings = ", ".join((character_list if characters_merge_enabled else []) + prepend_list + sorted_general_list + append_list).replace("(", "\(").replace(")", "\)")

                txt_file = self.create_file(sorted_general_strings, output_dir, image_name + ".txt")
                txt_infos.append({"path":txt_file, "name": image_name + ".txt"})

                tag_results[image_path] = { "strings": sorted_general_strings, "rating": rating, "character_res": character_res, "general_res": general_res }

            except Exception as e:
                print(traceback.format_exc())
                print("Error predict: " + str(e))

        # Result
        download = []
        if txt_infos is not None and len(txt_infos) > 0:
            downloadZipPath = os.path.join(output_dir, "images-tagger-" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".zip")
            with zipfile.ZipFile(downloadZipPath, 'w', zipfile.ZIP_DEFLATED) as taggers_zip:
                for info in txt_infos:
                    # Get file name from lookup
                    taggers_zip.write(info["path"], arcname=info["name"])
            download.append(downloadZipPath)

        return download, sorted_general_strings, rating, character_res, general_res
    
def get_selection_from_gallery(gallery: list, selected_state: gr.SelectData):
    if not selected_state:
        return selected_state

    tag_result = { "strings": "", "rating": "", "character_res": "", "general_res": "" }
    if selected_state.value["image"]["path"] in tag_results:
        tag_result = tag_results[selected_state.value["image"]["path"]]

    return (selected_state.value["image"]["path"], selected_state.value["caption"]), tag_result["strings"], tag_result["rating"], tag_result["character_res"], tag_result["general_res"]

def add_images_to_gallery(gallery: list, images):
    if gallery is None:
        gallery = []
    if not images:
        return gallery
    
    # Combine the new images with the existing gallery images
    gallery.extend(images)
    return gallery

def remove_image_from_gallery(gallery: list, selected_image: str):
    if not gallery or not selected_image:
        return gallery

    selected_image = eval(selected_image)
    # Remove the selected image from the gallery
    if selected_image in gallery:
        gallery.remove(selected_image)
    return gallery


def main():
    args = parse_args()

    predictor = Predictor()

    dropdown_list = [
        SWINV2_MODEL_DSV3_REPO,
        EVA02_LARGE_MODEL_DSV3_REPO,
        CONV_MODEL_DSV3_REPO,
        VIT_MODEL_DSV3_REPO,
        VIT_LARGE_MODEL_DSV3_REPO,
        MOAT_MODEL_DSV2_REPO,
        SWIN_MODEL_DSV2_REPO,
        CONV_MODEL_DSV2_REPO,
        CONV2_MODEL_DSV2_REPO,
        VIT_MODEL_DSV2_REPO,
    ]

    with gr.Blocks(title=TITLE) as demo:
        with gr.Column():
            gr.Markdown(
                value=f"<h1 style='text-align: center; margin-bottom: 1rem'>{TITLE}</h1>"
            )
            gr.Markdown(value=DESCRIPTION)
            with gr.Row():
                with gr.Column(variant="panel"):
                    with gr.Row():
                        submit = gr.Button(value="Submit", variant="primary", size="lg")
                    with gr.Row():
                        gallery = gr.Gallery(columns=5, rows=5, show_share_button=False, interactive=True, height="500px", label="Input")
                    with gr.Row():
                        upload_button = gr.UploadButton("Upload Images", file_types=["image"], file_count="multiple", size="sm")
                        remove_button = gr.Button("Remove Selected Image", size="sm")

                    model_repo = gr.Dropdown(
                        dropdown_list,
                        value=SWINV2_MODEL_DSV3_REPO,
                        label="Model",
                    )
                    with gr.Row():
                        general_thresh = gr.Slider(
                            0,
                            1,
                            step=args.score_slider_step,
                            value=args.score_general_threshold,
                            label="General Tags Threshold",
                            scale=3,
                        )
                        general_mcut_enabled = gr.Checkbox(
                            value=False,
                            label="Use MCut threshold",
                            scale=1,
                        )
                    with gr.Row():
                        character_thresh = gr.Slider(
                            0,
                            1,
                            step=args.score_slider_step,
                            value=args.score_character_threshold,
                            label="Character Tags Threshold",
                            scale=3,
                        )
                        character_mcut_enabled = gr.Checkbox(
                            value=False,
                            label="Use MCut threshold",
                            scale=1,
                        )
                    with gr.Row():
                        characters_merge_enabled = gr.Checkbox(
                            value=True,
                            label="Merge characters into the string output",
                            scale=1,
                        )
                    with gr.Row():
                        additional_tags_prepend = gr.Text(label="Prepend Additional tags (comma split)")
                        additional_tags_append  = gr.Text(label="Append Additional tags (comma split)")
                    with gr.Row():
                        clear = gr.ClearButton(
                            components=[
                                gallery,
                                model_repo,
                                general_thresh,
                                general_mcut_enabled,
                                character_thresh,
                                character_mcut_enabled,
                                characters_merge_enabled,
                                additional_tags_prepend,
                                additional_tags_append,
                            ],
                            variant="secondary",
                            size="lg",
                        )
                with gr.Column(variant="panel"):
                    download_file = gr.File(label="Output (Download)")
                    sorted_general_strings = gr.Textbox(label="Output (string)", show_label=True, show_copy_button=True)
                    rating = gr.Label(label="Rating")
                    character_res = gr.Label(label="Output (characters)")
                    general_res = gr.Label(label="Output (tags)")
                    clear.add(
                        [
                            download_file,
                            sorted_general_strings,
                            rating,
                            character_res,
                            general_res,
                        ]
                    )

                # When the upload button is clicked, add the new images to the gallery
                upload_button.upload(add_images_to_gallery, inputs=[gallery, upload_button], outputs=gallery)
                # Event to update the selected image when an image is clicked in the gallery
                selected_image = gr.Textbox(label="Selected Image", visible=False)
                gallery.select(get_selection_from_gallery, inputs=gallery, outputs=[selected_image, sorted_general_strings, rating, character_res, general_res])
                # Event to remove a selected image from the gallery
                remove_button.click(remove_image_from_gallery, inputs=[gallery, selected_image], outputs=gallery)

        submit.click(
            predictor.predict,
            inputs=[
                gallery,
                model_repo,
                general_thresh,
                general_mcut_enabled,
                character_thresh,
                character_mcut_enabled,
                characters_merge_enabled,
                additional_tags_prepend,
                additional_tags_append,
            ],
            outputs=[download_file, sorted_general_strings, rating, character_res, general_res],
        )
        
        # gr.Examples(
        #     [["power.jpg", SWINV2_MODEL_DSV3_REPO, 0.35, False, 0.85, False]], 
        #     inputs=[
        #         gallery,
        #         model_repo,
        #         general_thresh,
        #         general_mcut_enabled,
        #         character_thresh,
        #         character_mcut_enabled,
        #         characters_merge_enabled,
        #         additional_tags_prepend,
        #         additional_tags_append,
        #     ],
        # )

    demo.queue(max_size=10)
    demo.launch(inbrowser=True)


if __name__ == "__main__":
    main()
