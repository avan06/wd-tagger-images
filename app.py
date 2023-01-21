#!/usr/bin/env python

from __future__ import annotations

import argparse
import functools
import html
import os

import gradio as gr
import huggingface_hub
import numpy as np
import onnxruntime as rt
import pandas as pd
import piexif
import piexif.helper
import PIL.Image

from Utils import dbimutils

TITLE = "WaifuDiffusion v1.4 Tags"
DESCRIPTION = """
Demo for:
- [SmilingWolf/wd-v1-4-swinv2-tagger-v2](https://huggingface.co/SmilingWolf/wd-v1-4-convnext-tagger-v2)
- [SmilingWolf/wd-v1-4-convnext-tagger-v2](https://huggingface.co/SmilingWolf/wd-v1-4-convnext-tagger-v2)
- [SmilingWolf/wd-v1-4-vit-tagger-v2](https://huggingface.co/SmilingWolf/wd-v1-4-vit-tagger-v2)

Includes "ready to copy" prompt and a prompt analyzer.

Modified from [NoCrypt/DeepDanbooru_string](https://huggingface.co/spaces/NoCrypt/DeepDanbooru_string)  
Modified from [hysts/DeepDanbooru](https://huggingface.co/spaces/hysts/DeepDanbooru)

PNG Info code forked from [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)

Example image by [ほし☆☆☆](https://www.pixiv.net/en/users/43565085)
"""

HF_TOKEN = os.environ["HF_TOKEN"]
SWIN_MODEL_REPO = "SmilingWolf/wd-v1-4-swinv2-tagger-v2"
CONV_MODEL_REPO = "SmilingWolf/wd-v1-4-convnext-tagger-v2"
VIT_MODEL_REPO = "SmilingWolf/wd-v1-4-vit-tagger-v2"
MODEL_FILENAME = "model.onnx"
LABEL_FILENAME = "selected_tags.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-slider-step", type=float, default=0.05)
    parser.add_argument("--score-general-threshold", type=float, default=0.35)
    parser.add_argument("--score-character-threshold", type=float, default=0.85)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def load_model(model_repo: str, model_filename: str) -> rt.InferenceSession:
    path = huggingface_hub.hf_hub_download(
        model_repo, model_filename, use_auth_token=HF_TOKEN
    )
    model = rt.InferenceSession(path)
    return model


def change_model(model_name):
    global loaded_models

    if model_name == "SwinV2":
        model = load_model(SWIN_MODEL_REPO, MODEL_FILENAME)
    elif model_name == "ConvNext":
        model = load_model(CONV_MODEL_REPO, MODEL_FILENAME)
    elif model_name == "ViT":
        model = load_model(VIT_MODEL_REPO, MODEL_FILENAME)

    loaded_models[model_name] = model
    return loaded_models[model_name]


def load_labels() -> list[str]:
    path = huggingface_hub.hf_hub_download(
        SWIN_MODEL_REPO, LABEL_FILENAME, use_auth_token=HF_TOKEN
    )
    df = pd.read_csv(path)

    tag_names = df["name"].tolist()
    rating_indexes = list(np.where(df["category"] == 9)[0])
    general_indexes = list(np.where(df["category"] == 0)[0])
    character_indexes = list(np.where(df["category"] == 4)[0])
    return tag_names, rating_indexes, general_indexes, character_indexes


def plaintext_to_html(text):
    text = (
        "<p>" + "<br>\n".join([f"{html.escape(x)}" for x in text.split("\n")]) + "</p>"
    )
    return text


def predict(
    image: PIL.Image.Image,
    model_name: str,
    general_threshold: float,
    character_threshold: float,
    tag_names: list[str],
    rating_indexes: list[np.int64],
    general_indexes: list[np.int64],
    character_indexes: list[np.int64],
):
    global loaded_models

    rawimage = image

    model = loaded_models[model_name]
    if model is None:
        model = change_model(model_name)

    _, height, width, _ = model.get_inputs()[0].shape

    # Alpha to white
    image = image.convert("RGBA")
    new_image = PIL.Image.new("RGBA", image.size, "WHITE")
    new_image.paste(image, mask=image)
    image = new_image.convert("RGB")
    image = np.asarray(image)

    # PIL RGB to OpenCV BGR
    image = image[:, :, ::-1]

    image = dbimutils.make_square(image, height)
    image = dbimutils.smart_resize(image, height)
    image = image.astype(np.float32)
    image = np.expand_dims(image, 0)

    input_name = model.get_inputs()[0].name
    label_name = model.get_outputs()[0].name
    probs = model.run([label_name], {input_name: image})[0]

    labels = list(zip(tag_names, probs[0].astype(float)))

    # First 4 labels are actually ratings: pick one with argmax
    ratings_names = [labels[i] for i in rating_indexes]
    rating = dict(ratings_names)

    # Then we have general tags: pick any where prediction confidence > threshold
    general_names = [labels[i] for i in general_indexes]
    general_res = [x for x in general_names if x[1] > general_threshold]
    general_res = dict(general_res)

    # Everything else is characters: pick any where prediction confidence > threshold
    character_names = [labels[i] for i in character_indexes]
    character_res = [x for x in character_names if x[1] > character_threshold]
    character_res = dict(character_res)

    b = dict(sorted(general_res.items(), key=lambda item: item[1], reverse=True))
    a = (
        ", ".join(list(b.keys()))
        .replace("_", " ")
        .replace("(", "\(")
        .replace(")", "\)")
    )
    c = ", ".join(list(b.keys()))

    items = rawimage.info
    geninfo = ""

    if "exif" in rawimage.info:
        exif = piexif.load(rawimage.info["exif"])
        exif_comment = (exif or {}).get("Exif", {}).get(piexif.ExifIFD.UserComment, b"")
        try:
            exif_comment = piexif.helper.UserComment.load(exif_comment)
        except ValueError:
            exif_comment = exif_comment.decode("utf8", errors="ignore")

        items["exif comment"] = exif_comment
        geninfo = exif_comment

        for field in [
            "jfif",
            "jfif_version",
            "jfif_unit",
            "jfif_density",
            "dpi",
            "exif",
            "loop",
            "background",
            "timestamp",
            "duration",
        ]:
            items.pop(field, None)

    geninfo = items.get("parameters", geninfo)

    info = f"""
<p><h4>PNG Info</h4></p>    
"""
    for key, text in items.items():
        info += (
            f"""
<div>
<p><b>{plaintext_to_html(str(key))}</b></p>
<p>{plaintext_to_html(str(text))}</p>
</div>
""".strip()
            + "\n"
        )

    if len(info) == 0:
        message = "Nothing found in the image."
        info = f"<div><p>{message}<p></div>"

    return (a, c, rating, character_res, general_res, info)


def main():
    global loaded_models
    loaded_models = {"SwinV2": None, "ConvNext": None, "ViT": None}

    args = parse_args()

    change_model("SwinV2")

    tag_names, rating_indexes, general_indexes, character_indexes = load_labels()

    func = functools.partial(
        predict,
        tag_names=tag_names,
        rating_indexes=rating_indexes,
        general_indexes=general_indexes,
        character_indexes=character_indexes,
    )

    gr.Interface(
        fn=func,
        inputs=[
            gr.Image(type="pil", label="Input"),
            gr.Radio(["SwinV2", "ConvNext", "ViT"], value="SwinV2", label="Model"),
            gr.Slider(
                0,
                1,
                step=args.score_slider_step,
                value=args.score_general_threshold,
                label="General Tags Threshold",
            ),
            gr.Slider(
                0,
                1,
                step=args.score_slider_step,
                value=args.score_character_threshold,
                label="Character Tags Threshold",
            ),
        ],
        outputs=[
            gr.Textbox(label="Output (string)"),
            gr.Textbox(label="Output (raw string)"),
            gr.Label(label="Rating"),
            gr.Label(label="Output (characters)"),
            gr.Label(label="Output (tags)"),
            gr.HTML(),
        ],
        examples=[["power.jpg", "SwinV2", 0.5]],
        title=TITLE,
        description=DESCRIPTION,
        allow_flagging="never",
    ).launch(
        enable_queue=True,
        share=args.share,
    )


if __name__ == "__main__":
    main()
