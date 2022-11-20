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
Demo for [SmilingWolf/wd-v1-4-vit-tagger](https://huggingface.co/SmilingWolf/wd-v1-4-vit-tagger) with "ready to copy" prompt and a prompt analyzer.

Modified from [NoCrypt/DeepDanbooru_string](https://huggingface.co/spaces/NoCrypt/DeepDanbooru_string)  
Modified from [hysts/DeepDanbooru](https://huggingface.co/spaces/hysts/DeepDanbooru)

PNG Info code forked from [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
"""

HF_TOKEN = os.environ["HF_TOKEN"]
MODEL_REPO = "SmilingWolf/wd-v1-4-vit-tagger"
MODEL_FILENAME = "ViTB16_11_07_2022_18h19m14s.onnx"
LABEL_FILENAME = "selected_tags.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-slider-step", type=float, default=0.05)
    parser.add_argument("--score-threshold", type=float, default=0.35)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def load_model() -> rt.InferenceSession:
    path = huggingface_hub.hf_hub_download(
        MODEL_REPO, MODEL_FILENAME, use_auth_token=HF_TOKEN
    )
    model = rt.InferenceSession(path)
    return model


def load_labels() -> list[str]:
    path = huggingface_hub.hf_hub_download(
        MODEL_REPO, LABEL_FILENAME, use_auth_token=HF_TOKEN
    )
    df = pd.read_csv(path)["name"].tolist()
    return df


def plaintext_to_html(text):
    text = (
        "<p>" + "<br>\n".join([f"{html.escape(x)}" for x in text.split("\n")]) + "</p>"
    )
    return text


def predict(
    image: PIL.Image.Image,
    score_threshold: float,
    model: rt.InferenceSession,
    labels: list[str],
):
    rawimage = image
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

    labels = list(zip(labels, probs[0].astype(float)))

    # First 4 labels are actually ratings: pick one with argmax
    ratings_names = labels[:4]
    rating = dict(ratings_names)

    # Everything else is tags: pick any where prediction confidence > threshold
    tags_names = labels[4:]
    res = [x for x in tags_names if x[1] > score_threshold]
    res = dict(res)

    b = dict(sorted(res.items(), key=lambda item: item[1], reverse=True))
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

    return (a, c, rating, res, info)


def main():
    args = parse_args()
    model = load_model()
    labels = load_labels()

    func = functools.partial(predict, model=model, labels=labels)

    gr.Interface(
        fn=func,
        inputs=[
            gr.Image(type="pil", label="Input"),
            gr.Slider(
                0,
                1,
                step=args.score_slider_step,
                value=args.score_threshold,
                label="Score Threshold",
            ),
        ],
        outputs=[
            gr.Textbox(label="Output (string)"),
            gr.Textbox(label="Output (raw string)"),
            gr.Label(label="Rating"),
            gr.Label(label="Output (label)"),
            gr.HTML(),
        ],
        examples=[["power.jpg", 0.5]],
        title=TITLE,
        description=DESCRIPTION,
        allow_flagging="never",
    ).launch(
        enable_queue=True,
        share=args.share,
    )


if __name__ == "__main__":
    main()
