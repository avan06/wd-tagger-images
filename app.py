#!/usr/bin/env python

from __future__ import annotations

import argparse
import functools
import os
import html
import pathlib
import tarfile

import deepdanbooru as dd
import gradio as gr
import huggingface_hub
import numpy as np
import PIL.Image
import tensorflow as tf
import piexif
import piexif.helper

TITLE = 'DeepDanbooru String'

TOKEN = os.environ['TOKEN']
MODEL_REPO = 'NoCrypt/DeepDanbooru_string'
MODEL_FILENAME = 'model-resnet_custom_v3.h5'
LABEL_FILENAME = 'tags.txt'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--score-slider-step', type=float, default=0.05)
    parser.add_argument('--score-threshold', type=float, default=0.5)
    parser.add_argument('--theme', type=str, default='dark-grass')
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--port', type=int)
    parser.add_argument('--disable-queue',
                        dest='enable_queue',
                        action='store_false')
    parser.add_argument('--allow-flagging', type=str, default='never')
    return parser.parse_args()


def load_sample_image_paths() -> list[pathlib.Path]:
    image_dir = pathlib.Path('images')
    if not image_dir.exists():
        dataset_repo = 'hysts/sample-images-TADNE'
        path = huggingface_hub.hf_hub_download(dataset_repo,
                                               'images.tar.gz',
                                               repo_type='dataset',
                                               use_auth_token=TOKEN)
        with tarfile.open(path) as f:
            f.extractall()
    return sorted(image_dir.glob('*'))


def load_model() -> tf.keras.Model:
    path = huggingface_hub.hf_hub_download(MODEL_REPO,
                                           MODEL_FILENAME,
                                           use_auth_token=TOKEN)
    model = tf.keras.models.load_model(path)
    return model


def load_labels() -> list[str]:
    path = huggingface_hub.hf_hub_download(MODEL_REPO,
                                           LABEL_FILENAME,
                                           use_auth_token=TOKEN)
    with open(path) as f:
        labels = [line.strip() for line in f.readlines()]
    return labels

def plaintext_to_html(text):
    text = "<p>" + "<br>\n".join([f"{html.escape(x)}" for x in text.split('\n')]) + "</p>"
    return text

def predict(image: PIL.Image.Image, score_threshold: float,
            model: tf.keras.Model, labels: list[str]) -> dict[str, float]:
    rawimage = image
    _, height, width, _ = model.input_shape
    image = np.asarray(image)
    image = tf.image.resize(image,
                            size=(height, width),
                            method=tf.image.ResizeMethod.AREA,
                            preserve_aspect_ratio=True)
    image = image.numpy()
    image = dd.image.transform_and_pad_image(image, width, height)
    image = image / 255.
    probs = model.predict(image[None, ...])[0]
    probs = probs.astype(float)
    res = dict()
    for prob, label in zip(probs.tolist(), labels):
        if prob < score_threshold:
            continue
        res[label] = prob
    b = dict(sorted(res.items(),key=lambda item:item[1], reverse=True))
    a = ', '.join(list(b.keys())).replace('_',' ').replace('(','\(').replace(')','\)')
    c = ', '.join(list(b.keys()))
    
    items = rawimage.info
    geninfo = ''
    
    if "exif" in rawimage.info:
        exif = piexif.load(rawimage.info["exif"])
        exif_comment = (exif or {}).get("Exif", {}).get(piexif.ExifIFD.UserComment, b'')
        try:
            exif_comment = piexif.helper.UserComment.load(exif_comment)
        except ValueError:
            exif_comment = exif_comment.decode('utf8', errors="ignore")
    
        items['exif comment'] = exif_comment
        geninfo = exif_comment
    
        for field in ['jfif', 'jfif_version', 'jfif_unit', 'jfif_density', 'dpi', 'exif',
                      'loop', 'background', 'timestamp', 'duration']:
            items.pop(field, None)
    
    geninfo = items.get('parameters', geninfo)
    
    info = f"""
<p><h4>PNG Info</h4></p>    
"""
    for key, text in items.items():
        info += f"""
<div>
<p><b>{plaintext_to_html(str(key))}</b></p>
<p>{plaintext_to_html(str(text))}</p>
</div>
""".strip()+"\n"
    
    if len(info) == 0:
        message = "Nothing found in the image."
        info = f"<div><p>{message}<p></div>"
    
    return (a,c,res,info)


def main():
    args = parse_args()
    model = load_model()
    labels = load_labels()

    func = functools.partial(predict, model=model, labels=labels)
    func = functools.update_wrapper(func, predict)

    gr.Interface(
        func,
        [
            gr.inputs.Image(type='pil', label='Input'),
            gr.inputs.Slider(0,
                             1,
                             step=args.score_slider_step,
                             default=args.score_threshold,
                             label='Score Threshold'),
        ],
        [
            gr.outputs.Textbox(label='Output (string)'), 
            gr.outputs.Textbox(label='Output (raw string)'), 
            gr.outputs.Label(label='Output (label)'),
            gr.outputs.HTML()
        ],
        examples=[
        ['miku.jpg',0.5],
        ['miku2.jpg',0.5]
        ],
        title=TITLE,
        description='''
Demo for [KichangKim/DeepDanbooru](https://github.com/KichangKim/DeepDanbooru) with "ready to copy" prompt and a prompt analyzer.

Modified from [hysts/DeepDanbooru](https://huggingface.co/spaces/hysts/DeepDanbooru)

PNG Info code forked from [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
        ''',
        theme=args.theme,
        allow_flagging=args.allow_flagging,
        live=args.live,
    ).launch(
        enable_queue=args.enable_queue,
        server_port=args.port,
        share=args.share,
    )


if __name__ == '__main__':
    main()
