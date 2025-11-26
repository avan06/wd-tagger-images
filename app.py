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
import re
import ast
import time
from datetime import datetime
from collections import defaultdict
from classifyTags import classify_tags
from collections import Counter # Import Counter for statistics

TITLE = "WaifuDiffusion Tagger multiple images/texts"
DESCRIPTION = """
Demo for the WaifuDiffusion tagger models and text processing.
Select input type below. For images, it will generate tags. For text files, it will process existing tags.
Example image by [ほし☆☆☆](https://www.pixiv.net/en/users/43565085)

This project was duplicated from the Space of [wd-tagger](https://huggingface.co/spaces/SmilingWolf/wd-tagger) by the author SmilingWolf.  
Features of This Modified Version:
- Supports batch processing of multiple images or text files.
- Displays tag results in categorized groups: the generated tags will now be analyzed and categorized into corresponding groups. (for images)
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

# IdolSankaku series of models:
EVA02_LARGE_MODEL_IS_DSV1_REPO = "deepghs/idolsankaku-eva02-large-tagger-v1"
SWINV2_MODEL_IS_DSV1_REPO = "deepghs/idolsankaku-swinv2-tagger-v1"

# Files to download from the repos
MODEL_FILENAME = "model.onnx"
LABEL_FILENAME = "selected_tags.csv"

# LLAMA model
META_LLAMA_3_3B_REPO = "jncraton/Llama-3.2-3B-Instruct-ct2-int8"
META_LLAMA_3_8B_REPO = "avans06/Meta-Llama-3.2-8B-Instruct-ct2-int8_float16"

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

class Timer:
    def __init__(self):
        self.start_time  = time.perf_counter()  # Record the start time
        self.checkpoints = [("Start", self.start_time)]  # Store checkpoints

    def checkpoint(self, label="Checkpoint"):
        """Record a checkpoint with a given label."""
        now = time.perf_counter()
        self.checkpoints.append((label, now))

    def report(self, is_clear_checkpoints = True):
        # Determine the max label width for alignment
        max_label_length = max(len(label) for label, _ in self.checkpoints) if self.checkpoints else 0

        if len(self.checkpoints) > 1:
            prev_time = self.checkpoints[0][1]
            for label, curr_time in self.checkpoints[1:]:
                elapsed = curr_time - prev_time
                print(f"{label.ljust(max_label_length)}: {elapsed:.3f} seconds")
                prev_time = curr_time

        if is_clear_checkpoints:
            self.checkpoints = [("Start", time.perf_counter())]

    def report_all(self):
        """Print all recorded checkpoints and total execution time with aligned formatting."""
        print("\n> Execution Time Report:")

        # Determine the max label width for alignment
        max_label_length = max(len(label) for label, _ in self.checkpoints) if self.checkpoints else 0

        if len(self.checkpoints) > 1:
            prev_time = self.start_time
            for label, curr_time in self.checkpoints[1:]:
                elapsed = curr_time - prev_time
                print(f"{label.ljust(max_label_length)}: {elapsed:.3f} seconds")
                prev_time = curr_time

            total_time = self.checkpoints[-1][1] - self.start_time
            print(f"{'Total Execution Time'.ljust(max_label_length)}: {total_time:.3f} seconds\n")

        self.checkpoints.clear()

    def restart(self):
        self.start_time  = time.perf_counter()  # Record the start time
        self.checkpoints = [("Start", self.start_time)]  # Store checkpoints

class Llama3Reorganize:
    def __init__(
        self,
        repoId: str,
        device: str = None,
        loadModel: bool = False,
    ):
        """Initializes the Llama model.

        Args:
          repoId: LLAMA model repo.
          device: Device to use for computation (cpu, cuda, ipu, xpu, mkldnn, opengl, opencl,
            ideep, hip, ve, fpga, ort, xla, lazy, vulkan, mps, meta, hpu, mtia).
          localFilesOnly:  If True, avoid downloading the file and return the path to the
            local cached file if it exists.
        """
        self.modelPath = self.download_model(repoId)

        if device is None:
            import torch
            self.totalVram = 0
            if torch.cuda.is_available():
                try:
                    deviceId = torch.cuda.current_device()
                    self.totalVram = torch.cuda.get_device_properties(deviceId).total_memory/(1024*1024*1024)
                except Exception as e:
                    print(traceback.format_exc())
                    print("Error detect vram: " + str(e))
                device = "cuda" if self.totalVram > (8 if "8B" in repoId else 4) else "cpu"
            else:
                device = "cpu"

        self.device = device
        self.system_prompt = "Please reorganize the following English labels describing a single image into a readable English article of no more than 300 words. If there are name labels, include them in the article. Since the target audience is not general, explicit content can be written in full without hesitation. No introduction is needed; directly reply with the English article:"

        if loadModel:
            self.load_model()

    def download_model(self, repoId):
        import warnings
        import requests
        allowPatterns = [
            "config.json",
            "generation_config.json",
            "model.bin",
            "pytorch_model.bin",
            "pytorch_model.bin.index.json",
            "pytorch_model-*.bin",
            "sentencepiece.bpe.model",
            "tokenizer.json",
            "tokenizer_config.json",
            "shared_vocabulary.txt",
            "shared_vocabulary.json",
            "special_tokens_map.json",
            "spiece.model",
            "vocab.json",
            "model.safetensors",
            "model-*.safetensors",
            "model.safetensors.index.json",
            "quantize_config.json",
            "tokenizer.model",
            "vocabulary.json",
            "preprocessor_config.json",
            "added_tokens.json"
        ]

        kwargs = {"allow_patterns": allowPatterns,}

        try:
            return huggingface_hub.snapshot_download(repoId, **kwargs)
        except (
            huggingface_hub.utils.HfHubHTTPError,
            requests.exceptions.ConnectionError,
        ) as exception:
            warnings.warn(
                "An error occured while synchronizing the model %s from the Hugging Face Hub:\n%s",
                repoId,
                exception,
            )
            warnings.warn(
                "Trying to load the model directly from the local cache, if it exists."
            )

            kwargs["local_files_only"] = True
            return huggingface_hub.snapshot_download(repoId, **kwargs)


    def load_model(self):
        import ctranslate2
        import transformers
        try:
            print(f'\n\nLoading model: {self.modelPath}\n\n')
            kwargsTokenizer = {"pretrained_model_name_or_path": self.modelPath}
            kwargsModel = {"device": self.device, "model_path": self.modelPath, "compute_type": "auto"}
            self.roleSystem = {"role": "system", "content": self.system_prompt}
            self.Model = ctranslate2.Generator(**kwargsModel)

            self.Tokenizer = transformers.AutoTokenizer.from_pretrained(**kwargsTokenizer)
            self.terminators = [self.Tokenizer.eos_token_id, self.Tokenizer.convert_tokens_to_ids("<|eot_id|>")]

        except Exception as e:
            self.release_vram()
            raise e


    def release_vram(self):
        try:
            import torch
            if torch.cuda.is_available():
                if hasattr(self, "Model") and hasattr(self.Model, "unload_model"):
                    self.Model.unload_model()
                if hasattr(self, "Tokenizer"):
                    del self.Tokenizer
                if hasattr(self, "Model"):
                    del self.Model
                import gc
                gc.collect()
                try:
                    torch.cuda.empty_cache()
                except Exception as e:
                    print(traceback.format_exc())
                    print(f"\tcuda empty cache, error: {e}")
                print("release vram end.")
        except Exception as e:
            print(traceback.format_exc())
            print(f"Error release vram: {e}")

    def reorganize(self, text: str, max_length: int = 400):
        result = None
        try:
            input_ids = self.Tokenizer.apply_chat_template([self.roleSystem, {"role": "user", "content": text + "\n\nHere's the reorganized English article:"}], tokenize=False, add_generation_prompt=True)
            source = self.Tokenizer.convert_ids_to_tokens(self.Tokenizer.encode(input_ids))
            output = self.Model.generate_batch([source], max_length=max_length, max_batch_size=2, no_repeat_ngram_size=3, beam_size=2, sampling_temperature=0.7, sampling_topp=0.9, include_prompt_in_result=False, end_token=self.terminators)
            target = output[0]
            result = self.Tokenizer.decode(target.sequences_ids[0])
            if len(result) > 2:
                if result[0] == '"' and result[-1] == '"':
                    result = result[1:-1]
                elif result[0] == "'" and result[-1] == "'":
                    result = result[1:-1]
                elif result[0] == '「' and result[-1] == '」':
                    result = result[1:-1]
                elif result[0] == '『' and result[-1] == '』':
                    result = result[1:-1]
        except Exception as e:
            print(traceback.format_exc())
            print(f"Error reorganize text: {e}")

        return result


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
        self.tag_names, self.rating_indexes, self.general_indexes, self.character_indexes = sep_tags
        model = rt.InferenceSession(model_path)
        _, height, _, _ = model.get_inputs()[0].shape
        self.model_target_size = height
        self.last_loaded_repo = model_repo
        self.model = model

    def prepare_image(self, path):
        image = Image.open(path).convert("RGBA")
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
        if max_dim != self.model_target_size:
            padded_image = padded_image.resize(
                (self.model_target_size, self.model_target_size),
                Image.BICUBIC,
            )

        # Convert to numpy array
        image_array = np.asarray(padded_image, dtype=np.float32)

        # Convert PIL-native RGB to BGR
        image_array = image_array[:, :, ::-1]

        return np.expand_dims(image_array, axis=0)

    def create_file(self, text: str, directory: str, fileName: str) -> str:
        # Write the text to a file
        filepath = os.path.join(directory, fileName)
        with open(filepath, 'w', encoding="utf-8") as file:
            file.write(text)
        return filepath

    def predict_from_images(
        self,
        gallery,
        model_repo,
        general_thresh,
        general_mcut_enabled,
        character_thresh,
        character_mcut_enabled,
        characters_merge_enabled,
        llama3_reorganize_model_repo,
        additional_tags_prepend,
        additional_tags_append,
        tags_to_remove,
        tag_results,
        progress=gr.Progress()
    ):
        if not gallery:
             gr.Warning("No images in the gallery to process.")
             return None, "", "{}", "", "", "", "{}", {}, ""

        gallery_len = len(gallery)
        print(f"Predict from images: load model: {model_repo}, gallery length: {gallery_len}")

        timer = Timer()  # Create a timer
        progressRatio = 0.5 if llama3_reorganize_model_repo else 1
        progressTotal = gallery_len + (1 if llama3_reorganize_model_repo else 0) + 1 # +1 for model load
        current_progress = 0

        self.load_model(model_repo)
        current_progress += 1 / progressTotal
        progress(current_progress, desc="Initialize wd model finished")
        timer.checkpoint(f"Initialize wd model")

        # Result
        txt_infos = []
        output_dir = tempfile.mkdtemp()

        last_sorted_general_strings = ""
        last_classified_tags, last_unclassified_tags = {}, {}
        last_rating, last_character_res, last_general_res = None, None, None

        # Initialize counter for statistics
        tag_counter = Counter()

        llama3_reorganize = None
        if llama3_reorganize_model_repo:
            print(f"Llama3 reorganize load model {llama3_reorganize_model_repo}")
            llama3_reorganize = Llama3Reorganize(llama3_reorganize_model_repo, loadModel=True)
            current_progress += 1 / progressTotal
            progress(current_progress, desc="Initialize llama3 model finished")
            timer.checkpoint(f"Initialize llama3 model")

        timer.report()

        prepend_list = [tag.strip() for tag in additional_tags_prepend.split(",") if tag.strip()]
        append_list = [tag.strip() for tag in additional_tags_append.split(",") if tag.strip()]
        remove_list = [tag.strip() for tag in tags_to_remove.split(",") if tag.strip()] # Parse remove tags
        if prepend_list and append_list:
            append_list = [item for item in append_list if item not in prepend_list]

        # Dictionary to track counters for each filename
        name_counters = defaultdict(int)
        for idx, value in enumerate(gallery):
            try:
                image_path = value[0]
                image_name = os.path.splitext(os.path.basename(image_path))[0]

                # Increment the counter for the current name
                name_counters[image_name] += 1
                if name_counters[image_name] > 1:
                    image_name = f"{image_name}_{name_counters[image_name]:02d}"

                image = self.prepare_image(image_path)

                input_name = self.model.get_inputs()[0].name
                label_name = self.model.get_outputs()[0].name
                print(f"Gallery {idx+1}/{gallery_len}: Starting run wd model...")
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
                general_res = dict([x for x in general_names if x[1] > general_thresh])

                # Everything else is characters: pick any where prediction confidence > threshold
                character_names = [labels[i] for i in self.character_indexes]

                if character_mcut_enabled:
                    character_probs = np.array([x[1] for x in character_names])
                    character_thresh = mcut_threshold(character_probs)
                    character_thresh = max(0.15, character_thresh)
                character_res = dict([x for x in character_names if x[1] > character_thresh])
                character_list = list(character_res.keys())

                sorted_general_list = sorted(general_res.items(), key=lambda x: x[1], reverse=True)
                sorted_general_list = [x[0] for x in sorted_general_list]
                #Remove values from character_list that already exist in sorted_general_list
                character_list = [item for item in character_list if item not in sorted_general_list]
                #Remove values from sorted_general_list that already exist in prepend_list or append_list
                if prepend_list:
                    sorted_general_list = [item for item in sorted_general_list if item not in prepend_list]
                if append_list:
                    sorted_general_list = [item for item in sorted_general_list if item not in append_list]

                final_tags_list = prepend_list + sorted_general_list + append_list
                if characters_merge_enabled:
                    final_tags_list = character_list + final_tags_list

                # Apply removal logic
                if remove_list:
                    remove_set = set(remove_list)
                    final_tags_list = [tag for tag in final_tags_list if tag not in remove_set]

                # Update counter with the final list of tags for this image
                tag_counter.update(final_tags_list)

                sorted_general_strings = ", ".join(final_tags_list).replace("(", "\(").replace(")", "\)")
                classified_tags, unclassified_tags = classify_tags(final_tags_list)

                current_progress += progressRatio / progressTotal
                progress(current_progress, desc=f"Image {idx+1}/{gallery_len}, predict finished")
                timer.checkpoint(f"Image {idx+1}/{gallery_len}, predict finished")

                if llama3_reorganize:
                    print(f"Starting reorganize with llama3...")
                    reorganize_strings = llama3_reorganize.reorganize(sorted_general_strings)
                    if reorganize_strings:
                        reorganize_strings = re.sub(r" *Title: *", "", reorganize_strings)
                        reorganize_strings = re.sub(r"\n+", ",", reorganize_strings)
                        reorganize_strings = re.sub(r",,+", ",", reorganize_strings)
                        sorted_general_strings += "," + reorganize_strings

                    current_progress += progressRatio / progressTotal
                    progress(current_progress, desc=f"Image {idx+1}/{gallery_len}, llama3 reorganize finished")
                    timer.checkpoint(f"Image {idx+1}/{gallery_len}, llama3 reorganize finished")

                txt_file = self.create_file(sorted_general_strings, output_dir, image_name + ".txt")
                txt_infos.append({"path": txt_file, "name": image_name + ".txt"})

                tag_results[image_path] = { "strings": sorted_general_strings, "classified_tags": classified_tags, "rating": rating, "character_res": character_res, "general_res": general_res, "unclassified_tags": unclassified_tags }

                # Merge Unclassified into Classified for frontend display
                display_classified = classified_tags.copy()
                if unclassified_tags:
                    # If it is a list (common case), put it into the "Unclassified" category
                    if isinstance(unclassified_tags, list):
                        display_classified["Unclassified"] = unclassified_tags
                    # Just to be safe, if it is a dict, use update
                    elif isinstance(unclassified_tags, dict):
                        display_classified.update(unclassified_tags)

                # Store last result for UI display
                last_sorted_general_strings = sorted_general_strings
                last_classified_tags = display_classified # Use the merged result
                last_rating = rating
                last_character_res = character_res
                last_general_res = general_res
                last_unclassified_tags = unclassified_tags
                timer.report()

            except Exception as e:
                print(traceback.format_exc())
                print("Error predicting image: " + str(e))
                gr.Warning(f"Failed to process image {os.path.basename(value[0])}. Error: {e}")

        # Result
        download = []
        if txt_infos:
            zip_filename = "images-tagger-" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".zip"
            downloadZipPath = os.path.join(output_dir, zip_filename)
            with zipfile.ZipFile(downloadZipPath, 'w', zipfile.ZIP_DEFLATED) as taggers_zip:
                for info in txt_infos:
                    # Get file name from lookup
                    taggers_zip.write(info["path"], arcname=info["name"])
            download.append(downloadZipPath)

        if llama3_reorganize:
            llama3_reorganize.release_vram()

        progress(1, desc="Image processing completed")
        timer.report_all()
        print("Image prediction is complete.")
        
        # Format statistics for output
        stats_list = [f"{tag}: {count}" for tag, count in tag_counter.most_common()]
        statistics_output = "\n".join(stats_list)

        return download, last_sorted_general_strings, last_classified_tags, last_rating, last_character_res, last_general_res, last_unclassified_tags, tag_results, statistics_output

    # Method to process text files
    def predict_from_text(
        self,
        text_files,
        llama3_reorganize_model_repo,
        additional_tags_prepend,
        additional_tags_append,
        tags_to_remove,
        progress=gr.Progress()
    ):
        if not text_files:
             gr.Warning("No text files uploaded to process.")
             return None, "", "{}", "", "", "", "{}", {}, ""

        files_len = len(text_files)
        print(f"Predict from text: processing {files_len} files.")

        timer = Timer()
        progressRatio = 0.5 if llama3_reorganize_model_repo else 1.0
        progressTotal = files_len + (1 if llama3_reorganize_model_repo else 0)
        current_progress = 0

        txt_infos = []
        output_dir = tempfile.mkdtemp()
        last_processed_string = ""
        
        # Initialize counter for statistics
        tag_counter = Counter()

        llama3_reorganize = None
        if llama3_reorganize_model_repo:
            print(f"Llama3 reorganize load model {llama3_reorganize_model_repo}")
            llama3_reorganize = Llama3Reorganize(llama3_reorganize_model_repo, loadModel=True)
            current_progress += 1 / progressTotal
            progress(current_progress, desc="Initialize llama3 model finished")
            timer.checkpoint(f"Initialize llama3 model")

        timer.report()

        prepend_list = [tag.strip() for tag in additional_tags_prepend.split(",") if tag.strip()]
        append_list = [tag.strip() for tag in additional_tags_append.split(",") if tag.strip()]
        remove_list = [tag.strip() for tag in tags_to_remove.split(",") if tag.strip()] # Parse remove tags
        if prepend_list and append_list:
            append_list = [item for item in append_list if item not in prepend_list]

        name_counters = defaultdict(int)
        for idx, file_obj in enumerate(text_files):
            try:
                file_path = file_obj.name
                file_name_base = os.path.splitext(os.path.basename(file_path))[0]

                name_counters[file_name_base] += 1
                if name_counters[file_name_base] > 1:
                    output_file_name = f"{file_name_base}_{name_counters[file_name_base]:02d}.txt"
                else:
                    output_file_name = f"{file_name_base}.txt"

                with open(file_path, 'r', encoding='utf-8') as f:
                    original_content = f.read()

                # Process tags
                tags_list = [tag.strip() for tag in original_content.split(',') if tag.strip()]

                if prepend_list:
                    tags_list = [item for item in tags_list if item not in prepend_list]
                if append_list:
                    tags_list = [item for item in tags_list if item not in append_list]

                final_tags_list = prepend_list + tags_list + append_list

                # Apply removal logic
                if remove_list:
                    remove_set = set(remove_list)
                    final_tags_list = [tag for tag in final_tags_list if tag not in remove_set]
                
                # Update counter with the final list of tags for this file
                tag_counter.update(final_tags_list)

                processed_string = ", ".join(final_tags_list)

                current_progress += progressRatio / progressTotal
                progress(current_progress, desc=f"File {idx+1}/{files_len}, base processing finished")
                timer.checkpoint(f"File {idx+1}/{files_len}, base processing finished")

                if llama3_reorganize:
                    print(f"Starting reorganize with llama3...")
                    reorganize_strings = llama3_reorganize.reorganize(processed_string)
                    if reorganize_strings:
                        reorganize_strings = re.sub(r" *Title: *", "", reorganize_strings)
                        reorganize_strings = re.sub(r"\n+", ",", reorganize_strings)
                        reorganize_strings = re.sub(r",,+", ",", reorganize_strings)
                        processed_string += "," + reorganize_strings

                    current_progress += progressRatio / progressTotal
                    progress(current_progress, desc=f"File {idx+1}/{files_len}, llama3 reorganize finished")
                    timer.checkpoint(f"File {idx+1}/{files_len}, llama3 reorganize finished")

                txt_file_path = self.create_file(processed_string, output_dir, output_file_name)
                txt_infos.append({"path": txt_file_path, "name": output_file_name})
                last_processed_string = processed_string
                timer.report()

            except Exception as e:
                print(traceback.format_exc())
                print("Error processing text file: " + str(e))
                gr.Warning(f"Failed to process file {os.path.basename(file_obj.name)}. Error: {e}")

        download = []
        if txt_infos:
            zip_filename = "texts-processed-" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".zip"
            downloadZipPath = os.path.join(output_dir, zip_filename)
            with zipfile.ZipFile(downloadZipPath, 'w', zipfile.ZIP_DEFLATED) as processed_zip:
                for info in txt_infos:
                    processed_zip.write(info["path"], arcname=info["name"])
            download.append(downloadZipPath)

        if llama3_reorganize:
            llama3_reorganize.release_vram()

        progress(1, desc="Text processing completed")
        timer.report_all()  # Print all recorded times
        print("Text processing is complete.")
        
        # Format statistics for output
        stats_list = [f"{tag}: {count}" for tag, count in tag_counter.most_common()]
        statistics_output = "\n".join(stats_list)
        
        # Return values in the same structure as the image path, with placeholders for unused outputs
        return download, last_processed_string, "{}", "", "", "", "{}", {}, statistics_output

def get_selection_from_gallery(gallery: list, tag_results: dict, selected_state: gr.SelectData):
    if not selected_state:
        return selected_state

    # Default unclassified_tags to list (because classifyTags usually returns a list)
    tag_result = tag_results.get(selected_state.value["image"]["path"],
                                {"strings": "", "classified_tags": {}, "rating": "", "character_res": "", "general_res": "", "unclassified_tags": []})

    # Retrieve original data
    c_tags = tag_result["classified_tags"]
    u_tags = tag_result["unclassified_tags"]

    # Error handling: Ensure correct types
    if isinstance(c_tags, str): 
        try: c_tags = ast.literal_eval(c_tags)
        except: c_tags = {}
    if isinstance(u_tags, str): 
        try: u_tags = ast.literal_eval(u_tags)
        except: u_tags = []

    # Merge: Copy Classified, and append Unclassified if it exists
    display_classified = c_tags.copy() if isinstance(c_tags, dict) else {}
    
    if u_tags:
        if isinstance(u_tags, list):
            display_classified["Unclassified"] = u_tags
        elif isinstance(u_tags, dict):
            display_classified.update(u_tags)

    return (selected_state.value["image"]["path"], selected_state.value["caption"]), tag_result["strings"], display_classified, tag_result["rating"], tag_result["character_res"], tag_result["general_res"], tag_result["unclassified_tags"]

def append_gallery(gallery: list, image: str):
    if gallery is None:
        gallery = []
    if not image:
        return gallery, None

    gallery.append(image)

    return gallery, None


def extend_gallery(gallery: list, images):
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

    try:
        selected_image_tuple = ast.literal_eval(selected_image) #Use ast.literal_eval to parse text into a tuple.
        # Remove the selected image from the gallery
        if selected_image_tuple in gallery:
            gallery.remove(selected_image_tuple)
    except (ValueError, SyntaxError):
        # Handle cases where the string is not a valid literal
        print(f"Warning: Could not parse selected_image string: {selected_image}")

    return gallery


def main():
    # Custom CSS to set the height of the gr.Dropdown menu
    css = """
    div.progress-level div.progress-level-inner {
        text-align: left !important;
        width: 55.5% !important;
    }
    textarea[rows]:not([rows="1"]) {
        overflow-y: auto !important;
        scrollbar-width: thin !important;
    }
    textarea[rows]:not([rows="1"])::-webkit-scrollbar {
        all: initial !important;
        background: #f1f1f1 !important;
    }
    textarea[rows]:not([rows="1"])::-webkit-scrollbar-thumb {
        all: initial !important;
        background: #a8a8a8 !important;
    }
    /* Make the Dropdown options display more compactly */
    .tag-dropdown span.svelte-1f354aw {
        font-family: monospace;
    }
    """
    args = parse_args()

    predictor = Predictor()

    dropdown_list = [
        EVA02_LARGE_MODEL_DSV3_REPO,
        SWINV2_MODEL_DSV3_REPO,
        CONV_MODEL_DSV3_REPO,
        VIT_MODEL_DSV3_REPO,
        VIT_LARGE_MODEL_DSV3_REPO,
        # ---
        MOAT_MODEL_DSV2_REPO,
        SWIN_MODEL_DSV2_REPO,
        CONV_MODEL_DSV2_REPO,
        CONV2_MODEL_DSV2_REPO,
        VIT_MODEL_DSV2_REPO,
        # ---
        SWINV2_MODEL_IS_DSV1_REPO,
        EVA02_LARGE_MODEL_IS_DSV1_REPO,
    ]

    llama_list = [
        META_LLAMA_3_3B_REPO,
        META_LLAMA_3_8B_REPO,
    ]

    # Wrapper function to decide which prediction method to call
    def run_prediction(
        input_type, gallery, text_files, model_repo, general_thresh,
        general_mcut_enabled, character_thresh, character_mcut_enabled,
        characters_merge_enabled, llama3_reorganize_model_repo,
        additional_tags_prepend, additional_tags_append, tags_to_remove,
        tag_results, progress=gr.Progress()
    ):
        if input_type == 'Image':
            return predictor.predict_from_images(
                gallery, model_repo, general_thresh, general_mcut_enabled,
                character_thresh, character_mcut_enabled, characters_merge_enabled,
                llama3_reorganize_model_repo, additional_tags_prepend,
                additional_tags_append, tags_to_remove, tag_results, progress
            )
        else: # 'Text file (.txt)'
            # For text files, some parameters are not used, but we must return
            # a tuple of the same size. `predict_from_text` handles this.
            return predictor.predict_from_text(
                text_files, llama3_reorganize_model_repo,
                additional_tags_prepend, additional_tags_append, tags_to_remove, progress
            )

    with gr.Blocks(title=TITLE, css=css) as demo:
        gr.Markdown(f"<h1 style='text-align: center; margin-bottom: 1rem'>{TITLE}</h1>")
        gr.Markdown(value=DESCRIPTION)

        with gr.Row():
            with gr.Column():
                submit = gr.Button(value="Submit", variant="primary", size="lg")

                # Input type selector
                input_type_radio = gr.Radio(
                    choices=['Image', 'Text file (.txt)'],
                    value='Image',
                    label="Input Type"
                )

                # Group for image inputs, initially visible
                with gr.Column(visible=True) as image_inputs_group:
                    with gr.Column(variant="panel"):
                        # Create an Image component for uploading images
                        image_input = gr.Image(label="Upload an Image or clicking paste from clipboard button", type="filepath", sources=["upload", "clipboard"], height=150)
                        with gr.Row():
                            upload_button = gr.UploadButton("Upload multiple images", file_types=["image"], file_count="multiple", size="sm")
                            remove_button = gr.Button("Remove Selected Image", size="sm")
                        gallery = gr.Gallery(columns=5, rows=5, show_share_button=False, interactive=True, height=500, label="Gallery that displaying a grid of images")

                # Group for text file inputs, initially hidden
                with gr.Column(visible=False) as text_inputs_group:
                    text_files_input = gr.Files(
                        label="Upload .txt files",
                        file_types=[".txt"],
                        file_count="multiple",
                        height=500
                    )

                # Image-specific settings
                model_repo = gr.Dropdown(
                    dropdown_list,
                    value=EVA02_LARGE_MODEL_DSV3_REPO,
                    label="Model (for Images)",
                )
                with gr.Row(visible=True) as general_thresh_row:
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
                with gr.Row(visible=True) as character_thresh_row:
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
                characters_merge_enabled = gr.Checkbox(
                    value=True,
                    label="Merge characters into the string output",
                    scale=1,
                    visible=True,
                )

                # Common settings
                with gr.Row():
                    llama3_reorganize_model_repo = gr.Dropdown(
                        [None] + llama_list,
                        value=None,
                        label="Use the Llama3 model to reorganize the article",
                        info="(Note: very slow)",
                    )
                with gr.Row():
                    additional_tags_prepend = gr.Text(label="Prepend Additional tags (comma split)")
                    additional_tags_append  = gr.Text(label="Append Additional tags (comma split)")
                
                # Add the remove tags input box
                with gr.Row():
                    tags_to_remove = gr.Text(label="Remove tags (comma split)")

                with gr.Row():
                    clear = gr.ClearButton(
                        components=[
                            gallery,
                            text_files_input,
                            model_repo,
                            general_thresh,
                            general_mcut_enabled,
                            character_thresh,
                            character_mcut_enabled,
                            characters_merge_enabled,
                            llama3_reorganize_model_repo,
                            additional_tags_prepend,
                            additional_tags_append,
                            tags_to_remove,
                        ],
                        variant="secondary",
                        size="lg",
                    )

            with gr.Column(variant="panel"):
                download_file = gr.File(label="Output (Download)")
                sorted_general_strings = gr.Textbox(label="Output (string for last processed item)", show_label=True, show_copy_button=True, lines=5)
                
                # Use State to store categorized data
                categorized_state = gr.State({})
                
                # Wrap the dynamically rendered area with Accordion
                with gr.Accordion("Categorized (tags) - Interactive", open=False) as categorized_accordion:
                    # Use @gr.render to dynamically generate UI based on the content of categorized_state
                    @gr.render(inputs=categorized_state)
                    def render_categorized_tags(categories_data):
                        if not categories_data:
                            gr.Markdown("No categorized tags to display yet.")
                            return
                        
                        for category_name, tags_list in categories_data.items():
                            # Ensure tags_list is of type list
                            current_tags = tags_list if isinstance(tags_list, list) else str(tags_list).split(',')
                            current_tags = [t.strip() for t in current_tags if t.strip()]

                            with gr.Group():
                                with gr.Row(variant="compact", equal_height=True):
                                    # 1. Multiselect Dropdown (Main editing area)
                                    dd = gr.Dropdown(
                                        choices=current_tags,     # Default choices are the current tags
                                        value=current_tags,       #  Default value are the current tags
                                        label=f"{category_name} ({len(current_tags)})",
                                        multiselect=True,         # Enable multiselect (shows X button)
                                        allow_custom_value=True,  # Allow custom values (add new tags)
                                        interactive=True,
                                        scale=5,
                                        elem_classes=["tag-dropdown"]
                                    )
                                    
                                    # 2. Read-only Textbox (Used to provide a copy button)
                                    # Since Dropdown cannot directly copy raw strings, we use this Textbox to "sync display" the string
                                    txt_copy = gr.Textbox(
                                        value=", ".join(current_tags),
                                        label="Copy String",
                                        show_copy_button=True, # Copy button is here
                                        interactive=False,     # Disable manual editing, only sync from Dropdown
                                        scale=1,
                                        min_width=100,
                                        max_lines=1
                                    )

                                # 3. Event binding: Update Textbox when Dropdown changes
                                def sync_tags_to_text(selected_tags):
                                    return ", ".join(selected_tags)

                                dd.change(fn=sync_tags_to_text, inputs=dd, outputs=txt_copy)

                with gr.Accordion("Detailed Output (for last processed item)", open=False):
                    rating = gr.Label(label="Rating", visible=True)
                    character_res = gr.Label(label="Output (characters)", visible=True)
                    general_res = gr.Label(label="Output (tags)", visible=True)
                    unclassified = gr.JSON(label="Unclassified (tags)", visible=False)
                
                with gr.Accordion("Tags Statistics (All files)", open=False):
                    tags_statistics = gr.Text(
                        label="Statistics", 
                        autoscroll=False,
                        show_label=False,
                        show_copy_button=True, 
                        lines=10,
                    )

                clear.add(
                    [
                        download_file,
                        sorted_general_strings,
                        categorized_state,
                        rating,
                        character_res,
                        general_res,
                        unclassified,
                        tags_statistics,
                    ]
                )

            tag_results = gr.State({})
            selected_image = gr.Textbox(label="Selected Image", visible=False)

            # Event Listeners
            # Define the event listener to add the uploaded image to the gallery
            image_input.change(append_gallery, inputs=[gallery, image_input], outputs=[gallery, image_input])
            # When the upload button is clicked, add the new images to the gallery
            upload_button.upload(extend_gallery, inputs=[gallery, upload_button], outputs=gallery)
            # Event to update the selected image when an image is clicked in the gallery
            gallery.select(
                get_selection_from_gallery, 
                inputs=[gallery, tag_results], 
                outputs=[selected_image, sorted_general_strings, categorized_state, rating, character_res, general_res, unclassified]
            )
            # Event to remove a selected image from the gallery
            remove_button.click(remove_image_from_gallery, inputs=[gallery, selected_image], outputs=gallery)

            # Logic to show/hide input groups based on radio selection
            def change_input_type(input_type):
                is_image = (input_type == 'Image')
                return {
                    image_inputs_group: gr.update(visible=is_image),
                    text_inputs_group: gr.update(visible=not is_image),
                    # Also update visibility of image-specific settings
                    model_repo: gr.update(visible=is_image),
                    general_thresh_row: gr.update(visible=is_image),
                    character_thresh_row: gr.update(visible=is_image),
                    characters_merge_enabled: gr.update(visible=is_image),
                    # Update visibility of categorized_accordion
                    categorized_accordion: gr.update(visible=is_image),
                    rating: gr.update(visible=is_image),
                    character_res: gr.update(visible=is_image),
                    general_res: gr.update(visible=is_image),
                    unclassified: gr.update(visible=is_image),
                }

            # Connect the radio button to the visibility function
            input_type_radio.change(
                fn=change_input_type,
                inputs=input_type_radio,
                outputs=[
                    image_inputs_group, text_inputs_group, model_repo,
                    general_thresh_row, character_thresh_row, characters_merge_enabled,
                    categorized_accordion, rating, character_res, general_res, unclassified
                ]
            )

            # submit click now calls the wrapper function
            submit.click(
                fn=run_prediction,
                inputs=[
                    input_type_radio,
                    gallery,
                    text_files_input,
                    model_repo,
                    general_thresh,
                    general_mcut_enabled,
                    character_thresh,
                    character_mcut_enabled,
                    characters_merge_enabled,
                    llama3_reorganize_model_repo,
                    additional_tags_prepend,
                    additional_tags_append,
                    tags_to_remove,
                    tag_results,
                ],
                outputs=[download_file, sorted_general_strings, categorized_state, rating, character_res, general_res, unclassified, tag_results, tags_statistics],
            )

        gr.Examples(
            [["power.jpg", SWINV2_MODEL_DSV3_REPO, 0.35, False, 0.85, False]],
            inputs=[
                image_input,
                model_repo,
                general_thresh,
                general_mcut_enabled,
                character_thresh,
                character_mcut_enabled,
            ],
        )

    demo.queue(max_size=2)
    demo.launch(inbrowser=True)


if __name__ == "__main__":
    main()
