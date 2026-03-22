# inference_utils.py

import os
import glob
import torch
from PIL import Image
from transformers import pipeline
from image_gen_aux import DepthPreprocessor
from utils import resolve_flat_paths


# ---------------- Depth model setup ---------------- #
def setup_depth_model(config):
    depth_model = DepthPreprocessor.from_pretrained(
        "LiheYoung/depth-anything-large-hf"
    )
    return depth_model


# ---------------- Iterate dataset ---------------- #
def iterate_dataset_stems(input_dir):
    image_dir = os.path.join(input_dir, "image")
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.*")))

    stems = []
    for img_path in image_paths:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        stems.append(stem)

    return stems


# ---------------- Load image + caption ---------------- #
def load_image_and_caption(config, stem):
    source_image_path, annotation_path, _ = resolve_flat_paths(config, stem)

    if (source_image_path is None) or (not os.path.exists(annotation_path)):
        return None, None

    image = Image.open(source_image_path).convert("RGB")

    with open(annotation_path, "r") as f:
        caption = f.read().strip()

    return image, caption


# ---------------- Run FLUX inference ---------------- #
def run_flux_depth_inference(pipe, prompt, depth_map, config, width):
    image = pipe(
        prompt=prompt,
        control_image=depth_map,
        height=config.height,
        width=width,
        guidance_scale=config.cfg,
        num_inference_steps=config.num_steps,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(config.seed),
    ).images[0]

    return image