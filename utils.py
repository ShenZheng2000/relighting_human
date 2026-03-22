from PIL import Image, ImageDraw, ImageFont, ImageFilter
import argparse
from diffusers import FluxControlPipeline
from diffusers.utils import load_image
from omegaconf import OmegaConf
import os
import torch
import numpy as np
from image_gen_aux import DepthPreprocessor
from PIL import ImageOps

relighting_prompts_6 = {
    "noon_sunlight_1": "Relit with bright noon sunlight in a clear outdoor setting, casting soft natural shadows and surrounding the subject in crisp white light to create a clean, vibrant daytime mood.",
    "golden_sunlight_1": "Relit with warm golden sunlight during the late afternoon, casting gentle directional shadows and surrounding the subject in soft amber tones to create a calm, radiant mood.",
    "foggy_1": "Relit with dense fog in a muted outdoor setting, casting soft diffused shadows and surrounding the subject in pale gray light to create a quiet, atmospheric mood.",
    "moonlight_1": "Relit with cold moonlight in a minimalist nighttime scene, casting crisp soft shadows and bathing the subject in icy blue highlights to create a tranquil, distant mood.",

}


# Register available prompt versions
relighting_prompt_versions = {
    "6": relighting_prompts_6,
}


def concat_images_side_by_side(image1, image2):
    """
    Concatenates two images side-by-side, resizing both to the height of image2,
    
    Args:
        image1 (PIL.Image): The first image (left side).
        image2 (PIL.Image): The second image (right side).
    Returns:
        PIL.Image: The concatenated image
    """
    # Resize both images to the height of image2, preserving aspect ratio
    target_height = image2.height
    
    # Resize image1
    aspect_ratio1 = image1.width / image1.height
    new_width1 = int(target_height * aspect_ratio1)
    image1 = image1.resize((new_width1, target_height))
    
    # Create a new blank image with combined width
    concatenated_image = Image.new('RGB', (image1.width + image2.width, target_height))
    
    # Paste the images side-by-side
    concatenated_image.paste(image1, (0, 0))
    concatenated_image.paste(image2, (image1.width, 0))
    
    return concatenated_image


def parse_arguments():
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", type=str, default="configs/base.yaml", help="Path to the base configuration file")
    parser.add_argument("--exp_config", type=str, required=True, help="Path to the experiment-specific configuration file")
    parser.add_argument("--relight_type", type=str, required=True, help="Specify relighting type")
    parser.add_argument("--gpu", type=int, required=True, help="GPU ID to use")
    parser.add_argument('--num_seeds', type=int, default=1)
    parser.add_argument('--seed_offset', type=int, default=0, help='Starting seed value (default is 0)')

    # NOTE: parse args for prepare data only. 
    parser.add_argument("--root_dir", type=str,
        default="/home/shenzhen/Relight_Projects/relighting/outputs",
        help="Root directory containing relighting outputs")

    parser.add_argument("--output_base", type=str,
        default="/ssd0/shenzhen/Datasets/relighting",
        help="Base directory for saving prepared dataset")

    parser.add_argument("--train_ratio", type=float, default=0.8,
        help="Train split ratio")

    parser.add_argument("--img_dim", type=int, default=784,
        help="Crop size for base/relight images")


    return parser.parse_args()

def load_config(args):
    """Load and merge configuration files and inject CLI arguments."""
    base_cfg = OmegaConf.load(args.base_config)
    exp_cfg = OmegaConf.load(args.exp_config)
    config = OmegaConf.merge(base_cfg, exp_cfg)
    config.relight_type = args.relight_type
    config.gpu = args.gpu
    config.output_dir = os.path.splitext(os.path.basename(args.exp_config))[0]
    config.background_override_text = getattr(config, "background_override_text", "")
    print(OmegaConf.to_yaml(config))
    return config

def load_depth_map(subfolder_path, config, relight_id):
    # Only allow outpaint depth mode
    if "outpaint" not in str(config.depth_mode):
        raise ValueError(
            f"depth_mode={config.depth_mode} not supported. "
            "Only modes containing 'outpaint' are allowed."
        )

    # subfolder_path can be:
    #  - flat: "00000_00" (already a stem)
    #  - spreeai: "/.../dataset/Adidas_R2_Men_Jackets_216" (folder path)
    #  - spreeai: "Adidas_R2_Men_Jackets_216" (folder name)
    stem = os.path.basename(subfolder_path.rstrip("/"))

    outpaint_folder = os.path.join("outpaint", config.output_dir, relight_id, stem)

    if config.relight_image_only:
        relight_path = os.path.join(outpaint_folder, "depth_relight.png")
        return Image.open(relight_path).convert("RGB")
    else:
        base_path = os.path.join(outpaint_folder, "depth_base.png")
        relight_path = os.path.join(outpaint_folder, "depth_relight.png")

        base = Image.open(base_path).convert("RGB")
        relight = Image.open(relight_path).convert("RGB")
        assert base.size == relight.size, "Depth maps must have the same size!"
        return Image.fromarray(np.hstack([np.array(base), np.array(relight)]))

def extract_background(prompt: str) -> str:
    # Find the index where "background" starts, ignoring case.
    index = prompt.lower().find("background")
    if index != -1:
        return prompt[index:]
    return ""


def process_depth_map(input_path, output_path, depth_model):
    """
    Process an image to generate its depth map using the provided depth model.
    
    Args:
        input_path (str): Path to the input image.
        output_path (str): Path to save the generated depth map.
        depth_model: A pre-loaded depth estimation model (pipeline or processor).
        use_v2 (bool): Flag indicating if the new pipeline is used.
    """
    image = Image.open(input_path).convert("RGB")

    control_image = load_image(input_path)
    depth_map = depth_model(control_image)[0].convert("RGB")

    depth_map.save(output_path)
    print(f"Saved depth map to {output_path}")


def prepare_canvas_and_mask(image, target_width, target_height, apply_fg_mask=False, body_mask=None, crop_to_foreground=False, upper_crop=False):
    '''
    Resizes and centers an image on a fixed-size canvas with black padding, optionally creating a matching mask.
    If crop_to_foreground is True and a body_mask is provided, the image is tightly cropped to the foreground.
    '''

    # --- Crop to foreground if enabled ---
    if crop_to_foreground and body_mask is not None:
        # Invert, since the foreground is black
        inverted_mask = ImageOps.invert(body_mask)
        bbox = inverted_mask.getbbox()
        if bbox is not None:
            left, upper, right, lower = bbox

            if upper_crop:
                # keep only top 50% height
                mid = upper + (lower - upper) // 2
                lower = mid

            image = image.crop((left, upper, right, lower))
            body_mask = body_mask.crop((left, upper, right, lower))

    # --- Resize image and paste to canvas ---
    orig_w, orig_h = image.size
    scale = min(target_width / orig_w, target_height / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    x_offset = (target_width - new_w) // 2
    y_offset = (target_height - new_h) // 2

    canvas = Image.new("RGB", (target_width, target_height), color=(0, 0, 0))
    canvas.paste(image.resize((new_w, new_h), Image.LANCZOS), (x_offset, y_offset))

    # --- Create the mask ---
    if apply_fg_mask and body_mask is not None:
        body_mask_resized = body_mask.resize((new_w, new_h), Image.NEAREST)
        mask_canvas = Image.new("L", (target_width, target_height), color=255)
        mask_canvas.paste(body_mask_resized, (x_offset, y_offset))
    else:
        mask_canvas = Image.new("L", (target_width, target_height), color=255)
        from PIL import ImageDraw
        draw = ImageDraw.Draw(mask_canvas)
        draw.rectangle((x_offset, y_offset, x_offset + new_w, y_offset + new_h), fill=0)

    return canvas, mask_canvas, scale, x_offset, y_offset


def resolve_flat_paths(config, stem):
    root = config.input_dir
    image_dir = os.path.join(root, "image")
    caption_dir = os.path.join(root, "caption")
    mask_dir = os.path.join(root, "fg_masks")

    img_candidates = [
        os.path.join(image_dir, f"{stem}.jpg"),
        os.path.join(image_dir, f"{stem}.jpeg"),
        os.path.join(image_dir, f"{stem}.png"),
    ]
    source_image_path = next((p for p in img_candidates if os.path.exists(p)), None)

    annotation_path = os.path.join(caption_dir, f"{stem}.txt")

    mask_candidates = [
        os.path.join(mask_dir, f"{stem}.png"),
        os.path.join(mask_dir, f"{stem}.jpg"),
        os.path.join(mask_dir, f"{stem}.jpeg"),
    ]
    black_mask_path = next((p for p in mask_candidates if os.path.exists(p)), None)

    return source_image_path, annotation_path, black_mask_path


def apply_background_override(prompt: str, config):
    if not getattr(config, "background_override_text", ""):
        return prompt

    idx = prompt.lower().find("background")
    if idx == -1:
        return prompt.rstrip(",. ") + ", " + config.background_override_text

    return prompt[:idx].rstrip(",. ") + ", " + config.background_override_text


def tile_2x1_pil(img: Image.Image) -> Image.Image:
    w, h = img.size
    out = Image.new(img.mode, (w * 2, h))
    out.paste(img, (0, 0))
    out.paste(img, (w, 0))
    return out


def center_crop_pil(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """
    Center-crop to (target_w, target_h). If img is smaller, first resize up preserving aspect.
    Minimal + robust.
    """
    w, h = img.size
    if w < target_w or h < target_h:
        scale = max(target_w / w, target_h / h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        img = img.resize((new_w, new_h), Image.BICUBIC)
        w, h = img.size

    left = (w - target_w) // 2
    top = (h - target_h) // 2
    return img.crop((left, top, left + target_w, top + target_h))