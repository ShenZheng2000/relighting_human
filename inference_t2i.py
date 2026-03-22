# NOTE: unified inference version for both human relighting and driving relighting.
#
# Human mode:
#   - uses saved depth maps from outpainting step
#
# Driving mode:
#   - computes depth on the fly from the source image

import os
import torch
from diffusers import FluxControlPipeline

from utils import (
    relighting_prompt_versions,
    concat_images_side_by_side,
    parse_arguments,
    load_config,
    load_depth_map,
    apply_background_override,
    tile_2x1_pil,
    center_crop_pil,
)

from inference_utils import (
    setup_depth_model,
    iterate_dataset_stems,
    load_image_and_caption,
    run_flux_depth_inference,
)


# ---------------- Helper ---------------- #
def get_inference_mode(config):
    """
    Minimal-change mode switch.

    Expected:
      config.inference_mode == "human"   -> use saved outpaint depth maps
      config.inference_mode == "driving" -> compute depth online

    Default: human
    """
    return getattr(config, "inference_mode", "human")


# ---------------- Inference functions ---------------- #
def process_subfolder_inference(subfolder_path, config, pipe_inference, prompts, depth_model=None):
    """
    Processes a subfolder by running T2I inference.

    Human mode:
      - loads saved outpaint depth maps

    Driving mode:
      - predicts depth directly from source image

    The final output is a concatenated image saved in the outputs directory.
    """
    stem = subfolder_path
    source_image, base_prompt = load_image_and_caption(config, stem)

    if source_image is None:
        print(f"Skipping inference for {stem} due to missing image/caption.")
        return

    mode = get_inference_mode(config)
    relight_id = config.relight_type

    if relight_id not in prompts:
        raise KeyError(f"relight_type '{relight_id}' not found in prompt_version={config.prompt_version}")

    relight_prompt = prompts.get(relight_id, "")

    # ---------------- Human mode ---------------- #
    if mode == "human":
        base_prompt = apply_background_override(base_prompt, config)

        final_prompt = (
            f"A 2x1 image grid; "
            f"On the left, {base_prompt} "
            f"On the right, the same person {relight_prompt}."
        )
        output_width = config.width * 2

        if "outpaint" in config.depth_mode and not config.save_depth_maps:
            raise ValueError("depth_mode contains 'outpaint' but save_depth_maps is False")

        depth_map_2x1 = load_depth_map(stem, config, relight_id)

    # ---------------- Driving mode ---------------- #
    elif mode == "driving":
        if config.center_crop:
            source_image = center_crop_pil(source_image, config.width, config.height)

        phrase = " in neutral daytime lighting"
        final_prompt = (
            "A 2x1 image grid; "
            f"On the left, {base_prompt}{phrase}. "
            f"On the right, the same scene {relight_prompt}."
        )
        output_width = config.width * 2

        if depth_model is None:
            raise ValueError("depth_model is required in driving mode")

        depth = depth_model(source_image)[0].convert("RGB")
        depth_map_2x1 = tile_2x1_pil(depth)
        depth_map_2x1 = depth_map_2x1.resize((output_width, config.height))

        depth_dir_final = os.path.join("depth", config.output_dir, relight_id)
        os.makedirs(depth_dir_final, exist_ok=True)
        depth_save_path = os.path.join(depth_dir_final, f"{stem}.png")
        depth.save(depth_save_path)

    else:
        raise ValueError(f"Unknown inference_mode: {mode}. Expected 'human' or 'driving'.")

    image = run_flux_depth_inference(
        pipe=pipe_inference,
        prompt=final_prompt,
        depth_map=depth_map_2x1,
        config=config,
        width=output_width,
    )

    output_dir_final = os.path.join("outputs", config.output_dir, relight_id)
    os.makedirs(output_dir_final, exist_ok=True)

    output_filename = f"{stem}.png"
    output_path = os.path.join(output_dir_final, output_filename)

    concatenated_image = concat_images_side_by_side(source_image, image)
    concatenated_image.save(output_path)
    print(f"Saved final inference image: {output_path}")


def run_inference_loop(config, pipe_inference, prompts, depth_model=None):
    count = 0

    for stem in iterate_dataset_stems(config.input_dir):
        process_subfolder_inference(stem, config, pipe_inference, prompts, depth_model)

        count += 1
        if config.max_images and count >= config.max_images:
            print(f"Reached max_images limit: {config.max_images}. Stopping inference.")
            return


# ---------------- Main function ---------------- #
def main():
    args = parse_arguments()
    base_config = load_config(args)
    device = f"cuda:{base_config.gpu}" if torch.cuda.is_available() else "cpu"

    mode = get_inference_mode(base_config)

    depth_model = None
    if mode == "driving":
        depth_model = setup_depth_model(base_config)

    for i in range(args.num_seeds):
        seed = args.seed_offset + i
        config = load_config(args)
        config.seed = seed
        config.output_dir = f"{os.path.splitext(os.path.basename(args.exp_config))[0]}_seed{seed}"

        print(f"\n=== Running seed {seed} ===")
        print(f"Inference mode: {get_inference_mode(config)}")

        print("Loading inference pipeline...")
        pipe_inference = FluxControlPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Depth-dev",
            torch_dtype=torch.bfloat16,
        ).to(device)
        pipe_inference.set_progress_bar_config(disable=True)

        prompts = relighting_prompt_versions[str(config.prompt_version)]
        run_inference_loop(config, pipe_inference, prompts, depth_model)

        del pipe_inference
        torch.cuda.empty_cache()

    print("✅ T2I inference finished.")


if __name__ == "__main__":
    main()