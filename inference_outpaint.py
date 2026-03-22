import os
import torch
from PIL import Image
from diffusers import FluxFillPipeline

from utils import (
    relighting_prompt_versions,
    parse_arguments,
    load_config,
    prepare_canvas_and_mask,
    extract_background,
    process_depth_map,
    resolve_flat_paths,
    apply_background_override,
)

from inference_utils import (
    setup_depth_model,
    iterate_dataset_stems,
)


# ---------------- Outpainting functions ---------------- #
def run_outpainting(subfolder_path, config, pipe_outpaint, outpaint_prompts, depth_model):
    """
    Performs outpainting on a given subfolder.

    Saves:
      - Base outpaint (without fg mask) using the base prompt.
      - Relight outpaint (with fg mask) using the selected relight prompt.
      - Base depth map
      - Relight depth map

    Returns paths to the base and relight outpainted images.
    """
    stem = subfolder_path
    source_image_path, annotation_path, black_mask_path = resolve_flat_paths(config, stem)

    if (source_image_path is None) or (not os.path.exists(annotation_path)) or (black_mask_path is None):
        print(f"Skipping {stem} due to missing image/caption/mask.")
        return None, None

    body_mask = Image.open(black_mask_path).convert("L")
    source_image = Image.open(source_image_path).convert("RGB")

    with open(annotation_path, "r") as f:
        base_prompt = f.read().strip()

    base_prompt = apply_background_override(base_prompt, config)

    if config.extract_bg_from_base_prompt:
        base_prompt = extract_background(base_prompt)
        print("Extracted base_prompt:", base_prompt)

    target_width = config.width
    target_height = config.height
    cfg_outpaint = config.cfg_outpaint
    num_inference_steps = config.num_inference_steps
    crop_to_foreground = config.crop_to_foreground
    upper_crop = config.upper_crop

    # -------- Base outpaint --------
    canvas_base_no, mask_base_no, _, _, _ = prepare_canvas_and_mask(
        source_image,
        target_width,
        target_height,
        apply_fg_mask=False,
        body_mask=body_mask,
        crop_to_foreground=crop_to_foreground,
        upper_crop=upper_crop,
    )

    img_out_base_no = pipe_outpaint(
        prompt=base_prompt,
        image=canvas_base_no,
        mask_image=mask_base_no,
        height=target_height,
        width=target_width,
        guidance_scale=cfg_outpaint,
        num_inference_steps=num_inference_steps,
        generator=torch.Generator("cpu").manual_seed(42),
    ).images[0]

    # -------- Relight outpaint --------
    relight_id = config.relight_type
    relight_prompt = outpaint_prompts.get(relight_id, "")

    canvas_relight, mask_relight, _, _, _ = prepare_canvas_and_mask(
        source_image,
        target_width,
        target_height,
        apply_fg_mask=True,
        body_mask=body_mask,
        crop_to_foreground=crop_to_foreground,
        upper_crop=upper_crop,
    )

    img_out_relight = pipe_outpaint(
        prompt=relight_prompt,
        image=canvas_relight,
        mask_image=mask_relight,
        height=target_height,
        width=target_width,
        guidance_scale=cfg_outpaint,
        num_inference_steps=num_inference_steps,
        generator=torch.Generator("cpu").manual_seed(config.seed),
    ).images[0]

    outpaint_folder = os.path.join("outpaint", config.output_dir, relight_id, stem)
    os.makedirs(outpaint_folder, exist_ok=True)

    base_no_path = os.path.join(outpaint_folder, "img_out_base.png")
    relight_path = os.path.join(outpaint_folder, "img_out_relight.png")

    # IMPORTANT: save BOTH images
    img_out_base_no.save(base_no_path)
    img_out_relight.save(relight_path)
    print(f"Saved outpaint images in {outpaint_folder}")

    if config.save_depth_maps:
        depth_base_path = os.path.join(outpaint_folder, "depth_base.png")
        depth_relight_path = os.path.join(outpaint_folder, "depth_relight.png")

        # IMPORTANT: save BOTH depth maps
        process_depth_map(base_no_path, depth_base_path, depth_model)
        process_depth_map(relight_path, depth_relight_path, depth_model)

    return base_no_path, relight_path


def run_outpainting_loop(config, pipe_outpaint, outpaint_prompts, depth_model):
    count = 0

    for stem in iterate_dataset_stems(config.input_dir):
        run_outpainting(stem, config, pipe_outpaint, outpaint_prompts, depth_model)

        count += 1
        if config.max_images and count >= config.max_images:
            print(f"Reached max_images limit: {config.max_images}. Stopping outpainting.")
            return


# ---------------- Main function ---------------- #
def main():
    args = parse_arguments()
    base_config = load_config(args)
    device = f"cuda:{base_config.gpu}" if torch.cuda.is_available() else "cpu"

    depth_model = setup_depth_model(base_config)

    for i in range(args.num_seeds):
        seed = args.seed_offset + i
        config = load_config(args)
        config.seed = seed
        config.output_dir = f"{os.path.splitext(os.path.basename(args.exp_config))[0]}_seed{seed}"

        print(f"\n=== Running seed {seed} ===")

        print("Loading outpainting pipeline...")
        pipe_outpaint = FluxFillPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Fill-dev",
            torch_dtype=torch.bfloat16,
        ).to(device)
        pipe_outpaint.set_progress_bar_config(disable=True)

        outpaint_prompts = relighting_prompt_versions[str(config.prompt_version)]
        run_outpainting_loop(config, pipe_outpaint, outpaint_prompts, depth_model)

        del pipe_outpaint
        torch.cuda.empty_cache()

    print("✅ Outpainting + depth estimation finished.")


if __name__ == "__main__":
    main()