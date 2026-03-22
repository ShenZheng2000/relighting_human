import os
import random
from PIL import Image
import json
from tqdm import tqdm  # ✅ progress bar

# ---- import shared config + prompt dict ----
from utils import parse_arguments, load_config, relighting_prompt_versions


def save_crops(image_paths, base_folder, relight_folder, start_idx=0, img_dim=784):
    idx = start_idx
    for img_path in tqdm(image_paths, desc=f"Cropping {os.path.basename(base_folder)}"):
        try:
            img = Image.open(img_path)
            w, h = img.size
            base_crop = img.crop((w - 2 * img_dim, 0, w - img_dim, img_dim))
            relight_crop = img.crop((w - img_dim, 0, w, img_dim))
            fname = f"{idx}.png"
            base_crop.save(os.path.join(base_folder, fname))
            relight_crop.save(os.path.join(relight_folder, fname))
            idx += 1
        except Exception as e:
            print(f"Failed to process {img_path}: {e}")
    return idx


def create_json(num_images, prompt, output_path):
    data = {f"{i}.png": prompt for i in range(num_images)}
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Saved JSON to {output_path}")


def collect_valid_paths(root_dir, target_prefix, relight_type):
    valid_image_paths = []
    for exp_folder in sorted(os.listdir(root_dir)):

        if (exp_folder != target_prefix) and (not exp_folder.startswith(target_prefix + "_seed")):
            continue

        subfolder = os.path.join(root_dir, exp_folder, relight_type)
        if not os.path.isdir(subfolder):
            continue

        invalid_path = os.path.join(subfolder, "invalid.txt")
        invalid_files = set()
        if os.path.exists(invalid_path):
            with open(invalid_path, "r") as f:
                invalid_files = set(line.strip() for line in f.readlines())

        for fname in sorted(os.listdir(subfolder)):
            if fname.lower().endswith(".png") and fname not in invalid_files:
                valid_image_paths.append(os.path.join(subfolder, fname))

    return valid_image_paths


def main():
    args = parse_arguments()
    config = load_config(args)

    # ------------------------------------------------
    # Load experiment settings
    # ------------------------------------------------
    target_prefix = config.output_dir
    relight_type = config.relight_type
    prompt = relighting_prompt_versions[str(config.prompt_version)][relight_type]

    # ------------------------------------------------
    # Dataset input / output paths
    # ------------------------------------------------
    root_dir = args.root_dir
    output_dir = f"{args.output_base}/{target_prefix}/{relight_type}"

    # ------------------------------------------------
    # Create dataset folder structure
    # ------------------------------------------------
    save_base_train_A = os.path.join(output_dir, "train_A")
    save_relight_train_B = os.path.join(output_dir, "train_B")
    save_base_test_A = os.path.join(output_dir, "test_A")
    save_relight_test_B = os.path.join(output_dir, "test_B")

    for d in [save_base_train_A, save_relight_train_B, save_base_test_A, save_relight_test_B]:
        os.makedirs(d, exist_ok=True)

    # ------------------------------------------------
    # Collect valid images from experiment outputs
    # ------------------------------------------------
    valid_image_paths = collect_valid_paths(root_dir, target_prefix, relight_type)

    # ------------------------------------------------
    # Shuffle and split dataset into train / test
    # ------------------------------------------------
    random.seed(0)
    random.shuffle(valid_image_paths)

    split_idx = int(args.train_ratio * len(valid_image_paths))
    train_files = valid_image_paths[:split_idx]
    test_files = valid_image_paths[split_idx:]

    # ------------------------------------------------
    # Save training data
    # ------------------------------------------------
    save_crops(train_files, save_base_train_A, save_relight_train_B, img_dim=args.img_dim)
    create_json(len(train_files), prompt, os.path.join(output_dir, "train_prompts.json"))

    # ------------------------------------------------
    # Save testing data
    # ------------------------------------------------
    save_crops(test_files, save_base_test_A, save_relight_test_B, img_dim=args.img_dim)
    create_json(len(test_files), prompt, os.path.join(output_dir, "test_prompts.json"))

    print("Done.")


if __name__ == "__main__":
    main()