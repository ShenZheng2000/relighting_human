import os
import cv2
import numpy as np
import random
from tqdm import tqdm

def find_bdy_images(root_dir):
    """Find all 'bdy_' images in immediate subfolders (no recursion)"""
    bdy_files = []
    for subfolder in os.listdir(root_dir):
        subfolder_path = os.path.join(root_dir, subfolder)
        if not os.path.isdir(subfolder_path):
            continue
        for filename in os.listdir(subfolder_path):
            if filename.lower().startswith('bdy_') and not filename.lower().endswith('.json'):
                bdy_files.append(os.path.join(subfolder_path, filename))
    return bdy_files

def sample_images(image_list, num_samples=100, seed=0):
    random.seed(seed)
    return random.sample(image_list, min(num_samples, len(image_list)))

def compute_rgb_means(image_paths):
    """Compute average R, G, B values (normalized to [0,1])"""
    total_rgb = np.zeros(3)
    count = 0

    for path in tqdm(image_paths, desc="Processing", leave=False):
        img = cv2.imread(path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mean_rgb = np.mean(img, axis=(0, 1)) / 255.0
        total_rgb += mean_rgb
        count += 1
    
    if count == 0:
        return None
    return total_rgb / count

# ---- Main ----
root_dir = "/home/shenzhen/Datasets/dataset_with_garment"
bdy_files = find_bdy_images(root_dir)
print(f"Total available images: {len(bdy_files)}")

bias_labels = ["🔴 RED bias", "🟢 GREEN bias", "🔵 BLUE bias"]

for seed in range(10): # NOTE: itreate over 10 seeds
    print(f"\n--- Seed {seed} ---")
    sampled_files = sample_images(bdy_files, num_samples=100, seed=seed)
    if not sampled_files:
        print("No images sampled.")
        continue

    avg_rgb = compute_rgb_means(sampled_files)
    if avg_rgb is None:
        print("All sampled images failed to load.")
        continue

    r, g, b = avg_rgb
    print(f"Red:   {r:.4f}")
    print(f"Green: {g:.4f}")
    print(f"Blue:  {b:.4f}")
    dominant_idx = np.argmax(avg_rgb)
    print(bias_labels[dominant_idx])


# "/home/shenzhen/Datasets/dataset_with_garment_debug_100" (full-size)
# Average RGB values:
# Red:   0.5975
# Green: 0.5573
# Blue:  0.5277
# 🔴 RED bias


# Total available images: 33163

# --- Seed 0 ---
# Red:   0.5727
# Green: 0.5311
# Blue:  0.4960
# 🔴 RED bias

# --- Seed 1 ---
# Red:   0.5855
# Green: 0.5489
# Blue:  0.5168
# 🔴 RED bias

# --- Seed 2 ---
# Red:   0.5600
# Green: 0.5220
# Blue:  0.4855
# 🔴 RED bias

# --- Seed 3 ---
# Red:   0.5610
# Green: 0.5261
# Blue:  0.4942
# 🔴 RED bias

# --- Seed 4 ---
# Red:   0.5866
# Green: 0.5491
# Blue:  0.5241
# 🔴 RED bias

# --- Seed 5 ---
# Red:   0.5831
# Green: 0.5347
# Blue:  0.4960
# 🔴 RED bias

# --- Seed 6 ---
# Red:   0.5746
# Green: 0.5330
# Blue:  0.4986
# 🔴 RED bias

# --- Seed 7 ---
# Red:   0.5814
# Green: 0.5401
# Blue:  0.5024
# 🔴 RED bias

# --- Seed 8 ---
# Red:   0.5686
# Green: 0.5303
# Blue:  0.4936
# 🔴 RED bias

# --- Seed 9 ---
# Red:   0.5641
# Green: 0.5241
# Blue:  0.4873
# 🔴 RED bias