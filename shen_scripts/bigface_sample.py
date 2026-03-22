import os
import shutil

# ----------------------------
# Settings
# ----------------------------

src_dir = "/home/shenzhen/Datasets/dataset_with_garment"

ranking_file = "/home/shenzhen/Relight_Projects/relighting/shen_scripts/face_area_ranking.txt"

# choose range directly
start_num = 100       # inclusive
end_num   = 200       # exclusive

# new output dir
dst_dir = f"/home/shenzhen/Datasets/dataset_with_garment_bigface_start_{start_num}_end_{end_num}"
os.makedirs(dst_dir, exist_ok=True)

print(f"Selecting ranked folders from {start_num} to {end_num}...")
# ----------------------------

# Read ranking file and extract folder names
ranked_folders = []
with open(ranking_file, "r") as f:
    for line in f:
        parts = line.strip().split(",")
        if len(parts) >= 2:
            folder_name = parts[1].strip().strip("'\"")  # clean quotes
            ranked_folders.append(folder_name)

# slice based on start and end
subset = ranked_folders[start_num : end_num]
print(f"Found {len(subset)} folders.")

# Copy folders
for i, folder_name in enumerate(subset, 1):
    src_path = os.path.join(src_dir, folder_name)
    dst_path = os.path.join(dst_dir, folder_name)

    if not os.path.exists(src_path):
        print(f"[Skip] Missing: {folder_name}")
        continue

    print(f"[{i}/{len(subset)}] Copying: {folder_name}")
    shutil.copytree(src_path, dst_path)

print("\n✅ DONE!")