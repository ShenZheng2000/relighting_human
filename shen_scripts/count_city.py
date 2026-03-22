import json
import os
from collections import Counter

TRAIN_JSON = "/ssd0/shenzhen/Datasets/depth/workzone_segm/annotations/instances_train_gps_split.json"
VAL_JSON   = "/ssd0/shenzhen/Datasets/depth/workzone_segm/annotations/instances_val_gps_split.json"


def get_city_prefix(filename: str) -> str:
    stem = os.path.splitext(os.path.basename(filename))[0]
    parts = stem.split("_")
    return parts[0] if parts else "misc"


def count_cities(json_path):
    with open(json_path, "r") as f:
        coco = json.load(f)

    counter = Counter()

    for img in coco.get("images", []):
        fname = img.get("file_name", "")
        if fname:
            city = get_city_prefix(fname)
            counter[city] += 1

    return counter


train_counts = count_cities(TRAIN_JSON)
val_counts   = count_cities(VAL_JSON)

print("===== TRAIN =====")
for city, n in sorted(train_counts.items()):
    print(f"{city}: {n}")

print("\n===== VAL =====")
for city, n in sorted(val_counts.items()):
    print(f"{city}: {n}")


# ===== TRAIN =====
# IMG: 295
# boston: 477
# charlotte: 139
# chicago: 67
# columbus: 62
# denver: 272
# detroit: 302
# houston: 37
# indianapolis: 58
# jacksonville: 23
# los: 422
# minneapolis: 66
# new: 73
# pgh01: 177
# pgh02: 153
# pgh03: 597
# pgh04: 1116
# philadelphia: 90
# phoenix: 25
# pittsburgh: 341
# san: 369
# washington: 157

# ===== VAL =====
# boston: 382
# charlotte: 84
# chicago: 51
# columbus: 48
# denver: 193
# detroit: 220
# houston: 27
# indianapolis: 37
# jacksonville: 15
# los: 251
# minneapolis: 50
# new: 51
# philadelphia: 71
# phoenix: 17
# pittsburgh: 256
# san: 236
# washington: 109