import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm
from ultralytics.utils.ops import xyxy2xywhn

in_dir = Path("data/xView")

labels_dir = in_dir.joinpath("labels", "train")
labels_dir.mkdir(parents=True, exist_ok=True)
# in_dir.joinpath("images").mkdir()
# images_dir = in_dir.joinpath("train_images")
# images_dir = images_dir.rename(in_dir.joinpath("images", "train"))
images_dir = in_dir.joinpath("images", "train")

with open(in_dir.joinpath("xView_train.geojson")) as in_file:
    data = json.loads(in_file.read())

# Finding unique labels
class_labels = set([feature["properties"]["type_id"] for feature in data["features"]])
# Mapping them to 0-N
labels_to_index = {label: i for i, label in enumerate(class_labels)}

features = data["features"]
for feature in tqdm(features, desc="Processing features."):
    image_id = feature["properties"]["image_id"]
    try:
        # Extracting bounding box
        bbox = [
            int(coord) for coord in feature["properties"]["bounds_imcoords"].split(",")
        ]
        if len(bbox) != 4:
            raise ValueError("Bounding box has an incorrect number of coordinates.")
        class_id = labels_to_index[feature["properties"]["type_id"]]
        with Image.open(images_dir.joinpath(image_id)) as in_image:
            width, height, *_ = in_image.size
        # Converting bounding box to YOLO format
        yolo_bbox = xyxy2xywhn(
            x=np.array(bbox, dtype=np.float64), w=width, h=height, clip=True
        )
        yolo_bbox_str = [f"{coord:.6f}" for coord in yolo_bbox]
        # Producing YOLO format entry
        entry = " ".join([str(class_id), *yolo_bbox_str]) + "\n"
        with labels_dir.joinpath(image_id).with_suffix(".txt").open("a") as label_file:
            label_file.write(entry)
    except Exception as e:
        print(f"WARNING: Image ID: {image_id} skipped due to exception: {e}")
