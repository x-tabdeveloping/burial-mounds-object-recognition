import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
from radicli import Arg
from tqdm import tqdm
from ultralytics.data.utils import autosplit

from burial_mounds.cli import cli
from burial_mounds.utils import convert_bbox, image_with_annotations

XVIEW_CONFIG = """
# Ultralytics YOLO 🚀, AGPL-3.0 license

path: ../data/xView
train: images/autosplit_train.txt
val: images/autosplit_val.txt

# Classes
names:
  0: Fixed-wing Aircraft
  1: Small Aircraft
  2: Cargo Plane
  3: Helicopter
  4: Passenger Vehicle
  5: Small Car
  6: Bus
  7: Pickup Truck
  8: Utility Truck
  9: Truck
  10: Cargo Truck
  11: Truck w/Box
  12: Truck Tractor
  13: Trailer
  14: Truck w/Flatbed
  15: Truck w/Liquid
  16: Crane Truck
  17: Railway Vehicle
  18: Passenger Car
  19: Cargo Car
  20: Flat Car
  21: Tank car
  22: Locomotive
  23: Maritime Vessel
  24: Motorboat
  25: Sailboat
  26: Tugboat
  27: Barge
  28: Fishing Vessel
  29: Ferry
  30: Yacht
  31: Container Ship
  32: Oil Tanker
  33: Engineering Vehicle
  34: Tower crane
  35: Container Crane
  36: Reach Stacker
  37: Straddle Carrier
  38: Mobile Crane
  39: Dump Truck
  40: Haul Truck
  41: Scraper/Tractor
  42: Front loader/Bulldozer
  43: Excavator
  44: Cement Mixer
  45: Ground Grader
  46: Hut/Tent
  47: Shed
  48: Building
  49: Aircraft Hangar
  50: Damaged Building
  51: Facility
  52: Construction Site
  53: Vehicle Lot
  54: Helipad
  55: Storage Tank
  56: Shipping container lot
  57: Shipping Container
  58: Pylon
  59: Tower
"""

xview_to_yolo_label = {
    11: 0,
    12: 1,
    13: 2,
    15: 3,
    17: 4,
    18: 5,
    19: 6,
    20: 7,
    21: 8,
    23: 9,
    24: 10,
    25: 11,
    26: 12,
    27: 13,
    28: 14,
    29: 15,
    32: 16,
    33: 17,
    34: 18,
    35: 19,
    36: 20,
    37: 21,
    38: 22,
    40: 23,
    41: 24,
    42: 25,
    44: 26,
    45: 27,
    47: 28,
    49: 29,
    50: 30,
    51: 31,
    52: 32,
    53: 33,
    54: 34,
    55: 35,
    56: 36,
    57: 37,
    59: 38,
    60: 39,
    61: 40,
    62: 41,
    63: 42,
    64: 43,
    65: 44,
    66: 45,
    71: 46,
    72: 47,
    73: 48,
    74: 49,
    76: 50,
    77: 51,
    79: 52,
    83: 53,
    84: 54,
    86: 55,
    89: 56,
    91: 57,
    93: 58,
    94: 59,
}


@cli.command(
    "preprocess_xview",
    data_dir=Arg("--data_dir", "-d", help="Data where xView data is located."),
)
def preprocess_xview(data_dir: str = "data/xView"):
    in_dir = Path(data_dir)

    labels_dir = in_dir.joinpath("labels", "train")
    labels_dir.mkdir(parents=True, exist_ok=True)

    in_dir.joinpath("images").mkdir(exist_ok=True)
    # Checking if images are still in the original folder structure
    # and if so, changing it
    images_dir = in_dir.joinpath("train_images")
    if images_dir.is_dir():
        images_dir = images_dir.rename(in_dir.joinpath("images", "train"))
    else:
        images_dir = in_dir.joinpath("images", "train")

    with open(in_dir.joinpath("xView_train.geojson")) as in_file:
        data = json.loads(in_file.read())
    features = data["features"]

    all_images = set([feature["properties"]["image_id"] for feature in features])
    image_sizes = {}
    for image_id in tqdm(all_images, desc="Reading all image sizes."):
        try:
            with Image.open(images_dir.joinpath(image_id)) as in_image:
                image_sizes[image_id] = in_image.size
        except FileNotFoundError:
            print(f"WARNING: Image not found in directory {image_id}")

    image_labels: dict[str, list[str]] = defaultdict(list)
    for feature in tqdm(features, desc="Processing features."):
        image_id = feature["properties"]["image_id"]
        try:
            # Extracting bounding box
            bbox = [
                int(coord)
                for coord in feature["properties"]["bounds_imcoords"].split(",")
            ]
            if len(bbox) != 4:
                raise ValueError("Bounding box has an incorrect number of coordinates.")
            class_id = xview_to_yolo_label[feature["properties"]["type_id"]]
            width, height, *_ = image_sizes[image_id]
            # Converting bounding box to YOLO format
            bbox = convert_bbox(bbox, width=width, height=height)
            bbox_str = [f"{coord:.6f}" for coord in bbox]
            # Producing YOLO format entry
            entry = " ".join([str(class_id), *bbox_str]) + "\n"
            image_labels[image_id].append(entry)
        except KeyError as e:
            print(f"WARNING: Image, or Feature type not recognized: {e}")
        except Exception as e:
            print(
                f"WARNING: Feature in Image ID: {image_id} skipped due to exception: {e}"
            )
    annotated_path = images_dir.parent.joinpath("annotated")
    annotated_path.mkdir(exist_ok=True)
    out_data = list(image_labels.items())
    for image_id, entries in tqdm(out_data, desc="Saving all entries for all images."):
        with labels_dir.joinpath(image_id).with_suffix(".txt").open("w") as label_file:
            label_file.write("\n".join(entries))
    # Saving the annotated images for a sanity check
    for image_id, entries in random.sample(out_data, 30):
        with Image.open(images_dir.joinpath(image_id)) as image:
            entries = [entry.split(" ") for entry in entries]
            annotated = image_with_annotations(np.array(image), entries)
            Image.fromarray(annotated, "RGB").save(
                annotated_path.joinpath(f"{image_id}.png")
            )

    print("Producing training and validation splits.")
    autosplit(images_dir)

    print("Saving config")
    configs_dir = Path("configs")
    configs_dir.mkdir(exist_ok=True)
    with configs_dir.joinpath("xview.yaml").open("w") as config_file:
        config_file.write(XVIEW_CONFIG)
