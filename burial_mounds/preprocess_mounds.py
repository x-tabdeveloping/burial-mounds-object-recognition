from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import rasterio
import shapely
from PIL import Image, ImageEnhance
from radicli import Arg
from rasterio.features import geometry_window
from rasterio.windows import Window
from tqdm import tqdm
from ultralytics.data.utils import autosplit
from ultralytics.utils.ops import xywhn2xyxy, xyxy2xywhn
from ultralytics.utils.plotting import Annotator

from burial_mounds.cli import cli

MOUNDS_CONFIG = """
# Ultralytics YOLO 🚀, AGPL-3.0 license

path: ../data/TRAP_Data
train: autosplit_train.txt
val: autosplit_val.txt

# Classes
names:
  0: Mound
"""


def minmax(channel: np.ndarray) -> np.ndarray:
    """Minmax normalizes a whole array and keeps it the same shape."""
    return (channel - np.min(channel)) / (np.max(channel) - np.min(channel))


def iterate_windows(dataset, size: int = 1024) -> Iterable[tuple[int, int, Window]]:
    """Iterates through windows in a rasterio dataset."""
    w, h = dataset.shape
    for i_horizontal in range(w // size):
        for i_vertical in range(h // size):
            yield i_horizontal, i_vertical, Window(
                i_vertical * size, i_horizontal * size, size, size
            )


def to_yolo_entry(bbox, width, height) -> list[str]:
    """Turns bounding boxes to YOLO annotation entries."""
    yolo_bbox = xyxy2xywhn(
        x=np.array(bbox, dtype=np.float64), w=width, h=height, clip=True
    )
    yolo_bbox_str = [f"{coord:.6f}" for coord in yolo_bbox]
    return ["0", *yolo_bbox_str]


def image_with_annotations(
    image: np.ndarray, annotations: list[list[str]]
) -> np.ndarray:
    """Produces image with annotations on them."""
    annotator = Annotator(image)
    w, h, *_ = image.shape
    for label, *xywhn in annotations:
        xywhn = np.array([float(coord) for coord in xywhn])
        xyxy = xywhn2xyxy(xywhn, w=w, h=h)
        annotator.box_label(box=xyxy, label=label)
    return annotator.result()


@cli.command(
    "preprocess_mounds",
    data_dir=Arg("--data_dir", "-d", help="Data where mounds data is located."),
    image_size=Arg(
        "--image_size", "-s", help="Size of the square shaped images to produce."
    ),
)
def preprocess_mounds(data_dir: str = "data/TRAP_Data", image_size: int = 640):
    """Preprocesses the mounds dataset.
    Creates square images with annotations, corrects satellite color
    channels and produces train and test splits."""
    data_path = Path(data_dir)
    print("Loading bounding boxes")
    boxes = gpd.read_file(data_path.joinpath("Kaz_mndbbox.geojson"))
    boxes = boxes.set_crs(epsg=32635, allow_override=True)

    files = {
        "east": data_path.joinpath("East/kaz_e_fuse.img"),
        "west": data_path.joinpath("East/kaz_w_fuse.img"),
        "joint": data_path.joinpath("East/kaz_fuse.img"),
    }
    images_path = data_path.joinpath("images")
    images_path.mkdir(exist_ok=True)
    labels_path = data_path.joinpath("labels")
    labels_path.mkdir(exist_ok=True)
    annotated_path = data_path.joinpath("annotated")
    annotated_path.mkdir(exist_ok=True)
    bands = []
    # YOLO entries for each image
    entries: list[list[list[str]]] = []
    for name, file in files.items():
        print(f"Processing {name}")
        with rasterio.open(file) as dataset:
            windows = list(iterate_windows(dataset))
            for i, j, window in tqdm(windows, desc="Going through windows."):
                window_bounds = shapely.box(*dataset.window_bounds(window))
                # YOLO entries for window
                window_entries: list[list[str]] = []
                for certainty, polygon in zip(boxes.Certainty, boxes.geometry):
                    if not polygon.intersects(window_bounds):
                        continue
                    polygon = polygon.intersection(window_bounds)
                    enclosing_window = geometry_window(dataset, [polygon])
                    bbox = [
                        enclosing_window.col_off - window.col_off,
                        enclosing_window.row_off - window.row_off,
                        enclosing_window.col_off
                        + enclosing_window.width
                        - window.col_off,
                        enclosing_window.row_off
                        + enclosing_window.height
                        - window.row_off,
                    ]
                    window_entries.append(
                        to_yolo_entry(bbox, width=image_size, height=image_size)
                    )
                # Go to next window if there are no mounds in it
                if not window_entries:
                    continue
                image = np.stack([dataset.read(ch, window=window) for ch in (1, 2, 3)])
                bands.append(image)
                entries.append(window_entries)

        images = np.stack(bands)
        images = minmax(images) * 256
        idx = tqdm(np.arange(len(images)), desc="Saving images")
        for i, image, window_entries in zip(idx, images, entries):
            image = Image.merge(
                "RGB", [Image.fromarray(ch.astype(np.uint8)) for ch in image]
            )
            image = ImageEnhance.Brightness(image).enhance(2.0)
            image = ImageEnhance.Contrast(image).enhance(2.0)
            image.save(images_path.joinpath(f"{name}_{i}.png"))
            with labels_path.joinpath(f"{name}_{i}.txt").open("w") as labels_file:
                for entry in window_entries:
                    labels_file.write(" ".join(entry) + "\n")
            annotated = image_with_annotations(np.array(image), window_entries)
            Image.fromarray(annotated, "RGB").save(
                annotated_path.joinpath(f"{name}_{i}.png")
            )

    print("Producing training and validation splits.")
    autosplit(images_path)

    print("Saving config")
    configs_dir = Path("configs")
    configs_dir.mkdir(exist_ok=True)
    with configs_dir.joinpath("mounds.yaml").open("w") as config_file:
        config_file.write(MOUNDS_CONFIG)
