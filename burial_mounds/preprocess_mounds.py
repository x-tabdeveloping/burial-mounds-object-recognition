from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import rasterio
import shapely
from PIL import Image
from rasterio.windows import Window
from tqdm import tqdm
from ultralytics.utils.ops import xyxy2xywhn


def minmax(channel: np.ndarray) -> np.ndarray:
    return (channel - np.min(channel)) / (np.max(channel) - np.min(channel))


def iterate_windows(dataset, size: int = 1024) -> Iterable[tuple[int, int, Window]]:
    w, h = dataset.shape
    for i_horizontal in range(w // size):
        for i_vertical in range(h // size):
            yield i_horizontal, i_vertical, Window(
                i_vertical * size, i_horizontal * size, size, size
            )


out_path = Path("images")
out_path.mkdir(exist_ok=True)
print("Loading bounding boxes")
boxes = gpd.read_file("data/TRAP_Data/Kaz_mndbbox.geojson")
boxes = boxes.set_crs(epsg=32635, allow_override=True)

with rasterio.open("data/TRAP_Data/East/kaz_e_fuse.img") as east:
    n_windows = len(list(iterate_windows(east)))
    for i, j, window in tqdm(
        iterate_windows(east), total=n_windows, desc="Processing windows in raster."
    ):
        window_bounds = shapely.box(*east.window_bounds(window))
        # YOLO entries for window
        window_entries: list[list[str]] = []
        for certainty, polygon in zip(boxes.Certainty, boxes.geometry):
            if not polygon.intersects(window_bounds):
                continue
            polygon = polygon.intersection(window_bounds)
            projected_points = [east.index(*point) for point in polygon.exterior.coords]
            bbox = shapely.Polygon(projected_points).bounds
            # substracting window's starting point
            bbox = np.array(bbox) - np.array([window.row_off, window.col_off] * 2)
            # Converting to YOLO format
            yolo_bbox = xyxy2xywhn(bbox, w=window.width, h=window.height, clip=True)
            # Adding YOLO entry for window
            window_entries.append(["mound", *[f"{coord:.6f}" for coord in yolo_bbox]])
        # Go to next window if there are no mounds in it
        if not window_entries:
            continue
        # Stacking RGB channels from the window
        image = np.stack([east.read(ch, window=window) for ch in (4, 3, 2)])
        # Colors need to be normalized, I will do a minmax
        image = np.stack([minmax(channel) for channel in image])
        image = image * 255
        image = image.astype(np.uint8)
        # Transposing so that the color channel is last
        image = image.transpose((1, 2, 0))
        image = Image.fromarray(image, "RGB")
        image.save(out_path.joinpath(f"east_{i}_{j}.png"))
