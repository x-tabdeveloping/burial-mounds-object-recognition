from typing import Iterable

import numpy as np
from ultralytics.utils.ops import xywhn2xyxy
from ultralytics.utils.plotting import Annotator


def image_with_annotations(
    image: np.ndarray, annotations: list[list[str]]
) -> np.ndarray:
    """Produces image with annotations on them."""
    annotator = Annotator(image)
    h, w, *_ = image.shape
    for label, *xywhn in annotations:
        xywhn = np.array([float(coord) for coord in xywhn])
        xyxy = xywhn2xyxy(xywhn, w=w, h=h)
        annotator.box_label(box=xyxy, label=label)
    return annotator.result()


def convert_bbox(
    bbox: Iterable[int], width: int, height: int
) -> tuple[float, float, float, float]:
    """Convert bounding box to YOLO format."""
    x_min, y_min, x_max, y_max = bbox
    x_min = max(x_min, 0)
    x_max = min(x_max, width)
    y_min = max(y_min, 0)
    y_max = min(y_max, height)
    x_center = (x_min + x_max) / (2 * width)
    y_center = (y_min + y_max) / (2 * height)
    h = (y_max - y_min) / height
    w = (x_max - x_min) / width
    if (y_max < y_min) or (x_max < x_min):
        raise ValueError(f"Invalid bounding box: {x_min}, {y_min}, {x_max}, {y_max}")
    return x_center, y_center, w, h
