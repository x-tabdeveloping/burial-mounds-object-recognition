from typing import Iterable, Literal

import numpy as np
from shapely import Polygon
from ultralytics.utils.ops import xywhn2xyxy
from ultralytics.utils.plotting import Annotator


def image_with_annotations(
    image: np.ndarray, annotations: list[list[str]]
) -> np.ndarray:
    """Produces image with annotations on them."""
    annotator = Annotator(image)
    h, w, *_ = image.shape
    for label, *coordinates in annotations:
        # If the length of the coordinates is 4 we can be sure the task is 'detect'
        coordinates = np.array([float(coord) for coord in coordinates])
        if len(coordinates) == 4:
            xyxy = xywhn2xyxy(coordinates, w=w, h=h)
        # If the length of the coordinates is 8 we can be sure the task is 'obb'
        elif len(coordinates) == 8:
            polygon = Polygon(np.reshape(coordinates, (4, 2)))
            minx, miny, maxx, maxy = polygon.bounds
            minx = int(minx * w)
            maxx = int(maxx * w)
            miny = int(miny * h)
            maxy = int(maxy * h)
            xyxy = np.array([minx, miny, maxx, maxy])
        else:
            raise ValueError(
                f"Label format unkown. Should contain 4 or 8 coordinates, recieved: {len(coordinates)}"
            )
        annotator.box_label(box=xyxy, label=label)

    return annotator.result()


def convert_bbox(
    bbox: Iterable[int],
    width: int,
    height: int,
    format: Literal["obb", "detect"] = "detect",
) -> tuple[float, ...]:
    """Convert bounding box to YOLO format."""
    x_min, y_min, x_max, y_max = bbox
    x_min = max(x_min, 0)
    x_max = min(x_max, width)
    y_min = max(y_min, 0)
    y_max = min(y_max, height)
    if format == "detect":
        # The format is x, y, width, height normalized to image size
        x_center = (x_min + x_max) / (2 * width)
        y_center = (y_min + y_max) / (2 * height)
        h = (y_max - y_min) / height
        w = (x_max - x_min) / width
        if (y_max < y_min) or (x_max < x_min):
            raise ValueError(
                f"Invalid bounding box: {x_min}, {y_min}, {x_max}, {y_max}"
            )
        return x_center, y_center, w, h
    elif format == "obb":
        # The format is an outline normalized to image size
        x_min = x_min / width
        x_max = x_max / width
        y_min = y_min / width
        y_max = y_max / width
        return x_min, y_min, x_min, y_max, x_max, y_max, x_max, y_min
    else:
        raise ValueError(
            f"Only the formats 'obb' and 'detect' are supported, given: {format}"
        )
