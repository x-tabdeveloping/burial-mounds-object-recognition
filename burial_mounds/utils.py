import numpy as np
from ultralytics.utils.ops import xywhn2xyxy
from ultralytics.utils.plotting import Annotator


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
