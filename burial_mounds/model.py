import warnings
from pathlib import Path
from typing import Union

from huggingface_hub import snapshot_download
from PIL import Image
from shapely import Polygon, box
from ultralytics.models.yolo.model import YOLO
from ultralytics.utils.plotting import Annotator


class MoundDetector(YOLO):
    @classmethod
    def load_from_hub(cls, repo_id: str):
        """Loads model from Huggingface repository."""
        in_dir = snapshot_download(repo_id=repo_id)
        model_files = list(Path(in_dir).glob("model.*"))
        if not model_files:
            raise ValueError("Repo does not contain model file.")
        if len(model_files) > 1:
            warnings.warn(f"Multiple model files in repo, loading {model_files[0]}")
        return cls(model_files[0])

    def detect_mounds(
        self, image: Union[str, Path, Image], normalized: bool = False
    ) -> list[Polygon]:
        """Detects burial mounds from image.
        Returns shapely polygons around the mound.

        Parameters
        ----------
        image: str or Path or Image
            Image path or PIL image.
        normalized: bool, default False
            Indicates whether the results should be normalized
            to image size.

        Returns
        -------
        list[Polgyon]
            List of polygons enclosing predicted burial mounds in
            the image.
        """
        results = self(image)[0]
        if results.obb is not None:
            coords = results.obb.xyxyxyxyn if normalized else results.obb.xyxyxyxy
            return [Polygon(coord.numpy()) for coord in coords]
        elif results.boxes is not None:
            coords = results.boxes.xyxyn if normalized else results.boxes.xyxy
            return [box(coord.numpy()) for coord in coords]

    def annotate_image(self, image: Union[str, Path, Image]) -> Image:
        """Annotates image with the model's predictions.

        Parameters
        ----------
        image: str or Path or Image
            Image path or PIL image.

        Returns
        -------
        Image
            Annotated image with bounding boxes of burial mounds.
        """
        if isinstance(image, (Path, str)):
            image = Image.open(image)
        detections = self.detect_mounds(image, normalized=False)
        annotator = Annotator(image)
        for detection in detections:
            annotator.box_label(box=detection.bounds, label="mound")
        return Image.fromarray(annotator.result())
