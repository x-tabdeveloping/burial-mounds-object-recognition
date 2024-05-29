import tempfile
import warnings
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from radicli import Arg
from ultralytics import YOLO

from burial_mounds.cli import cli
from burial_mounds.model import MoundDetector

DEFAULT_README = """
---
language:
    - en
tags:
    - yolo
    - object-recognition
library_name: burial_mounds
---

# {repo}

This repository contains a YOLO model that has been finetuned by the `burial_mounds` Python package on the `{dataset}` dataset.

## Usage
```python
# pip install burial_mounds

from burial_mounds.hub import load_from_hub

model = load_from_hub("{repo}")

mounds = model(["data/mounds/images/east_30.png"])

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename="result.jpg")  # save to disk
```

For a more detailed guide consult the [YOLOv8 documentation](https://docs.ultralytics.com/modes/predict/#key-features-of-predict-mode).
"""


@cli.command(
    "push_to_hub",
    model_path=Arg("--model_path", "-m", help="Path to model to upload."),
    repo_id=Arg(
        "--repo_id",
        "-r",
        help="Repository to upload the model to.",
    ),
    skip_readme=Arg(
        "--skip_readme", help="If passed no readme will be uploaded to the repository."
    ),
)
def push_to_hub(model_path: str, repo_id: str, skip_readme: bool = False) -> None:
    api = HfApi()
    suffix = Path(model_path).suffix
    api.create_repo(repo_id, exist_ok=True)
    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=f"model.{suffix}",
        repo_id=repo_id,
        repo_type="model",
    )
    dataset = Path(model_path).stem.split("_")[0]
    with tempfile.TemporaryDirectory() as tmp_dir:
        if not skip_readme:
            readme_path = Path(tmp_dir).joinpath("README.md")
            with open(readme_path, "w") as readme_f:
                readme_f.write(DEFAULT_README.format(repo=repo_id, dataset=dataset))
            api.upload_file(
                path_or_fileobj=readme_path,
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="model",
            )


def load_from_hub(repo_id: str):
    return MoundDetector.load_from_hub(repo_id)
