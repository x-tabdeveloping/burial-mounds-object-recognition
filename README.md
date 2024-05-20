# burial-mounds-object-recognition
Finetuning Object recognition models to recognize burial mounds.

## Usage

This repo was built around using YOLOv8 for detecting objects in satellite images.
I chose to organize it as a CLI so that it is easy to get started in completely fresh environments.

### Installation

To just use the CLI for preprocessing and finetuning, you can install the package from PyPI.

```bash
pip install burial_mounds
```

Make sure to also install OpenCV on your computer. Here's how you would do that for Debian-based systems (which I have used):

```bash
sudo apt update && sudo apt install python3-opencv
```

### Preprocessing

#### xView

The CLI has code for preprocessing the data in the [xView](http://xviewdataset.org/) dataset, which contains high quality annotated satellite imagery formulated as an object detection task.

Download the dataset, and arrange it in the following folder structure:

```
- data/
    - xView/
        - train_images/
            - 10.tiff
            ...
        xView_train.geojson
```

Then run the command:

```bash
python3 -m burial_mounds preprocess_xview --data_dir data/xView
```

This will convert all labels in the geoJSON file to YOLO format and output a config file for YOLO training under `configs/xview.yaml`

#### Burial Mounds

To preprocess the burial mounds dataset, you can also utilise the CLI.

The preprocessing pipeline will split the large geoTIFF files into smaller images with annotations.
The script can either prepare data for simple object detection or for [OBB](https://docs.ultralytics.com/datasets/obb/) training.

The preprocessing consists of the following steps:
1. Splitting the large raster files into smaller windows. (`--image_size` parameter controls the size of the windows)
2. Minmax color normalization.
3. Producing bounding box labels (either oriented or non-oriented formats, `--format` parameter)

To prepare the dataset for OBB:

```bash
python3 -m burial_mounds preprocess_mounds --data_dir data/TRAP_Data --out_dir data/mounds --image_size 1024 --format obb
```

For simple object detection:

```bash
python3 -m burial_mounds preprocess_mounds --data_dir data/TRAP_Data --out_dir data/mounds --image_size 640 --format detect
```

### Finetuning

There are two types of models you can choose from for finetuning.
Either models that have been trained on OBB, or simple object detection.

> OBB models have been pretrained on the DOTA satellite object recognition dataset and are therefore more likely to perform better on satellite images without finetuning.

Finetuning the models also comes with data augmentation built-in thereby increasing the robustness of trained models.

#### Detection

These are the models that you can finetune on a detection task:

| Model   | Size (pixels) | mAPval 50-95 | Speed CPU ONNX (ms) | Speed A100 TensorRT (ms) | Params (M) | FLOPs (B) |
|---------|---------------|--------------|---------------------|--------------------------|------------|-----------|
| YOLOv8n | 640           | 18.4         | 142.4               | 1.21                     | 3.5        | 10.5      |
| YOLOv8s | 640           | 27.7         | 183.1               | 1.40                     | 11.4       | 29.7      |
| YOLOv8m | 640           | 33.6         | 408.5               | 2.26                     | 26.2       | 80.6      |
| YOLOv8l | 640           | 34.9         | 596.9               | 2.43                     | 44.1       | 167.4     |
| YOLOv8x | 640           | 36.3         | 860.6               | 3.56                     | 68.7       | 260.6     |

These have been pretrained on the Open Images V7 dataset with all sorts of objects the models have to recognize.

You can finetune an already existing model on object detection with the `finetune` command.
If you want to go down this route I recommend that you finetune on xView first, so that the model will have seen satellite images before encountering the mound problem.

```bash
python3 -m burial_mounds finetune "yolov8n.pt" "configs/xview.yaml" --epochs 300 --image_size 640
```

#### OBB

These are the models that you can finetune on OBB detection:

| Model       | Size (pixels) | mAPtest 50 | Speed CPU ONNX (ms) | Speed A100 TensorRT (ms) | Params (M) | FLOPs (B) |
|-------------|---------------|------------|---------------------|--------------------------|------------|-----------|
| YOLOv8n-obb | 1024          | 78.0       | 204.77              | 3.57                     | 3.1        | 23.3      |
| YOLOv8s-obb | 1024          | 79.5       | 424.88              | 4.07                     | 11.4       | 76.3      |
| YOLOv8m-obb | 1024          | 80.5       | 763.48              | 7.61                     | 26.4       | 208.6     |
| YOLOv8l-obb | 1024          | 80.7       | 1278.42             | 11.83                    | 44.5       | 433.8     |
| YOLOv8x-obb | 1024          | 81.36      | 1759.10             | 13.23                    | 69.5       | 676.7     |

These models have been pretrained on the DOTA dataset, which contains satellite imagery, and these models are therefore more likely to be better at mound recognition.

```bash
python3 -m burial_mounds finetune "yolov8n.pt" "configs/mounds.yaml" --epochs 300 --image_size 1024
```

To run these finetuning scripts in the background (on Ucloud for instance), I recommend that you use `nohup` and store the logs.

```bash
nohup python3 -m burial_mounds finetune "yolov8n.pt" "configs/mounds.yaml" --epochs 300 --image_size 1024 &> "nano_mounds_finetune.log" &
```

### Publishing

If you intend to publish a trained model to the HuggingFace Hub you can use the `push_to_hub` command.

```bash
python3 -m burial_mounds push_to_hub --model_path "models/mounds_base-yolov8n_best.pt" --repo_id "chcaa/burial-mounds_yolov8n"
```

### Inference

You can use the models trained with this package like any other YOLO model.
The package also comes with a convenience function for loading models from HuggingFace repositories.

```python
from burial_mounds.hub import load_from_hub

model = load_from_hub("chcaa/burial_mounds_yolov8n")

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
