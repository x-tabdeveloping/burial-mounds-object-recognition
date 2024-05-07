from pathlib import Path

from radicli import Arg
from ultralytics import YOLO

from burial_mounds.cli import cli


@cli.command(
    "finetune",
    config=Arg(help="Name of the config to finetune the model on or path to config."),
    base_model=Arg("--base_model", "-b", help="Base Model to finetune."),
    epochs=Arg("--epochs", "-e", help="Number of epochs for training."),
)
def finetune(
    config: str,
    base_model: str = "yolov8n.pt",
    epochs: int = 100,
):
    # User may either specify a path to a config file, or just a name like "mounds", "xview"
    if config.endswith(".yaml") or config.endswith(".yml"):
        config_path = Path(config)
    else:
        config_path = Path("configs").joinpath(f"{config}.yaml")

    models_dir = Path("models/")
    models_dir.mkdir(exist_ok=True)

    print("Training model")
    # Load a model
    model = YOLO("yolov8n.pt")
    # Train the model
    results = model.train(data=config_path, epochs=epochs, imgsz=640)

    print("Validating model:")
    model.val()
    success = model.export(format="onnx")
    success = Path(success)

    extension = success.suffix
    out_path = models_dir.joinpath(
        f"{config_path.stem}_base-{base_model}_best.{extension}"
    )
    # Moving model to output path
    success.rename(out_path)
