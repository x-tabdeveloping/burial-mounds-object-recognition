from pathlib import Path

from radicli import Arg
from ultralytics import YOLO

from burial_mounds.cli import cli


@cli.command(
    "finetune",
    base_model=Arg(help="Base Model to finetune."),
    config=Arg(help="Name of the config to finetune the model on or path to config."),
    epochs=Arg("--epochs", "-e", help="Number of epochs for training."),
    image_size=Arg("--image_size", "-s", help="Size of the images to use in training."),
)
def finetune(
    base_model: str,
    config: str,
    epochs: int = 100,
    image_size: int = 640,
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
    model = YOLO(base_model)
    # Train the model
    results = model.train(
        data=config_path,
        epochs=epochs,
        imgsz=image_size,
        degrees=180,
        flipud=0.3,
        optimizer="Adam",
        lr0=0.01,
    )

    print("Validating model:")
    model.val()
    success = model.export()
    success = Path(success)

    extension = success.suffix
    base_model_name = Path(base_model).stem
    out_path = models_dir.joinpath(
        f"{config_path.stem}_base-{base_model_name}_best{extension}"
    )
    # Moving model to output path
    success.rename(out_path)
    print(f"Saved best model to {out_path}")
