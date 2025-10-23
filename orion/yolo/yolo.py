import logging
from pathlib import Path
from typing import Annotated

import typer
from ultralytics import (
    YOLO,  # pyright: ignore[reportPrivateImportUsage]
)
from ultralytics import (
    settings as yolo_settings,
)
from ultralytics.engine.results import Results

from orion.config.settings import settings

app = typer.Typer()
LOGGER = logging.getLogger(__name__)


@app.command()
def train(
    base_model: Annotated[str, typer.Argument(metavar="model", help="base model name.")],
    data: Annotated[
        Path,
        typer.Option(
            "--data",
            "-d",
            help="training data.",
            file_okay=True,
            exists=True,
        ),
    ] = settings.ORION_HOME_DIR
    / "dataset"
    / "dataset.yaml",
    output: Annotated[
        Path,
        typer.Option(
            "--output", "-o", file_okay=False, dir_okay=True, help="save directory."
        ),
    ] = Path.cwd()
    / "runs/train",
    exist_ok: Annotated[bool, typer.Option(help="override results in save dir.")] = False,
    epochs: Annotated[int, typer.Option("--epochs", "-e", help="epochs.")] = 60,
    imgsz: Annotated[int, typer.Option("--imgsz", "-i", help="image size.")] = 640,
    batch: Annotated[int, typer.Option("--batch", "-b", help="batch size.")] = 16,
    device: Annotated[str, typer.Option(help="device.")] = "",
    plots: Annotated[bool, typer.Option(help="plot metrics during training.")] = True,
):
    """
    Fine-tune a base Yolo model on given dataset.

    Args:
        base_model (str | Path): base model name.
        data (Path): training data.
        output (Path, optional): save directory. Defaults to Path() / runs.
        exist_ok (bool, optional): override results in save dir if exists.
            Defaults to False.
        epochs (int, optional): epochs. Defaults to 60.
        imgsz (int, optional): image size. Defaults to 640.
        batch (int, optional): batch size. Defaults to 16.
        device (str, optional): device to use. Defaults to ''.
        plots (bool, optional): plot metrics during training. Defaults to True.
    """
    LOGGER.info(f"Loading model from {base_model}...")
    yolo_settings.update({"tensorboard": True})
    model = YOLO(base_model)
    project = output.parent
    name = output.name

    LOGGER.info(f"Running training. Output saved to [bold green]{output}[/].")
    results = model.train(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        exist_ok=exist_ok,
        plots=plots,
    )
    LOGGER.info(f"Training complete. Output saved to [bold green]{output}[/].")
    return results


@app.command()
def predict(
    model_path: Annotated[
        Path,
        typer.Argument(help="model path.", file_okay=True, exists=True),
    ],
    data: Annotated[
        Path,
        typer.Argument(
            help="data to make predictions on.",
            file_okay=True,
            dir_okay=True,
            exists=True,
        ),
    ],
    save_txt: Annotated[
        bool, typer.Option(help="save detection results in a txt file.")
    ] = True,
    save_conf: Annotated[
        bool, typer.Option(help="save confidence score for each detection.")
    ] = True,
    output: Annotated[
        Path,
        typer.Option(
            "--output", "-o", file_okay=False, dir_okay=True, help="save directory."
        ),
    ] = Path.cwd()
    / "runs/predict",
) -> list[Results]:
    """
    Run predictions on a set of images using the given model.

    Args:
        model_path (str | Path): the model to use for prediction.
        data (str | Path): data to make predictions on.
        save_txt (bool, optional): save detection results in a txt file. Defaults to True.
        save_conf (bool, optional): save confidence score for each detection.
            Defaults to True.
        output (Path, optional): Output directory.
            Defaults to Path.cwd() / "runs/predict".

    Returns:
        list[Results]: A list of detection results.
    """
    LOGGER.info(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    project = output.parent
    name = output.name

    LOGGER.info(f"Running prediction on {data}. Output saved to [bold green]{output}[/].")
    results = model.predict(
        data,
        stream=False,
        save_txt=save_txt,
        save_conf=save_conf,
        project=project,
        name=name,
    )
    LOGGER.info(f"Predictions complete. Output saved to [bold green]{output}[/].")
    return results


@app.command()
def track(
    model_path: Annotated[
        Path,
        typer.Argument(help="model path.", file_okay=True, exists=True),
    ],
    data: Annotated[
        Path,
        typer.Argument(
            help="input video.",
            file_okay=True,
            exists=True,
        ),
    ],
    conf: Annotated[
        float, typer.Option("--conf", "-c", help="confidence threshold for detections.")
    ] = 0.5,
    tracker: Annotated[
        str, typer.Option("--tracker", "-t", help="tracker configuration file.")
    ] = "botsort.yaml",
    output: Annotated[
        Path,
        typer.Option(
            "--output", "-o", file_okay=False, dir_okay=True, help="save directory."
        ),
    ] = Path.cwd()
    / "runs/track",
):
    """
    Track tanks in a video using a YOLO model and specified tracker.

    Args:
        model_path (str | Path): Path to the YOLO model weights file.
        source (str | Path): Path to the source of video to track
        conf (float, optional): Confidence threshold for detections . Defaults to 0.5.
        tracker (str | Path, optional): The tracker configuration file.
            Defaults to "botsort.yaml".
        output (Path, optional): Output directory. Defaults to Path.cwd() / "runs/track".

    Returns:
        results: The tracking results, typically including information on
            detected and tracked tanks.
    """
    LOGGER.info(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    project = output.parent
    name = output.name

    LOGGER.info(f"Running tracking on {data}. Output saved to [bold green]{output}[/].")
    results = model.track(
        source=data,
        conf=conf,
        tracker=tracker,
        save=True,
        stream=True,
        project=project,
        name=name,
        exist_ok=True,
    )
    list(results)
    LOGGER.info(f"Tracking complete. Output saved to [bold green]{output}[/].")
