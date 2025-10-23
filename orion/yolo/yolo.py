import logging
from pathlib import Path
from typing import Annotated

import typer
from ultralytics import YOLO

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
    epochs: Annotated[int, typer.Option("--epochs", "-e", help="epochs.")] = 60,
    imgsz: Annotated[int, typer.Option("--imgsz", "-i", help="image size.")] = 640,
    batch: Annotated[int, typer.Option("--batch", "-b", help="batch size.")] = 16,
    device: Annotated[str, typer.Option(help="device.")] = "cpu",
):
    """
    Train a Yolo base model with data in the data directory.

    Args:
        base_model (str | Path): base model name.
        data (Path): training data.
        epochs (int, optional): epochs. Defaults to 60.
        imgsz (int, optional): image size. Defaults to 640.
        batch (int, optional): batch size. Defaults to 16.
        device (str, optional): device to use. Defaults to 'cpu'.
    """
    model = YOLO(base_model)  # load a pretrained model
    results = model.train(
        data=data, epochs=epochs, imgsz=imgsz, batch=batch, device=device
    )
    return results


def predict(
    model_path: str | Path,
    source: str | Path,
    save_txt: bool = True,
    save_conf: bool = True,
):
    """
    Run prediction on a source using the given model

    Args:
        model_path (str | Path): the model to use for prediction
        source (str | Path): the source of data to predict
        save_txt (bool, optional): save detection results in a txt file. Defaults to True.
        save_conf (bool, optional): save confidence score for each detection.
            Defaults to True.

    Returns:
        list: A list of detection results.
    """
    model = YOLO(model_path)

    # Run inference on the source
    results = model.predict(source, stream=False, save_txt=save_txt, save_conf=save_conf)
    return results


def track(
    model_path: str | Path,
    source: str | Path,
    conf: float = 0.5,
    save: bool = True,
    tracker: str | Path = "botsort.yaml",
):
    """
    Track tanks in a video using a YOLO model and specified tracker.

    Args:
        model_path (str | Path): Path to the YOLO model weights file.
        source (str | Path): Path to the source of video to track
        conf (float, optional): Confidence threshold for detections . Defaults to 0.5.
        save (bool, optional): Save the processed video with tracked tanks.
            Defaults to True.
        tracker (str | Path, optional): The tracker configuration file.
            Defaults to "botsort.yaml".

    Returns:
        results: The tracking results, typically including information on
            detected and tracked tanks.
    """
    model = YOLO(model_path)
    results = model.track(source=source, conf=conf, save=save, tracker=tracker)
    return results
