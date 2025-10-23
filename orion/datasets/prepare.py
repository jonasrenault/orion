import logging
from pathlib import Path
from typing import Annotated

import fiftyone as fo
import fiftyone.utils.random as four
import fiftyone.zoo as foz
import typer
from fiftyone.types.dataset_types import VOCDetectionDataset, YOLOv4Dataset

from orion.config.settings import settings
from orion.datasets.imagenet import download as download_imagenet
from orion.datasets.roboflow import LABEL_MAPPING
from orion.datasets.roboflow import download as download_roboflow
from orion.utils import download_and_extract
from orion.yolo.utils import export_yolo_data

app = typer.Typer()
LOGGER = logging.getLogger(__name__)

GOOGLE_DATASET_URL = "https://github.com/jonasrenault/adomvi/releases/download/v1.2.0/military-vehicles-dataset.tar.gz"


@app.command()
def prepare(
    dir: Annotated[
        Path,
        typer.Option(
            "--dir",
            "-d",
            help="Orion home directory.",
            file_okay=False,
            dir_okay=True,
        ),
    ] = settings.ORION_HOME_DIR,
    imagenet_ids: Annotated[
        list[str], typer.Option("--ids", help="List of class ids to download.")
    ] = ["n04389033"],
):
    """
    Prepare a dataset of annotated military vehicle images.

    Args:
        dir (Path, optional): directory where files will be downloaded.
            Defaults to ORION_HOME_DIR / "imagenet".
    """
    LOGGER.info("========== Downloading images from ImageNet dataset ==========")
    # Download ImageNet images for classes imagenet_ids
    imagenet_dir = dir / "imagenet"
    download_imagenet(imagenet_ids, imagenet_dir)

    # Create a dataset
    dataset = fo.Dataset.from_dir(
        dataset_dir=imagenet_dir, dataset_type=VOCDetectionDataset
    )
    dataset.map_labels("ground_truth", {"n04389033": "AFV"}).save()
    LOGGER.info(
        f"========== Dataset currently contains {dataset.count()} samples =========="
    )

    LOGGER.info("========== Downloading images from OpenImage dataset ==========")
    # Add OpenImages dataset
    oi_samples = foz.load_zoo_dataset(
        "open-images-v7", classes=["Tank"], only_matching=True, label_types="detections"
    ).map_labels("ground_truth", {"Tank": "AFV"})
    dataset.merge_samples(oi_samples)
    LOGGER.info(
        f"========== Dataset currently contains {dataset.count()} samples =========="
    )

    LOGGER.info("========== Downloading images from Roboflow dataset ==========")
    # Add roboflow dataset
    roboflow_dir = dir / "roboflow"
    download_roboflow(roboflow_dir)
    dataset_rf = fo.Dataset.from_dir(
        dataset_dir=roboflow_dir, dataset_type=VOCDetectionDataset
    )
    dataset_rf.map_labels("ground_truth", LABEL_MAPPING).save()
    dataset.merge_samples(dataset_rf)
    LOGGER.info(
        f"========== Dataset currently contains {dataset.count()} samples =========="
    )

    LOGGER.info("========== Downloading images from Google Images dataset ==========")
    # Add google images dataset
    google_dir = dir / "google"
    download_and_extract(
        GOOGLE_DATASET_URL, "military-vehicles-dataset.tar.gz", google_dir
    )
    dataset_google = fo.Dataset.from_dir(
        dataset_dir=google_dir / "dataset", dataset_type=YOLOv4Dataset
    )
    dataset.merge_samples(dataset_google)
    LOGGER.info(
        f"========== Dataset currently contains {dataset.count()} samples =========="
    )

    export_dir = dir / "dataset"
    LOGGER.info(f"========== Exporting dataset to {export_dir} ==========")
    # delete existing tags
    dataset.untag_samples(dataset.distinct("tags"))

    # split into train, test and val
    four.random_split(dataset, {"train": 0.8, "val": 0.1, "test": 0.1})
    export_yolo_data(
        dataset,
        export_dir,
        ["AFV", "APC", "MEV", "LAV"],
        split=["train", "val", "test"],
        overwrite=True,
    )
