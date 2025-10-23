import logging
import shutil
from pathlib import Path

from orion.config.settings import settings
from orion.utils import download_and_extract

LOGGER = logging.getLogger(__name__)

DATASET_URL = "https://universe.roboflow.com/ds/P2jPq32qKU?key=E4MIo8mavP"
LABEL_MAPPING = {
    "bm-21": "AFV",
    "t-80": "AFV",
    "t-64": "AFV",
    "t-72": "AFV",
    "bmp-1": "AFV",
    "bmp-2": "AFV",
    "bmd-2": "AFV",
    "btr-70": "APC",
    "btr-80": "APC",
    "mt-lb": "APC",
}


def download(dir: Path = settings.ORION_HOME_DIR / "roboflow"):
    download_and_extract(DATASET_URL, "dataset_rf.zip", dir)
    restructure_dataset(dir)


def restructure_dataset(dir: Path):
    """
    Restructure dataset by copying jpg images to the 'data' directory
    and xml files to the 'labels' directory from the source directory.

    Args:
        dir (Path): The source directory containing the dataset.
    """
    LOGGER.info(f"Restructuring dataset directory {dir}")
    # Create target subdirectories if they don't exist
    (dir / "data").mkdir(parents=True, exist_ok=True)
    (dir / "labels").mkdir(parents=True, exist_ok=True)

    # Define the source subdirectories
    source_subdirs = ["test", "train", "valid"]

    # Copy jpg and xml files from source to target
    for source_subdir in source_subdirs:
        source_path = dir / source_subdir

        # Copy jpg files to 'data'
        for jpg_file in source_path.glob("*.jpg"):
            shutil.copy(jpg_file, dir / "data")

        # Copy xml files to 'labels'
        for xml_file in source_path.glob("*.xml"):
            shutil.copy(xml_file, dir / "labels")

        # Delete the original subdirectory
        shutil.rmtree(source_path)

    LOGGER.info("Dataset directory restructured successfully.")
