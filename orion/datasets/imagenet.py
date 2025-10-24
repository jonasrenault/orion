import logging
import re
import shutil
import tarfile
from pathlib import Path

from orion.config.settings import settings
from orion.utils import download_file

LOGGER = logging.getLogger(__name__)

CLASS_ID_FILE = "imagenet21k_wordnet_ids.txt"
CLASS_NAME_FILE = "imagenet21k_wordnet_lemmas.txt"


def get_class_names(dir: Path) -> dict[str, str]:
    """
    Download class ids and names for ImageNet (if not already present in directory)
    and return a dict of class id to class name.

    Args:
        dir (Path): dataset directory

    Returns:
        dict[str, str]: dict of class id to class name
    """
    id_file = dir / CLASS_ID_FILE
    name_file = dir / CLASS_NAME_FILE

    download_file(
        "https://storage.googleapis.com/bit_models/imagenet21k_wordnet_ids.txt", id_file
    )
    download_file(
        "https://storage.googleapis.com/bit_models/imagenet21k_wordnet_lemmas.txt",
        name_file,
    )

    with open(id_file, "r") as f:
        ids = f.readlines()

    with open(name_file, "r") as f:
        names = f.readlines()

    classes = {ids[i].strip(): names[i].strip() for i in range(len(ids))}
    return classes


def download_annotations(class_ids: list[str], dir: Path) -> list[str]:
    """
    Download ImageNet annotations for given class ids, into dir.

    Args:
        class_ids (list[str]): the class ids
        dir (Path): the dataset directory

    Returns:
        list[str]: list of classes with annotations available
    """
    # Download zipfile with detections for all classes
    annotations_file = dir / "bboxes_annotations.tar.gz"
    annotations_dir = dir / "bboxes_annotations"
    download_file(
        "https://image-net.org/data/bboxes_annotations.tar.gz", annotations_file
    )

    # Extract annotations
    with tarfile.open(annotations_file, "r:gz") as tf:
        tf.extractall(annotations_dir)

    # Extract annotations for each class
    annoted_classes = []
    for class_id in class_ids:
        class_label_dir = dir / "labels" / class_id
        if class_label_dir.exists():
            LOGGER.info(
                f"Annotations directory {class_label_dir} already exists. "
                "Skipping extract."
            )
        else:
            annotations_class_file = annotations_dir / f"{class_id}.tar.gz"
            if annotations_class_file.exists():
                with tarfile.open(annotations_class_file, "r:gz") as tf:
                    tf.extractall(annotations_dir)
                shutil.move(annotations_dir / "Annotation" / class_id, class_label_dir)
                LOGGER.info(f"Extracted annotations for {class_id} to {class_label_dir}")
                annoted_classes.append(class_id)
            else:
                LOGGER.info(f"There are no annotations for class {class_id}.")

    # Delete annotations directory
    LOGGER.info("Deleting annotations dir.")
    shutil.rmtree(annotations_dir)
    return annoted_classes


def download_imagenet_detections(class_ids: list[str], dir: Path):
    """
    Download ImageNet images and annotations for given class ids into dir

    Args:
        class_ids (list[str]): class_ids to download
        dir (Path): the directory to save images into
    """
    # Create dataset_dir
    dir.mkdir(exist_ok=True)
    data_dir = dir / "data"
    data_dir.mkdir(exist_ok=True)
    labels_dir = dir / "labels"
    labels_dir.mkdir(exist_ok=True)

    annoted_classes = download_annotations(class_ids, dir)

    # Download synset images for each class with annotations
    for class_id in annoted_classes:
        class_dir = data_dir / class_id
        if class_dir.exists():
            LOGGER.info(f"Directory {class_dir} already exists. Skipping download.")
        else:
            tarfilename = dir / f"{class_id}.tar"
            url = f"https://image-net.org/data/winter21_whole/{class_id}.tar"
            download_file(url, tarfilename)
            with tarfile.open(tarfilename) as tf:
                tf.extractall(class_dir)
            LOGGER.info(f"Extracted {class_dir}.")


def cleanup_labels_without_images(dir: Path):
    """
    Remove labels without images from dataset dir

    Args:
        dir (Path): the ImageNet dataset directory
    """
    data_dir = dir / "data"
    labels_dir = dir / "labels"
    classes = [path.name for path in data_dir.iterdir() if path.is_dir()]
    for class_id in classes:
        images = {
            path.stem for path in (data_dir / class_id).iterdir() if not path.is_dir()
        }
        labels = {
            path.stem for path in (labels_dir / class_id).iterdir() if not path.is_dir()
        }
        LOGGER.info(f"Deleting {len(labels.difference(images))} labels without images")
        for label_id in labels.difference(images):
            filename = labels_dir / class_id / (label_id + ".xml")
            filename.unlink()


def search(
    keywords: list[str],
    dir: Path = settings.ORION_HOME_DIR / "imagenet",
):
    """
    Search image net classes matching the given keywords.

    Args:
        keywords (list[str]): List of keywords to search for.
        dir (Path, optional): directory where files will be downloaded.
            Defaults to ORION_HOME_DIR / "imagenet".
    """
    classes = get_class_names(dir)
    filtered = {
        id: lemma
        for id, lemma in classes.items()
        if any([re.search(query, lemma, re.IGNORECASE) for query in keywords])
    }
    LOGGER.info(f"Displaying all ImageNet classes containing one of {keywords}")
    LOGGER.info("\n".join([f"{id}: \t{name}" for id, name in filtered.items()]))


def download(
    ids: list[str],
    dir: Path = settings.ORION_HOME_DIR / "imagenet",
):
    """
    Download ImageNet images and annotations for the given class ids.

    Args:
        ids (list[str]): the class ids to download.
        dir (Path, optional): the dataset directory.
            Defaults to settings.ORION_HOME_DIR / "imagenet".
    """
    download_imagenet_detections(ids, dir)
    cleanup_labels_without_images(dir)
