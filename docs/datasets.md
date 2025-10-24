# Datasets

Orion uses images from open source object-detection datasets to create a dataset of military vehicles and format it correctly for [YOLO](https://github.com/ultralytics/ultralytics) training. This page describes the various datasets used by Orion.

## ImageNet

The first dataset Orion uses is `ImageNet21k`. The `ImageNet21k` dataset is available from the [image-net website](https://image-net.org/download-images.php). You need to register and be granted access to download the images. We use the Winter 21 version since it gives the option of downloading the images for a single synset (a class) and we're only interested in images of specific classes (military vehicles).

* The processed version of ImageNet21k is available on the [ImageNet21k repository](https://github.com/Alibaba-MIIL/ImageNet21K).
* The class ids and names are available in [this issue](https://github.com/google-research/big_transfer/issues/7#issuecomment-640048775).

Orion provides a `search` function to search ImageNet class names for a given query.

```python
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
```

!!! note
    The `search` function will download the list of class names and ids in the dataset `dir` if they are not already present.

Orion also provides a convenience `download` function to download images and annotations for a specific class id.

```python
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
```

!!! note
    The `download` function will only download images for classes that actually have object detection annotations (a lot of classes in the ImageNet21k dataset do not have annotations).

!!! info
    Orion uses **378 annotated images** from the ImageNet dataset, coming from the `n04389033` class (tank, army tank, armored combat vehicle, armoured combat vehicle), which are all mapped to the `AFV` class.`

## OpenImage

The second dataset Orion uses is [Open Images](https://storage.googleapis.com/openimages/web/index.html) which contains images with `Tank` detection labels. Images from OpenImages are downloaded and managed with [fiftyone](https://docs.voxel51.com/integrations/open_images.html).

!!! info
    Orion uses **1246 annotated images** from the OpenImage dataset which are all mapped to the `AFV` class.

## Russian Military annotated dataset

Another dataset Orion uses is the [Russian Military vehicles](https://universe.roboflow.com/capstoneproject/russian-military-annotated) annotated dataset provided by Tuomo Hiippala from Digital Geography Lab on Roboflow. It contains 1042 annotated images of russing military vehicles with 10 classes which we map to either the `AFC` or the `APC` class.

```python
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
```

Orion provides a `download` function to download the images and annotations from this dataset and structure the directory to be imported into a `fiftyone` dataset.

```python
def download(dir: Path = settings.ORION_HOME_DIR / "roboflow"):
    """
    Downlad images and annotations from the russian military annotated dataset
    on roboflow and format them to be imported into a fo.Dataset.

    Args:
        dir (Path, optional): the dataset dir.
            Defaults to settings.ORION_HOME_DIR / "roboflow".
    """
```

!!! info
    Orion uses **1042 annotated images** from the Russian Military vehicles dataset which are mapped to the `AFV` or `APC` class.

## Google Images

To improve our training dataset, we also scraped images of military vehicles from Google Image and annotated them by hand. This sample dataset is available for download from Orion's github repository and contains 669 images of vehicles from all four classes (`AFV`, `APC`, `MEV` and `LAV`).

!!! info
    Orion uses **669 annotated images** scraped from Google Images for all four classes.

## The Search 2

The [The Search_2](https://figshare.com/articles/dataset/The_Search_2_dataset/1041463) consist of 44 high-resolution digital color images of different complex natural scenes, with each scene (image) containing a single military vehicle that serves as a search target. This dataset is not used by Orion for training; it is used instead for evaluating the models on **realistic long range automatic target recognition (ATR) samples**.

## Command-line

Orion provides a CLI command to download and setup a dataset of annotated military vehicles for training and development of automatic target recognition models. The `prepare` command will download images from the [ImageNet](#imagenet), [OpenImages](#openimage), [Russian military](#russian-military-annotated-dataset) and [Google Images](#google-images) sources and combine them into a single dataset on disk.

The `prepare` command takes as an option the directory where all the source images will be downloaded and where the full combined dataset will be saved (by default, `~/.cache/orion`).

```bash
orion prepare --help

 Usage: orion prepare [OPTIONS]

 Prepare a dataset of annotated military vehicle images.

 Args:     dir (Path, optional): directory where files will be downloaded.         Defaults to
 ORION_HOME_DIR.

╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────╮
│ --dir   -d      DIRECTORY  Orion home directory. [default: ~/.cache/orion]                              │
│ --ids           TEXT       List of class ids to download. [default: n04389033]                          │
│ --help                     Show this message and exit.                                                  │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```
