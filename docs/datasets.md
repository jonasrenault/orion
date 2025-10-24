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
