import hashlib
import logging
import tarfile
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36",  # noqa: E501
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",  # noqa: E501
    "Accept-Charset": "ISO-8859-1,utf-8;q=0.7,*;q=0.3",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Accept-Language": "en-US,en;q=0.8",
    "Connection": "keep-alive",
}


def download_and_extract(url: str, filename: str, save_dir: Path):
    """
    Download .tar or .zip file and extract it to given directory

    Args:
        url (str): the file's url
        filename (str): the downloaded file name
        save_dir (Path): the destination directory
    """
    save_dir.mkdir(exist_ok=True)
    dest_file = save_dir / filename
    download_file(url, dest_file)

    if dest_file.suffix == ".zip":
        with zipfile.ZipFile(dest_file, "r") as zip:
            zip.extractall(save_dir)
    else:
        with tarfile.open(dest_file) as tf:
            tf.extractall(save_dir)

    LOGGER.info(f"Extracted {dest_file} to {save_dir}.")


def download_file(
    url: str,
    file_path: Path,
    chunk_size: int = 1024,
    force: bool = False,
    sha256: str | None = None,
) -> Path:
    """
    Download a file from given url while displaying a progress bar.

    Args:
        url (str): the url to download.
        file_path (Path): path of file to save data to.
        chunk_size (int, optional): streaming chunk size in bytes. Defaults to 1024.
        force (bool, optional): if force is True and file path already exists,
            download file again. Defaults to False.
        sha256 (str | None, optional): checksum. Defaults to None.

    Returns:
        Path: the downloaded file path
    """
    if file_path.exists() and not file_path.is_file():
        raise RuntimeError(
            f"{file_path} already exists but is not a file. Cannot download."
        )

    if file_path.is_file() and not force:
        LOGGER.info(f"{file_path} already exists. Not downloading.")
        return file_path

    resp = requests.get(url, stream=True, headers=HEADERS)
    resp.raise_for_status()
    total = int(resp.headers.get("Content-Length", 0))

    with (
        open(file_path, "wb") as file,
        tqdm(
            desc=file_path.name,
            total=total,
            unit="iB",
            unit_scale=True,
            unit_divisor=chunk_size,
        ) as bar,
    ):
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)

    if sha256 is not None:
        with open(file_path, "rb") as f:
            model_bytes = f.read()
        if hashlib.sha256(model_bytes).hexdigest() != sha256:
            raise RuntimeError(
                f"Invalid checksum for downloaded file {file_path}."
                " Please retry download."
            )

    return file_path
