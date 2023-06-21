from typing import Tuple, Callable
import numpy.typing as npt

import os
import cv2
import tifffile
import mrcfile

from grace.io import read_graph
from grace.simulator.simulate_image import montage_from_image

import torch
from torch.utils.data import Dataset

from pathlib import Path


class ImageGraphDataset(Dataset):
    """Creating a Torch dataset from an image directory and
    annotation (.grace file) directory.

    Parameters
    ----------
    image_dir: str
        Directory of the image files
    grace_dir: str
        Directory of the annotation (.grace) files
    image_reader_fn: Callable
        Function to read images from image filenames
    transform : Callable
        Transformation added to the images and targets
    image_filetype : str
        File extension of the image files
    """

    def __init__(
        self,
        image_dir: os.PathLike,
        grace_dir: os.PathLike,
        *,
        transform: Callable = lambda x, g: (x, g),
        image_filetype: str = "mrc",
    ) -> None:
        self.image_reader_fn = FILETYPES[image_filetype]
        self.transform = transform

        image_paths = list(Path(image_dir).glob(f"*.{image_filetype}"))
        grace_paths = list(Path(grace_dir).glob("*.grace"))

        if not image_paths:
            raise ValueError(
                "No images have been found in image_dir. Are you sure"
                " you have the right filetype?"
            )

        image_names = [p.stem for p in image_paths]
        grace_names = [p.stem for p in grace_paths]
        common_names = set(image_names).intersection(set(grace_names))

        self.image_paths = sorted(
            [p for p in image_paths if p.stem in common_names]
        )
        self.grace_paths = sorted(
            [p for p in grace_paths if p.stem in common_names]
        )

    def __len__(self) -> int:
        return len(self.grace_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        img_path = self.image_paths[idx]
        grace_path = self.grace_paths[idx]

        image = torch.tensor(
            self.image_reader_fn(img_path), dtype=torch.float32
        )

        grace_dataset = read_graph(grace_path)

        target = {}
        target["graph"] = grace_dataset.graph
        target["metadata"] = grace_dataset.metadata
        assert img_path.stem == target["metadata"]["image_filename"]

        # Transpose for training?
        image = image.t()
        print(grace_dataset.metadata["image_filename"])
        montage_from_image(
            G=grace_dataset.graph, image=image, crop_shape=(224, 224)
        )

        image, target = self.transform(image, target)

        return image, target


def mrc_reader(fn: os.PathLike) -> npt.NDArray:
    """Reads a .mrc image file

    Parameters
    ----------
    fn: str
        Image filename

    Returns
    -------
    image_data: np.ndarray
        Image array
    """
    with mrcfile.open(fn, "r") as mrc:
        # image_data = mrc.data.astype(int)
        image_data = mrc.data
    return image_data


def tiff_reader(fn: os.PathLike) -> npt.NDArray:
    """Reads a .tiff image file

    Parameters
    ----------
    fn: str
        Image filename

    Returns
    -------
    image_data: np.ndarray
        Image array
    """
    return tifffile.imread(fn).astype(int)


def png_reader(fn: os.PathLike) -> npt.NDArray:
    """Reads a .png image file

    Parameters
    ----------
    fn: str
        Image filename

    Returns
    -------
    image_data: np.ndarray
        Image array
    """
    return cv2.imread(fn, cv2.IMREAD_GREYSCALE)


FILETYPES = {
    "mrc": mrc_reader,
    "tiff": tiff_reader,
    "png": png_reader,
}
