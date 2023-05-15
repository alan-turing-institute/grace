from typing import Tuple, Callable
import numpy.typing as npt

import os

import mrcfile
import warnings

from grace.io import read_graph

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
    """

    def __init__(
        self,
        image_dir: os.PathLike,
        grace_dir: os.PathLike,
        image_reader_fn: Callable,
        *,
        transform: Callable = lambda x, g: (x, g),
    ) -> None:
        self.image_reader_fn = image_reader_fn
        self.transform = transform

        image_paths = list(Path(image_dir).iterdir())
        grace_paths = list(Path(grace_dir).glob("*.grace"))
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
        image_data = mrc.data.astype(int)
    return image_data


def read_mrc(path):
    """
    Takes a path and read a mrc file convert the data to np array
    """
    warnings.simplefilter(
        "ignore"
    )  # to mute some warnings produced when opening the tomos
    with mrcfile.open(path, mode="r+", permissive=True) as mrc:
        mrc.update_header_from_data()
        mrc.header.map = mrcfile.constants.MAP_ID
        mrc = mrc.data
    with mrcfile.open(path) as mrc:
        data = mrc.data.astype(int)
    return data
