from typing import Tuple, Callable
import numpy.typing as npt

import os

import mrcfile

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
        transform: Callable = lambda x,g: (x,g), 
    ) -> None:
        self.image_paths = list(Path(image_dir).iterdir())
        self.grace_paths = list(Path(grace_dir).glob("*.grace"))
        self.image_reader_fn = image_reader_fn
        self.transform = transform

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


def mrc_reader(fn: os.PathLike,
               **kwargs,) -> npt.NDArray:
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
    with mrcfile.open(fn, "r", **kwargs) as mrc:
        image_data = mrc.data.astype(int)
    return image_data
