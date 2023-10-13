from typing import Callable
import numpy.typing as npt
import numpy as np
import networkx as nx

import os
import cv2
import tifffile
import mrcfile

from grace.io import read_graph
from grace.base import GraphAttrs, Annotation
from grace.styling import LOGGER

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
    keep_node_unknown_labels : bool
        If True, the Annotation.UNKNOWN will remain in graph.
        If False, all UNKNOWN nodes are relabelled to TRUE_NEGATIVE
    keep_edge_unknown_labels : bool
        If True, the Annotation.UNKNOWN will remain in graph.
        If False, all UNKNOWN edges are relabelled to TRUE_NEGATIVE
    verbose : bool
        Whether to print out the image node & edge statistics.
    """

    def __init__(
        self,
        image_dir: os.PathLike,
        grace_dir: os.PathLike,
        *,
        transform: Callable = lambda x, g: (x, g),
        image_filetype: str = "mrc",
        keep_node_unknown_labels: bool = True,
        keep_edge_unknown_labels: bool = True,
        verbose: bool = True,
    ) -> None:
        self.image_reader_fn = FILETYPES[image_filetype]
        self.keep_node_unknown_labels = keep_node_unknown_labels
        self.keep_edge_unknown_labels = keep_edge_unknown_labels
        self.transform = transform
        self.verbose = verbose

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

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        img_path = self.image_paths[idx]
        grace_path = self.grace_paths[idx]

        image = torch.tensor(self.image_reader_fn(img_path).astype("float32"))
        grace_dataset = read_graph(grace_path)
        graph = grace_dataset.graph

        # Print original graph label statistics:
        if self.verbose is True:
            LOGGER.info(img_path.stem)
            log_graph_label_statistics(graph)

        # Relabel Annotation.UNKNOWN in nodes:
        if self.keep_node_unknown_labels is False:
            relabel_unknown_node_labels(G=graph)

        # Relabel Annotation.UNKNOWN in edges:
        if self.keep_edge_unknown_labels is False:
            relabel_unknown_edge_labels(G=graph)

        # Print updated statistics:
        if self.verbose is True:
            if (
                self.keep_node_unknown_labels is False
                or self.keep_edge_unknown_labels is False
            ):
                LOGGER.info("Relabelled 'Annotation.UNKNOWN'")
                log_graph_label_statistics(graph)

        # Package together:
        target = {}
        target["graph"] = graph
        target["metadata"] = grace_dataset.metadata
        target["annotation"] = grace_dataset.annotation
        assert img_path.stem == target["metadata"]["image_filename"]

        image, target = self.transform(image, target)

        return image, target


def relabel_unknown_node_labels(G: nx.Graph):
    """Relabels all Annotation.UNKNOWN nodes
    to Annotation.TRUE_NEGATIVE by in-place graph
    modification. Good for exhaustive labelling.
    """
    for _, node in G.nodes(data=True):
        if node[GraphAttrs.NODE_GROUND_TRUTH] == Annotation.UNKNOWN:
            node[GraphAttrs.NODE_GROUND_TRUTH] = Annotation.TRUE_NEGATIVE


def relabel_unknown_edge_labels(G: nx.Graph):
    """Relabels all Annotation.UNKNOWN edges
    to Annotation.TRUE_NEGATIVE by in-place graph
    modification. Good for exhaustive labelling.
    """
    for _, _, edge in G.edges(data=True):
        if edge[GraphAttrs.EDGE_GROUND_TRUTH] == Annotation.UNKNOWN:
            edge[GraphAttrs.EDGE_GROUND_TRUTH] = Annotation.TRUE_NEGATIVE


def log_graph_label_statistics(G: nx.Graph) -> None:
    graph_attributes = ["nodes", "edges"]
    component_list = [G.nodes(data=True), G.edges(data=True)]
    gt_label_keys = [
        GraphAttrs.NODE_GROUND_TRUTH,
        GraphAttrs.EDGE_GROUND_TRUTH,
    ]

    for a, attribute in enumerate(graph_attributes):
        counter = [0 for _ in range(len(Annotation))]

        for comp in component_list[a]:
            label = comp[-1][gt_label_keys[a]]
            counter[label.value] += 1

        perc = [item / np.sum(counter) for item in counter]
        perc = [float("%.2f" % (elem * 100)) for elem in perc]
        string = f"{attribute.capitalize()} count | {counter} x | {perc} %"
        LOGGER.info(string)


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
    # return tifffile.imread(fn).astype(int)
    return tifffile.imread(fn)


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
