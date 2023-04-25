from typing import Any, Dict, List, Tuple, Union, Optional

import torch

from torchvision.transforms import functional

from grace.base import GraphAttrs

import numpy as np
import numpy.typing as npt
import networkx as nx


class RandomEdgeCrop:
    """Trim and pad an image at one of its edges to simulate the edge
    of a field-of-view.

    Accepts an image stack of size (C,W,H) or (B,C,W,H)

    Crops all images in a stack uniformly.

    Parameters
    ----------
    max_fraction : float
        Maximum fraction of image dimension that can be cropped.
    """

    def __init__(self, max_fraction: float = 0.5):
        self.max_fraction = max_fraction

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        fraction = torch.rand((1,)) * self.max_fraction * 100
        fraction = fraction.type(torch.IntTensor)
        dims = x.size()
        vignettes = torch.cat(
            (
                torch.zeros(tuple(dims[:-1]) + (fraction,)),
                torch.ones(tuple(dims[:-1]) + (100 - fraction,)),
            ),
            dim=-1,
        )
        num_rot = torch.randint(4, (1,))[0]
        vignettes = torch.rot90(vignettes, k=num_rot, dims=[-2, -1])
        vignettes = functional.resize(vignettes, dims[-2:])
        vignettes = vignettes.reshape(x.size())
        y = torch.mul(x, vignettes)

        return y


def rotate_coordinates(
    coords: npt.NDArray[np.float32],
    rot_center: npt.NDArray[np.float32],
    rot_angle: float,
) -> npt.NDArray[np.float32]:
    """Rotates point coordinates by a certain angle.

    Parameters
    ----------
    coords : numpy.array
        Input coordinates; shape (num_points, 2)
    rot_center : numpy.array
        center around which to rotate ; shape (2,)
    rot_angle : float
        Counter-clockwise rotation angle in degrees

    Returns
    -------
    rotated_coords : numpy.array
        Output coordinates
    """
    rot_rads = np.deg2rad(rot_angle)
    transform_matrix = np.array(
        [
            [np.cos(rot_rads), -np.sin(rot_rads)],
            [np.sin(rot_rads), np.cos(rot_rads)],
        ]
    )
    rotated_coords = np.matmul(transform_matrix, (coords - rot_center).T).T
    rotated_coords += rot_center
    return rotated_coords


def rotate_image_and_graph(
    image: torch.Tensor,
    target: Dict[Union[str, GraphAttrs], Any],
    rot_angle: float,
    rot_center: List[int],
) -> Tuple[torch.Tensor, Dict[Union[str, GraphAttrs], Any]]:
    """Rotate the image and graph in tandem.

    I.e., the graph x-y coordinates will be transformed to reflect
    the image rotation.

    Parameters
    ----------
    image : torch.Tensor
        Input image; shape (B,W,H) or (B,C,W,H)
    target : dict
        Input target dict.
    rot_angle : float
        Angle through which to rotate counter-clockwise
    rot_center: List[int]
        x-y coordinates of the center of the rotation

    Returns
    -------
    image : torch.Tensor
        Output image; same shape as input image
    target : dict
        Output target dict
    """

    if image.ndim < 4:
        image = image[[None] * (4 - len(image.size()))]

    image = functional.rotate(image, float(rot_angle), center=rot_center)
    image = image.squeeze()

    if rot_center is None:
        rot_center = np.array(image.size())[-2:] / 2.0

    coords = np.array(
        [
            (f[GraphAttrs.NODE_X], f[GraphAttrs.NODE_Y])
            for f in target["graph"].nodes.values()
        ],
        dtype=np.float32,
    )
    transformed_coords = rotate_coordinates(
        coords, np.array(rot_center, dtype=np.float32), rot_angle
    )

    update_coords = {
        n: {GraphAttrs.NODE_X: coords[0], GraphAttrs.NODE_Y: coords[1]}
        for n, coords in enumerate(transformed_coords)
    }

    nx.set_node_attributes(target["graph"], update_coords)

    return image, target


class RandomImageGraphRotate:
    """Rotate the image and graph in tandem; i.e., the graph x-y coordinates
    will be transformed to reflect the image rotation.

    Accepts an image stack of size (C,W,H) or (B,C,W,H) and a dictionary
    which includes the graph object.

    Parameters
    ----------
    rot_center : List[int]
        center of rotation, in x-y coordinates
    rot_angle_range : Tuple[int]
        Lower and upper limits on the rotation angle, in degrees
    rng : numpy.random.Generator
        Random number generator
    """

    def __init__(
        self,
        rot_center: Optional[List[int]] = None,
        rot_angle_range: Tuple[float, float] = (0.0, 360.0),
        rng: np.random.Generator = np.random.default_rng(),
    ):
        self.rng = rng
        self.rot_center = rot_center
        self.rot_angle_range = rot_angle_range

    def __call__(
        self, x: torch.Tensor, graph: dict
    ) -> Tuple[torch.Tensor, dict]:
        random_angle = self.rng.uniform(
            low=self.rot_angle_range[0], high=self.rot_angle_range[-1]
        )
        return rotate_image_and_graph(x, graph, random_angle, self.rot_center)
