import torch

from torchvision.transforms import functional

from grace.base import GraphAttrs

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


def rotate_image_and_graph(image: torch.Tensor, target: dict):
    """Rotate the image and graph in tandem.

    I.e., the graph x-y coordinates will be transformed to reflect
    the image rotation.

    Parameters
    ----------
    image : torch.Tensor
        Input image; shape (B,W,H) or (B,C,W,H)
    target : dict
        Input target dict.

    Returns
    -------
    image : torch.Tensor
        Output image; same shape as input image
    target : dict
        Output target dict
    """

    rot_angle = torch.rand(1) * 360

    if len(image.size()) < 4:
        image = image[[None] * (4 - len(image.size()))]

    image = functional.rotate(image, float(rot_angle))
    image = image.squeeze()

    rot_rads = torch.deg2rad(rot_angle)
    centre_coords = torch.tensor(image.size())[-2:] / 2.0
    transform_matrix = torch.tensor(
        [
            [torch.cos(rot_rads), -torch.sin(rot_rads)],
            [torch.sin(rot_rads), torch.cos(rot_rads)],
        ]
    )
    transform = (
        lambda x: torch.matmul(transform_matrix, (x - centre_coords).T).T
        + centre_coords
    )

    coords = torch.tensor(
        [
            (f[GraphAttrs.NODE_X], f[GraphAttrs.NODE_Y])
            for f in target["graph"].nodes.values()
        ],
        dtype=torch.float32,
    )
    transformed_coords = transform(coords)  # N,2

    update_coords = {
        n: {GraphAttrs.NODE_X: coords[0], GraphAttrs.NODE_Y: coords[1]}
        for n, coords in enumerate(transformed_coords)
    }

    nx.set_node_attributes(target["graph"], update_coords)

    return image, target
