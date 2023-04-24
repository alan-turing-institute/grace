from unittest.mock import patch

import torch
import pytest
import torchvision.transforms as transforms

import numpy as np

from grace.base import GraphAttrs
from grace.utils.augment_image import RandomEdgeCrop, rotate_image_and_graph

from _utils import random_image_and_graph


expected_outputs_random_edge_crop = [
    np.array(
        [
            [0.0, 0.0, 0.0, 1.4, 2.7, 8.8],
            [0.0, 0.0, 0.0, 4.2, 8.9, 4.3],
            [0.0, 0.0, 0.0, 3.8, 5.1, 2.2],
            [0.0, 0.0, 0.0, 3.2, 9.1, 2.3],
        ],
        dtype=np.float32,
    ),
    np.array(
        [
            [3.6, 7.1, 8.2, 1.4, 2.7, 8.8],
            [0.3, 1.5, 5.2, 4.2, 8.9, 4.3],
            [9.8, 8.5, 1.3, 3.8, 5.1, 2.2],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    ),
    np.array(
        [
            [3.6, 7.1, 8.2, 1.4, 2.7, 8.8],
            [0.3, 1.5, 5.2, 4.2, 8.9, 4.3],
            [9.8, 8.5, 1.3, 3.8, 5.1, 2.2],
            [2.2, 1.2, 7.4, 3.2, 9.1, 2.3],
        ],
        dtype=np.float32,
    ),
    np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [2.2, 1.2, 7.4, 3.2, 9.1, 2.3],
        ],
        dtype=np.float32,
    ),
    np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [9.8, 8.5, 1.3, 3.8, 5.1, 2.2],
            [2.2, 1.2, 7.4, 3.2, 9.1, 2.3],
        ],
        dtype=np.float32,
    ),
]


@pytest.mark.parametrize(
    "n, max_fraction, fraction, num_rot",
    [
        (0, 0.5, 1.0, 0),
        (1, 0.5, 0.5, 1),
        (2, 1.0, 0.0, 2),
        (3, 1.0, 0.75, 3),
        (4, 0.5, 1.0, 3),
    ],
)
def test_augment_random_edge_crop(n, max_fraction, fraction, num_rot):
    img = np.array(
        [
            [3.6, 7.1, 8.2, 1.4, 2.7, 8.8],
            [0.3, 1.5, 5.2, 4.2, 8.9, 4.3],
            [9.8, 8.5, 1.3, 3.8, 5.1, 2.2],
            [2.2, 1.2, 7.4, 3.2, 9.1, 2.3],
        ],
        dtype=np.float32,
    )
    img_3d = transforms.ToTensor()(img)
    img_4d = img_3d[None, ...]
    random_edge_crop = RandomEdgeCrop(max_fraction=max_fraction)

    with (
        patch(
            "torch.rand",
            return_value=torch.tensor(
                [
                    fraction,
                ],
                dtype=torch.float32,
            ),
        ),
        patch(
            "torch.randint",
            return_value=torch.tensor(
                [
                    num_rot,
                ],
                dtype=torch.int32,
            ),
        ),
    ):
        augmented_3d = random_edge_crop(img_3d).numpy()
        augmented_4d = random_edge_crop(img_4d).numpy()

    expected = expected_outputs_random_edge_crop[n]
    assert np.allclose(augmented_3d[0, ...], expected, atol=0.01)
    assert np.allclose(augmented_4d[0, 0, ...], expected, atol=0.01)


augment_rotate_coords = [
    np.array([[4, 3], [1, 2], [5, 3], [3, 3]]),
    np.array([[0, 1], [1, 0], [0, 0], [1, 1]]),
    np.array([[0, 1], [1, 0], [0, 0], [1, 1]]),
    np.array([[2, 5], [5, 3], [3, 5], [2, 2]]),
    np.array([[0, 1], [1, 0], [0, 0], [1, 1]]),
]
augment_rotate_angles = [0, 90, 45, 30, 28]
expected_end_coords_img = [
    (np.array([1, 3, 4, 5]), np.array([2, 3, 3, 3])),
    (np.array([4, 4, 5, 5]), np.array([0, 1, 0, 1])),
    (np.array([2, 3]), np.array([0, 0])),
    (np.array([1, 1, 2, 2, 4, 5]), np.array([4, 5, 2, 5, 4, 4])),
    (np.array([1, 2, 2]), np.array([0, 0, 1])),
]
expected_end_coords_float = [
    np.array([[3.0, 4.0], [2.0, 1.0], [3.0, 5.0], [3.0, 3.0]]),
    np.array([[6.0, 1.0], [5.0, 0.0], [6.0, 0.0], [5.0, 1.0]]),
    np.array([[3.71, -0.54], [2.29, -0.54], [3.0, -1.24], [3.0, 0.17]]),
    np.array([[5.23, 3.13], [2.0, 4.73], [4.73, 4.0], [2.63, 1.63]]),
    np.array([[2.64, -0.59], [1.29, -0.17], [1.76, -1.06], [2.17, 0.3]]),
]


@pytest.mark.parametrize(
    "n",
    [0, 1, 2, 3, 4],
)
def test_augment_rotate_image_and_graph(n):
    with patch("numpy.random.default_rng") as mock:
        rng = mock.return_value
        rng.uniform.return_value = 0
        rng.integers.side_effect = [augment_rotate_coords[n], [1] * 4]
        image, graph = random_image_and_graph(rng, image_size=(6, 6))

    image = torch.tensor(image.astype("int16"))
    target = {"graph": graph}

    with patch("torch.rand") as mock:
        mock.return_value = (
            torch.tensor(augment_rotate_angles[n], dtype=torch.float32) / 360.0
        )
        image, target = rotate_image_and_graph(image, target)

    augmented_img_coords = np.where(image.squeeze().numpy())
    augmented_float_coords = np.array(
        [
            [f[GraphAttrs.NODE_X], f[GraphAttrs.NODE_Y]]
            for f in target["graph"].nodes.values()
        ],
        dtype=np.float32,
    )

    assert np.array_equal(expected_end_coords_img[n], augmented_img_coords)
    assert np.allclose(
        expected_end_coords_float[n], augmented_float_coords, atol=0.01
    )
