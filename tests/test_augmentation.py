from unittest.mock import patch

import torch
import pytest
import torchvision.transforms as transforms

import numpy as np

from grace.utils.augment_image import RandomEdgeCrop

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
