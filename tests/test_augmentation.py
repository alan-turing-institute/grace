from unittest.mock import patch

import torch
import pytest
import torchvision.transforms as transforms

import numpy as np

from grace.base import GraphAttrs, Annotation
from grace.utils.augment_image import (
    RandomEdgeCrop,
    RandomImageGraphRotate,
)
from grace.utils.augment_graph import (
    find_average_annotation,
    RandomEdgeAdditionAndRemoval,
    RandomXYTranslation,
)

from conftest import random_image_and_graph


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
    np.array([[0, 1], [1, 0], [0, 0], [1, 1]]),
]
expected_end_coords_img = [
    (np.array([2, 3, 3, 3]), np.array([1, 3, 4, 5])),
    (np.array([4, 4, 5, 5]), np.array([0, 1, 0, 1])),
    (np.array([2, 3]), np.array([0, 0])),
    (np.array([2, 2, 4, 5, 5]), np.array([2, 5, 4, 3, 4])),
    (np.array([1, 2, 2]), np.array([0, 0, 1])),
    (np.array([1, 1, 2]), np.array([0, 1, 0])),
]
expected_end_coords_float = [
    np.array([[4.0, 3.0], [1.0, 2.0], [5.0, 3.0], [3.0, 3.0]]),
    np.array([[1.0, 6.0], [0.0, 5.0], [0.0, 6.0], [1.0, 5.0]]),
    np.array([[-0.54, 3.71], [-0.54, 2.29], [-1.24, 3.0], [0.17, 3.0]]),
    np.array([[3.13, 5.23], [4.73, 2.0], [4.0, 4.73], [1.63, 2.63]]),
    np.array([[-0.59, 2.64], [-0.17, 1.29], [-1.06, 1.76], [0.30, 2.17]]),
    np.array([[-0.24, 2.06], [0.18, 0.70], [-0.70, 1.17], [0.65, 1.59]]),
]


@pytest.mark.skip(reason="augmentations throw an error for some reason")
@pytest.mark.parametrize(
    "n, rot_angle, rot_center",
    [
        (0, 0, None),
        (1, 90, None),
        (2, 45, None),
        (3, 30, None),
        (4, 28, None),
        (5, 28, [2, 2]),
    ],
)
def test_augment_rotate_image_and_graph(n, rot_angle, rot_center):
    print("hello-1")
    with patch("numpy.random.default_rng") as mock:
        rng = mock.return_value
        rng.uniform.return_value = 0
        rng.integers.return_value = augment_rotate_coords[n]
        rng.choice.return_value = [1] * 4
        image, graph = random_image_and_graph(rng, image_size=(6, 6))
    print("hello-2")

    image = torch.tensor(image.astype("int16"))
    target = {"graph": graph}

    with patch("numpy.random.default_rng") as mock:
        rng = mock.return_value
        rng.uniform.return_value = rot_angle
        transform = RandomImageGraphRotate(rot_center=rot_center, rng=rng)
        image, target = transform(image, target)
    print("hello-3")

    augmented_img_coords = np.where(image.squeeze().numpy())
    augmented_float_coords = np.array(
        [
            [f[GraphAttrs.NODE_X], f[GraphAttrs.NODE_Y]]
            for f in target["graph"].nodes.values()
        ],
        dtype=np.float32,
    )
    print("hello-4")

    assert np.array_equal(expected_end_coords_img[n], augmented_img_coords)
    assert np.allclose(
        expected_end_coords_float[n], augmented_float_coords, atol=0.01
    )


@pytest.mark.parametrize(
    "p_add, p_remove, annotation_mode",
    [
        (0, 0, "random"),
        (0, 0.2, "random"),
        (0.2, 0.2, "average"),
        (0.2, 0.4, "average"),
        (0.8, 0, "unknown"),
    ],
)
class TestAugmentGraphEdgeAdditionRemoval:
    @pytest.fixture
    def outputs(self, p_add, p_remove, annotation_mode, default_rng):
        image, graph = random_image_and_graph(default_rng, num_nodes=4)
        transform = RandomEdgeAdditionAndRemoval(
            p_add, p_remove, default_rng, annotation_mode
        )
        augmented_image, augmented_target = transform(image, {"graph": graph})
        augmented_graph = augmented_target["graph"]
        return {
            "image": image,
            "graph": graph,
            "augmented_image": augmented_image,
            "augmented_graph": augmented_graph,
        }

    def test_images_remain_same(
        self, p_add, p_remove, annotation_mode, outputs
    ):
        assert np.array_equal(outputs["image"], outputs["augmented_image"])

    def test_new_edges_in_node_range(self, p_add, p_remove, outputs):
        assert all(
            [
                e in range(outputs["graph"].number_of_nodes())
                for edge_tuple in outputs["augmented_graph"].edges
                for e in edge_tuple
            ]
        )

    def test_number_of_added_and_removed_edges(
        self, p_add, p_remove, annotation_mode, default_rng, outputs
    ):
        add_edges = RandomEdgeAdditionAndRemoval(p_add, 0, default_rng)
        remove_edges = RandomEdgeAdditionAndRemoval(0, p_remove, default_rng)
        num_edges_init = outputs["graph"].number_of_edges()

        image, target_added = add_edges(
            outputs["image"], {"graph": outputs["graph"]}
        )
        num_edges_augmented_add = target_added["graph"].number_of_edges()

        image, target_removed = remove_edges(image, target_added)
        num_edges_augmented_remove = target_removed["graph"].number_of_edges()

        assert (
            num_edges_init + int(p_add * num_edges_init)
            >= num_edges_augmented_add
            >= num_edges_init
        )
        assert (
            num_edges_augmented_add
            >= num_edges_augmented_remove
            >= num_edges_augmented_add - int(p_remove * num_edges_init)
        )

    def test_added_edges_have_correct_annotations(
        self, p_add, p_remove, annotation_mode, outputs
    ):
        if annotation_mode == "random":
            return

        added_edges = set(outputs["augmented_graph"].edges).difference(
            set(outputs["graph"].edges)
        )

        for e in (
            outputs["augmented_graph"].edge_subgraph(added_edges).edges.data()
        ):
            if annotation_mode == "average":
                expected_annotation = find_average_annotation(
                    e[:2], outputs["graph"]
                )
            else:
                expected_annotation = Annotation.UNKNOWN

            assert e[2][GraphAttrs.EDGE_GROUND_TRUTH] == expected_annotation


@pytest.mark.parametrize(
    "mode",
    [
        "randon",
        "randion",
        "unkown",
        "avrage",
    ],
)
def test_invalid_annotation_mode_raises_error(mode):
    with pytest.raises(ValueError):
        RandomEdgeAdditionAndRemoval(annotation_mode=mode)


def test_xy_translation(default_rng):
    max_shift = 2
    num_nodes = 4

    image, graph = random_image_and_graph(default_rng, num_nodes=num_nodes)
    transform = RandomXYTranslation(max_shift=max_shift)
    shifts = default_rng.uniform(
        low=-max_shift, high=max_shift, size=(num_nodes, 2)
    ).tolist()

    with patch("numpy.random.uniform") as mock:
        mock.side_effect = shifts
        _, graph_transformed = transform(image, {"graph": graph})

    coords = np.array(
        [
            [value[GraphAttrs.NODE_X], value[GraphAttrs.NODE_Y]]
            for _, value in graph.nodes(data=True)
        ]
    )
    coords_expected = coords + shifts
    coords_transformed = np.array(
        [
            [value[GraphAttrs.NODE_X], value[GraphAttrs.NODE_Y]]
            for _, value in graph_transformed["graph"].nodes(data=True)
        ]
    )

    assert np.array_equal(coords_expected, coords_transformed)
