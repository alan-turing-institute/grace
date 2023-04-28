from grace.models.classifier import GCN
from grace.models.feature_extractor import FeatureExtractor

import torch
import pytest

from _utils import random_image_and_graph

from grace.base import GraphAttrs


@pytest.mark.parametrize("input_dims", [1, 2, 4])
@pytest.mark.parametrize("embedding_dims", [16, 32, 64])
@pytest.mark.parametrize("output_dims", [1, 2, 4])
def test_model_building(input_dims, embedding_dims, output_dims):
    """Test building the model with different dimension."""

    model = GCN(
        input_dims=input_dims,
        embedding_dims=embedding_dims,
        output_dims=output_dims,
    )

    assert model.conv1.in_channels == input_dims
    assert model.linear.in_features == embedding_dims
    assert model.linear.out_features == output_dims


@pytest.mark.parametrize(
    "bbox_size, model",
    [
        ((2, 2), lambda x: torch.sum(x)),
        ((3, 3), lambda x: torch.mean(x)),
    ],
)
class TestFeatureExtractor:
    @pytest.fixture
    def vars(self, bbox_size, model, default_rng):
        image, graph = random_image_and_graph(default_rng)
        return {
            "extractor": FeatureExtractor(
                bbox_size=bbox_size,
                model=model,
                transforms=lambda x: x,
                augmentations=lambda x: x,
            ),
            "image": torch.tensor(image.astype("float32")),
            "graph": graph,
        }

    def test_feature_extractor_forward(self, bbox_size, model, vars):
        image_out, target_out = vars["extractor"](
            vars["image"], {"graph": vars["graph"]}
        )
        graph_out = target_out["graph"]

        assert torch.equal(vars["image"], image_out)

        for node in graph_out.nodes.data():
            y, x = int(node[1][GraphAttrs.NODE_X]), int(
                node[1][GraphAttrs.NODE_Y]
            )
            features = node[1][GraphAttrs.NODE_FEATURES]

            x_low = int(x - bbox_size[0] / 2)
            x_box = slice(x_low, x_low + bbox_size[0])

            y_low = int(y - bbox_size[1] / 2)
            y_box = slice(y_low, y_low + bbox_size[1])

            bbox_image = vars["image"][..., x_box, y_box]

            assert features == model(bbox_image)
