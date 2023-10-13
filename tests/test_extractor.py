import math
import torch
import pytest

from grace.base import GraphAttrs, Annotation
from grace.models.feature_extractor import FeatureExtractor

from conftest import random_image_and_graph


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
                keep_patch_fraction=0.0,
                normalize_func=lambda x: x,
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

        for _, node_attrs in graph_out.nodes.data():
            x, y = node_attrs[GraphAttrs.NODE_X], node_attrs[GraphAttrs.NODE_Y]
            features = node_attrs[GraphAttrs.NODE_FEATURES]

            x_low = int(x - bbox_size[0] / 2)
            x_box = slice(x_low, x_low + bbox_size[0])

            y_low = int(y - bbox_size[1] / 2)
            y_box = slice(y_low, y_low + bbox_size[1])

            bbox_image = vars["image"][..., y_box, x_box]

            if bbox_image.shape == bbox_size:
                assert features == model(bbox_image)

    @pytest.mark.skip(reason="keep patch fraction needs to be re-implemented")
    @pytest.mark.parametrize("keep_patch_fraction", [0.3, 0.5, 0.7])
    def test_feature_extractor_rejects_edge_touching_boxes(
        self, bbox_size, model, vars, keep_patch_fraction
    ):
        """TODO: There's a bug here if parametrised with:
        "keep_patch_fraction", [0.3, 0.5, 0.7]"""
        extractor = vars["extractor"]
        image = vars["image"]
        graph = vars["graph"]

        setattr(extractor, "keep_patch_fraction", keep_patch_fraction)
        graph.add_node(
            4,
            **{
                GraphAttrs.NODE_X: bbox_size[0] * 0.5
                + math.ceil(
                    image.size(-1) - bbox_size[0] * keep_patch_fraction * 0.99
                ),
                GraphAttrs.NODE_Y: 0,
                GraphAttrs.NODE_GROUND_TRUTH: Annotation.TRUE_POSITIVE,
            },
        )
        graph.add_node(
            5,
            **{
                GraphAttrs.NODE_X: 0,
                GraphAttrs.NODE_Y: bbox_size[1] * 0.5
                + math.floor(-bbox_size[1] * keep_patch_fraction * 1.01),
                GraphAttrs.NODE_GROUND_TRUTH: Annotation.TRUE_POSITIVE,
            },
        )

        _, target_out = extractor(image, {"graph": graph})
        graph_out = target_out["graph"]
        print(bbox_size)
        print(graph_out.number_of_nodes(), graph_out.nodes(data=True))
        labels = [
            node_attr[GraphAttrs.NODE_GROUND_TRUTH]
            for _, node_attr in graph_out.nodes(data=True)
        ]

        num_unknown = len([lab for lab in labels if lab == Annotation.UNKNOWN])

        assert num_unknown == 2
