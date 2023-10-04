import numpy as np

from grace.base import GraphAttrs

from grace.evaluation.process import (
    update_graph_with_dummy_predictions,
)

from _utils import random_image_and_graph


def test_update_dummy_graph_predictions(default_rng):
    """Test that the dataset loader can import test images and grace annotations."""

    _, graph = random_image_and_graph(default_rng)
    update_graph_with_dummy_predictions(graph)

    for _, node in graph.nodes(data=True):
        lbl = node[GraphAttrs.NODE_PREDICTION].label
        assert isinstance(lbl, int)

        probabs = np.array(
            [
                node[GraphAttrs.NODE_PREDICTION].prob_TN,
                node[GraphAttrs.NODE_PREDICTION].prob_TP,
                node[GraphAttrs.NODE_PREDICTION].prob_UNKNOWN,
            ]
        )
        assert probabs[2] == 0
        assert np.sum(probabs) == 1

        arg = np.argmax(probabs)
        assert arg == lbl
