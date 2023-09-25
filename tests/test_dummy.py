import numpy as np

from grace.base import GraphAttrs, Prediction

from grace.evaluation.process import (
    update_graph_with_dummy_predictions,
)

from _utils import random_image_and_graph


def test_update_dummy_graph_predictions(default_rng):
    """Test that the dataset loader can import test images and grace annotations."""

    _, graph = random_image_and_graph(default_rng)
    update_graph_with_dummy_predictions(graph)

    for _, node in graph.nodes(data=True):
        lbl = node[GraphAttrs.NODE_PREDICTION][Prediction.LABEL]
        assert isinstance(lbl, int)

        logits = np.array(
            [
                node[GraphAttrs.NODE_PREDICTION][Prediction.PROB_TN],
                node[GraphAttrs.NODE_PREDICTION][Prediction.PROB_TP],
            ]
        )
        assert np.sum(logits) == 1

        arg = np.argmax(logits)
        assert arg == lbl
