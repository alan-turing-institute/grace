from grace.base import GraphAttrs
from grace.models.optimiser import optimise_graph
from grace.evaluation.process import update_graph_with_dummy_predictions

from _utils import random_image_and_graph


def test_optimisation_of_dummy_graph(default_rng):
    """Test that the dataset loader can import test images and grace annotations."""

    _, graph = random_image_and_graph(default_rng)
    update_graph_with_dummy_predictions(graph)

    node_attr = GraphAttrs.NODE_GROUND_TRUTH
    edge_attr = GraphAttrs.EDGE_GROUND_TRUTH

    # Make sure updated graph still has its GT labels for nodes & edges:
    assert all([node_attr in n for _, n in graph.nodes(data=True)])
    assert all([edge_attr in e for _, _, e in graph.edges(data=True)])

    # Number of nodes must agree, only edge count differs:
    optim = optimise_graph(graph)
    assert graph.number_of_nodes() == optim.number_of_nodes()

    # Make sure all graph attributes are still in the optimised graph:
    assert all([node_attr in n for _, n in optim.nodes(data=True)])
    assert all([edge_attr in e for _, _, e in optim.edges(data=True)])
