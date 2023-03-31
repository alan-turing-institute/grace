from grace import graph_from_dataframe
from grace.base import GraphAttrs

SIMPLE_GRAPH_NUM_EDGES = 3
REQUIRED_NODE_ATTRIBUTES = set(
    [GraphAttrs.NODE_X, GraphAttrs.NODE_Y, GraphAttrs.NODE_PROB_DETECTION]
)
REQUIRED_EDGE_ATTRIBUTES = set([GraphAttrs.EDGE_PROB_LINK])


def test_graph_from_dataframe(simple_graph_dataframe):
    """Test construction of a grace graph from a dataframe"""
    G = graph_from_dataframe(simple_graph_dataframe)

    # first check that we have all the correct number of nodes and edges
    assert G.number_of_nodes() == len(
        simple_graph_dataframe.loc[:, GraphAttrs.NODE_X]
    )
    assert G.number_of_edges() == SIMPLE_GRAPH_NUM_EDGES


def test_graph_node_attrs_from_dataframe(simple_graph_dataframe):
    """Test that a grace graph has the appropriate attributes"""
    G = graph_from_dataframe(simple_graph_dataframe)

    for _, node_attrs in G.nodes(data=True):
        node_keys = set(list(node_attrs.keys()))
        assert REQUIRED_NODE_ATTRIBUTES.issubset(node_keys)


def test_graph_edge_attrs_from_dataframe(simple_graph_dataframe):
    """Test that a grace graph has the appropriate attributes"""
    G = graph_from_dataframe(simple_graph_dataframe)

    for source, target, edge_attrs in G.edges(data=True):
        edge_keys = set(list(edge_attrs.keys()))
        assert REQUIRED_EDGE_ATTRIBUTES.issubset(edge_keys)
