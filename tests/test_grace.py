import networkx as nx
import numpy as np

from pathlib import Path
from grace.base import GraphAttrs, graph_from_dataframe
from grace.io import read_graph, write_graph


SIMPLE_GRAPH_NUM_EDGES = 3
REQUIRED_NODE_ATTRIBUTES = set(
    [GraphAttrs.NODE_X, GraphAttrs.NODE_Y, GraphAttrs.NODE_GROUND_TRUTH]
)
REQUIRED_EDGE_ATTRIBUTES = set([GraphAttrs.EDGE_GROUND_TRUTH])


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

    for _, _, edge_attrs in G.edges(data=True):
        edge_keys = set(list(edge_attrs.keys()))
        assert REQUIRED_EDGE_ATTRIBUTES.issubset(edge_keys)


def test_write_graph_roundtrip(tmp_path, simple_graph):
    """Test writing a graph."""

    filename = Path(tmp_path) / "test.grace"
    write_graph(filename, graph=simple_graph)
    assert filename.exists()

    data = read_graph(filename)
    assert data.graph is not None
    assert isinstance(data.graph, nx.Graph)

    assert simple_graph.number_of_nodes() == data.graph.number_of_nodes()
    assert simple_graph.number_of_edges() == data.graph.number_of_edges()

    assert nx.utils.nodes_equal(simple_graph.nodes, data.graph.nodes)
    assert nx.utils.edges_equal(simple_graph.edges, data.graph.edges)


def test_write_annotation_roundtrip(tmp_path, default_rng):
    """Test writing an annotation."""
    annotation = default_rng.uniform(size=(64, 64)).astype(int)

    filename = Path(tmp_path) / "test.grace"
    write_graph(filename, annotation=annotation)
    assert filename.exists()

    data = read_graph(filename)
    assert data.annotation is not None
    np.testing.assert_equal(annotation, data.annotation)


def test_write_metadata_roundtrip(tmp_path):
    """Test writing metadata to a file."""
    metadata = {
        "image_filename": "test.mrc",
        "detections_filename": "test.cbox",
    }

    filename = Path(tmp_path) / "test.grace"
    write_graph(filename, metadata=metadata)
    assert filename.exists()

    data = read_graph(filename)
    assert data.metadata is not None
    assert data.metadata == metadata
