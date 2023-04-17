from grace.io import read_graph, write_graph
from pathlib import Path

import networkx as nx
import numpy as np


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
