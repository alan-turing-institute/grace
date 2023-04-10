from grace.io import write_graph
from pathlib import Path


def test_write_graph(tmp_path, simple_graph):
    """Test writing a graph."""

    filename = Path(tmp_path) / "test.grace"
    write_graph(filename, simple_graph)
    assert filename.exists()


def test_write_read_graph(tmp_path, simple_graph):
    """Test round-trip writing and reading graph."""
    pass


def test_write_annotations(tmp_path):
    pass
