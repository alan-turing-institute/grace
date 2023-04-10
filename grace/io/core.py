from __future__ import annotations

import dataclasses
import json
import os
import pyarrow as pa
import pyarrow.parquet as pq
import networkx as nx
import numpy as np
import numpy.typing as npt

from grace.base import GraphAttrs
from pathlib import Path
from typing import Any, Dict


NODE_SCHEMA = pa.schema(
    [
        pa.field(GraphAttrs.NODE_X, pa.float32()),
        pa.field(GraphAttrs.NODE_Y, pa.float32()),
        pa.field(GraphAttrs.NODE_CONFIDENCE, pa.float32()),
        pa.field(GraphAttrs.NODE_GROUND_TRUTH, pa.int64()),
        pa.field(GraphAttrs.NODE_FEATURES, pa.float32()),
    ],
    # metadata={"year": "2023"}
)

EDGE_SCHEMA = pa.schema(
    [
        pa.field(GraphAttrs.EDGE_SOURCE, pa.int64()),
        pa.field(GraphAttrs.EDGE_TARGET, pa.int64()),
        pa.field(GraphAttrs.EDGE_GROUND_TRUTH, pa.int64()),
    ],
    # metadata={"year": "2023"}
)


@dataclasses.dataclass
class GraceFile:
    """GraceFile storage of graphs.

    Parameters
    ----------
    filename : str, pathlike
        The filename to write the data to.

    """

    filename: os.PathLike

    def __enter__(self) -> GraceFile:
        self._create_file(self.filename)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        pass

    def write(
        self,
        *,
        graph: nx.Graph | None = None,
        annotation: npt.NDArray | None = None,
        metadata: dict[str, str] | None = None,
    ) -> None:
        """Write a graph to a `.grace` file."""

        if graph is not None:
            # write the graph nodes
            nodes_df = pa.Table.from_pylist(
                [node_attr for _, node_attr in graph.nodes(data=True)],
                schema=NODE_SCHEMA,
            )
            nodes_filename = self.filename / "nodes.parquet"
            pq.write_table(nodes_df, nodes_filename)

            # write the graph edges
            edge_list = []
            for source, target, edge_attrs in graph.edges(data=True):
                edge_dict = {
                    GraphAttrs.EDGE_SOURCE: source,
                    GraphAttrs.EDGE_TARGET: target,
                    **edge_attrs,
                }
                edge_list.append(edge_dict)

            edges_df = pa.Table.from_pylist(edge_list, schema=EDGE_SCHEMA)
            edges_filename = self.filename / "edges.parquet"
            pq.write_table(edges_df, edges_filename)

        # write out the annotation
        if annotation is not None:
            annotation_filename = self.filename / "annotation.npz"
            np.savez_compressed(annotation_filename, annotation)

        # add any metadata
        if metadata is not None:
            metadata_filename = self.filename / "metadata.json"
            with open(metadata_filename, "w") as metadata_file:
                json.dump(metadata, metadata_file)

    def read(self) -> nx.Graph:
        """Read a graph from a `.grace` file."""
        nodes_filename = self.filename / "nodes.parquet"
        nodes_df = pq.read_table(nodes_filename).to_pandas()
        nodes = [
            (
                idx,
                {
                    GraphAttrs.NODE_X: nodes_df[GraphAttrs.NODE_X][idx],
                    GraphAttrs.NODE_Y: nodes_df[GraphAttrs.NODE_Y][idx],
                },
            )
            for idx in range(nodes_df.shape[0])
        ]

        edges_filename = self.filename / "edges.parquet"
        edges_df = pq.read_table(edges_filename).to_pandas()
        edges = [
            (
                edges_df[GraphAttrs.EDGE_SOURCE][idx],
                edges_df[GraphAttrs.EDGE_TARGET][idx],
                {
                    GraphAttrs.EDGE_GROUND_TRUTH: edges_df[
                        GraphAttrs.EDGE_GROUND_TRUTH
                    ][idx],
                },
            )
            for idx in range(edges_df.shape[0])
        ]

        graph = nx.Graph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)
        return graph

    def _create_file(self, filename: os.PathLike) -> None:
        self.filepath = Path(filename)

        if not self.filepath.suffix == ".grace":
            self.filepath = self.filepath.with_suffix(".grace")

        if self.filepath.exists():
            return

        self.filepath.mkdir()


def write_graph(
    filename: os.PathLike, graph: nx.Graph, metadata: Dict[str, Any]
) -> None:
    """Write graph to file."""
    with GraceFile(filename) as gfile:
        gfile.write(graph=graph, metadata=metadata)


def write_annotation(
    filename: os.PathLike, *, annotation: npt.NDArray
) -> None:
    """Write annotations to a file."""
    with GraceFile(filename) as gfile:
        gfile.write(annotation=annotation)


def read_graph(filename: os.PathLike) -> nx.Graph:
    """Read graph from file."""
    with GraceFile(filename) as gfile:
        graph = gfile.read()
    return graph
