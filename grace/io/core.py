from __future__ import annotations

import dataclasses
import json
import os
import pyarrow as pa
import pyarrow.parquet as pq
import networkx as nx
import numpy as np
import numpy.typing as npt
import starfile
import pathlib

from grace.base import Annotation, GraphAttrs, graph_from_dataframe
from grace.io.schema import NODE_SCHEMA, EDGE_SCHEMA
from pathlib import Path
from typing import Any, Dict


@dataclasses.dataclass
class GraceFileDataset:
    metadata: Dict[str, Any] | None = None
    graph: nx.Graph | None = None
    annotation: npt.NDArray = None


@dataclasses.dataclass
class GraceFile:
    """GraceFile storage of graphs and annotations.

    Nodes and edges are stored in parquet tabular format.
    Annotations as compressed numpy arrays.
    Metadata as json data.
    Image data and annotations could be stored as zarr.

    Parameters
    ----------
    filename : str, pathlike
        The filename to write the data to.
    archive: bool, (default: False)
        [Not functional] Whether to write as zip archive.

    Notes
    -----
    General format of grace file is:

    annotations.grace/
    ├─ metadata.json
    ├─ nodes.parquet
    ├─ edges.parquet
    ├─ annotation.npz
    ├─ images.zarr/
    │  ├─ ...
    """

    filename: os.PathLike
    archive: bool = False

    def __enter__(self) -> GraceFile:
        self._create_file(self.filename)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        pass

    @property
    def nodes_filename(self) -> Path:
        return Path(self.filename) / "nodes.parquet"

    @property
    def edges_filename(self) -> Path:
        return Path(self.filename) / "edges.parquet"

    @property
    def annotation_filename(self) -> Path:
        return Path(self.filename) / "annotation.npz"

    @property
    def metadata_filename(self) -> Path:
        return Path(self.filename) / "metadata.json"

    def write(self, data: GraceFileDataset) -> None:
        """Write a graph to a `.grace` file."""

        if data.graph is not None:
            # write the graph nodes
            nodes_df = pa.Table.from_pylist(
                [node_attr for _, node_attr in data.graph.nodes(data=True)],
                schema=NODE_SCHEMA,
            )
            pq.write_table(nodes_df, self.nodes_filename)

            # write the graph edges
            edge_list = []
            for source, target, edge_attrs in data.graph.edges(data=True):
                edge_dict = {
                    GraphAttrs.EDGE_SOURCE: source,
                    GraphAttrs.EDGE_TARGET: target,
                    **edge_attrs,
                }
                edge_list.append(edge_dict)

            edges_df = pa.Table.from_pylist(edge_list, schema=EDGE_SCHEMA)
            pq.write_table(edges_df, self.edges_filename)

        # write out the annotation
        if data.annotation is not None:
            np.savez_compressed(
                self.annotation_filename, annotation=data.annotation
            )

        # add any metadata
        if data.metadata is not None:
            with open(self.metadata_filename, "w") as metadata_file:
                json.dump(data.metadata, metadata_file)

    def read(self) -> GraceFileDataset:
        """Read a graph from a `.grace` file."""

        data: GraceFileDataset = GraceFileDataset()
        graph: nx.Graph = None

        if self.nodes_filename.exists():
            nodes_df = pq.read_table(
                self.nodes_filename, schema=NODE_SCHEMA
            ).to_pandas()
            graph = graph_from_dataframe(nodes_df, triangulate=False)

        # TODO: fix this to deal with other edge attributes
        if self.edges_filename.exists():
            edges_df = pq.read_table(
                self.edges_filename, schema=EDGE_SCHEMA
            ).to_pandas()
            edges = [
                (
                    edges_df[GraphAttrs.EDGE_SOURCE][idx],
                    edges_df[GraphAttrs.EDGE_TARGET][idx],
                    {
                        GraphAttrs.EDGE_GROUND_TRUTH: Annotation(
                            edges_df[GraphAttrs.EDGE_GROUND_TRUTH][idx]
                        ),
                    },
                )
                for idx in range(edges_df.shape[0])
            ]

            if not graph:
                raise IOError("Graph nodes are missing.")

            graph.add_edges_from(edges)
            data.graph = graph

        if self.annotation_filename.exists():
            annotation = np.load(self.annotation_filename)
            data.annotation = annotation["annotation"]

        if self.metadata_filename.exists():
            with open(self.metadata_filename, "r") as metadata_file:
                metadata = json.load(metadata_file)
            data.metadata = metadata

        return data

    def _create_file(self, filename: os.PathLike) -> None:
        self.filepath = Path(filename)

        if not self.filepath.suffix == ".grace":
            self.filepath = self.filepath.with_suffix(".grace")

        if self.filepath.exists():
            return

        self.filepath.mkdir()


def write_graph(
    filename: os.PathLike,
    *,
    graph: nx.Graph | None = None,
    metadata: Dict[str, Any] | None = None,
    annotation: npt.NDArray | None = None,
) -> None:
    """Write graph to file."""
    with GraceFile(filename) as gfile:
        data = GraceFileDataset(
            graph=graph,
            metadata=metadata,
            annotation=annotation,
        )

        gfile.write(data)


def read_graph(filename: os.PathLike) -> GraceFileDataset:
    """Read graph from file."""
    with GraceFile(filename) as gfile:
        data = gfile.read()
    return data


def star_to_graph(filename: os.PathLike) -> nx.Graph:
    """Reads a starfile into a graph."""
    star_df = starfile.read(filename)
    star_df = star_df.rename(
        columns={"rlnCoordinateX": "x", "rlnCoordinateY": "y"}
    )

    return graph_from_dataframe(star_df)


def mkdir_grace_from_star(
    stardir: os.PathLike, gracedir: os.PathLike = None
) -> None:
    """Make and populate a grace directory from a directory of starfiles.

    Inputs:
     stardir: path to starfile directory
     gracedir (optional): path to grace directory"""

    # Read all the files in starfile directory
    p = pathlib.Path(stardir)
    star_list = [f for f in p.iterdir() if f.is_file()]

    # Create grace directory if none is provided
    if gracedir is None:
        pathlib.Path(str(p.parent) + "/grace").mkdir(exist_ok=True)
        gracedir = pathlib.Path(str(p.parent) + "/grace")
    # Scrape through the star file directory

    for file in star_list:
        # Per file, read to dataframe and get graph from star_to_graph
        temp_graph = star_to_graph(file)

        # Write grace with the new filename (if existing, overwrite)
        grace_name = str(gracedir) + "/" + file.stem + ".grace"
        write_graph(grace_name, graph=temp_graph)
