import starfile
import pathlib
import os
import networkx as nx

from grace.io.core import write_graph, graph_from_dataframe


def star_to_graph(filename: os.PathLike) -> nx.Graph:
    """Reads a starfile into a graph.

    Parameters
    ----------
    filename : str, pathlike
       Path to starfile.

    Returns
    -------
    G : nx.Graph
        A graph of the nodes connected by edges determined using Delaunay
        triangulation.

    """
    star_df = starfile.read(str(filename))
    star_df = star_df.rename(
        columns={"rlnCoordinateX": "x", "rlnCoordinateY": "y"}
    )

    return graph_from_dataframe(star_df)


def mkdir_grace_from_star(
    stardir: os.PathLike, gracedir: os.PathLike = None
) -> None:
    """Make and populate a grace directory from a directory of starfiles.

    Parameters
    ----------
    stardir : str, PathLike
        Path to starfile directory.
    gracedir : str, PathLike (optional)
        Path to grace directory.
    """

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
