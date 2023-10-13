import click

from pathlib import Path
from grace.base import GraphAttrs
from grace.io import write_graph
from grace.io.image_dataset import ImageGraphDataset
from grace.models.property_cruncher import EdgePropertyCruncher


def store_edge_properties_in_graph(
    data_path: str | Path,
) -> None:
    # Process the check the paths:
    if isinstance(data_path, str):
        data_path = Path(data_path)
    assert data_path.is_dir()

    # Organise the image + grace annotation pairs:
    dataset = ImageGraphDataset(
        image_dir=data_path,
        grace_dir=data_path,
    )

    # Unwrap each item & store the node features:
    for _, target in dataset:
        fn = target["metadata"]["image_filename"]
        graph = target["graph"]

        # Calculate graph edge attributes:
        graph = EdgePropertyCruncher(graph).process()

        # Make sure that EDGE_PROPERTIES are in the right format:
        for _, _, edge in graph.edges(data=True):
            dictionary = edge[GraphAttrs.EDGE_PROPERTIES]
            edge["edge_properties_keys"] = dictionary.property_keys
            edge["edge_properties_values"] = dictionary.property_vals
            del edge[GraphAttrs.EDGE_PROPERTIES]

        write_graph(
            filename=data_path / f"{fn}.grace",
            graph=graph,
            metadata=target["metadata"],
            annotation=target["annotation"],
        )


# Define a click command to input the file name directly:
@click.command(name="Graph Edge Storage")
@click.option(
    "--data_path",
    type=click.Path(exists=True),
    help="Path to images and grace annotations",
)
def run_edge_storage(data_path: str | Path) -> None:
    store_edge_properties_in_graph(data_path)


if __name__ == "__main__":
    run_edge_storage()
