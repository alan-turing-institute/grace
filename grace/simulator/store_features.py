import click
import torch

from pathlib import Path
from grace.base import GraphAttrs
from grace.io import write_graph
from grace.io.image_dataset import ImageGraphDataset
from grace.models.feature_extractor import FeatureExtractor


# Define a click command to input the file name directly:
@click.command(name="Graph Storage")
@click.option(
    "--data_path",
    type=click.Path(exists=True),
    help="Path to images and grace annotations",
)
@click.option(
    "--extractor_fn",
    type=click.Path(exists=True),
    help="Path to feature extractor model",
)
@click.option(
    "--bbox_size",
    type=tuple[int, int],
    help="Image patch shape for feature extraction",
    default=(224, 224),
)
def store_node_features_in_graph(
    data_path: str | Path,
    extractor_fn: str | Path,
    bbox_size: tuple[int, int] = (224, 224),
) -> None:
    # Process the check the paths:
    if isinstance(data_path, str):
        data_path = Path(data_path)
    assert data_path.is_dir()

    if isinstance(extractor_fn, str):
        extractor_fn = Path(extractor_fn)
    assert extractor_fn.is_file()

    # Load the feature extractor:
    pre_trained_resnet = torch.load(extractor_fn)
    feature_extractor = FeatureExtractor(
        model=pre_trained_resnet,
        bbox_size=bbox_size,
    )

    # Organise the image + grace annotation pairs:
    dataset = ImageGraphDataset(
        image_dir=data_path, grace_dir=data_path, transform=feature_extractor
    )

    # Unwrap each item & store the node features:
    for _, target in dataset:
        fn = target["metadata"]["image_filename"]
        graph = target["graph"]

        for _, node in graph.nodes(data=True):
            node[GraphAttrs.NODE_FEATURES] = node[
                GraphAttrs.NODE_FEATURES
            ].numpy()

        write_graph(
            filename=data_path / f"{fn}.grace",
            graph=graph,
            metadata=target["metadata"],
            annotation=target["annotation"],
        )


if __name__ == "__main__":
    store_node_features_in_graph()
