import click
import napari

import pandas as pd
import numpy as np

from pathlib import Path

from grace.base import GraphAttrs
from grace.io.image_dataset import FILETYPES


# Define a click command to input the file name directly:
@click.command(name="Napari Annotator")
@click.option(
    "--image_path",
    type=click.Path(exists=True),
    help="Path to the image to open in napari annotator",
)
def run_napari_annotator(image_path=str) -> None:
    # Expects the image data & H5 node positions in the same folder.
    # Use identical naming convention for files & specify whole path to mrc file:
    # e.g. /Users/kulicna/Desktop/dataset/shape_squares/MRC_Synthetic_File_000.mrc

    # Process the image data + load nodes:
    suffix = image_path.split(".")[-1]
    assert suffix in FILETYPES, f"Choose these filetypes: {FILETYPES.keys()}"

    image_reader = FILETYPES[suffix]
    image_data = image_reader(Path(image_path))

    nodes_path = image_path.replace(".mrc", ".h5")
    nodes_data = pd.read_hdf(Path(nodes_path))

    data_name = f"{Path(image_path).stem}"

    # Start a napari window:
    viewer = napari.Viewer()
    mn, mx = np.min(image_data), np.max(image_data)
    viewer.add_image(image_data, name=data_name, contrast_limits=(mn, mx))

    # Locate the nodes as points:
    points = np.asarray(
        nodes_data.loc[:, [GraphAttrs.NODE_Y, GraphAttrs.NODE_X]]
    )
    # Process the features information - TSNE:
    # features = {
    #     GraphAttrs.NODE_FEATURES:
    #     [np.squeeze(f.numpy()) for f in nodes_data.loc[:, "features"]]
    # }
    features = None

    viewer.add_points(
        points, features=features, size=32, name=f"nodes_{data_name}"
    )

    viewer.window.add_plugin_dock_widget(
        plugin_name="grace", widget_name="GRACE"
    )
    napari.run()


if __name__ == "__main__":
    # The napari event loop needs to be run under here to allow the window
    # to be spawned from a Python script
    run_napari_annotator()
