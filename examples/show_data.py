import click
import napari

import pandas as pd
import numpy as np

from pathlib import Path

from grace.base import GraphAttrs
from grace.io.image_dataset import (
    FILETYPES,
    mrc_reader,
    tiff_reader,
    png_reader,
)


# Define a click command to input the file name directly:
@click.command(name="Napari Annotator")
@click.option("--image_path", type=click.Path(exists=True))
def run_napari_annotator(image_path=str) -> None:
    # Expects the image data & H5 node positions in the same folder.
    # Use identical naming convention for files & specify whole path to mrc file:
    # e.g. /Users/kulicna/Desktop/dataset/shape_squares/MRC_Synthetic_File_000.mrc

    suffix = image_path.split(".")[-1]
    assert suffix in FILETYPES, f"Choose these filetypes: {FILETYPES.keys()}"
    IMAGE_PATH = Path(image_path)

    if suffix == "mrc":
        image_data = mrc_reader(IMAGE_PATH)
    elif suffix == "png":
        image_data = png_reader(IMAGE_PATH)
    elif suffix == "tiff":
        image_data = tiff_reader(IMAGE_PATH)

    nodes_path = image_path.replace(".mrc", ".h5")
    nodes_data = pd.read_hdf(Path(nodes_path))

    points = np.asarray(
        nodes_data.loc[:, [GraphAttrs.NODE_Y, GraphAttrs.NODE_X]]
    )
    # features = {
    #     GraphAttrs.NODE_FEATURES:
    #     [np.squeeze(f.numpy()) for f in nodes_data.loc[:, "features"]]
    # }
    features = None
    data_name = f"{IMAGE_PATH.stem}"

    mn, mx = np.min(image_data), np.max(image_data)

    viewer = napari.Viewer()
    viewer.add_image(image_data, name=data_name, contrast_limits=(mn, mx))
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
