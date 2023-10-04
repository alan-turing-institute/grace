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
@click.option(
    "--show_features",
    type=bool,
    default=True,
    help="Show feature representation of individual nodes",
)
def run_napari_annotator(
    image_path: str | Path,
    show_features: bool = False,
) -> None:
    """Function to open an image & annotate it in napari.

    Parameters
    ----------
    image_path : str | Path
        Absolute filename of the image to be opened.
        Expects the image data & H5 node positions in the same folder.
        Use identical naming convention for these files to pair them up.
    show_features : bool
        Whether to display node features stored in the `h5` file.
        Defaults to False.

    Notes
    -----
    - expected file organisation:
        /path/to/your/image/MRC_Synthetic_File_000.mrc
        ...identical to...
        /path/to/your/nodes/MRC_Synthetic_File_000.h5
    """
    # Process the image data + load nodes:
    suffix = str(image_path).split(".")[-1]
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

    # Process the node features:
    if show_features is True:
        features = {
            "node_central_pixel_value": [
                image_data[int(point[0]), int(point[1])] for point in points
            ]
        }
    else:
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
