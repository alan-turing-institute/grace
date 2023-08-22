import napari

import mrcfile
import pandas as pd
import numpy as np

from grace.base import GraphAttrs
from pathlib import Path


# Expects the image data & H5 node positions in the same folder.
# Use identical naming convention for files & specify whole path to mrc file:
# e.g. /Users/kulicna/Desktop/dataset/shape_squares/MRC_Synthetic_File_000.mrc

IMAGE_PATH = Path(
    input(
        "Enter absolute path to your file "
        "(e.g. /Users/path/to/your/data/image.mrc, omit ''): "
    )
)
NODES_PATH = Path(str(IMAGE_PATH).replace(".mrc", ".h5"))


with mrcfile.open(IMAGE_PATH, "r") as mrc:
    # image_data = mrc.data.astype(int)
    image_data = mrc.data

nodes_data = pd.read_hdf(NODES_PATH)
points = np.asarray(nodes_data.loc[:, [GraphAttrs.NODE_Y, GraphAttrs.NODE_X]])
# features = {
#     GraphAttrs.NODE_FEATURES:
#     [np.squeeze(f.numpy()) for f in nodes_data.loc[:, "features"]]
# }
features = None
mn, mx = np.min(image_data), np.max(image_data)

data_name = f"{IMAGE_PATH.stem}"

viewer = napari.Viewer()
img_layer = viewer.add_image(
    image_data, name=data_name, contrast_limits=(mn, mx)
)
pts_layer = viewer.add_points(
    points, features=features, size=32, name=f"nodes_{data_name}"
)


_, widget = viewer.window.add_plugin_dock_widget(
    plugin_name="grace", widget_name="GRACE"
)

if __name__ == "__main__":
    # The napari event loop needs to be run under here to allow the window
    # to be spawned from a Python script
    napari.run()
