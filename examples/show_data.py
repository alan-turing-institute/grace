import napari

import mrcfile
import pandas as pd
import numpy as np

from grace.base import GraphAttrs
from pathlib import Path


DATA_PATH = Path(
    # "/Users/kulicna/Desktop/classifier/data_fake/train/padded/"
    "/Users/kulicna/Desktop/classifier/data_fake/infer/padded"
)
IMAGE_FILE = (
    # "FoilHole_24680421_Data_24671727_24671728_20181024_2216-78563_noDW"
    "MRC_Synthetic_File_000"
    # "MRC_File_001"
)
IMAGE_PATH = DATA_PATH / f"{IMAGE_FILE}.mrc"
NODES_PATH = DATA_PATH / f"{IMAGE_FILE}.h5"


with mrcfile.open(IMAGE_PATH, "r") as mrc:
    image_data = mrc.data.astype(int)  # works
    # image_data = mrc.data

nodes_data = pd.read_hdf(NODES_PATH)
points = np.asarray(nodes_data.loc[:, [GraphAttrs.NODE_Y, GraphAttrs.NODE_X]])
# features = {
#     GraphAttrs.NODE_FEATURES:
#     [np.squeeze(f.numpy()) for f in nodes_data.loc[:, "features"]]
# }
features = None


data_name = f"{IMAGE_PATH.stem}"

viewer = napari.Viewer()
img_layer = viewer.add_image(image_data, name=data_name)
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
