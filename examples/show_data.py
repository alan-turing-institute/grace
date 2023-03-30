import napari

import mrcfile
import pandas as pd
import numpy as np

from pathlib import Path

DATA_PATH = Path("/Users/arl/Documents/Turing/Data/Bea/")
IMAGE_PATH = (
    DATA_PATH
    / "FoilHole_24680421_Data_24671727_24671728_20181024_2216-78563_noDW.mrc"
)
NODES_PATH = DATA_PATH / "data.h5"


with mrcfile.open(IMAGE_PATH, "r") as mrc:
    image_data = mrc.data.astype(int)
nodes_data = pd.read_hdf(NODES_PATH)
points = np.asarray(nodes_data.loc[:, ["y", "x"]])


viewer = napari.Viewer()
img_layer = viewer.add_image(image_data, name=f"{IMAGE_PATH.stem[:10]}...")
pts_layer = viewer.add_points(points, size=32, name="nodes")


_, widget = viewer.window.add_plugin_dock_widget(
    plugin_name="grace", widget_name="GRACE"
)

if __name__ == "__main__":
    # The napari event loop needs to be run under here to allow the window
    # to be spawned from a Python script
    napari.run()
