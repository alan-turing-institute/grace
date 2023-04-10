import napari

import mrcfile
import networkx as nx
import numpy as np
import starfile

from grace.base import GraphAttrs
from grace.napari.utils import graph_to_napari_layers

# from grace.io.cbox import read_cbox_as_nodes
from pathlib import Path

DATA_PATH = Path("/Users/arl/Documents/Turing/Data/Bea/")
FILE_STEM = Path(
    "FoilHole_24680421_Data_24671727_24671728_20181024_2216-78563_noDW"
)
IMAGE_PATH = DATA_PATH / FILE_STEM.with_suffix(".mrc")
CBOX_PATH = DATA_PATH / FILE_STEM.with_suffix(".cbox")


with mrcfile.open(IMAGE_PATH, "r") as mrc:
    image_data = mrc.data.astype(int)


cbox_df = starfile.read(CBOX_PATH)["cryolo"]
num_nodes = cbox_df.shape[0]

nodes = [
    (
        idx,
        {
            GraphAttrs.NODE_X: cbox_df["CoordinateX"][idx]
            + cbox_df["Width"][idx] / 2,
            GraphAttrs.NODE_Y: cbox_df["CoordinateY"][idx]
            + cbox_df["Height"][idx] / 2,
            GraphAttrs.NODE_CONFIDENCE: cbox_df["Confidence"][idx],
            # GraphAttrs.NODE_WIDTH: cbox_df["Width"][idx],
            # GraphAttrs.NODE_HEIGHT: cbox_df["Height"][idx],
        },
    )
    for idx in range(num_nodes)
]

graph = nx.Graph()
graph.add_nodes_from(nodes)
points, _ = graph_to_napari_layers(graph)

features = {"features": np.asarray(cbox_df["Confidence"])}

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
