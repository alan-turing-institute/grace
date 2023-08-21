import napari

import mrcfile
import numpy as np
import starfile

from grace.base import GraphAttrs
from pathlib import Path


# Expects the image data & H5 node positions in the same folder.
# Use identical naming convention for files & specify whole path to mrc file:
# e.g. /Users/kulicna/Desktop/dataset/shape_squares/MRC_Synthetic_File_000.mrc

IMAGE_PATH = Path(input("Enter absolute path to your file: "))
C_BOX_PATH = Path(str(IMAGE_PATH).replace(".mrc", ".cbox"))


with mrcfile.open(IMAGE_PATH, "r") as mrc:
    image_data = mrc.data.astype(int)


cbox_df = starfile.read(C_BOX_PATH)["cryolo"]

points = np.stack(
    [
        cbox_df["CoordinateY"] + cbox_df["Height"] / 2,
        cbox_df["CoordinateX"] + cbox_df["Width"] / 2,
    ],
    axis=-1,
)

features = {GraphAttrs.NODE_CONFIDENCE: np.asarray(cbox_df["Confidence"])}

data_name = f"{IMAGE_PATH.stem}"

viewer = napari.Viewer()
img_layer = viewer.add_image(image_data, name=data_name)
pts_layer = viewer.add_points(
    points,
    features=features,
    size=32,
    name=f"nodes_{data_name}",
    # text= {
    #     "string": "{confidence:.2f}",
    #     "size": 8,
    #     "color": "r",
    # }
)


_, widget = viewer.window.add_plugin_dock_widget(
    plugin_name="grace", widget_name="GRACE"
)

if __name__ == "__main__":
    # The napari event loop needs to be run under here to allow the window
    # to be spawned from a Python script
    napari.run()
