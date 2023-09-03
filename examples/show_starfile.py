import napari


from grace.io import starfile, core
from grace.io.image_dataset import mrc_reader
from pathlib import Path

#
path_input = input("Input data path:")

if len(path_input) < 1:
    DATA_PATH = Path(
        "/Users/bcostagomes/Documents/2dEM_images/annotation_dataset"
    )
else:
    DATA_PATH = Path(path_input)


IMAGE_PATH = DATA_PATH / "images"
STAR_PATH = DATA_PATH / "detections"

# Makes the grace files/directory from the starfile folder, for every starfile.
starfile.mkdir_grace_from_star(STAR_PATH)

# Gets list of available files in the image
name_list = [f for f in IMAGE_PATH.iterdir() if f.is_file()]


GRACE_PATH = DATA_PATH / "grace"

# Runs only the first one
# TODO: run a batch

FILE_STEM = Path(name_list[1].stem)

print("Loading image: " + str(FILE_STEM))


IMAGE = IMAGE_PATH / FILE_STEM.with_suffix(".mrc")
DETECTION = GRACE_PATH / FILE_STEM.with_suffix(".grace")

image_data = mrc_reader(IMAGE)

grace_file = core.GraceFile(DETECTION)
grace_file = grace_file.read()

graph = grace_file.graph

# make points from node attributes
points = []
for i in range(0, len(graph.nodes)):
    x, y = graph.nodes[i]["x"], graph.nodes[i]["y"]
    points.append([y, x])


features = None

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
