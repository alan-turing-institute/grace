from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # import numpy.typing as npt
    from magicgui.widgets import Container, Widget

import magicgui
import napari
import numpy as np

from grace.base import _edges_from_delaunay
from scipy.spatial import Delaunay


def selection_widget() -> Widget:
    image_tooltip = "Select an 'Image' layer to use for annotation."
    image_widget = magicgui.widgets.create_widget(
        annotation=napari.layers.Image,
        name="image",
        label="image: ",
        options={"tooltip": image_tooltip},
    )
    return [
        image_widget,
    ]


def process_widget() -> Widget:
    triangulate_tooltip = "Build the graph by triangulation."
    triangulate_widget = magicgui.widgets.create_widget(
        name="triangulate_button",
        label="triangulate",
        widget_type="PushButton",
        options={"tooltip": triangulate_tooltip},
    )

    cut_tooltip = "Cut the graph using the mask."
    cut_widget = magicgui.widgets.create_widget(
        name="cut_button",
        label="cut",
        widget_type="PushButton",
        options={"tooltip": cut_tooltip},
    )

    train_tooltip = "Train GRACE using the annotations"
    train_widget = magicgui.widgets.create_widget(
        name="train_button",
        label="train GRACE",
        widget_type="PushButton",
        options={"tooltip": train_tooltip},
    )

    inference_tooltip = "Process the graph using GRACE"
    inference_widget = magicgui.widgets.create_widget(
        name="inference_button",
        label="run GRACE",
        widget_type="PushButton",
        options={"tooltip": inference_tooltip},
    )

    return [triangulate_widget, cut_widget, train_widget, inference_widget]


def status_widget() -> Widget:
    bar_widget = magicgui.widgets.create_widget(
        name="progress",
        label="status: ",
        widget_type="ProgressBar",
    )
    return [
        bar_widget,
    ]


class GraceManager:
    def __init__(self, viewer: napari.Viewer):
        self.viewer = viewer
        self.edge_layer = None
        self.annotation_layer = None
        self.widgets = []
        self.selected_layer = None

        self.graph = None

    def selected_layer(self, selected_layer: str):
        self.selected_layer = self.viewer.layers[str(selected_layer)]

    def create_layers(self):
        image_layer = self.selected_layer

        self.annotation_layer = self.viewer.add_labels(
            np.zeros_like(image_layer.data).astype(int),
            name=f"annotation_{image_layer.name}",
        )
        self.annotation_layer.brush_size = 100

    def triangulate(self) -> None:
        points = self.viewer.layers["nodes"].data
        tri = Delaunay(points)
        edges = [points[(i, j), :] for i, j in _edges_from_delaunay(tri)]

        image_layer = self.selected_layer

        if self.edge_layer is None:
            self.edge_layer = self.viewer.add_shapes(
                ndim=2,
                name=f"edges_{image_layer.name}",
                shape_type="line",
                edge_width=5,
                edge_color="blue",
            )

        self.edge_layer.data = []
        self.edge_layer.add_lines(edges)

    def cut_graph(self, *, progress=None) -> None:
        for i in range(1000):
            print(progress.value)
            progress.value = i
            # yield
        pass

    def __del__(self):
        print("Goodbye!")


def create_grace_widget() -> Container:
    """Create widgets for the grace plugin."""

    # First create our UI along with some default configs for the widgets
    widgets = [
        *selection_widget(),
        *process_widget(),
        *status_widget(),
    ]
    grace_widget = magicgui.widgets.Container(
        widgets=widgets,
        scrollable=False,
    )
    grace_widget.viewer = napari.current_viewer()

    grace_manager = GraceManager(grace_widget.viewer)
    grace_manager.selected_layer = grace_widget.image.value
    grace_manager.create_layers()

    # grace_widget.image.changed.connect(
    #     lambda: grace_widget.create_layers(str(grace_widget.image.value)),
    # )

    grace_widget.triangulate_button.changed.connect(
        lambda: grace_manager.triangulate(),
    )

    grace_widget.cut_button.changed.connect(
        lambda: grace_manager.cut_graph(progress=grace_widget.progress)
    )

    # grace_widget.progress.value = 500

    # grace_widget.run_button.changed.connect(
    #     lambda: print("run"),
    # )

    return grace_widget
