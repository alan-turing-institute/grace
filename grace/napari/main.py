from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # import numpy.typing as npt
    import networkx as nx
    from magicgui.widgets import Container, Widget

import enum
import magicgui
import napari
import numpy as np
import pandas as pd

from grace.base import graph_from_dataframe, GraphAttrs
from grace.io import write_graph
from grace.napari.utils import graph_to_napari_layers, cut_graph_using_mask
from pathlib import Path

from qtpy.QtWidgets import QFileDialog

# from scipy.spatial import Delaunay

LOGO_WIDTH = 200
LOGO_HEIGHT = 60


class EdgeColor(str, enum.Enum):
    TRUE_POSITIVE = "green"
    TRUE_NEGATIVE = "magenta"
    UNKNOWN = "blue"


def branding_widget() -> Widget:
    logo_widget = magicgui.widgets.create_widget(
        value=Path(__file__).parent / "logo.png",
        widget_type="Image",
    )
    logo_widget.min_width = LOGO_WIDTH
    logo_widget.min_height = LOGO_HEIGHT
    return [
        logo_widget,
    ]


def selection_widget() -> Widget:
    image_tooltip = "Select an 'Image' layer to use for annotation."
    image_widget = magicgui.widgets.create_widget(
        annotation=napari.layers.Image,
        name="selected_image",
        label="image: ",
        options={"tooltip": image_tooltip},
    )
    return [
        image_widget,
    ]


def process_widget() -> Widget:
    build_tooltip = "Build the graph by triangulation."
    build_widget = magicgui.widgets.create_widget(
        name="build_button",
        label="build graph",
        widget_type="PushButton",
        options={"tooltip": build_tooltip},
    )

    cut_tooltip = "Cut the graph using the mask."
    cut_widget = magicgui.widgets.create_widget(
        name="cut_button",
        label="cut graph",
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

    return [
        build_widget,
        cut_widget,
        train_widget,
        inference_widget,
    ]


def status_widget() -> Widget:
    bar_widget = magicgui.widgets.create_widget(
        name="progress",
        label="status: ",
        widget_type="ProgressBar",
    )
    return [
        bar_widget,
    ]


def io_widget() -> Widget:
    export_tooltip = "Export the annotations."
    export_widget = magicgui.widgets.create_widget(
        name="export_button",
        label="export...",
        widget_type="PushButton",
        options={"tooltip": export_tooltip},
    )

    import_tooltip = "Import annotations."
    import_widget = magicgui.widgets.create_widget(
        name="import_button",
        label="import...",
        widget_type="PushButton",
        options={"tooltip": import_tooltip},
    )

    return [
        import_widget,
        export_widget,
    ]


def color_edges(graph: nx.Graph) -> str:
    """Color an edge based on the set it belongs too."""
    edge_colors = []
    for source, target, edge_attr in graph.edges(data=True):
        edge_annotation = edge_attr[GraphAttrs.EDGE_GROUND_TRUTH].name
        color = EdgeColor[edge_annotation]
        edge_colors.append(color.value)
    return edge_colors


class GraceManager:
    def __init__(self, viewer: napari.Viewer):
        self.viewer = viewer
        self.edge_layer = None
        self.annotation_layer = None
        self.widgets = []
        self._selected_layer = None

        self.graph = None

    def selected_layer(self, selected_layer: str):
        self._selected_layer = self.viewer.layers[str(selected_layer)]

    def node_layer(self) -> napari.Layer:
        return self.viewer.layers[f"nodes_{self._selected_layer.name}"]

    def create_layers(self):
        """Create new annotation laters based on the selected image layer."""
        image_layer = self._selected_layer

        self.annotation_layer = self.viewer.add_labels(
            np.zeros_like(image_layer.data).astype(int),
            name=f"annotation_{image_layer.name}",
        )
        self.annotation_layer.brush_size = 100
        self.annotation_layer.mode = "PAINT"

    def build_graph(self, *, progress: Widget | None = None) -> None:
        """Build a graph using the node layer."""
        points = self.node_layer().data
        features = self.node_layer().features

        # TODO(arl): this is pretty ugly right now
        df = pd.DataFrame(
            {
                GraphAttrs.NODE_Y: points[:, 0],
                GraphAttrs.NODE_X: points[:, 1],
                GraphAttrs.NODE_FEATURES: features["features"],
            }
        )

        self.graph = graph_from_dataframe(df)
        _, edges = graph_to_napari_layers(self.graph)

        image_layer = self._selected_layer

        if self.edge_layer is None:
            self.edge_layer = self.viewer.add_shapes(
                ndim=2,
                name=f"edges_{image_layer.name}",
                shape_type="line",
                edge_width=5,
                edge_color=EdgeColor.UNKNOWN.value,
            )

        self.edge_layer.data = []
        self.edge_layer.add_lines(edges)

    def cut_graph(self, *, progress: Widget | None = None) -> None:
        """Cut the graph according to the annotation layer."""
        idx, enclosed, cut = cut_graph_using_mask(
            self.graph,
            self.annotation_layer.data,
            update_graph=True,
        )

        self.edge_layer.edge_color = color_edges(self.graph)

    def train(self, *, progress: Widget | None = None) -> None:
        pass

    def predict(self, *, progress: Widget | None = None) -> None:
        pass

    def __del__(self):
        print("Goodbye!")

    def export(self) -> None:
        """Export the tree as an SVG."""
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export annotations",
            "annotations.zip",
            "ZIP Files(*.zip)",
            options=options,
        )
        if filename:
            write_graph(filename, self.graph)


def create_grace_widget() -> Container:
    """Create widgets for the grace plugin."""

    # First create our UI along with some default configs for the widgets
    widgets = [
        *branding_widget(),
        *selection_widget(),
        *process_widget(),
        *status_widget(),
        *io_widget(),
    ]
    grace_widget = magicgui.widgets.Container(
        widgets=widgets,
        scrollable=False,
    )
    grace_widget.viewer = napari.current_viewer()

    grace_manager = GraceManager(grace_widget.viewer)
    grace_manager.selected_layer(grace_widget.selected_image.value)
    grace_manager.create_layers()

    # if we choose another input image, create new annotation layers
    grace_widget.selected_image.changed.connect(
        lambda: grace_widget.create_layers(),
    )

    # connect buttons to methods
    grace_widget.build_button.changed.connect(
        lambda: grace_manager.build_graph(progress=grace_widget.progress),
    )

    grace_widget.cut_button.changed.connect(
        lambda: grace_manager.cut_graph(progress=grace_widget.progress)
    )

    grace_widget.export_button.changed.connect(lambda: grace_manager.export())
    # grace_widget.progress.value = 500

    # grace_widget.run_button.changed.connect(
    #     lambda: print("run"),
    # )

    return grace_widget
