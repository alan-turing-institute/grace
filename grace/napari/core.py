from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy.typing as npt
    from magicgui.widgets import Container, Widget

import magicgui
import napari
import networkx as nx
import numpy as np
import pandas as pd

from grace.base import Annotation, graph_from_dataframe, GraphAttrs
from grace.io import read_graph, write_graph
from grace.napari.utils import (
    EdgeColor,
    color_edges,
    graph_to_napari_layers,
    cut_graph_using_mask,
)
from grace.napari.widgets import (
    branding_widget,
    io_widget,
    process_widget,
    status_widget,
    selection_widget,
)
from pathlib import Path

from qtpy.QtWidgets import QFileDialog


class GraceManager:
    """GraceManager acts to coordinate grace functionality and provide an
    interface to the napari plugin.

    Parameters
    ----------
    viewer : napari.Viewer
        An instance of the napari viewer.
    """

    def __init__(self, viewer: napari.Viewer):
        self.viewer = viewer
        self.graph = None

        # the currently selected layer in the napari viewer
        self._selected_layer = None

    @property
    def selected_layer(self) -> napari.Layer:
        return self._selected_layer

    @selected_layer.setter
    def selected_layer(self, selected_layer: str) -> None:
        self._selected_layer = self.viewer.layers[str(selected_layer)]
        self.node_layer.events.data.connect(self.build_graph)

    @property
    def node_layer(self) -> napari.Layer:
        node_layer_name = f"nodes_{self.selected_layer.name}"
        return self.viewer.layers[node_layer_name]

    @property
    def annotation_layer(self) -> napari.Layer:
        annotation_layer_name = f"annotation_{self.selected_layer.name}"
        return self.viewer.layers[annotation_layer_name]

    @property
    def edge_layer(self) -> napari.Layer:
        """The layer containing edges."""
        edge_layer_name = f"edges_{self.selected_layer.name}"
        if edge_layer_name not in self.viewer.layers:
            self.viewer.add_shapes(
                ndim=2,
                name=f"edges_{self.selected_layer.name}",
                shape_type="line",
                edge_width=5,
                edge_color=EdgeColor.UNKNOWN.value,
            )
        return self.viewer.layers[edge_layer_name]

    def create_layers(
        self,
        *,
        graph: nx.Graph | None = None,
        annotation: npt.NDArray | None = None,
    ):
        """Create new annotation laters based on the selected image layer."""

        if graph is not None:
            self.graph = graph
            _, edges = graph_to_napari_layers(self.graph)
            self.edge_layer.data = []
            self.edge_layer.add_lines(edges)
            self.edge_layer.edge_color = color_edges(self.graph)

        if annotation is None:
            annotation = np.zeros_like(self.selected_layer.data).astype(int)
            self.viewer.add_labels(
                annotation,
                name=f"annotation_{self.selected_layer.name}",
            )
        else:
            self.annotation_layer.data = annotation
        self.annotation_layer.brush_size = 100
        self.annotation_layer.mode = "PAINT"

    def clear_graph(self) -> None:
        """Clear the graph."""
        self.graph = None
        self.edge_layer.data = []

    def build_graph(self, *, progress: Widget | None = None) -> None:
        """Build a graph using the node layer."""

        points = self.node_layer.data
        features = self.node_layer.features

        # TODO(arl): this is pretty ugly right now
        df = pd.DataFrame(
            {
                GraphAttrs.NODE_Y: points[:, 0],
                GraphAttrs.NODE_X: points[:, 1],
                **features,
            }
        )

        self.graph = graph_from_dataframe(df)
        _, edges = graph_to_napari_layers(self.graph)

        self.edge_layer.data = []
        self.edge_layer.add_lines(edges)

    def cut_graph(self, *, progress: Widget | None = None) -> None:
        """Cut the graph according to the annotation layer."""

        # reset all of the edge annotations
        nx.set_edge_attributes(
            self.graph, Annotation.UNKNOWN, GraphAttrs.EDGE_GROUND_TRUTH
        )

        # recalculate based on the current mask
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

    def export_data(self, widget: Widget) -> None:
        """Export the data as a `.grace` file."""
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getSaveFileName(
            widget,
            "Export annotations",
            "annotations.grace",
            "GRACE Files(*.grace)",
            options=options,
        )
        if filename:
            metadata = {
                "image_filename": str(self.selected_layer.name),
            }

            filename = Path(filename)
            write_graph(
                filename,
                graph=self.graph,
                metadata=metadata,
                annotation=self.annotation_layer.data,
            )

    def import_data(self, widget: Widget) -> None:
        """Export the data as a `.grace` file."""
        options = QFileDialog.Options()
        filename = QFileDialog.getExistingDirectory(
            widget,
            "Import annotations",
            "GRACE Files(*.grace)",
            options=options,
        )
        if filename:
            filename = Path(filename)
            data = read_graph(filename)

            self.create_layers(graph=data.graph, annotation=data.annotation)


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
    grace_manager.selected_layer = grace_widget.selected_image.value
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

    grace_widget.export_button.changed.connect(
        lambda: grace_manager.export_data(grace_widget.native)
    )

    grace_widget.import_button.changed.connect(
        lambda: grace_manager.import_data(grace_widget.native)
    )
    # grace_widget.progress.value = 500

    # grace_widget.run_button.changed.connect(
    #     lambda: print("run"),
    # )

    return grace_widget
