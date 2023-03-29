from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # import numpy.typing as npt
    from magicgui.widgets import Container, Widget

import magicgui
import napari

from grace.base import _edges_from_delaunay
from scipy.spatial import Delaunay


def image_selection_widget() -> Widget:
    image_tooltip = "Select an 'Image' layer to use for annotation."
    _widget = magicgui.widgets.create_widget(
        annotation=napari.layers.Image,
        name="image",
        label="image: ",
        options={"tooltip": image_tooltip},
    )
    return _widget


def triangulate_widget() -> Widget:
    triangulate_tooltip = "Build the graph by triangulation."
    _widget = magicgui.widgets.create_widget(
        name="triangulate_button",
        label="triangulate",
        widget_type="PushButton",
        options={"tooltip": triangulate_tooltip},
    )
    return _widget


def cut_widget() -> Widget:
    pass


def triangulate(viewer: napari.Viewer) -> None:
    """Ugly function to triangulate."""
    points = viewer.layers["Detections"].data
    tri = Delaunay(points)
    edges = [points[(i, j), :] for i, j in _edges_from_delaunay(tri)]

    viewer.add_shapes(
        edges, shape_type="line", edge_width=5, edge_color="blue"
    )
    return


def create_grace_widget() -> Container:
    """Create widgets for the grace plugin."""

    # First create our UI along with some default configs for the widgets
    widgets = [
        image_selection_widget(),
        triangulate_widget(),
    ]
    grace_widget = magicgui.widgets.Container(
        widgets=widgets,
        scrollable=False,
    )
    grace_widget.viewer = napari.current_viewer()

    grace_widget.triangulate_button.changed.connect(
        lambda: triangulate(grace_widget.viewer),
    )

    return grace_widget
