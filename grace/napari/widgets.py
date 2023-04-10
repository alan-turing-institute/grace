from __future__ import annotations

import magicgui
import napari

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from magicgui.widgets import Widget


LOGO_WIDTH = 200
LOGO_HEIGHT = 60


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
    image_widget.max_width = 200
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

    optimise_tooltip = "Optimise the graph after inference."
    optimise_widget = magicgui.widgets.create_widget(
        name="optimise_option",
        label="optimise graph",
        widget_type="CheckBox",
        value=True,
        options={"tooltip": optimise_tooltip},
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
        optimise_widget,
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
