import pandas as pd
import numpy as np
import networkx as nx
import numpy.typing as npt

import mrcfile
from pathlib import Path

from grace.base import GraphAttrs


def create_canvas_image(
    background_pixel_value: int | float,
    image_shape: tuple[int, int],
    image_padding: tuple[int, int] = None,
) -> npt.NDArray:
    """Synthesize a 2D blank canvas image of specified shape.

    Parameters
    ----------
    background_pixel_value : int | float
        Value of the canvas image pixels (background).
    image_shape : tuple[int, int]
        Shape of the canvas image.
    image_padding : tuple[int, int]
        Shape of the canvas padding. Defaults to None.

    Returns
    -------
    image : npt.NDArray
        Blank canvas image.
    """
    # Define an image canvas:
    image = np.full(
        shape=image_shape, fill_value=background_pixel_value, dtype=np.float32
    )
    # Pad corners if needed:
    if image_padding is not None:
        image = np.pad(
            image,
            pad_width=image_padding,
            mode="constant",
            constant_values=background_pixel_value,
        )
    return image


def _insert_fading_square_into_patch(
    original_patch: npt.NDArray,
    central_pixel_value: int | float,
    background_pixel_value: int | float,
) -> npt.NDArray:
    """Inserts a fading shape at the centre coordinate.
    Creates a gradient of values from the central pixel to the boundary (background).
    Blends the modified patch by averaging with the original crop."""

    # Identify the image shape
    patch_size = original_patch.shape[0]
    square_size = patch_size // 2

    # Create a grid of coordinates
    y, x = np.ogrid[:patch_size, :patch_size]
    center_x, center_y = patch_size // 2, patch_size // 2

    # Calculate distances from the center
    distances = np.maximum(np.abs(x - center_x), np.abs(y - center_y))

    # Calculate the fading factor
    fading_factor = np.clip((square_size - distances) / square_size, 0, 1)

    # Calculate the pixel values with fading squares
    faded_patch = (
        central_pixel_value - background_pixel_value
    ) * fading_factor + background_pixel_value

    # Average values with the original patch
    average_image = (original_patch + faded_patch) / 2

    return average_image


def _insert_fading_circle_into_patch(
    original_patch: npt.NDArray,
    central_pixel_value: int | float,
    background_pixel_value: int | float,
) -> npt.NDArray:
    """Inserts a fading shape at the centre coordinate.
    Creates a gradient of values from the central pixel to the boundary (background).
    Blends the modified patch by averaging with the original crop."""

    # Create a grid of coordinates
    patch_size = min(original_patch.shape)
    y, x = np.ogrid[:patch_size, :patch_size]
    center_x, center_y = patch_size // 2, patch_size // 2

    # Calculate distances from the center
    distances = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

    # Calculate the smooth transition
    normalized_distances = distances / (patch_size // 2)
    transition = (
        1 - normalized_distances
    ) * central_pixel_value + normalized_distances * background_pixel_value

    # Ensure values are within the specified range
    smooth_patch = np.clip(
        transition,
        min(central_pixel_value, background_pixel_value),
        max(central_pixel_value, background_pixel_value),
    )

    # Average values with the original patch
    average_image = (original_patch + smooth_patch) / 2

    return average_image


def synthesize_image_from_graph(
    G: nx.Graph,
    drawing_type: str,
    background_pixel_value: int | float,
    image_shape: tuple[int, int],
    patch_shape: tuple[int, int],
    image_padding: tuple[int, int] = None,
):
    """Synthesize a 2D fake image of specified shape with grey patch centres
        of real nodes (belonging to an object) and black patch centres
        of fake nodes (random noisy nodes).

    Parameters
    ----------
    G : nx.Graph
        A (synthetic) networkx graph.
    drawing_type : str
        Type of the geometric object to draw under the node.
        Choose between "circle(s)", "square(s)" or "star(s)".
    background_pixel_value : int | float
        Value of the canvas image (background).
    image_shape : tuple[int, int]
        Shape of the canvas image (background).
    patch_shape : tuple[int, int]
        Shape of the patch drawing under each node.
    image_padding : tuple[int, int]
        Padding of the image around the corners. Defaults to None.

    Returns
    -------
    image : npt.NDArray
        Simulated image.
    G : nx.Graph
        A (synthetic) networkx graph.
    """
    # Make the shape a plural form:
    drawing_type = (
        f"{drawing_type}s" if not drawing_type.endswith("s") else drawing_type
    )

    # Create a fake blank image:
    canvas_image = create_canvas_image(
        background_pixel_value, image_shape, image_padding
    )

    # Iterate through the node positions:
    for _, node in G.nodes.data():
        coords_max = [image_shape[-1], image_shape[-2]]

        if image_padding is not None:
            node[GraphAttrs.NODE_X] += image_padding[-1]
            node[GraphAttrs.NODE_Y] += image_padding[-2]
            coords_max[0] += 2 * image_padding[-1]
            coords_max[1] += 2 * image_padding[-2]

        x, y = int(node[GraphAttrs.NODE_X]), int(node[GraphAttrs.NODE_Y])

        # Randomly choose the value of the pixel:
        random_value = np.random.random() * 0.5
        intensity_value = (
            1 - random_value if node["label"] > 0 else random_value
        )

        # Define the square type:
        if drawing_type == "squares":
            sq_st_x = max([x - patch_shape[-1] // 4, 0])
            sq_en_x = min([x + patch_shape[-1] // 4, coords_max[0]])
            sq_st_y = max([y - patch_shape[-2] // 4, 0])
            sq_en_y = min([y + patch_shape[-2] // 4, coords_max[1]])
            canvas_image[sq_st_x:sq_en_x, sq_st_y:sq_en_y] = intensity_value

        else:
            # Choose an object shape:
            if drawing_type == "circles":
                drawing_function = _insert_fading_circle_into_patch
            elif drawing_type == "stars":
                drawing_function = _insert_fading_square_into_patch
            else:
                pass

            # List the coords:
            sq_st_x = x - patch_shape[-1] // 2
            sq_en_x = x + patch_shape[-1] // 2
            sq_st_y = y - patch_shape[-2] // 2
            sq_en_y = y + patch_shape[-2] // 2
            coords = [sq_st_x, sq_en_x, sq_st_y, sq_en_y]

            # Shortlist (ignore, TODO: fix) boundary nodes:
            if min(coords) < 0 or max(coords) > max(coords_max):
                continue

            # Draw the object into the patch:
            patch_to_modify = canvas_image[sq_st_x:sq_en_x, sq_st_y:sq_en_y]
            modified_patch = drawing_function(
                patch_to_modify, intensity_value, background_pixel_value
            )
            canvas_image[sq_st_x:sq_en_x, sq_st_y:sq_en_y] = modified_patch

    return canvas_image, G


def save_image_and_graph_combo(
    G: nx.Graph, image: np.ndarray, folder_path: str, file_name: str
) -> None:
    """Saves the image (as .mrc) and list of its node coordinates (as .h5)
    into two accompanying files into the same folder."""
    # Create the path:
    Path(folder_path).mkdir(parents=True, exist_ok=True)

    # Save the image:
    SAVE_IMAGE = Path(folder_path) / file_name
    with mrcfile.new(SAVE_IMAGE, overwrite=True) as mrc:
        mrc.set_data(image)

    # Extract nodes coords:
    SAVE_NODES = SAVE_IMAGE.with_suffix(".h5")

    # SAVE_NODES = SAVE_IMAGE.replace(".mrc", ".h5")
    node_coords = np.array(
        [
            [node[GraphAttrs.NODE_X], node[GraphAttrs.NODE_Y]]
            for _, node in G.nodes(data=True)
        ]
    )

    # Export as H5:
    coords_df = pd.DataFrame(node_coords, columns=["x", "y"])
    coords_df.to_hdf(SAVE_NODES, key="df", mode="w")
