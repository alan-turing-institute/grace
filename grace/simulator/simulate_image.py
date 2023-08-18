import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import numpy.typing as npt

import mrcfile
from pathlib import Path
from skimage.util import montage

from grace.base import GraphAttrs, Annotation


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


def _insert_simple_square_into_patch(
    original_patch: npt.NDArray,
    square_pixel_value: int | float,
) -> npt.NDArray:
    """TODO: Fill in."""

    # Get image dimensions
    height, width = original_patch.shape

    # Calculate the size of the square
    square_size = min(height, width) // 2

    # Calculate the coordinates of the square's top-left corner
    top_left_x = (width - square_size) // 2
    top_left_y = (height - square_size) // 2

    # Insert a black square in the center
    original_patch[
        top_left_y : top_left_y + square_size,
        top_left_x : top_left_x + square_size,
    ] = square_pixel_value

    return original_patch


def _insert_fading_square_into_patch(
    original_patch: npt.NDArray,
    square_pixel_value: int | float,
    background_pixel_value: int | float,
) -> npt.NDArray:
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
        square_pixel_value - background_pixel_value
    ) * fading_factor + background_pixel_value

    return faded_patch


# def _insert_circle_into_patch(
#     original_patch: npt.NDArray,
#     circle_pixel_value: int | float,
# ) -> npt.NDArray:
#     """TODO: Fill in."""

#     # Get image dimensions
#     height, width = original_patch.shape

#     # Insert a chosen pixel value at the center
#     center_x, center_y = height // 2, width // 2
#     # print (height, width, center_x, center_y, circle_pixel_value)
#     original_patch[center_y, center_x] = circle_pixel_value

#     # Create a Gaussian kernel for blurring
#     blur_radius = min(height, width) // 2
#     gaussian_kernel = np.outer(
#         np.exp(-np.linspace(-1, 1, blur_radius * 2 + 1)**2),
#         np.exp(-np.linspace(-1, 1, blur_radius * 2 + 1)**2)
#     )
#     # Normalize the Gaussian kernel
#     gaussian_kernel /= np.sum(gaussian_kernel)

#     # Apply radial blur from the center towards the edges
#     blurred_patch = gaussian_filter(original_patch, sigma=blur_radius/2)

#     return blurred_patch


def _insert_circle_into_patch(
    original_patch: npt.NDArray,
    circle_pixel_value: int | float,
    background_pixel_value: int | float,
) -> npt.NDArray:
    """TODO: Fill in."""

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
    ) * circle_pixel_value + normalized_distances * background_pixel_value

    # Ensure values are within the specified range
    smooth_patch = np.clip(
        transition,
        min(circle_pixel_value, background_pixel_value),
        max(circle_pixel_value, background_pixel_value),
    )

    return smooth_patch


def synthesize_image_from_graph(
    G: nx.Graph,
    square_type: str,
    background_pixel_value: int | float,
    image_shape: tuple[int, int],
    patch_shape: tuple[int, int],
    image_padding: tuple[int, int] = None,
):
    """Synthesize a 2D fake image of specified shape with grey patch centres
        of real nodes (belonging to an object) and black patch centres
        of fake nodes (random noisy nodes).

    Parameters -> TODO: Fix!!!
    ----------
    G : nx.Graph
        A (synthetic) networkx graph.
    image_value : float
        Value of the blank image pixels (e.g. 255).
    image_shape : tuple[int, int]
        Shape of the image to create, without padding.
    crop_shape : tuple[int, int]
        Shape of the cropped patches to train on (e.g. (224, 224)
        - compatible to resnet input).
    mask_shape : tuple[int, int]
        Shape of the mask to mark the positive nodes (i.e. parts of an object)
        from negative nodes (random noise).

    Returns
    -------
    image : npt.NDArray
        Simulated image.
    """

    # Create a fake blank image:
    canvas_image = create_canvas_image(background_pixel_value, image_shape)

    # Iterate through the node positions:
    for _, node in G.nodes.data():
        # Randomly choose the value of the pixel:
        random_value = np.random.random() * 0.5
        intensity_value = (
            1 - random_value if node["label"] > 0 else random_value
        )

        # Define the square type:
        if square_type == "simple":
            sq_st_x = max(
                [int(node[GraphAttrs.NODE_X]) - patch_shape[-1] // 4, 0]
            )
            sq_en_x = min(
                [
                    int(node[GraphAttrs.NODE_X]) + patch_shape[-1] // 4,
                    image_shape[-1],
                ]
            )
            sq_st_y = max(
                [int(node[GraphAttrs.NODE_Y]) - patch_shape[-2] // 4, 0]
            )
            sq_en_y = min(
                [
                    int(node[GraphAttrs.NODE_Y]) + patch_shape[-2] // 4,
                    image_shape[-2],
                ]
            )
            canvas_image[sq_st_x:sq_en_x, sq_st_y:sq_en_y] = intensity_value

        elif square_type == "fading":
            sq_st_x = int(node[GraphAttrs.NODE_X]) - patch_shape[-1] // 2
            sq_en_x = int(node[GraphAttrs.NODE_X]) + patch_shape[-1] // 2
            sq_st_y = int(node[GraphAttrs.NODE_Y]) - patch_shape[-2] // 2
            sq_en_y = int(node[GraphAttrs.NODE_Y]) + patch_shape[-2] // 2
            coords = [sq_st_x, sq_en_x, sq_st_y, sq_en_y]
            if min(coords) < 0 or max(coords) > max(image_shape):
                continue

            patch_to_modify = canvas_image[sq_st_x:sq_en_x, sq_st_y:sq_en_y]
            modified_patch = _insert_fading_square_into_patch(
                patch_to_modify, intensity_value, background_pixel_value
            )
            canvas_image[sq_st_x:sq_en_x, sq_st_y:sq_en_y] = modified_patch

        else:
            pass

    return canvas_image, G


def montage_from_image(
    G: nx.Graph, image: npt.NDArray, crop_shape: tuple[int, int]
) -> None:
    """Visualise the montages of some true negative (0) and true positive (1) nodes.

    Parameters
    ----------
    G : nx.Graph
        A (synthetic) networkx graph.
    image : np.array
        Simulated image corresponding to the graph.
    crop_shape : tuple[int, int]
        Shape of the cropped patches to train on
        (e.g. (224, 224) - compatible to resnet input).
    """

    plt.figure(figsize=(10, 5))
    crops = [[], []]

    for _, node in G.nodes.data():
        coords = node[GraphAttrs.NODE_X], node[GraphAttrs.NODE_Y]
        st_x, en_x = (
            int(coords[0]) - crop_shape[0] // 2,
            int(coords[0]) + crop_shape[0] // 2,
        )
        st_y, en_y = (
            int(coords[1]) - crop_shape[1] // 2,
            int(coords[1]) + crop_shape[1] // 2,
        )

        # Sort crops based on labels:
        crop = image[st_x:en_x, st_y:en_y].numpy()
        label = node[GraphAttrs.NODE_GROUND_TRUTH]
        if label < Annotation.UNKNOWN:
            crops[label].append(crop)

    for c, crop_collection in enumerate(crops):
        mont = montage(
            crop_collection[:49],
            grid_shape=(7, 7),
            padding_width=10,
            fill=np.max([np.max(c) for c in crop_collection[0]]),
        )
        plt.subplot(1, 2, c + 1)
        plt.imshow(mont, cmap="binary_r")
        plt.colorbar(fraction=0.045)
        plt.title(f"Montage of patches\nwith 'node_label' = {c}")
        plt.axis("off")
    plt.show()
    plt.close()


def save_image_and_graph_combo(
    G: nx.Graph, image: np.ndarray, folder_path: str, file_name: str
) -> None:
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
