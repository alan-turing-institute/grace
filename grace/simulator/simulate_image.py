import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import numpy.typing as npt

import mrcfile
from pathlib import Path
from skimage.util import montage

from grace.base import GraphAttrs, Annotation
from grace.simulator.simulate_graph import random_graph


VALUE = 0.5
SCALE = 3500
PATCH_SIZE = 224

CROP_SHAPE = (PATCH_SIZE, PATCH_SIZE)
MASK_SHAPE = tuple([p // 2 for p in CROP_SHAPE])


def create_blank_image(
    image_value: float,
    image_shape: tuple[int, int],
    image_padding: tuple[int, int] = None,
) -> npt.NDArray:
    image = np.zeros(shape=image_shape, dtype=np.float32)
    image += image_value
    if image_padding is not None:
        image = np.pad(
            image,
            pad_width=image_padding,
            mode="constant",
            constant_values=image_value,
        )
    return image


def synthesize_image_from_graph(
    G: nx.Graph,
    image_value: float,
    image_shape: tuple[int, int],
    crop_shape: tuple[int, int],
    mask_shape: tuple[int, int],
):
    """Synthesize a 2D fake image of specified shape with grey patch centres
        of real nodes (belonging to an object) and black patch centres
        of fake nodes (random noisy nodes).

    Parameters
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
    image = create_blank_image(
        image_value=image_value,
        image_shape=image_shape,
        image_padding=crop_shape,
    )
    patch_white_value = np.mean(image) * 2

    # Define patches - grey for object-belonging nodes, black for random noisy nodes:
    patch_black = np.zeros(shape=mask_shape, dtype=image.dtype)
    patch_white = (
        np.ones(shape=mask_shape, dtype=image.dtype) * patch_white_value
    )

    for _, node in G.nodes.data():
        # Update the coords of the node by padding:
        node[GraphAttrs.NODE_X] += crop_shape[0]
        node[GraphAttrs.NODE_Y] += crop_shape[1]

        # Position the mask patch:
        coords = node[GraphAttrs.NODE_X], node[GraphAttrs.NODE_Y]
        st_x, en_x = (
            int(coords[0]) - mask_shape[0] // 2,
            int(coords[0]) + mask_shape[0] // 2,
        )
        st_y, en_y = (
            int(coords[1]) - mask_shape[1] // 2,
            int(coords[1]) + mask_shape[1] // 2,
        )

        # Paint the image according to the node label:
        if node["label"] > 0:  # real node with GT = 1
            image[st_x:en_x, st_y:en_y] = patch_white
        else:
            image[st_x:en_x, st_y:en_y] = patch_black

    return image, G


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


def synthetic_image_dataset(num_images: int, motif: str, folder_path: str):
    for iteration in range(num_images):
        # Create a random graph with motif:
        G = random_graph(
            n_motifs=5, n_chaff=100, scale=SCALE, density=0.025, motif=motif
        )

        # Synthesize a corresponding image:
        image, G = synthesize_image_from_graph(
            G,
            image_value=VALUE,
            image_shape=(SCALE, SCALE),
            crop_shape=CROP_SHAPE,
            mask_shape=MASK_SHAPE,
        )

        # Save the image & node coordinates:
        file_name = f"MRC_Synthetic_File_{str(iteration).zfill(3)}.mrc"
        save_image_and_graph_combo(G, image.T, folder_path, file_name)
