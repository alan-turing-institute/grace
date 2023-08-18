from grace.simulator.simulate_graph import random_graph
from grace.simulator.simulate_image import (
    synthesize_image_from_graph,
    save_image_and_graph_combo,
)


def synthetic_image_dataset(
    folder_path: str,
    num_images: int = 10,
    square_type: str = "simple",
    object_motif: str = "lines",
) -> None:
    """TODO: Fill in."""

    for iteration in range(num_images):
        # Create a random graph with motif:
        G = random_graph(
            n_motifs=5,
            n_chaff=100,
            scale=SCALE,
            density=0.025,
            motif=object_motif,
        )

        # Synthesize a corresponding image:
        image, G = synthesize_image_from_graph(
            G,
            square_type=square_type,
            background_pixel_value=VALUE,
            image_shape=(SCALE, SCALE),
            patch_shape=(PATCH_SIZE, PATCH_SIZE),
        )

        # Save the image & node coordinates:
        file_name = f"MRC_Synthetic_File_{str(iteration).zfill(3)}.mrc"
        save_image_and_graph_combo(G, image.T, folder_path, file_name)


if __name__ == "__main__":
    VALUE = 0.5
    SCALE = 3500
    PATCH_SIZE = 224
    SQUARE = "fading"

    # Create a dataset:
    synthetic_image_dataset(
        folder_path=f"/Users/kulicna/Desktop/dataset/data_squares_{SQUARE}",
        num_images=10,
        square_type=SQUARE,
        object_motif="lines",
    )
