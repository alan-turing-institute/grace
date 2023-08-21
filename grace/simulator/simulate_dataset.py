from grace.simulator.simulate_graph import random_graph
from grace.simulator.simulate_image import (
    synthesize_image_from_graph,
    save_image_and_graph_combo,
)


def synthetic_image_dataset(
    folder_path: str,
    num_images: int = 10,
    drawing_type: str = "square",
    object_motif: str = "lines",
    image_padding: tuple[int, int] = None,
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
            drawing_type=drawing_type,
            background_pixel_value=VALUE,
            image_shape=(SCALE, SCALE),
            patch_shape=(PATCH_SIZE, PATCH_SIZE),
            image_padding=image_padding,
        )

        # Save the image & node coordinates:
        file_name = f"MRC_Synthetic_File_{str(iteration).zfill(3)}.mrc"
        save_image_and_graph_combo(G, image.T, folder_path, file_name)


if __name__ == "__main__":
    VALUE = 0.5
    SCALE = 3500
    NUM_IMAGES = 10
    PATCH_SIZE = 224
    MOTIF = "lines"
    DRAWING = "stars"
    PADDING = (112, 112)

    # Create a dataset:
    synthetic_image_dataset(
        folder_path=f"/Users/kulicna/Desktop/dataset/shape_{DRAWING}",
        num_images=NUM_IMAGES,
        drawing_type=DRAWING,
        object_motif=MOTIF,
        image_padding=PADDING,
    )
