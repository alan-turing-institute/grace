import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from pathlib import Path
from tqdm.auto import tqdm
from grace.logger import LOGGER


def animate_valid_graph_plots(
    plots_path: str | Path, plots_file: str, verbose: bool = True
) -> None:
    LOGGER.info(f"Animation for file {plots_file} launched...")

    # # Define the output video file name
    video_filename = plots_path / f"{plots_file}-Animation.mp4"

    # Get a list of all the image files in the folder
    images = [
        img for img in plots_path.glob("*.png") if plots_file in img.stem
    ]

    # Custom sorting function to sort by the numeric part of the filename
    def sort_by_numeric_part(filename):
        numeric_part = "".join(filter(str.isdigit, filename.stem))
        return int(numeric_part)

    # Sort the images based on the numeric part of the filename
    images.sort(key=lambda x: sort_by_numeric_part(x))

    # Turn off interactive mode to prevent figure display
    plt.ioff()

    # Create a writer for the video
    writer = FFMpegWriter(fps=5)  # Adjust the fps as needed

    # Get the dimensions of the first image to set the figure size
    first_image_path = images[0]
    first_image = plt.imread(first_image_path)
    height, width, _ = first_image.shape

    # Create a figure with adjusted figsize and axis limits
    fig, ax = plt.subplots(figsize=(width / 100, height / 100))
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)

    with writer.saving(fig, video_filename, dpi=100):
        desc = f"Creating animation with {len(images)} frames..."
        for image_path in tqdm(images, desc=desc, disable=not verbose):
            img = plt.imread(image_path)
            ax.imshow(img, extent=[0, width, 0, height])
            ax.axis("off")  # Turn off the axis

            # Set the title of the figure to be the filename
            filename = image_path.stem  # Get the filename without extension
            ax.set_title(filename)

            writer.grab_frame()

        # Close the figure to prevent it from being displayed
        plt.close(fig)

    LOGGER.info(f"Animation for file {plots_file} complete...")


def animate_entire_valid_set(
    plots_path: str | Path, verbose: bool = True
) -> None:
    unique_images = set(
        [str(img.stem).split("-")[0] for img in plots_path.glob("*.png")]
    )
    desc = (
        f"Processing {len(unique_images)} validation images for animation..."
    )
    for image in tqdm(unique_images, desc=desc, disable=not verbose):
        animate_valid_graph_plots(plots_path, plots_file=image)
