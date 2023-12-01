import torch
import networkx as nx

from grace.io.image_dataset import ImageGraphDataset


def test_image_graph_dataset(mrc_image_and_annotations_dir):
    """Test that the dataset loader can import test images and grace annotations."""
    num_images = len(list(mrc_image_and_annotations_dir.glob("*.mrc")))

    dataset = ImageGraphDataset(
        mrc_image_and_annotations_dir,
        mrc_image_and_annotations_dir,
        image_filetype="mrc",
    )
    assert len(dataset) == num_images

    # unwrap sample image & target
    image, target = dataset[0]
    graph = target["graph"]
    metadata = target["metadata"]
    filename = metadata["image_filename"]

    assert isinstance(image, torch.Tensor)
    assert image.shape == (128, 128)  # 2D image

    assert isinstance(metadata, dict)
    assert isinstance(filename, str)

    assert isinstance(graph, nx.Graph)
    assert graph.number_of_nodes() == 4


def test_dataset_only_takes_common_filenames(tmp_path):
    image_fns = ["b", "a", "aj", "jj"]
    label_fns = ["jj", "b", "kk"]

    image_fns = [f + ".png" for f in image_fns]
    label_fns = [f + ".grace" for f in label_fns]

    image_dir = tmp_path / "images"
    label_dir = tmp_path / "labels"

    image_dir.mkdir()
    label_dir.mkdir()

    for fn in image_fns:
        file = image_dir / fn
        file.touch()

    for fn in label_fns:
        file = label_dir / fn
        file.touch()

    dataset = ImageGraphDataset(image_dir, label_dir, image_filetype="png")

    expected_image_paths = [
        tmp_path / "images" / "b.png",
        tmp_path / "images" / "jj.png",
    ]

    expected_label_paths = [
        tmp_path / "labels" / "b.grace",
        tmp_path / "labels" / "jj.grace",
    ]

    assert dataset.image_paths == expected_image_paths
    assert dataset.grace_paths == expected_label_paths
