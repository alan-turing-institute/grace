from typing import Union

import os
import click
import torch

from datetime import datetime
from tqdm.auto import tqdm

from grace.io.image_dataset import ImageGraphDataset
from grace.training.train import train_model
from grace.models.datasets import dataset_from_graph
from grace.models.classifier import GCN
from grace.models.feature_extractor import FeatureExtractor
from grace.training.config import write_config_file, load_config_params
from grace.utils.transforms import get_transforms


def run_grace(config_file: Union[str, os.PathLike]) -> None:
    """Runs the GRACE pipeline; going straight from images and .grace annotations
    to a trained node/edge classifier model.

    This function sacrifices some flexibility

    Parameters
    ----------
    config_file : str or Path
        Config file specifying the hyperparameters of the model, training run
        and dataset loading procedure. If this is a Path or string, then it is
        assumed, this will be the filename of a config file (.json or .yaml).
    """
    config = load_config_params(config_file)

    extractor_model = torch.load(config.extractor_fn)
    patch_augs = get_transforms(config, "patch")
    img_graph_augs = get_transforms(config, "graph")
    feature_extractor = FeatureExtractor(
        model=extractor_model,
        augmentations=patch_augs,
        bbox_size=config.patch_size,
        keep_patch_fraction=config.keep_patch_fraction,
    )

    def transform(img, grph):
        img_aug, grph_aug = img_graph_augs(img, grph)
        return feature_extractor(img_aug, grph_aug)

    input_data = ImageGraphDataset(
        image_dir=config.image_dir,
        grace_dir=config.grace_dir,
        image_filetype=config.filetype,
        transform=transform,
    )

    # TQDM progress bar:
    dataset = []
    for _, target in tqdm(
        input_data, desc="Extracting patch features from training data... "
    ):
        print(target["metadata"]["image_filename"])
        dataset.extend(dataset_from_graph(target["graph"], mode="sub"))

    classifier = GCN(
        input_channels=config.feature_dim,
        hidden_channels=config.hidden_channels,
        dropout=config.dropout,
        node_output_classes=config.num_node_classes,
        edge_output_classes=config.num_edge_classes,
    )

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = config.log_dir / current_time
    setattr(config, "run_dir", run_dir)

    train_model(
        classifier,
        dataset,
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        log_dir=run_dir,
        metrics=config.metrics,
        tensorboard_update_frequency=config.tensorboard_update_frequency,
    )

    model_save_fn = run_dir / "classifier.pt"
    torch.save(classifier, model_save_fn)
    write_config_file(config)


@click.command(name="GRACE Trainer")
@click.option("--config_file", type=click.Path(exists=True))
def run(config_file: Union[str, os.PathLike]) -> None:
    run_grace(config_file)


if __name__ == "__main__":
    run()
