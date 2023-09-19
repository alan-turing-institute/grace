from typing import Union, Callable
from functools import partial

import os
import click
import torch

from datetime import datetime
from tqdm.auto import tqdm

from grace.logger import LOGGER
from grace.io.image_dataset import ImageGraphDataset
from grace.training.train import train_model
from grace.models.datasets import dataset_from_graph
from grace.models.classifier import GCN
from grace.models.feature_extractor import FeatureExtractor
from grace.training.config import write_config_file, load_config_params
from grace.utils.transforms import get_transforms
from grace.evaluation.inference import GraphLabelPredictor
from grace.visualisation.plotting import (
    visualise_node_and_edge_probabilities,
    visualise_prediction_probs_hist,
)


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
    # Load all config parameters:
    config = load_config_params(config_file)

    # Prepare the feature extractor:
    extractor_model = torch.load(config.extractor_fn)
    patch_augs = get_transforms(config, "patch")
    img_graph_augs = get_transforms(config, "graph")
    feature_extractor = FeatureExtractor(
        model=extractor_model,
        augmentations=patch_augs,
        normalize=config.normalize,
        bbox_size=config.patch_size,
        keep_patch_fraction=config.keep_patch_fraction,
    )

    # Condition the augmentations to train mode only: TODO: type correctly:
    def transform(img, grph, in_train_mode: bool = True) -> Callable:
        if in_train_mode:
            img_aug, grph_aug = img_graph_augs(img, grph)
            return feature_extractor(img_aug, grph_aug)
        else:
            return feature_extractor(img, grph)

    # Create a transform function with frozen 'in_train_mode' parameter:
    transform_train_mode = partial(transform, in_train_mode=True)
    transform_valid_mode = partial(transform, in_train_mode=False)

    # Read the TRAIN data:
    train_input_data = ImageGraphDataset(
        image_dir=config.train_image_dir,
        grace_dir=config.train_grace_dir,
        image_filetype=config.filetype,
        keep_node_unknown_labels=config.keep_node_unknown_labels,
        keep_edge_unknown_labels=config.keep_edge_unknown_labels,
        transform=transform_train_mode,
    )

    # Iterate through images & extract node features:
    train_dataset = []
    for _, target in tqdm(
        train_input_data,
        desc="Extracting patch features from TRAIN images... ",
    ):
        file_name = target["metadata"]["image_filename"]
        LOGGER.info(f"Processing file: {file_name}")
        graph_dataset = dataset_from_graph(target["graph"], mode="sub")
        train_dataset.extend(graph_dataset)

    # Read the VALID data:
    valid_input_data = ImageGraphDataset(
        image_dir=config.valid_image_dir,
        grace_dir=config.valid_grace_dir,
        image_filetype=config.filetype,
        keep_node_unknown_labels=config.keep_node_unknown_labels,
        keep_edge_unknown_labels=config.keep_edge_unknown_labels,
        transform=transform_valid_mode,
    )

    # Iterate through images & extract node features:
    valid_target_list = []
    valid_dataset = []
    for _, target in tqdm(
        valid_input_data,
        desc="Extracting patch features from VALID images... ",
    ):
        LOGGER.info(f"Current file: {target['metadata']['image_filename']}")
        # Store the valid graph list:
        valid_target_list.append(target)
        # Chop graph into subgraphs & store:
        graph_dataset = dataset_from_graph(target["graph"], mode="sub")
        valid_dataset.extend(graph_dataset)

    # Define the GNN classifier model:
    classifier = GCN(
        input_channels=config.feature_dim,
        hidden_channels=config.hidden_channels,
        dropout=config.dropout,
        node_output_classes=config.num_node_classes,
        edge_output_classes=config.num_edge_classes,
    )

    # Define where you'll save the outputs:
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = config.log_dir / current_time
    setattr(config, "run_dir", run_dir)

    # Perform the training:
    train_model(
        classifier,
        train_dataset,
        valid_dataset,
        valid_target_list,
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        log_dir=run_dir,
        metrics=config.metrics,
        tensorboard_update_frequency=config.tensorboard_update_frequency,
        valid_graph_ploter_frequency=config.valid_graph_ploter_frequency,
    )

    # Save the model:
    model_save_fn = run_dir / "classifier.pt"
    torch.save(classifier, model_save_fn)
    write_config_file(config)

    # TODO: Run inference on the final, trained model on unseen data:
    # Create subdirectory to save out plots:
    infer_path = run_dir / "infer_plots"
    infer_path.mkdir(parents=True, exist_ok=True)

    # Read the INFER data:
    infer_input_data = ImageGraphDataset(
        image_dir=config.infer_image_dir,
        grace_dir=config.infer_grace_dir,
        image_filetype=config.filetype,
        keep_node_unknown_labels=config.keep_node_unknown_labels,
        keep_edge_unknown_labels=config.keep_edge_unknown_labels,
        transform=transform_valid_mode,
    )

    # Iterate through images & extract node features:
    infer_target_list = []
    for _, target in tqdm(
        infer_input_data,
        desc="Extracting patch features from INFER images... ",
    ):
        LOGGER.info(f"Current file: {target['metadata']['image_filename']}")
        infer_target_list.append(target)

    # Instantiate the model with frozen weights:
    GLP = GraphLabelPredictor(model_save_fn)

    # Iterate through all validation graphs & predict nodes / edges:
    for infer_target in infer_target_list:
        infer_graph = infer_target["graph"]

        # Filename:
        infer_name = infer_target["metadata"]["image_filename"]

        # Update probabs & visualise the graph:
        GLP.set_node_and_edge_probabilities(G=infer_graph)
        # TODO: Re-implement when plotting issues are solved:

        # GLP.visualise_model_performance_on_graph(
        #     G=infer_graph,
        #     filename=infer_path / f"{infer_name}-Areas_under_Curves.png"
        # )

        # HACK: Save out at least two plots:
        visualise_node_and_edge_probabilities(
            G=infer_graph,
            filename=infer_path / f"{infer_name}-Inference_Graph.png",
        )
        visualise_prediction_probs_hist(
            G=infer_graph,
            filename=infer_path / f"{infer_name}-Prediction_Hist.png",
        )

    # Close the run:
    LOGGER.info("Run complete... Done!")


@click.command(name="GRACE Trainer")
@click.option("--config_file", type=click.Path(exists=True))
def run(config_file: Union[str, os.PathLike]) -> None:
    run_grace(config_file)


if __name__ == "__main__":
    run()
