from typing import Union

import os
import click
import torch
from datetime import datetime
from tqdm.auto import tqdm

from grace.styling import LOGGER
from grace.io.image_dataset import ImageGraphDataset

from grace.models.feature_extractor import FeatureExtractor, SimpleDescriptor
from grace.models.datasets import dataset_from_graph
from grace.models.classifier import Classifier

from grace.training.train import train_model
from grace.training.assess import assess_training_performance
from grace.training.config import (
    validate_required_config_hparams,
    load_config_params,
    write_config_file,
)
from grace.visualisation.animation import animate_entire_valid_set


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
    validate_required_config_hparams(config)

    # Define where you'll save the outputs:
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = config.log_dir / current_time
    setattr(config, "run_dir", run_dir)

    # Create subdirectory to save out plots:
    subfolders = ["model", "valid", "infer"]
    if config.classifier_type == "GAT":
        subfolders.append("weights")

    for subfolder in subfolders:
        subfolder_path = run_dir / subfolder
        subfolder_path.mkdir(parents=True, exist_ok=True)

    # Process the datasets as desired:
    def prepare_dataset(
        image_dir: Union[str, os.PathLike],
        grace_dir: Union[str, os.PathLike],
        num_hops: int | str,
        connection: str = "spiderweb",
        verbose: bool = True,
    ) -> tuple[list]:
        # Read the data & terate through images & extract node features:
        # Choose extractor:
        if config.extractor_fn is not None:
            assert str(config.extractor_fn).endswith(".pt")
            extractor_model = torch.load(config.extractor_fn)
            feature_extractor = FeatureExtractor(
                model=extractor_model,
                normalize=config.normalize,
                bbox_size=config.patch_size,
            )
        else:
            feature_extractor = SimpleDescriptor(
                bbox_size=config.patch_size,
            )

        input_data = ImageGraphDataset(
            image_dir=image_dir,
            grace_dir=grace_dir,
            image_filetype=config.filetype,
            keep_node_unknown_labels=config.keep_node_unknown_labels,
            keep_edge_unknown_labels=config.keep_edge_unknown_labels,
            transform=feature_extractor,
        )

        # Process the (sub)graph data into torch_geometric dataset:
        target_list, subgraph_dataset = [], []
        desc = "Extracting patch features from images... "
        for _, target in tqdm(input_data, desc=desc, disable=not verbose):
            file_name = target["metadata"]["image_filename"]
            LOGGER.info(f"Processing file: {file_name}")

            # Store the valid graph list:
            target_list.append(target)

            # Chop graph into subgraphs & store:
            graph_data = dataset_from_graph(
                target["graph"],
                num_hops=num_hops,
                connection=connection,
            )
            subgraph_dataset.extend(graph_data)

        return target_list, subgraph_dataset

    # Read the respective datasets:
    _, train_dataset = prepare_dataset(
        config.train_image_dir,
        config.train_grace_dir,
        num_hops=config.num_hops,
        connection=config.connection,
    )
    valid_target_list, valid_dataset = prepare_dataset(
        config.valid_image_dir,
        config.valid_grace_dir,
        num_hops=config.num_hops,
        connection=config.connection,
    )
    infer_target_list, _ = prepare_dataset(
        config.infer_image_dir,
        config.infer_grace_dir,
        num_hops="whole",
    )

    # Define the Classifier model:
    classifier = Classifier().get_model(
        classifier_type=config.classifier_type,
        input_channels=config.node_embedding_ndim,
        hidden_graph_channels=config.hidden_graph_channels,
        hidden_dense_channels=config.hidden_dense_channels,
        node_output_classes=config.num_node_classes,
        edge_output_classes=config.num_edge_classes,
        num_heads=config.num_attention_heads,
        dropout=config.dropout,
    )

    # Log the model architecture:
    model_architecture_str = classifier.__str__()
    LOGGER.info(model_architecture_str)

    # Perform the training:
    train_model(
        config.classifier_type,
        classifier,
        train_dataset,
        valid_dataset,
        valid_target_list=valid_target_list,
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        scheduler_type=config.scheduler_type,
        scheduler_step=config.scheduler_step,
        scheduler_gamma=config.scheduler_gamma,
        log_dir=run_dir,
        metrics=config.metrics_classifier,
        tensorboard_update_frequency=config.tensorboard_update_frequency,
        valid_graph_ploter_frequency=config.valid_graph_ploter_frequency,
    )

    # Save the trained model:
    config.run_dir = run_dir / "model"
    model_save_fn = config.run_dir / "classifier.pt"
    torch.save(classifier, model_save_fn)

    # Save the training hyperparameters:
    write_config_file(config, filetype="json")
    write_config_file(config, filetype="yaml")

    # Archive the model architecture:
    model_architecture_dir = config.run_dir / "summary_architecture.txt"
    with open(model_architecture_dir, "w") as summary_file:
        summary_file.write(model_architecture_str)

    # Project the TSNE manifold:
    if config.visualise_tsne_manifold is True:
        LOGGER.warning(
            "WARNING; TSNE manifold visualisation not implemented yet"
        )

        # TODO: Implement
        pass

    # Animate the validation outputs:
    if config.animate_valid_progress is True:
        animate_entire_valid_set(run_dir / "valid", verbose=False)

    # Assess model performance on inference dataset:
    assess_training_performance(
        run_dir=run_dir,
        infer_target_list=infer_target_list,
        compute_exact_metrics="exact" in config.metrics_objects,
        compute_approx_metrics="approx" in config.metrics_objects,
    )

    # Close the run:
    LOGGER.info("Run complete... Done!")


@click.command(name="GRACE Trainer")
@click.option("--config_file", type=click.Path(exists=True))
def run(config_file: Union[str, os.PathLike]) -> None:
    run_grace(config_file)


if __name__ == "__main__":
    run()
