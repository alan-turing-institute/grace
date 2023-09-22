from typing import Union, Callable
from functools import partial

import os
import click
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm.auto import tqdm

from grace.logger import LOGGER
from grace.io.image_dataset import ImageGraphDataset
from grace.training.train import train_model
from grace.models.datasets import dataset_from_graph

from grace.models.classifier import GNNClassifier
from grace.models.feature_extractor import FeatureExtractor
from grace.models.optimiser import optimise_graph
from grace.training.config import (
    load_config_params,
    write_config_file,
    write_json_file,
)
from grace.utils.transforms import get_transforms
from grace.evaluation.inference import GraphLabelPredictor
from grace.evaluation.process import generate_ground_truth_graph
from grace.visualisation.animation import animate_entire_valid_set
from grace.visualisation.utils import plot_iou_histogram
from grace.visualisation.plotting import (
    plot_simple_graph,
    plot_connected_components,
)
from grace.evaluation.metrics_objects import (
    ExactMetricsComputer,
    ApproxMetricsComputer,
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

    # Define where you'll save the outputs:
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = config.log_dir / current_time
    setattr(config, "run_dir", run_dir)

    # Create subdirectory to save out plots:
    for subfolder in ["valid", "infer"]:
        subfolder_path = run_dir / subfolder
        subfolder_path.mkdir(parents=True, exist_ok=True)

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

    # Condition the augmentations to train mode only:
    def transform(
        image: torch.Tensor, graph: dict, *, in_train_mode: bool = True
    ) -> Callable:
        # Ensure augmentations are only run on train data:
        if in_train_mode:
            image_aug, graph_aug = img_graph_augs(image, graph)
            return feature_extractor(image_aug, graph_aug)
        else:
            return feature_extractor(image, graph)

    # Process the datasets as desired:
    def prepare_dataset(
        image_dir: Union[str, os.PathLike],
        grace_dir: Union[str, os.PathLike],
        transform_method: Callable,
        *,
        graph_processing: str = "sub",
        verbose: bool = True,
    ) -> tuple[list]:
        # Read the data & terate through images & extract node features:
        input_data = ImageGraphDataset(
            image_dir=image_dir,
            grace_dir=grace_dir,
            image_filetype=config.filetype,
            keep_node_unknown_labels=config.keep_node_unknown_labels,
            keep_edge_unknown_labels=config.keep_edge_unknown_labels,
            transform=transform_method,
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
            graphs = dataset_from_graph(target["graph"], mode=graph_processing)
            subgraph_dataset.extend(graphs)

        return target_list, subgraph_dataset

    # Create a transform function with frozen 'in_train_mode' parameter:
    transform_train_mode = partial(transform, in_train_mode=True)
    transform_valid_mode = partial(transform, in_train_mode=False)

    # Read the respective datasets:
    _, train_dataset = prepare_dataset(
        config.train_image_dir,
        config.train_grace_dir,
        transform_train_mode,
    )
    valid_target_list, valid_dataset = prepare_dataset(
        config.valid_image_dir,
        config.valid_grace_dir,
        transform_valid_mode,
    )
    infer_target_list, _ = prepare_dataset(
        config.infer_image_dir,
        config.infer_grace_dir,
        transform_valid_mode,
    )

    # Define the GNN classifier model:
    classifier = GNNClassifier().get_model(
        config.gnn_classifier_type,
        input_channels=config.feature_dim,
        hidden_channels=config.hidden_channels,
        dropout=config.dropout,
        node_output_classes=config.num_node_classes,
        edge_output_classes=config.num_edge_classes,
    )

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
        metrics=config.metrics_classifier,
        tensorboard_update_frequency=config.tensorboard_update_frequency,
        valid_graph_ploter_frequency=config.valid_graph_ploter_frequency,
    )

    # Save the trained model:
    model_save_fn = run_dir / "classifier.pt"
    torch.save(classifier, model_save_fn)
    write_config_file(config)

    # Animate the validation outputs:
    animate_entire_valid_set(run_dir / "valid", verbose=False)

    # Run inference on the final, trained model on unseen data:
    GLP = GraphLabelPredictor(model_save_fn)

    # Process entire batch & save the results:
    inference_metrics = GLP.batch_process_infer_dataset(
        infer_target_list,
        save_path=run_dir / "infer",
        save_figures=True,
        show_figures=False,
    )
    # Log inference metrics:
    LOGGER.info(f"Inference batch dataset metrics: {inference_metrics}")

    # Write out the batch metrics:
    batch_metrics_fn = (
        run_dir / "infer" / "Inference_Dataset-Batch_Metrics.json"
    )
    write_json_file(inference_metrics, batch_metrics_fn)

    # Define which metrics to calculate:
    metrics = [m.upper() for m in config.metrics_objects]

    # Process each file individually:
    desc = "Measuring inference metrics on images... "
    for graph_data in tqdm(infer_target_list, desc=desc):
        file_name = graph_data["metadata"]["image_filename"]
        LOGGER.info(f"Processing file: {file_name}")

        GLP.visualise_model_performance_on_graph(
            graph_data,
            save_path=run_dir / "infer",
            save_figures=True,
            show_figures=False,
        )

        # Generate GT & optimised graphs:
        graph = graph_data["graph"]
        GLP.set_node_and_edge_probabilities(graph)

        true_graph = generate_ground_truth_graph(graph)
        pred_graph = optimise_graph(graph)

        # Save out the connected components figure:
        # shape = 5
        _, axes = plt.subplots(1, 3, figsize=(15, 5))

        plot_simple_graph(graph, title="Triangulated graph", ax=axes[0])
        plot_connected_components(
            true_graph, title="Ground truth graph", ax=axes[1]
        )
        plot_connected_components(
            pred_graph, title="Optimised graph", ax=axes[2]
        )

        plt.tight_layout()
        plt.savefig(
            run_dir / "infer" / f"{file_name}-Graph_Connected_Components.png"
        )
        # plt.show()

        if "EXACT" in metrics:
            EMC = ExactMetricsComputer(
                G=graph,
                pred_optimised_graph=pred_graph,
                true_annotated_graph=true_graph,
            )
            results_dict = EMC.metrics()
            LOGGER.info(f"'EXACT' metrics: {file_name} | {results_dict}")
            save_fn = run_dir / "infer" / f"{file_name}-Metrics.json"
            write_json_file(results_dict, save_fn)

            EMC.visualise(
                save_path=run_dir / "infer",
                file_name=file_name,
                save_figures=True,
                show_figures=False,
            )

            # Append IoUs:
            if "Object IoUs" not in inference_metrics:
                inference_metrics["Object IoUs"] = []
            file_instance_ious = results_dict["Instance IoU [list]"]
            inference_metrics["Object IoUs"].extend(file_instance_ious)

        # Plot overal IoUs:
        plot_iou_histogram(iou_per_object=inference_metrics["Object IoUs"])
        plt.savefig(
            run_dir / "infer" / "Inference_Dataset-Object_IoU_Histogram.png"
        )
        plt.close()

        if "APPROX" in metrics:
            LOGGER.warning("WARNING; 'APPROX' metrics not implemented yet")

            # TODO: Implement:
            ApproxMetricsComputer

    # Close the run:
    LOGGER.info("Run complete... Done!")


@click.command(name="GRACE Trainer")
@click.option("--config_file", type=click.Path(exists=True))
def run(config_file: Union[str, os.PathLike]) -> None:
    run_grace(config_file)


if __name__ == "__main__":
    run()
