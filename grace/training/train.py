from typing import Optional, Callable

import networkx as nx
import random
import torch
import torch_geometric

from torch_geometric.loader import DataLoader

from grace.base import Annotation
from grace.styling import LOGGER
from grace.evaluation.metrics_classifier import get_metric
from grace.evaluation.inference import GraphLabelPredictor

from torch.utils.tensorboard import SummaryWriter


def train_model(
    model: torch.nn.Module,
    train_dataset: list[torch_geometric.data.Data],
    valid_dataset: list[torch_geometric.data.Data],
    *,
    valid_target_list: list[nx.Graph] = None,
    epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    weight_decay: float = 0.0,
    scheduler_type: str = "none",
    scheduler_step: int = 1,
    scheduler_gamma: float = 1.0,
    node_masked_class: Annotation = Annotation.UNKNOWN,
    edge_masked_class: Annotation = Annotation.UNKNOWN,
    log_dir: Optional[str] = None,
    metrics: list[str | Callable] = [],
    tensorboard_update_frequency: int = 1,
    valid_graph_ploter_frequency: int = 1,
):
    """Train the pytorch model.

    Parameters
    ----------
    model : torch.nn.Module
        Model to train
    train_dataset : list[torch_geometric.data.Data]
        Training dataset of tiny subgraphs
    valid_dataset : list[torch_geometric.data.Data]
        Validation dataset of tiny subgraphs
    valid_target_list : list[dict[str]]
        List of entire graph targets for validation visualisation
    epochs : int
        Number of epochs to train the model
    batch_size : int
        Batch size
    learning_rate : float
        (Base) learning rate to use during training
    weight_decay : float
        Weight decay (L2 penalty) (default: 0.0)
    scheduler_type : str
        Learning rate scheduler (default: "none")
    scheduler_step: int
        Period of learning rate decay in epochs.
    scheduler_gamma : float
        Multiplicative factor of learning rate decay (default: 1.0)
    node_masked_class : Annotation
        Target node class for which to set the loss to 0
    edge_masked_class : Annotation
        Target edge class for which to set the loss to 0
    log_dir : str or None
        Log folder for the current training run
    metrics : List[str or Callable]
        Metrics to be evaluated after every training epoch
    tensorboard_update_frequency : int
        Frequency (in epochs) at which to update tensorboard
    """
    # Instantiate the logger:
    writer = SummaryWriter(log_dir)

    # Shuffle the dataset to make sure subgraphs are unordered:
    random.seed(23)
    random.shuffle(train_dataset)
    random.shuffle(valid_dataset)

    # Split the datasets into respective dataloaders:
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False
    )

    # Define the optimiser:
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # Define the scheduler:
    if scheduler_type != "none":
        if scheduler_type == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=scheduler_step,
                gamma=scheduler_gamma,
            )
        elif scheduler_type == "expo":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=scheduler_gamma,
            )
        else:
            raise NotImplementedError(
                f"Scheduler type '{scheduler_type}' is not implemented."
            )

    # Specify node & edge criterion:
    # TODO: Implement class weighting
    node_criterion = torch.nn.CrossEntropyLoss(
        weight=None,
        reduction="mean",
        ignore_index=node_masked_class,
    )
    edge_criterion = torch.nn.CrossEntropyLoss(
        weight=None,
        reduction="mean",
        ignore_index=edge_masked_class,
    )

    # Train the model epoch:
    def train(loader: torch_geometric.loader.DataLoader) -> None:
        """Trains a single epoch & updates params based on loss."""
        model.train()

        for data in loader:
            node_x, edge_x = model(data.x, data.edge_index)

            loss_node = node_criterion(node_x, data.y)
            loss_edge = edge_criterion(edge_x, data.edge_label)
            loss = loss_node + loss_edge

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    # Validate the epoch, including evaluating metrics:
    def valid(
        loader: torch_geometric.loader.DataLoader,
    ) -> dict[str, torch.tensor]:
        """Evaluates the classifier on node & edge classification."""
        model.eval()

        node_pred = []
        edge_pred = []
        node_true = []
        edge_true = []

        # Iterate through the data loader contents:
        for data in loader:
            node_x, edge_x = model(data.x, data.edge_index)

            node_pred.extend(node_x)
            edge_pred.extend(edge_x)
            node_true.extend(data.y)
            edge_true.extend(data.edge_label)

        # Stack the results:
        node_pred = torch.stack(node_pred, axis=0)
        edge_pred = torch.stack(edge_pred, axis=0)
        node_true = torch.stack(node_true, axis=0)
        edge_true = torch.stack(edge_true, axis=0)

        # Calculate & record loss(es):
        loss_node = node_criterion(node_pred, node_true)
        loss_edge = edge_criterion(edge_pred, edge_true)
        loss = loss_node + loss_edge

        metric_values = {
            "loss": (float(loss_node), float(loss_edge), float(loss))
        }

        # Pre-process the predictions for metric calculations to
        # ensure all metrics fns receive the same values / dtypes:
        node_pred = node_pred.argmax(dim=-1).long()
        edge_pred = edge_pred.argmax(dim=-1).long()

        # Calculate the metrics:
        for m in metrics:
            if isinstance(m, str):
                m_call = get_metric(m)
                m_name = m
            else:
                m_call = m
                m_name = m.__name__

            m_node, m_edge = m_call(node_pred, edge_pred, node_true, edge_true)

            metric_values[m_name] = (m_node, m_edge)

        return metric_values

    # Iterate over all epochs:
    for epoch in range(1, epochs + 1):
        # Computes loss, backprop grads, update params:
        train(train_loader)

        # Get the current learning rate from the optimizer
        current_lr = optimizer.param_groups[0]["lr"]

        # Call the scheduler step after each epoch
        if scheduler_type != "none":
            scheduler.step()

        # Loss & metrics on both Dataloaders:
        train_metrics = valid(train_loader)
        valid_metrics = valid(valid_loader)

        # Log the loss & metrics data:
        logger_string = f"Epoch: {epoch:03d} | "
        logger_string += f"Learning rate: {current_lr} | "
        logger_string += f"Scheduler type: {scheduler_type} | "

        for metric in train_metrics:
            logger_string += "\n\t"

            for regime, metric_dict in [
                ("train", train_metrics),
                ("valid", valid_metrics),
            ]:
                node_value, edge_value = metric_dict[metric][:2]

                metric_name = f"{metric}/{regime}"
                metric_out = {
                    "node": node_value,
                    "edge": edge_value,
                }

                # Combine node & edge losses:
                if len(metric_dict[metric]) == 3:
                    metric_out["total"] = metric_dict[metric][2]

                # Record floating point values (loss & numerical metrics):
                if isinstance(node_value, float):
                    logger_string += (
                        f"{metric_name} (node): " f"{node_value:.4f} | "
                    )
                    logger_string += (
                        f"{metric_name} (edge): " f"{edge_value:.4f} | "
                    )

                    if epoch % tensorboard_update_frequency == 0:
                        writer.add_scalar(
                            f"{metric_name} (node)", metric_out["node"], epoch
                        )
                        writer.add_scalar(
                            f"{metric_name} (edge)", metric_out["edge"], epoch
                        )

                # Upload a figure to Tensorboard (confusion matrix):
                else:
                    if epoch % tensorboard_update_frequency == 0:
                        writer.add_figure(
                            f"{metric_name} (node)", metric_out["node"], epoch
                        )
                        writer.add_figure(
                            f"{metric_name} (edge)", metric_out["edge"], epoch
                        )

        # Print out the logging string:
        LOGGER.info(logger_string)

        # Choose whether to visualise the valudation progress or not:
        if valid_target_list is not None:
            # At chosen epochs, visualise the prediction probabs for whole graph:
            if epoch % valid_graph_ploter_frequency == 0:
                # Instantiate the model with frozen weights from current epoch:
                GLP = GraphLabelPredictor(model)

                # Iterate through all validation graphs & predict nodes / edges:
                for valid_target in valid_target_list:
                    valid_graph = valid_target["graph"]

                    # Filename:
                    valid_name = valid_target["metadata"]["image_filename"]
                    valid_name = f"{valid_name}-Epoch_{epoch}"

                    # Update probabs & visualise the graph:
                    GLP.set_node_and_edge_probabilities(G=valid_graph)
                    GLP.visualise_prediction_probs_on_graph(
                        G=valid_graph,
                        graph_filename=valid_name,
                        save_figure=log_dir / "valid",
                        show_figure=False,
                    )

    # Clear & close the tensorboard writer:
    writer.flush()
    writer.close()
