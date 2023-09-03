from typing import List, Union, Optional, Callable

import random
import torch
import torch_geometric

import matplotlib.pyplot as plt

from torch_geometric.loader import DataLoader

from grace.base import Annotation
from grace.utils.metrics import get_metric

from torch.utils.tensorboard import SummaryWriter


def train_model(
    model: torch.nn.Module,
    dataset: List[torch_geometric.data.Data],
    *,
    epochs: int = 100,
    batch_size: int = 64,
    train_fraction: float = 0.85,
    learning_rate: float = 0.001,
    node_masked_class: Annotation = Annotation.UNKNOWN,
    edge_masked_class: Annotation = Annotation.UNKNOWN,
    log_dir: Optional[str] = None,
    metrics: List[Union[str, Callable]] = [],
    tensorboard_update_frequency: int = 1,
):
    """Train the pytorch model.

    Parameters
    ----------
    model : torch.nn.Module
        Model to train
    dataset : List[torch_geometric.data.Data]
        Training and validation data
    epochs : int
        Number of epochs to train the model
    batch_size : int
        Batch size
    train_fraction : float
        Fraction of data to be used for training (rest for validation)
    learning_rate : float
        Learning rate to use during training
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
    random.shuffle(dataset)

    # Split the datasets:
    train_dataset = dataset[: round(train_fraction * len(dataset))]
    valid_dataset = dataset[round(train_fraction * len(dataset)) :]

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        # weight_decay=5e-4),
    )

    node_criterion = torch.nn.CrossEntropyLoss(
        ignore_index=node_masked_class, reduction="mean"
    )
    edge_criterion = torch.nn.CrossEntropyLoss(
        ignore_index=edge_masked_class, reduction="mean"
    )

    def train(loader):
        model.train()

        for data in loader:
            _, _, node_x, edge_x = model(data.x, data.edge_index)

            loss_node = node_criterion(node_x, data.y)
            loss_edge = edge_criterion(edge_x, data.edge_label)
            loss = loss_node + loss_edge

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    def valid(loader):
        """Evaluates the GCN on node classification."""
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

        # Pre-process the predictions for metric calculations:
        # TODO

        # Calculate the metrics:
        for m in metrics:
            if isinstance(m, str):
                m_call = get_metric(m)
                m_name = m
            # else:
            #     m_call = m
            #     m_name = m.__name__

            m_node, m_edge = m_call(node_pred, edge_pred, node_true, edge_true)

            metric_values[m_name] = (m_node, m_edge)

        return metric_values

    for epoch in range(1, epochs + 1):
        train(train_loader)
        train_metrics = valid(train_loader)
        valid_metrics = valid(valid_loader)

        print_string = f"Epoch: {epoch:03d} | "

        for metric in train_metrics:
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

                if len(metric_dict[metric]) == 3:
                    metric_out["total"] = metric_dict[metric][2]

                if isinstance(node_value, float):
                    print_string += (
                        f"{metric_name} (node): " f"{node_value:.4f} | "
                    )
                    print_string += (
                        f"{metric_name} (edge): " f"{edge_value:.4f} | "
                    )

                    if epoch % tensorboard_update_frequency == 0:
                        writer.add_scalars(metric_name, metric_out, epoch)

                elif isinstance(node_value, plt.Figure):
                    if epoch % tensorboard_update_frequency == 0:
                        writer.add_figure(
                            f"{metric_name} (node)", metric_out["node"], epoch
                        )
                        writer.add_figure(
                            f"{metric_name} (edge)", metric_out["edge"], epoch
                        )

        # Output the epoch stats:
        print(print_string)

    writer.flush()
    writer.close()