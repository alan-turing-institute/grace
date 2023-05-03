from typing import List, Dict, Tuple, Optional

import torch
import torch_geometric
from torch_geometric.loader import DataLoader

from grace.base import Annotation
from torch.utils.tensorboard import SummaryWriter

def train_model(
    model: torch.nn.Module,
    dataset: List[torch_geometric.data.Data],
    *,
    epochs: int = 100,
    batch_size: int = 64,
    val_fraction: float = 0.7,
    node_masked_class: Annotation = Annotation.UNKNOWN,
    edge_masked_class: Annotation = Annotation.UNKNOWN,
    log_dir: Optional[str] = None,
):
    """Train the pytorch model.
    
    Parameters
    ----------
    model : torch.nn.Module
        Model to train
    dataset : List[torch_geometric.data.Data]
        Training and validation data
    epochs: int
        Number of epochs to train the model
    batch_size: int
        Batch size
    val_fraction: float
        Fraction of data to be used for validation
    node_masked_class: Annotation
        Target node class for which to set the loss to 0
    edge_masked_class: Annotation
        Target edge class for which to set the loss to 0
    log_dir: str or None
        Log folder for the current training run
    """
    writer = SummaryWriter(log_dir)

    train_dataset = dataset[: round(val_fraction * len(dataset))]
    test_dataset = dataset[round(val_fraction * len(dataset)) :]

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01,
        # weight_decay=5e-4),
    )

    node_criterion = torch.nn.CrossEntropyLoss(
        ignore_index=node_masked_class, reduction="mean"
    )
    edge_criterion = torch.nn.CrossEntropyLoss(
        ignore_index=edge_masked_class, reduction="mean"
    )

    def calculate_batch_metrics(data: torch_geometric.data.Data) -> Tuple[torch.Tensor]:
        node_x, edge_x = model(data.x, data.edge_index, data.batch)

        loss_node = node_criterion(node_x, data.y)
        loss_edge = edge_criterion(edge_x[0], data.edge_label)
        loss = loss_node + loss_edge

        pred_node = node_x.argmax(dim=-1)
        pred_edge = edge_x.argmax(dim=-1)
        correct_nodes = (pred_node == data.y).sum()
        correct_edges = (pred_edge == data.edge_label).sum()

        return {
            "loss": loss,
            "loss_node": loss_node,
            "loss_edge": loss_edge,
            "correct_nodes": correct_nodes,
            "correct_edges": correct_edges,
            "num_nodes": len(node_x),
            "num_edges": torch.numel(data.edge_label),
        }
    
    def process_epoch_metrics(epoch_metrics: Dict[str, torch.Tensor]):
        return {
            "loss": epoch_metrics["loss"],
            "loss_node": epoch_metrics["loss_node"],
            "loss_edge": epoch_metrics["loss_edge"],
            "acc_node": epoch_metrics["correct_nodes"] / epoch_metrics["num_nodes"],
            "acc_edge": epoch_metrics["correct_edges"] / epoch_metrics["num_edges"],
        }

    def train(loader):
        model.train()

        epoch_metrics = {}

        for data in loader:
            metrics = calculate_batch_metrics(data)

            metrics["loss"].backward()
            optimizer.step()
            optimizer.zero_grad()

            if epoch_metrics:
                for key in epoch_metrics:
                    epoch_metrics[key] += metrics[key]
            else:
                epoch_metrics = metrics

        return process_epoch_metrics(epoch_metrics)
    
    def test(loader):
        """Evaluates the GCN on node classification."""
        model.eval()

        epoch_metrics = {}

        for data in loader:
            metrics = calculate_batch_metrics(data)

            if epoch_metrics:
                for key in epoch_metrics:
                    epoch_metrics[key] += metrics[key]
            else:
                epoch_metrics = metrics

        return process_epoch_metrics(epoch_metrics)

    for epoch in range(1, epochs):
        train_metrics = train(train_loader)
        test_metrics = test(test_loader)
        print(
            f"Epoch: {epoch:03d} | Loss: {train_metrics['loss']:.4f} |"
            f" Node Loss: {train_metrics['loss_node']:.4f} | Edge Loss: {train_metrics['loss_edge']:.4f} |"
            f" Acc (Node): {train_metrics['acc_node']:.4f} |"
            f" Acc (Edge): {train_metrics['acc_edge']:.4f} |"
            f" Val Acc (Node): {test_metrics['acc_node']:.4f} |"
            f" Val Acc (Edge): {test_metrics['acc_edge']:.4f}"
        )

        writer.add_scalars('Loss/train', {'total': train_metrics["loss"],
                                          'node': train_metrics["loss_node"],
                                          'edge': train_metrics["loss_edge"],}, 
                                          epoch)
        writer.add_scalars('Loss/test', {'total': test_metrics["loss"],
                                          'node': test_metrics["loss_node"],
                                          'edge': test_metrics["loss_edge"],}, 
                                          epoch)
        writer.add_scalars('Accuracy/train', {'node': train_metrics["acc_node"],
                                            'edge': train_metrics["acc_edge"],}, 
                                            epoch)
        writer.add_scalars('Accuracy/test', {'node': test_metrics["acc_node"],
                                            'edge': test_metrics["acc_edge"],}, 
                                            epoch)

    writer.flush()
    writer.close()