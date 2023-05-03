from typing import List, Tuple

import torch
import torch_geometric
from torch_geometric.loader import DataLoader

from grace.io.core import Annotation
from torch.utils.tensorboard import SummaryWriter

def train_model(
    model: torch.nn.Module,
    dataset: List[torch_geometric.data.Data],
    *,
    epochs: int = 100,
    batch_size: int = 64,
    masked_class: Annotation = Annotation.UNKNOWN,
):
    """Train the pytorch model."""
    writer = SummaryWriter()

    train_dataset = dataset[: round(0.7 * len(dataset))]
    test_dataset = dataset[round(0.7 * len(dataset)) :]

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
        ignore_index=Annotation.UNKNOWN, reduction="mean"
    )
    edge_criterion = torch.nn.CrossEntropyLoss(
        ignore_index=Annotation.UNKNOWN, reduction="mean"
    )

    def calculate_batch_metrics(data: torch_geometric.data.Data) -> Tuple[torch.Tensor]:
        node_x, edge_x = model(data.x, data.edge_index, data.batch)

        loss_node = node_criterion(node_x, data.y)
        loss_edge = edge_criterion(edge_x, data.edge_label)
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

    def train():
        model.train()

        for data in train_loader:
            node_x, edge_x = model(data.x, data.edge_index, data.batch)

            loss_node = node_criterion(node_x, data.y)
            loss_edge = edge_criterion(edge_x, data.edge_label)
            loss = loss_node + loss_edge

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    
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

        return {
            "loss": epoch_metrics["loss"],
            "loss_node": epoch_metrics["loss_node"],
            "loss_edge": epoch_metrics["loss_edge"],
            "acc_node": epoch_metrics["correct_nodes"] / epoch_metrics["num_nodes"],
            "acc_edge": epoch_metrics["correct_edges"] / epoch_metrics["num_edges"],
        }

    for epoch in range(1, epochs):
        train()
        train_metrics = test(train_loader)
        test_metrics = test(test_loader)
        '''print(
            f"Epoch: {epoch:03d}, Loss: {loss:.4f},"
            f" Node Loss: {loss_node:.4f}, Edge Loss: {loss_edge:.4f},"
            f" Train Acc (Node): {train_acc_node:.4f},"
            f" Train Acc (Edge): {train_acc_edge:.4f},"
            f" Test Acc (Node): {test_acc_node:.4f},"
            f" Test Acc (Edge): {test_acc_edge:.4f}"
        )'''

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
