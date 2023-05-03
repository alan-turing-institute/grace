from typing import List

import torch
import torch_geometric
from torch_geometric.loader import DataLoader

from grace.base import Annotation


def train_model(
    model: torch.nn.Module,
    dataset: List[torch_geometric.data.Data],
    *,
    epochs: int = 100,
    batch_size: int = 64,
    node_masked_class: Annotation = Annotation.UNKNOWN,
    edge_masked_class: Annotation = Annotation.UNKNOWN,
):
    """Train the pytorch model."""
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
        ignore_index=node_masked_class, reduction="mean"
    )
    edge_criterion = torch.nn.CrossEntropyLoss(
        ignore_index=edge_masked_class, reduction="mean"
    )

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

        return loss, loss_node, loss_edge

    def test(loader):
        """Evaluates the GCN on node classification."""
        model.eval()

        correct_nodes = 0
        correct_edges = 0
        num_edges = 0

        for data in loader:
            node_x, edge_x = model(data.x, data.edge_index, data.batch)
            num_edges += edge_x.size(-2)

            # node
            pred_node = node_x.argmax(
                dim=-1
            )  # Use the class with highest probability.
            correct_nodes += int((pred_node == data.y).sum())

            # edge
            pred_edge = edge_x.argmax(
                dim=-1
            )  # Use the class with highest probability.
            correct_edges += int((pred_edge == data.edge_label).sum())

        return correct_nodes / len(loader.dataset), correct_edges / num_edges

    for epoch in range(1, epochs):
        loss, loss_node, loss_edge = train()
        train_acc_node, train_acc_edge = test(train_loader)
        test_acc_node, test_acc_edge = test(test_loader)
        print(
            f"Epoch: {epoch:03d}, Loss: {loss:.4f},"
            f" Node Loss: {loss_node:.4f}, Edge Loss: {loss_edge:.4f},"
            f" Train Acc (Node): {train_acc_node:.4f},"
            f" Train Acc (Edge): {train_acc_edge:.4f},"
            f" Test Acc (Node): {test_acc_node:.4f},"
            f" Test Acc (Edge): {test_acc_edge:.4f}"
        )
