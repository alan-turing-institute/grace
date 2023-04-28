from typing import List

import torch
import torch_geometric
from torch_geometric.loader import DataLoader

import torch.nn.functional as F

from grace.base import Annotation


def train_model(
    model: torch.nn.Module,
    dataset: List[torch_geometric.data.Data],
    *,
    epochs: int = 100,
    batch_size: int = 64,
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

    node_criterion = torch.nn.CrossEntropyLoss(ignore_index=Annotation.UNKNOWN)

    def edge_criterion(pred, target, edge_index):
        src, dst = edge_index
        edge_score = (pred[src] * pred[dst]).sum(dim=-1)

        return F.cross_entropy(edge_score, target, ignore_index=Annotation.UNKNOWN)

    def train():
        model.train()

        for data in train_loader:
            out_x, out_embedding = model(data.x, data.edge_index, data.batch)
            loss_node = node_criterion(out_x, data.y)
            loss_edge = edge_criterion(
                out_embedding, data.edge_label, data.edge_index
            )

            loss = loss_node + loss_edge

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    def test(loader):
        """Evaluates the GCN on node classification."""
        model.eval()

        correct_nodes = 0
        correct_edges = 0
        num_edges = 0

        for data in loader:

            out_x, out_embedding = model(data.x, data.edge_index, data.batch)

            # node
            pred = out_x.argmax(
                dim=1
            )  # Use the class with highest probability.
            correct_nodes += int((pred == data.y).sum())

            # edge
            src, dst = data.edge_index
            edge_score = (out_embedding[src] * out_embedding[dst]).sum(dim=-1)
            num_edges += edge_score.size(dim=0)
            correct_edges += int((pred == data.y).sum())

        return correct_nodes / len(loader.dataset)

    for epoch in range(1, epochs):
        train()
        train_acc = test(train_loader)
        test_acc = test(test_loader)
        print(
            f"Epoch: {epoch:03d}, Train Acc: {train_acc:.4f},"
            f" Test Acc: {test_acc:.4f}"
        )
