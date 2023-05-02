from typing import List

import torch
import torch_geometric
from torch_geometric.loader import DataLoader

import torch.nn.functional as F

from grace.base import Annotation


def edge_criterion(
    embed: torch.Tensor,
    target: torch.Tensor,
    edge_index: torch.Tensor,
    masked_class: Annotation,
) -> torch.Tensor:
    src, dst = edge_index
    edge_score = (embed[src] * embed[dst]).sum(dim=-1)  # (num_edges,)

    mask = torch.where(target != masked_class, True, False)

    return F.cross_entropy(edge_score[mask], target[mask])


def train_model(
    model: torch.nn.Module,
    dataset: List[torch_geometric.data.Data],
    *,
    epochs: int = 100,
    batch_size: int = 64,
    masked_class: Annotation = Annotation.UNKNOWN,
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
        ignore_index=Annotation.UNKNOWN, reduction='mean'
    )

    def train():
        model.train()

        for data in train_loader:
            out_x, out_embedding = model(data.x, data.edge_index, data.batch)
            loss_node = node_criterion(out_x, data.y)
            loss_edge = edge_criterion(
                out_embedding, data.edge_label, data.edge_index, masked_class
            )

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
            correct_edges += int((torch.round(pred) == data.edge_label).sum())

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
